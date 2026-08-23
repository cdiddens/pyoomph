from __future__ import annotations
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

import math

from ..generic import Equations
from .generic import EnforcedBC
from ..expressions import *  # Import grad et al
from ..expressions.coordsys import AxisymmetricCoordinateSystem, AxisymmetryBreakingCoordinateSystem, CartesianCoordinateSystem
from ..expressions.tensor_funcs import (DiagonalizeSymmetricTensor, LogConfTensorDecompositionAxisymmetric,
                                        LogConfTensorDecompositionCartesian2d, SymmetricMatrixExponential)

from ..typings import *


##############################################################################
# Constitutive models
##############################################################################

# All models here are written in terms of the conformation tensor C (dimensionless,
# C=identity at equilibrium). The convention throughout this file is
#
#     upper_convected(C) = -g(C)/relaxation_time
#     polymer_stress     = polymer_viscosity/relaxation_time * h(C)
#
# with the model supplying g and h. For Oldroyd-B both are C-identity, for the other
# models they differ (Giesekus and PTT modify only the relaxation, FENE-P only the
# stress definition, FENE-CR both).
#
# Under Psi=log(C) the relaxation term of the conformation equation turns into C^-1*g(C), which is
# the third method below. Every model here has a C^-1*g(C) that is a plain linear combination of
# the identity, C and C^-1 with tr(C)-dependent coefficients -- no matrix products -- which is
# what lets the log-conformation formulation avoid an eigenvalue decomposition in its relaxation
# term. That matters for more than speed: see ViscoelasticEquations._log_conformation_residual for
# why the decomposition must be kept out of the terms that carry the Jacobian's diagonal.
def _exp(x: ExpressionOrNum) -> ExpressionOrNum:
    # The models are evaluated symbolically when assembling residuals, but also on plain floats
    # (the tests integrate the same model definitions in numpy as an independent reference).
    return exp(x) if isinstance(x, Expression) else math.exp(x)


def _root(x: ExpressionOrNum, order: int = 2) -> ExpressionOrNum:
    return square_root(x, order) if isinstance(x, Expression) else x ** (1.0 / order)


def _shear_thinning_root(u: ExpressionOrNum) -> ExpressionOrNum:
    r"""
    The unique real root :math:`Y\ge1` of :math:`Y^3-Y^2=2u` for :math:`u\ge0`.

    Both trace-dependent models whose steady shear solution is not already explicit end up here:
    linear PTT with :math:`u=\epsilon\mathrm{Wi}^2` and FENE-P with :math:`Y=f/a`, see their
    ``steady_shear_coefficients``.

    The discriminant of that cubic is negative for every u>0 and zero only at u=0, so there is
    exactly one real root and Cardano's formula can be used with real cube roots throughout -- no
    branch selection and no trigonometric form. Written in terms of s rather than as the textbook
    sum of two cube roots because the two Cardano radicands multiply to a constant, so the second
    is 1/s; that also makes the value exactly 1 at u=0 rather than 1+O(1e-16), which matters
    because u=0 is not a corner case but the symmetry line of every channel.
    """
    s = _root(1 + 27 * u + _root(54 * u + 729 * u ** 2), 3)
    return (1 + s + 1 / s) / 3


class ViscoelasticConstitutiveModel:
    """
    Base class for the differential constitutive models used by :py:class:`ViscoelasticEquations`.

    Subclasses supply the relaxation function g, its log-conformation counterpart
    :math:`\\mathbf{C}^{-1}g(\\mathbf{C})`, and the stress function h.
    """

    #: tr(identity), i.e. the trace of C at equilibrium. The conformation tensor is always treated
    #: as a full 3x3 object here, so this is 3 in both the planar and the axisymmetric case.
    equilibrium_trace: int = 3

    #: Whether the out-of-plane component of C is a genuine unknown in a planar (2d Cartesian)
    #: flow. It is not for most models, because their relaxation function vanishes at an eigenvalue
    #: of 1, so C_zz=1 solves its own equation identically. See FENE_P for the exception.
    requires_out_of_plane_component: bool = False

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        """g(C), for the plain conformation formulation."""
        raise NotImplementedError("Implement relaxation_matrix in " + self.__class__.__name__)

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        """C^-1*g(C), for the log-conformation formulation."""
        raise NotImplementedError("Implement log_relaxation_matrix in " + self.__class__.__name__)

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        """h(C), i.e. the polymer stress up to the factor polymer_viscosity/relaxation_time."""
        raise NotImplementedError("Implement stress_matrix in " + self.__class__.__name__)

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum) -> tuple[ExpressionOrNum, ExpressionOrNum, ExpressionOrNum, ExpressionOrNum] | None:
        r"""
        The steady viscometric solution of the model, i.e. the conformation tensor in simple shear.

        Rather than the tensor itself, this returns the four coefficients of

        .. math::

            \mathbf{C} = c_\mathrm{I}\mathbf{I} + c_\mathrm{s}\lambda\left(\mathbf{L}+\mathbf{L}^t\right)
                + c_\mathrm{f}\lambda^2\mathbf{L}\mathbf{L}^t + c_\mathrm{g}\lambda^2\mathbf{L}^t\mathbf{L}

        where :math:`\mathbf{L}=\nabla\vec{u}` is the (nilpotent) velocity gradient of the shear
        flow. That form is frame-free -- it does not have to be told which direction the flow is in
        and which one the gradient is in -- which is what lets
        :py:class:`ViscoelasticInflowBC` build the condition from the inflow profile alone, in
        planar as well as in axisymmetric coordinates. In the canonical frame (flow along x,
        gradient along y) it reads
        :math:`C_{xx}=c_\mathrm{I}+c_\mathrm{f}\mathrm{Wi}^2`, :math:`C_{xy}=c_\mathrm{s}\mathrm{Wi}`,
        :math:`C_{yy}=c_\mathrm{I}+c_\mathrm{g}\mathrm{Wi}^2` and :math:`C_{zz}=c_\mathrm{I}`.

        The argument is :math:`\mathrm{Wi}^2`, not :math:`\mathrm{Wi}`: every one of the four
        coefficients is an even function of the shear rate, and passing the square is what keeps
        them free of a 0/0 at zero shear rate -- which is where a channel's symmetry line sits, so
        it is the ordinary case rather than a corner one.

        Returns None if the model has no closed-form viscometric solution, in which case
        :py:class:`ViscoelasticInflowBC` falls back to enforcing the constitutive equation itself
        on the inflow boundary.
        """
        return None


class OldroydB(ViscoelasticConstitutiveModel):
    r"""
    Oldroyd-B (equivalently: the upper-convected Maxwell model plus a solvent viscosity):

    .. math::

        \stackrel{\triangledown}{\mathbf{C}} = -\frac{1}{\lambda}\left(\mathbf{C}-\mathbf{I}\right),
        \qquad
        \boldsymbol{\tau}_\mathrm{p} = \frac{\eta_\mathrm{p}}{\lambda}\left(\mathbf{C}-\mathbf{I}\right)

    Note that Oldroyd-B has an unbounded extension: in a steady planar elongational flow with
    rate :math:`\dot\varepsilon` the conformation tensor diverges at :math:`\lambda\dot\varepsilon=1/2`.
    Use one of the FENE models if that matters.
    """

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return C - identity

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return identity - Cinv

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return C - identity

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum):
        # C = [[1+2Wi^2, Wi], [Wi, 1]], the textbook viscometric solution.
        return 1, 1, 2, 0


class Giesekus(ViscoelasticConstitutiveModel):
    r"""
    Giesekus model with anisotropy (mobility) parameter :math:`\alpha`:

    .. math::

        \stackrel{\triangledown}{\mathbf{C}} = -\frac{1}{\lambda}\left[
            \left(\mathbf{C}-\mathbf{I}\right) + \alpha\left(\mathbf{C}-\mathbf{I}\right)^2\right]

    The polymer stress is the same as for Oldroyd-B. Unlike Oldroyd-B, this gives shear thinning
    and a bounded extensional viscosity for :math:`\alpha>0`. Physically :math:`0\le\alpha\le 1/2`.

    Args:
        alpha: the mobility parameter. alpha=0 reduces the model to Oldroyd-B.
    """

    def __init__(self, alpha: ExpressionOrNum = 0.1):
        self.alpha = alpha

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        CmI = C - identity
        return CmI + self.alpha * matproduct(CmI, CmI)

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        # C^-1[(C-I) + alpha(C-I)^2] = (I - C^-1) + alpha(C - 2I + C^-1), i.e. no matrix product.
        return (identity - Cinv) + self.alpha * (C - 2 * identity + Cinv)

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return C - identity

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum):
        # Giesekus' own viscometric solution, in the notation of Bird, Armstrong & Hassager. The
        # quadratic relaxation term couples the gradient direction to the shear stress, so this is
        # the only model here that stretches C along that direction at all and hence the only one
        # with a nonzero fourth coefficient.
        alpha = self.alpha
        S = _root(1 + 16 * alpha * (1 - alpha) * weissenberg_squared)
        # Lambda^2, written as 2/(1+S) rather than as the usual (S-1)/(8*alpha*(1-alpha)*Wi^2):
        # the two are the same number, but the latter is 0/0 both at zero shear rate and at
        # alpha=0, and neither is a case that may be excluded here.
        Lam = _root(2 / (1 + S))
        # chi divided by alpha*Wi^2, to which it is proportional. Same reason: alpha=0 is the
        # Oldroyd-B limit of the model and must come out of these formulas rather than blow up.
        chi_reduced = 16 * (1 - alpha) / ((S + 1) ** 2 * (1 + Lam) * (1 + (1 - 2 * alpha) * Lam))
        chi = alpha * weissenberg_squared * chi_reduced
        # The reduced shear viscosity, which is C_xy/Wi. Giesekus shear thins, so this is below 1.
        eta = (1 - chi) ** 2 / (1 + (1 - 2 * alpha) * chi)
        c_gradient = -2 * alpha * eta ** 2 / (1 + _root(1 - 4 * alpha ** 2 * weissenberg_squared * eta ** 2))
        # C_xx - C_yy is the first normal stress difference of the model, hence the c_gradient term.
        c_flow = c_gradient + 2 * chi_reduced * (1 - alpha * chi) / (1 - chi)
        return 1, eta, c_flow, c_gradient


class PTT(ViscoelasticConstitutiveModel):
    r"""
    Phan-Thien-Tanner model, i.e. Oldroyd-B with a trace-dependent relaxation rate

    .. math::

        \stackrel{\triangledown}{\mathbf{C}} = -\frac{Y(\operatorname{tr}\mathbf{C})}{\lambda}
            \left(\mathbf{C}-\mathbf{I}\right)

    with :math:`Y=1+\epsilon\left(\operatorname{tr}\mathbf{C}-3\right)` (linear PTT) or
    :math:`Y=\exp\left[\epsilon\left(\operatorname{tr}\mathbf{C}-3\right)\right]` (exponential PTT).

    Only the affine (:math:`\xi=0`) variant is implemented. A nonzero slip parameter replaces the
    upper-convected derivative by the Gordon-Schowalter one, which invalidates the
    Fattal-Kupferman decomposition used here, so it is rejected rather than silently ignored.

    Args:
        epsilon: extensibility parameter. epsilon=0 reduces the model to Oldroyd-B.
        kind: "linear" or "exponential".
        xi: slip parameter, must be 0.
    """

    def __init__(self, epsilon: ExpressionOrNum = 0.02, kind: Literal["linear", "exponential"] = "linear", xi: ExpressionOrNum = 0):
        if kind not in {"linear", "exponential"}:
            raise ValueError("PTT argument 'kind' must be either 'linear' or 'exponential', got " + str(kind))
        if not (isinstance(xi, (int, float)) and xi == 0):
            raise NotImplementedError(
                "Only the affine PTT model (xi=0) is implemented. A nonzero slip parameter uses the "
                "Gordon-Schowalter derivative, for which the log-conformation decomposition in "
                "pyoomph.expressions.tensor_funcs does not apply.")
        self.epsilon = epsilon
        self.kind: Literal["linear", "exponential"] = kind

    def _Y(self, trace: ExpressionOrNum) -> ExpressionOrNum:
        arg = self.epsilon * (trace - self.equilibrium_trace)
        return 1 + arg if self.kind == "linear" else _exp(arg)

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._Y(trace) * (C - identity)

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._Y(trace) * (identity - Cinv)

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return C - identity

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum):
        # In shear the PTT solution is Oldroyd-B with Wi replaced by Wi/Y, and tr(C)-3 = 2Wi^2/Y^2
        # closes the model: Y^3-Y^2 = 2*epsilon*Wi^2 for the linear kind.
        if self.kind != "linear":
            # Exponential PTT gives Y = exp(2*epsilon*Wi^2/Y^2), whose solution is a Lambert W and
            # not an elementary expression. ViscoelasticInflowBC enforces the constitutive equation
            # on the boundary instead.
            return None
        Y = _shear_thinning_root(self.epsilon * weissenberg_squared)
        return 1, 1 / Y, 2 / Y ** 2, 0


class FENE_CR(ViscoelasticConstitutiveModel):
    r"""
    FENE-CR (Chilcott-Rallison) model with finite extensibility :math:`L`:

    .. math::

        f = \frac{L^2-3}{L^2-\operatorname{tr}\mathbf{C}},\qquad
        \stackrel{\triangledown}{\mathbf{C}} = -\frac{f}{\lambda}\left(\mathbf{C}-\mathbf{I}\right),\qquad
        \boldsymbol{\tau}_\mathrm{p} = \frac{\eta_\mathrm{p}}{\lambda}f\left(\mathbf{C}-\mathbf{I}\right)

    :math:`f` is normalised to 1 at equilibrium, which is what makes the shear viscosity of this
    model constant (hence "CR"), unlike FENE-P. The trace is bounded by :math:`L^2`.

    Args:
        L: the extensibility, i.e. the maximum chain extension. Must exceed sqrt(3).
    """

    def __init__(self, L: ExpressionOrNum = 5):
        self.L = L

    def _f(self, trace: ExpressionOrNum) -> ExpressionOrNum:
        L2 = self.L ** 2
        return (L2 - self.equilibrium_trace) / (L2 - trace)

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._f(trace) * (C - identity)

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._f(trace) * (identity - Cinv)

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._f(trace) * (C - identity)

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum):
        # As for PTT, shear gives Oldroyd-B with Wi/f, and tr(C)-3 = 2Wi^2/f^2 closes it -- but here
        # that is only a quadratic, f^2-f = 2Wi^2/(L^2-3), because f depends on the trace through
        # (L^2-trC)^-1 rather than linearly. Hence the closed form where PTT needs a cubic.
        f = (1 + _root(1 + 8 * weissenberg_squared / (self.L ** 2 - 3))) / 2
        return 1, 1 / f, 2 / f ** 2, 0


class FENE_P(ViscoelasticConstitutiveModel):
    r"""
    FENE-P model with finite extensibility :math:`L`:

    .. math::

        f = \frac{1}{1-\operatorname{tr}\mathbf{C}/L^2},\quad a = \frac{1}{1-3/L^2},\qquad
        \stackrel{\triangledown}{\mathbf{C}} = -\frac{1}{\lambda}\left(f\mathbf{C}-a\mathbf{I}\right),\qquad
        \boldsymbol{\tau}_\mathrm{p} = \frac{\eta_\mathrm{p}}{\lambda}\left(f\mathbf{C}-a\mathbf{I}\right)

    The constant :math:`a=f(3)` is what puts the equilibrium at C=identity; without it the model
    would relax towards a multiple of the identity instead. FENE-P shear thins, FENE-CR does not.

    This is the one model here whose out-of-plane component is not slaved in a planar flow: its
    relaxation function does not vanish at an eigenvalue of 1 unless the flow is at equilibrium,
    because :math:`f` and :math:`a` differ as soon as :math:`\operatorname{tr}\mathbf{C}\ne3`. So
    :math:`C_{zz}` is solved for as an extra unknown, see ``solve_out_of_plane_component`` of
    :py:class:`ViscoelasticEquations`.

    Args:
        L: the extensibility, i.e. the maximum chain extension. Must exceed sqrt(3).
    """

    requires_out_of_plane_component: bool = True

    def __init__(self, L: ExpressionOrNum = 5):
        self.L = L

    def _f(self, trace: ExpressionOrNum) -> ExpressionOrNum:
        return 1 / (1 - trace / self.L ** 2)

    def _a(self) -> ExpressionOrNum:
        return 1 / (1 - self.equilibrium_trace / self.L ** 2)

    def relaxation_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._f(trace) * C - self._a() * identity

    def log_relaxation_matrix(self, C: Expression, Cinv: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self._f(trace) * identity - self._a() * Cinv

    def stress_matrix(self, C: Expression, trace: ExpressionOrNum, identity: Expression) -> Expression:
        return self.relaxation_matrix(C, trace, identity)

    def steady_shear_coefficients(self, weissenberg_squared: ExpressionOrNum):
        # f*C - a*I = 0 in every component that the shear does not drive, so C_yy = C_zz = a/f is
        # pulled below 1 here, unlike in any other model. Substituting the resulting trace back into
        # f gives (f/a)^3 - (f/a)^2 = 2*Wi^2/(a^3*(L^2-3)).
        a = self._a()
        ratio = _shear_thinning_root(weissenberg_squared / (a ** 3 * (self.L ** 2 - 3)))
        f = a * ratio
        return a / f, a / f ** 2, 2 * a / f ** 3, 0


##############################################################################
# The equation class
##############################################################################

class ViscoelasticEquations(Equations):
    r"""
    .. _ViscoelasticEquations:

    Evolution of the conformation tensor of a differential viscoelastic constitutive model,
    together with the polymer contribution to the momentum equation. Add it alongside
    :ref:`NavierStokesEquations <NavierStokesEquations>` (or :ref:`StokesEquations <StokesEquations>`),
    whose ``dynamic_viscosity`` then plays the role of the *solvent* viscosity:

    .. code-block:: python

        eqs = NavierStokesEquations(dynamic_viscosity=eta_s, mass_density=rho)
        eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=lam, polymer_viscosity=eta_p)

    By default the equations are solved in the log-conformation representation of
    Fattal and Kupferman, i.e. the unknown is :math:`\boldsymbol{\Psi}=\log\mathbf{C}` rather than
    :math:`\mathbf{C}` itself. This keeps :math:`\mathbf{C}` positive definite by construction and
    is what allows the high Weissenberg numbers at which the plain conformation formulation
    loses convergence. The evolution equation reads

    .. math::

        \partial_t\boldsymbol{\Psi} + \left(\vec{u}\cdot\nabla\right)\boldsymbol{\Psi}
        - \left(\boldsymbol{\Omega}\boldsymbol{\Psi}-\boldsymbol{\Psi}\boldsymbol{\Omega}\right)
        - 2\mathbf{B}
        + \frac{1}{\lambda}\mathbf{C}^{-1}g(\mathbf{C}) = 0

    where :math:`\boldsymbol{\Omega}` and :math:`\mathbf{B}` are the rotational and extensional
    parts of :math:`\nabla\vec{u}` in the eigenframe of :math:`\mathbf{C}`, and the constitutive
    model supplies :math:`g`. Setting ``formulation="conformation"`` instead solves the
    conventional equation for :math:`\mathbf{C}`, which is useful as a cross-check at low
    Weissenberg numbers, where both must agree.

    Only 2d Cartesian and axisymmetric coordinate systems are supported, since the eigenvalue
    decompositions with analytic derivatives in :py:mod:`pyoomph.expressions.tensor_funcs` are
    only implemented for these.

    In the 2d Cartesian case the flow is planar, so for most models :math:`C_{zz}=1` solves its own
    equation identically and is not made an unknown; it is nevertheless part of
    :math:`\operatorname{tr}\mathbf{C}`, which matters for the trace-dependent models (PTT, FENE).
    :py:class:`FENE_P` is the exception and gets :math:`C_{zz}` as an extra unknown, see
    ``solve_out_of_plane_component``.

    Args:
        model: the constitutive model, e.g. :py:class:`OldroydB` or :py:class:`Giesekus`.
        relaxation_time: the relaxation time :math:`\lambda` of the polymer.
        polymer_viscosity: the polymer viscosity :math:`\eta_\mathrm{p}`. The solvent viscosity is the ``dynamic_viscosity`` of the (Navier-)Stokes equations.
        formulation: "log-conf" (default) or "conformation".
        space: the finite element space of the conformation (or log-conformation) components.
        field_name: the name of the tensor field. Defaults to "log_conformation" or "conformation" depending on the formulation.
        velocity_name: the name of the velocity field to advect with and to add the polymer stress to.
        wind: the velocity that transports and deforms the conformation tensor. Defaults to ``var(velocity_name)``. It supplies both the advection and the velocity gradient, since a convected derivative is defined with respect to a single velocity field.
        add_polymer_stress_to_momentum: whether to add the polymer stress to the momentum equation. Disable to drive the constitutive equation by an imposed velocity field.
        solve_out_of_plane_component: whether to solve for the out-of-plane component in a planar flow. "auto" (default) leaves that to the constitutive model. Has no effect in axisymmetric coordinates, where the azimuthal component is always an unknown.
        stabilization: None (default) for plain Galerkin, or "SUPG" for streamline-upwind Petrov-Galerkin on the constitutive equation. The latter is residual-based and hence consistent, so it does not change a converged solution, but it damps the oscillations that pure advection with no diffusion produces on an under-resolved mesh.
        supg_factor: multiplies the SUPG intrinsic time tau. 1 is the standard choice; lower it to stabilise more weakly.

    .. warning::

        A *stationary* solve started from the rest state can fail with SUPG switched on at full
        strength, while the same solve converges without it. The cause is not the stabilisation: at
        rest the decomposition of :math:`\nabla\vec{u}` runs entirely through its degenerate branch,
        whose reported Jacobian is truncated (see the note above), and the SUPG term multiplies that
        same residual by :math:`\tau\left(\vec{u}\cdot\nabla w\right)`, amplifying the existing
        Jacobian error. Ramping ``supg_factor`` from 0 up to 1 on an already-converged solution walks
        past it without trouble, and changes the answer by O(1e-6) once there.
        eigen_epsilon: eigenvalues of :math:`\mathbf{C}` closer than this are considered degenerate, see the note on the initial condition below.
        use_FD: let the eigenvalue decompositions fill their Jacobian by finite differences instead of the analytic expressions.
        use_subexpression: wrap the decomposition results in subexpressions. Considerably reduces the size of the generated code.
        time_scheme: an optional time stepping scheme for the constitutive equation.
        output_conformation: add the conformation tensor components, its trace and the polymer stress as output fields.
        spatial_error_estimators: add the conformation components to the spatial error estimator.

    .. note::

        At rest :math:`\mathbf{C}=\mathbf{I}`, so all eigenvalues coincide and the first Newton step
        runs entirely through the degenerate branch of the decomposition of :math:`\nabla\vec{u}`.
        That branch returns the correct residual (:math:`\mathbf{B}=\operatorname{sym}\nabla\vec{u}`,
        :math:`\boldsymbol{\Omega}=0`) but a truncated Jacobian, which costs Newton iterations
        without affecting the converged solution. Everything that carries the diagonal of the
        Jacobian is deliberately kept out of that decomposition, so a stationary solve straight from
        the rest state does converge -- see ``_in_plane_exponential``.
    """

    def __init__(self, *, model: ViscoelasticConstitutiveModel | None = None,
                 relaxation_time: ExpressionOrNum = 1, polymer_viscosity: ExpressionOrNum = 1,
                 formulation: Literal["log-conf", "conformation"] = "log-conf",
                 space: "FiniteElementSpaceEnum" = "C2", field_name: str | None = None,
                 velocity_name: str = "velocity", wind: ExpressionNumOrNone = None,
                 add_polymer_stress_to_momentum: bool = True,
                 solve_out_of_plane_component: Literal["auto"] | bool = "auto",
                 stabilization: Literal["SUPG"] | None = None, supg_factor: ExpressionOrNum = 1,
                 eigen_epsilon: float = 1e-7, use_FD: bool | float = False, use_subexpression: bool = True,
                 time_scheme: TimeSteppingScheme | None = None,
                 output_conformation: bool = True,
                 spatial_error_estimators: bool = False):
        super().__init__()
        if formulation not in {"log-conf", "conformation"}:
            raise ValueError("Viscoelastic equations argument 'formulation' must be either "
                             "'log-conf' or 'conformation', got " + str(formulation))
        self.model: ViscoelasticConstitutiveModel = model if model is not None else OldroydB()
        self.relaxation_time = relaxation_time
        self.polymer_viscosity = polymer_viscosity
        self.formulation: Literal["log-conf", "conformation"] = formulation
        self.space: "FiniteElementSpaceEnum" = space
        self.field_name = field_name if field_name is not None else ("log_conformation" if self.formulation == "log-conf" else "conformation")
        self.velocity_name = velocity_name
        self.wind = wind
        self.add_polymer_stress_to_momentum = add_polymer_stress_to_momentum
        self.solve_out_of_plane_component: Literal["auto"] | bool = solve_out_of_plane_component
        if stabilization not in {None, "SUPG"}:
            raise ValueError("Viscoelastic equations argument 'stabilization' must be None or "
                             "'SUPG', got " + str(stabilization))
        self.stabilization: Literal["SUPG"] | None = stabilization
        self.supg_factor = supg_factor
        self.eigen_epsilon = eigen_epsilon
        self.use_FD = use_FD
        self.use_subexpression = use_subexpression
        self.time_scheme: TimeSteppingScheme | None = time_scheme
        self.output_conformation = output_conformation
        self.spatial_error_estimators = spatial_error_estimators

    # ---------------------------------------------------------------- helpers

    # Everything below is built as a full 3x3 tensor, because that is what pyoomph's grad() of a
    # vector field produces in every coordinate system (vector() always pads to three components
    # and matrix() to 3x3). Working at 3x3 also makes the planar case come out right for free:
    # the third row and column of Psi are zero, so C=exp(Psi) has C_zz=1 and tr(C) already
    # contains the out-of-plane contribution that the trace-dependent models need.
    def _is_axisymmetric(self) -> bool:
        cs = self.get_coordinate_system()
        if isinstance(cs, AxisymmetricCoordinateSystem):
            if isinstance(cs, AxisymmetryBreakingCoordinateSystem):
                raise RuntimeError("The viscoelastic equations are not implemented for the coordinate system " + str(cs))
            return True
        elif isinstance(cs, CartesianCoordinateSystem):
            if self.get_nodal_dimension() != 2:
                raise RuntimeError("The viscoelastic equations are only implemented for 2d Cartesian coordinates, "
                                   "but the nodal dimension is " + str(self.get_nodal_dimension()) + ". The eigenvalue "
                                   "decompositions in pyoomph.expressions.tensor_funcs would have to be extended first.")
            return False
        else:
            raise RuntimeError("The viscoelastic equations are not implemented for the coordinate system " + str(cs))

    # Name of the out-of-plane component: a genuine unknown in axisymmetric coordinates, and in
    # planar flow only for the models that need it (currently only FENE-P).
    def _out_of_plane_component(self) -> str | None:
        if self._is_axisymmetric():
            return "aa"
        if self.solve_out_of_plane_component == "auto":
            return "zz" if self.model.requires_out_of_plane_component else None
        return "zz" if self.solve_out_of_plane_component else None

    # (row, column) -> component suffix. Entries that are absent are identically zero, which for
    # the (2,2) entry is exactly right: Psi_zz=0 means C_zz=1.
    def _component_map(self) -> dict[tuple[int, int], str]:
        comps = {(0, 0): "xx", (0, 1): "xy", (1, 0): "xy", (1, 1): "yy"}
        oop = self._out_of_plane_component()
        if oop is not None:
            comps[(2, 2)] = oop
        return comps

    # Build a 3x3 matrix out of a per-component callback. absent_diagonal is what an unsolved
    # diagonal entry stands for: 0 for the log-conformation tensor and its time derivative
    # (Psi_zz=0), but 1 for the conformation tensor itself (C_zz=1).
    def _assemble(self, entry: Callable[[str], ExpressionOrNum], absent_diagonal: ExpressionOrNum = 0) -> Expression:
        comps = self._component_map()

        def get(i: int, j: int) -> ExpressionOrNum:
            if (i, j) in comps:
                return entry(comps[(i, j)])
            return absent_diagonal if i == j else 0

        return matrix([[get(i, j) for j in range(3)] for i in range(3)])

    def _diagonal(self, entries: Sequence[ExpressionOrNum]) -> Expression:
        return matrix([[entries[i] if i == j else 0 for j in range(3)] for i in range(3)])

    def _se(self, M: Expression) -> Expression:
        return subexpression(M) if self.use_subexpression else M

    # ---------------------------------------------------------------- fields

    def define_fields(self):
        # The conformation tensor is dimensionless in either formulation, so its scale is 1. That
        # is also required by the eigenvalue decomposition, whose degeneracy threshold
        # eigen_epsilon is an absolute one.
        name = self.field_name
        testscale = scale_factor("spatial") / scale_factor(self.velocity_name)
        self.define_tensor_field(name, self.space, symmetric=True, scale=1, testscale=testscale)
        oop = self._out_of_plane_component()
        if not self._is_axisymmetric() and oop is not None:
            # define_tensor_field only covers the in-plane block in Cartesian coordinates, so the
            # out-of-plane component is added by hand, with the same scales.
            self.define_scalar_field(name + "_" + oop, self.space)
            self.set_scaling({name + "_" + oop: 1})
            self.set_test_scaling({name + "_" + oop: testscale})
        # The tensor is written to the output as a single VTK tensor array, off the component grid
        # define_tensor_field built. That grid stops at the in-plane block in Cartesian coordinates,
        # so the out-of-plane slot has to be filled in or the output shows a zero there. A zero is
        # right for the log-conformation tensor, whose absent component means Psi_zz=0, but not for
        # the conformation tensor, where it means C_zz=1 -- a zero on that diagonal reads as a
        # collapsed configuration. The axisymmetric grid already carries its azimuthal entry.
        mst = self._master()
        assert isinstance(mst, Equations)
        grid = [list(row) + [""] * (3 - len(row)) for row in mst._tensorfields[name]]
        grid += [["", "", ""] for _ in range(3 - len(grid))]
        if not grid[2][2]:
            if oop is not None:
                grid[2][2] = name + "_" + oop
            elif self.formulation == "conformation":
                self.add_local_function(name + "_zz", 1)
                grid[2][2] = name + "_zz"
        mst._tensorfields[name] = grid
        if self.formulation == "conformation":
            for comp in set(self._component_map().values()):
                # C=identity at rest. In the log-conformation formulation the corresponding
                # Psi=0 is already the default, so nothing has to be set there.
                self.set_initial_condition(name + "_" + comp, 0 if comp == "xy" else 1)
        self.define_field_by_substitution("polymer_stress", self.get_polymer_stress())

    def define_scaling(self):
        if self.spatial_error_estimators:
            for comp in sorted(set(self._component_map().values())):
                self.add_spatial_error_estimator(grad(var(self.field_name + "_" + comp)))

    # ------------------------------------------------------------- residuals

    def define_residuals(self):
        name = self.field_name
        identity = identity_matrix(3)

        # An unsolved out-of-plane entry means Psi_zz=0 in the log-conformation formulation and
        # C_zz=1 in the conformation one; both are the rest state of that component.
        absent = 0 if self.formulation == "log-conf" else 1
        field = self._assemble(lambda c: var(name + "_" + c), absent_diagonal=absent)
        field_test = self._assemble(lambda c: testfunction(name + "_" + c))

        wind = self.wind if self.wind is not None else var(self.velocity_name)
        gradu = grad(wind)

        if self.formulation == "log-conf":
            # dt(Psi) + (u.grad)Psi. The rest of the equation is the Fattal-Kupferman decomposition
            # of grad(u), which is not an upper-convected derivative, so only the transport part
            # comes from the library here.
            transported = material_derivative(field, wind, ALE="auto", dt_scheme=self.time_scheme)
            residual, C, trC = self._log_conformation_residual(field, gradu, transported, identity)
        else:
            # dt(C) + (u.grad)C - grad(u)*C - C*grad(u)^t, exactly what upper_convected_derivative
            # gives with pyoomph's grad(u)[i,j] = d(u_i)/d(x_j) convention.
            transported = upper_convected_derivative(field, wind, ALE="auto", dt_scheme=self.time_scheme)
            residual, C, trC = self._conformation_residual(field, transported, identity)

        if self.time_scheme is not None:
            residual = time_scheme(self.time_scheme, residual)
        self.add_residual(weak(residual, field_test))

        if self.stabilization == "SUPG":
            # Streamline-upwind Petrov-Galerkin. The constitutive equation has no diffusion at all,
            # so plain Galerkin is unstabilised advection and oscillates once the element Peclet
            # number gets large; this perturbs the test function along the streamline instead.
            #
            # It is residual-based, i.e. the perturbation multiplies the *strong* residual, which
            # vanishes at the exact solution. The scheme is therefore consistent: it changes the
            # discrete solution but not the equation being solved, and switching it on must not move
            # a converged answer.
            #
            # "cartesian_element_length_h" rather than "element_length_h": the latter takes the
            # Eulerian element size to the power 1/dim, and in axisymmetric coordinates that size
            # carries the 2*pi*r factor, so it is not a length.
            h = var("cartesian_element_length_h")
            # tau -> h/(2|u|) where advection dominates, and -> relaxation_time where the flow is
            # slow. The second branch is what keeps it finite at stagnation points and on no-slip
            # walls, where h/(2|u|) alone would blow up; the relaxation time is the natural time
            # scale to fall back on, since 1/lambda is the reaction rate of this equation.
            #
            # Written with dot(u,u) under the one square root rather than as (2*|u|/h)^2, which is
            # the same number but not the same expression: squaring square_root(dot(u,u)) leaves the
            # inner root in place for GiNaC to differentiate, and d|u|/du = u/|u| is 0/0 at rest,
            # where u vanishes over most of the domain. That put NaNs in the Jacobian and the first
            # Newton step went to 1e105. Here the radicand is bounded below by (1/lambda)^2, so the
            # only root has a strictly positive argument and a bounded derivative everywhere.
            tau = self.supg_factor / square_root(4 * dot(wind, wind) / h ** 2
                                                 + (1 / self.relaxation_time) ** 2)
            tau=subexpression(tau) if self.use_subexpression else tau
            supg_test = self._assemble(lambda c: tau * dot(wind, grad(testfunction(name + "_" + c))))
            self.add_residual(weak(residual, supg_test))

        # Polymer contribution to the momentum equation. The (Navier-)Stokes equations assemble
        # their momentum residual as weak(stress, grad(v)), with the stress containing -p*identity,
        # so the polymer stress is simply added with the same sign.
        stress = self._polymer_stress(C, trC, identity)
        if self.add_polymer_stress_to_momentum:
            u_test = testfunction(self.velocity_name)
            if self.time_scheme is not None:
                stress = time_scheme(self.time_scheme, stress)
            self.add_residual(weak(stress, grad(u_test)))

        if self.output_conformation:
            # Registered as whole tensors, not component by component: only then do they land in
            # _tensorfields and get written as single tensor arrays (rather than loose scalars) to
            # the vtu. The component names follow from the matrix, so in axisymmetry the azimuthal
            # entry is now called "_zz" rather than "_aa" -- which is where it belongs in the output,
            # the azimuthal direction being the Cartesian z of the r-z plane.
            if self.formulation == "log-conf":
                # In the conformation formulation these are the unknowns themselves and are output
                # anyway; adding them again would clash with the field names.
                self.add_local_function("conformation", C)
            self.add_local_function("polymer_stress", stress)
            self.add_local_function("conformation_trace", trC)
            # The first normal stress difference, the quantity most rheological measurements report.
            self.add_local_function("polymer_N1", stress[0, 0] - stress[1, 1])

    # Plain conformation formulation: upper_convected_derivative(C, u) + g(C)/lambda = 0.
    def _conformation_residual(self, C: Expression, transported: Expression, identity: Expression):
        trC = trace(C)
        relax = self.model.relaxation_matrix(C, trC, identity) / self.relaxation_time
        return transported + relax, C, trC

    # Exponential of the in-plane block of a symmetric tensor, as a 2x2 list of entries. This uses
    # SymmetricMatrixExponential rather than the eigendecomposition on purpose, and that choice is
    # what makes a steady solve from rest possible at all:
    #
    # DiagonalizeSymmetricTensor takes a shortcut whenever the off-diagonal entry is tiny -- it
    # returns R=identity and, critically, a Jacobian in which every derivative of R is zero and the
    # eigenvalues are taken to be independent of the off-diagonal entry. At rest Psi=0, so the
    # relaxation and stress terms built from that decomposition end up with no dependence on Psi_xy
    # at all, and the Psi_xy row and column of the Jacobian are empty. The residual is still right,
    # so a transient run survives (its time derivative fills the diagonal), but a stationary solve
    # hits an exactly singular matrix.
    #
    # SymmetricMatrixExponential has a correct Jacobian in its own degenerate branch, so taking
    # C and C^-1 from it keeps the diagonal populated. The eigendecomposition is then only used to
    # feed the Fattal-Kupferman decomposition, which does not use R at all when the eigenvalues are
    # degenerate.
    def _in_plane_exponential(self, block: Expression) -> Expression:
        # The coordinate system is passed as Cartesian even in the axisymmetric case: this is the
        # exponential of a plain 2x2 symmetric block, which is the same operation either way, and
        # SymmetricMatrixExponential has no axisymmetric implementation.
        exponential = SymmetricMatrixExponential(CartesianCoordinateSystem(), 2, scale=1,
                                                 fill_to_max_vector_dim=False, use_FD=self.use_FD,
                                                 use_subexpression=self.use_subexpression)
        return exponential(block)

    # Log-conformation formulation of Fattal & Kupferman:
    #   dt(Psi) + (u.grad)Psi - (Omega*Psi - Psi*Omega) - 2B + C^-1 g(C)/lambda = 0
    # where Omega and B are the rotational and extensional parts of grad(u) in the eigenframe of C.
    # C and C^-1 from the log-conformation tensor. Only the in-plane block needs the matrix
    # exponential; the third direction is always an eigendirection here (Psi has no out-of-plane
    # shear components), so it is a scalar exponential.
    # Returns C, C^-1 and the out-of-plane eigenvalue C_zz, the last one because the decomposition
    # of grad(u) needs the eigenvalues of C and only the in-plane pair comes out of the
    # eigendecomposition.
    def _conformation_from_log(self, psi: Expression) -> tuple[Expression, Expression, ExpressionOrNum]:
        in_plane = matrix([[psi[0, 0], psi[0, 1]], [psi[1, 0], psi[1, 1]]], fill_to_max_vector_dim=False)
        expC = self._in_plane_exponential(in_plane)
        expCinv = self._in_plane_exponential(-in_plane)
        third, third_inverse = self._se(exp(psi[2, 2])), self._se(exp(-psi[2, 2]))
        C = self._se(matrix([[expC[0, 0], expC[0, 1], 0], [expC[1, 0], expC[1, 1], 0], [0, 0, third]]))
        Cinv = self._se(matrix([[expCinv[0, 0], expCinv[0, 1], 0], [expCinv[1, 0], expCinv[1, 1], 0], [0, 0, third_inverse]]))
        return C, Cinv, third

    def _log_conformation_residual(self, psi: Expression, gradu: Expression, transported: Expression, identity: Expression):
        axisym = self._is_axisymmetric()

        C, Cinv, third = self._conformation_from_log(psi)
        trC = trace(C)

        # The eigenframe, needed only for the decomposition of grad(u). scale=1 on purpose: Psi is
        # dimensionless, and the degeneracy threshold inside the decomposition is an absolute one,
        # so any other scale would silently move it.
        diagonalize = DiagonalizeSymmetricTensor(self.get_coordinate_system(), 2, scale=1,
                                                 fill_to_max_vector_dim=False, use_FD=self.use_FD)
        R2, eigPsi = diagonalize(psi)
        if axisym:
            # The axisymmetric branch already returns a full 3x3 rotation, with R[2,2]=1.
            R = self._se(R2)
            eigenvalues:list[ExpressionOrNum] = [exp(eigPsi[i, i]) for i in range(3)]
        else:
            # In Cartesian coordinates only the in-plane block is diagonalized. The third
            # eigendirection has to be put back by hand as R[2,2]=1, not the zero that
            # fill_to_max_vector_dim would leave there.
            R = self._se(matrix([[R2[0, 0], R2[0, 1], 0], [R2[1, 0], R2[1, 1], 0], [0, 0, 1]]))
            eigenvalues = [exp(eigPsi[0, 0]), exp(eigPsi[1, 1]), third]

        decomposition:CustomMultiReturnExpression
        if axisym:
            decomposition = LogConfTensorDecompositionAxisymmetric(epsilon=self.eigen_epsilon, use_FD=self.use_FD,
                                                                   use_subexpression=self.use_subexpression)
        else:
            decomposition = LogConfTensorDecompositionCartesian2d(epsilon=self.eigen_epsilon,
                                                                  use_subexpression=self.use_subexpression)
        B, Omega = decomposition(R, gradu, self._diagonal(eigenvalues))

        relax = self.model.log_relaxation_matrix(C, Cinv, trC, identity) / self.relaxation_time
        rotation = matproduct(Omega, psi) - matproduct(psi, Omega)
        return transported - rotation - 2 * B + relax, C, trC

    def _polymer_stress(self, C: Expression, trC: ExpressionOrNum, identity: Expression) -> Expression:
        return self.polymer_viscosity / self.relaxation_time * self.model.stress_matrix(C, trC, identity)

    # ------------------------------------------------------------ public API

    def get_conformation_tensor(self, domain: str | None = None) -> Expression:
        r"""
        The conformation tensor :math:`\mathbf{C}` as an expression, whichever formulation is in use.

        Pass ``domain=".."`` to build it from the bulk fields while on an interface, which is what a
        traction integral on a wall needs -- see :py:meth:`get_polymer_stress`.
        """
        absent = 0 if self.formulation == "log-conf" else 1
        field = self._assemble(lambda c: var(self.field_name + "_" + c, domain=domain),
                               absent_diagonal=absent)
        if self.formulation == "log-conf":
            return self._conformation_from_log(field)[0]
        return field

    def get_polymer_stress(self, domain: str | None = None) -> Expression:
        r"""
        The polymer stress :math:`\boldsymbol{\tau}_\mathrm{p}` as an expression.

        The total stress of the fluid is this plus the solvent and pressure contributions of the
        (Navier-)Stokes equations, so a drag or traction integral over a wall reads

        .. code-block:: python

            traction = matproduct(-p*identity_matrix(3) + 2*eta_s*sym(grad(u)) + tau_p, var("normal"))

        with ``tau_p = viscoelastic_equations.get_polymer_stress(domain="..")`` and ``u``, ``p``
        likewise taken from the bulk. Note that ``var("normal")`` points out of the fluid domain, so
        the force exerted *on* the wall is minus that traction.
        """
        C = self.get_conformation_tensor(domain=domain)
        return self._polymer_stress(C, trace(C), identity_matrix(3))


##############################################################################
# Boundary conditions
##############################################################################

class ViscoelasticInflowBC(EnforcedBC):
    r"""
    .. _ViscoelasticInflowBC:

    The fully developed conformation tensor on an inflow boundary, for the constitutive model of the
    :ref:`ViscoelasticEquations <ViscoelasticEquations>` of the adjacent bulk domain:

    .. code-block:: python

        inflow = 1.5*(1 - var("coordinate_y")**2/4)
        eqs += DirichletBC(velocity=vector(inflow, 0)) @ "inlet"
        eqs += ViscoelasticInflowBC(vector(inflow, 0)) @ "inlet"

    Prescribing only the velocity at an inlet leaves the polymer unstretched there, so the solution
    spends the whole upstream length relaxing towards the profile it should have had on entry. This
    imposes that profile instead. The argument is the same inflow velocity that the
    :py:class:`~pyoomph.equations.generic.DirichletBC` above uses; the velocity gradient is obtained from it
    by differentiating symbolically with respect to the coordinates. A fully developed profile is a
    function of the transverse coordinate alone, so that gives the shear rate and nothing else. Pass
    ``velocity_gradient`` instead to give :math:`\nabla\vec{u}` directly.

    Both the planar and the axisymmetric case are covered, and in either one it is the inflow
    profile that says which direction the flow is in -- an axisymmetric pipe inlet is
    ``ViscoelasticInflowBC(vector(0, w))`` with ``w`` a function of ``var("coordinate_x")``.

    There are two ways of imposing the condition, selected by ``mode``:

    * ``"dirichlet"`` pins the components of the (log-)conformation tensor to the model's
      viscometric solution, which :py:meth:`ViscoelasticConstitutiveModel.steady_shear_coefficients`
      supplies in closed form. This costs no additional degrees of freedom and is what
      ``mode="auto"`` uses whenever the model has such a solution. It assumes the inflow to be
      unidirectional, i.e. an actual shear flow.
    * ``"enforced"`` instead demands, through a field of Lagrange multipliers on the boundary, that
      the constitutive equation itself hold there with the imposed velocity gradient and no
      variation in the flow direction. That is the same condition, but stated rather than solved,
      so it needs no closed form and is not restricted to a unidirectional profile. It is what
      ``mode="auto"`` falls back to -- for the exponential :py:class:`PTT`, whose viscometric
      solution is a Lambert W function, and for any model of your own that does not implement
      ``steady_shear_coefficients``.

    The two are not equally sharp. The Dirichlet mode is exact at the nodes; the enforced one holds
    the condition only in the weak sense, since the constraint is integrated against the Lagrange
    multiplier's test functions and is not a polynomial, so it carries a discretisation error of the
    order of the finite element space (measured at O(h^3) on a C2 space). It is also a genuine
    unknown of the problem rather than a pinned value, and therefore rather less forgiving of a cold
    start: a stationary solve from rest at a large Weissenberg number may need continuation where
    the Dirichlet mode would converge straight away. Where a model offers both, prefer the default.

    Note that this does not touch the velocity itself: add the usual
    :py:class:`~pyoomph.equations.generic.DirichletBC` for that, with the same profile.

    Args:
        velocity: the imposed inflow velocity profile, as a vector expression.
        velocity_gradient: the velocity gradient, as a 3x3 matrix, instead of ``velocity``.
        mode: "auto" (default), "dirichlet" or "enforced", see above.
        space: the finite element space of the Lagrange multipliers of the enforced mode. Defaults to the space of the conformation tensor.
    """

    required_parent_type = ViscoelasticEquations

    def __init__(self, velocity: Expression | None = None, *,
                 velocity_gradient: Expression | None = None,
                 mode: Literal["auto", "dirichlet", "enforced"] = "auto",
                 space: FiniteElementSpaceEnum | None = None):
        super().__init__(space=space)
        if (velocity is None) == (velocity_gradient is None):
            raise ValueError("ViscoelasticInflowBC must be given either the inflow velocity profile "
                             "or its gradient, but not both")
        if mode not in {"auto", "dirichlet", "enforced"}:
            raise ValueError("ViscoelasticInflowBC argument 'mode' must be 'auto', 'dirichlet' or "
                             "'enforced', got " + str(mode))
        self.velocity = velocity
        self.velocity_gradient = velocity_gradient
        self.mode: Literal["auto", "dirichlet", "enforced"] = mode

    def _viscoelastic(self) -> ViscoelasticEquations:
        equations = self.get_parent_equations()
        if not isinstance(equations, ViscoelasticEquations):
            raise RuntimeError("ViscoelasticInflowBC must be attached to a boundary of a domain "
                               "with exactly one set of ViscoelasticEquations")
        return equations

    def _gradient(self) -> Expression:
        if self.velocity_gradient is not None:
            return self.velocity_gradient
        velocity = self.velocity
        assert velocity is not None
        # Differentiated symbolically with respect to the coordinates, not with grad(). grad() would
        # resolve the derivative through the shape functions of the boundary element, and the result
        # is then no longer a pointwise expression, which a strong Dirichlet value has to be.
        # Symbolic differentiation also has nothing to say about the flow direction: the profile is
        # a function of the transverse coordinate alone, so its streamwise derivative comes out as
        # zero by itself, which is exactly the fully developed assumption.
        def derivative(component: int, coordinate: str) -> ExpressionOrNum:
            return symbolic_diff(velocity[component], coordinate, hold_until_codegen=False)

        entries: list[list[ExpressionOrNum]] = [
            [derivative(i, "coordinate_x"), derivative(i, "coordinate_y"), 0] for i in range(2)]
        entries.append([0, 0, 0])
        coordinate_system = self.get_coordinate_system()
        if isinstance(coordinate_system, AxisymmetricCoordinateSystem):
            # The azimuthal entry of the axisymmetric velocity gradient, u_r/r. It vanishes for the
            # unidirectional inflow the closed-form mode assumes, but not for a general one.
            radial = 1 if coordinate_system.use_x_as_symmetry_axis else 0
            radius = var("coordinate_y" if coordinate_system.use_x_as_symmetry_axis else "coordinate_x")
            entries[2][2] = velocity[radial] / radius
        return matrix(entries)

    # The model's viscometric solution, rebuilt in whatever frame the inflow profile happens to be
    # in, or None if the model has no closed form for it.
    def _closed_form_conformation(self) -> Expression | None:
        equations = self._viscoelastic()
        L = self._gradient()
        Lt = transpose(L)
        relaxation_time = equations.relaxation_time
        # tr(L*L^t) is the squared Frobenius norm of the velocity gradient, i.e. the squared shear
        # rate of a unidirectional flow. Taking the Weissenberg number from it rather than from a
        # named component is what keeps this free of any assumption about the orientation.
        coefficients = equations.model.steady_shear_coefficients(relaxation_time ** 2 * trace(matproduct(L, Lt)))
        if coefficients is None:
            return None
        c_isotropic, c_shear, c_flow, c_gradient = coefficients
        return (c_isotropic * identity_matrix(3) + c_shear * relaxation_time * (L + Lt)
                + c_flow * relaxation_time ** 2 * matproduct(L, Lt)
                + c_gradient * relaxation_time ** 2 * matproduct(Lt, L))

    # (row, column) of each component of the symmetric conformation tensor, mirroring
    # ViscoelasticEquations._component_map.
    _component_indices = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1), "zz": (2, 2), "aa": (2, 2)}

    def _constrained_components(self) -> list[str]:
        # Which of them exist is read off the bulk domain rather than asked of the bulk equations:
        # their _component_map goes through the coordinate system and hence through their own code
        # generator, which is not the current one while an interface condition is being defined.
        name = self._viscoelastic().field_name
        domain = self.get_parent_domain()
        return [component for component in self._component_indices
                if domain.get_space_of_field(name + "_" + component) != ""]

    def _uses_lagrange_multipliers(self) -> bool:
        if self.mode == "dirichlet":
            return False
        return self.mode == "enforced" or self._closed_form_conformation() is None

    def define_fields(self):
        if self._uses_lagrange_multipliers():
            equations = self._viscoelastic()
            # Only the names of the constrained fields are needed to create the Lagrange multipliers;
            # the constraints themselves are assembled in define_residuals, since building the
            # conformation tensor out of the bulk fields belongs there.
            self.constraints = {equations.field_name + "_" + component: Expression(0)
                                for component in self._constrained_components()}
        super().define_fields()

    def define_residuals(self):
        equations = self._viscoelastic()
        name = equations.field_name
        conformation = None if self._uses_lagrange_multipliers() else self._closed_form_conformation()

        if conformation is None:
            if self.mode == "dirichlet":
                raise RuntimeError("The constitutive model " + type(equations.model).__name__ +
                                   " has no closed-form solution for steady shear, so "
                                   "ViscoelasticInflowBC cannot use mode='dirichlet' with it. Use "
                                   "mode='enforced' (or the default 'auto') instead.")
            C = equations.get_conformation_tensor()
            L = self._gradient()
            relaxation = equations.model.relaxation_matrix(C, trace(C), identity_matrix(3))
            # Fully developed means that C does not change along the flow, so its material
            # derivative vanishes and the constitutive equation collapses to the algebraic balance
            # between the upper-convected stretching and the relaxation. Multiplied by the
            # relaxation time, since the constraint has to be dimensionless.
            residual = equations.relaxation_time * (matproduct(L, C) + matproduct(C, transpose(L))) - relaxation
            self.constraints = {name + "_" + component: residual[self._component_indices[component]]
                                for component in self._constrained_components()}
            super().define_residuals()
            return

        if equations.formulation == "log-conf":
            # The out-of-plane direction is an eigendirection of the shear solution, so only the
            # in-plane block needs the matrix logarithm.
            in_plane = symmetric_2x2_matrix_log(conformation)
            values = {"xx": in_plane[0, 0], "xy": in_plane[0, 1], "yy": in_plane[1, 1]}
            out_of_plane = log(conformation[2, 2])
        else:
            values = {"xx": conformation[0, 0], "xy": conformation[0, 1], "yy": conformation[1, 1]}
            out_of_plane = conformation[2, 2]
        for component in self._constrained_components():
            self.set_Dirichlet_condition(name + "_" + component, values.get(component, out_of_plane))


##############################################################################
# Utilities
##############################################################################

def symmetric_2x2_matrix_log(M: Expression, epsilon: float = 1e-30) -> Expression:
    r"""
    Analytic matrix logarithm of a symmetric positive definite 2x2 matrix.

    Uses :math:`\log\mathbf{M}=a\mathbf{I}+b\mathbf{M}` with
    :math:`b=(\log\Lambda_+-\log\Lambda_-)/(\Lambda_+-\Lambda_-)` and
    :math:`a=\log\Lambda_+-b\Lambda_+`, which follows from the fact that any function of a 2x2
    matrix is a linear combination of the identity and the matrix itself (Cayley-Hamilton).
    This is a purely symbolic expression, so it can be used e.g. to prescribe the
    log-conformation tensor on an inflow boundary.

    Degenerate eigenvalues are handled, which matters because the obvious use of this function hits
    them: prescribing the inflow conformation of a channel makes it isotropic on the symmetry line,
    where the shear rate vanishes. The b coefficient is then 0/0. Regularising the denominator sends
    b to 0 there, and that happens to be exactly right rather than merely safe -- for an isotropic
    M = Lambda*I the result collapses to a*I = log(Lambda)*I, which is the correct logarithm. The
    epsilon only degrades the answer when the two eigenvalues are separated by an amount comparable
    to it, which for the default is far below anything representable in the surrounding computation.

    Args:
        M: the matrix, assumed symmetric and positive definite.
        epsilon: regularisation of the eigenvalue difference in the denominator.
    """
    t = M[0, 0] + M[1, 1]
    d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    root = square_root(t ** 2 / 4 - d)
    lp, lm = t / 2 + root, t / 2 - root
    b = (log(lp) - log(lm)) / (lp - lm + epsilon)
    a = log(lp) - b * lp
    return matrix([[a + b * M[0, 0], b * M[0, 1]], [b * M[1, 0], a + b * M[1, 1]]], fill_to_max_vector_dim=False)


def oldroyd_b_shear_conformation(weissenberg: ExpressionOrNum) -> Expression:
    r"""
    The steady conformation tensor of the Oldroyd-B model in simple shear at
    :math:`\mathrm{Wi}=\lambda\dot\gamma`, i.e.

    .. math::

        \mathbf{C}=\begin{pmatrix}1+2\mathrm{Wi}^2 & \mathrm{Wi}\\ \mathrm{Wi} & 1\end{pmatrix}

    Useful to impose a fully developed inflow profile: combine with
    :py:func:`symmetric_2x2_matrix_log` for the log-conformation formulation. The shear direction
    is x, the gradient direction y. See :py:func:`steady_shear_conformation` for the same thing for
    an arbitrary constitutive model, and :py:class:`ViscoelasticInflowBC` for the boundary condition
    that this is usually wanted for.
    """
    return matrix([[1 + 2 * weissenberg ** 2, weissenberg], [weissenberg, 1]], fill_to_max_vector_dim=False)


def steady_shear_conformation(model: ViscoelasticConstitutiveModel, weissenberg: ExpressionOrNum) -> Expression:
    r"""
    The steady conformation tensor of a constitutive model in simple shear at
    :math:`\mathrm{Wi}=\lambda\dot\gamma`, as a full 3x3 matrix, with the flow along x and the
    velocity gradient along y.

    This generalises :py:func:`oldroyd_b_shear_conformation` to the other models. Note that
    :math:`C_{zz}` is 1 for all of them except :py:class:`FENE_P`, whose relaxation does not vanish
    at an eigenvalue of 1, and that :math:`C_{yy}` is 1 for all of them except :py:class:`Giesekus`
    and again FENE-P.

    Args:
        model: the constitutive model.
        weissenberg: the Weissenberg number, i.e. the shear rate times the relaxation time.

    Raises:
        NotImplementedError: if the model has no closed-form viscometric solution, as is the case
            for the exponential :py:class:`PTT`. :py:class:`ViscoelasticInflowBC` can still impose
            the inflow condition there, by enforcing the constitutive equation on the boundary.
    """
    coefficients = model.steady_shear_coefficients(weissenberg ** 2)
    if coefficients is None:
        raise NotImplementedError("The constitutive model " + type(model).__name__ + " does not "
                                  "provide a closed-form solution for steady shear")
    c_isotropic, c_shear, c_flow, c_gradient = coefficients
    return matrix([[c_isotropic + c_flow * weissenberg ** 2, c_shear * weissenberg, 0],
                   [c_shear * weissenberg, c_isotropic + c_gradient * weissenberg ** 2, 0],
                   [0, 0, c_isotropic]])


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
