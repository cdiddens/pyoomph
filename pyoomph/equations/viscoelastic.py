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
        wind: the advection velocity. Defaults to ``var(velocity_name)``.
        add_polymer_stress_to_momentum: whether to add the polymer stress to the momentum equation. Disable to drive the constitutive equation by an imposed velocity field.
        solve_out_of_plane_component: whether to solve for the out-of-plane component in a planar flow. "auto" (default) leaves that to the constitutive model. Has no effect in axisymmetric coordinates, where the azimuthal component is always an unknown.
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

    def _diagonal(self, entries: list[ExpressionOrNum]) -> Expression:
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
        if not self._is_axisymmetric() and self._out_of_plane_component() is not None:
            # define_tensor_field only covers the in-plane block in Cartesian coordinates, so the
            # out-of-plane component is added by hand, with the same scales.
            self.define_scalar_field(name + "_zz", self.space)
            self.set_scaling({name + "_zz": 1})
            self.set_test_scaling({name + "_zz": testscale})
        if self.formulation == "conformation":
            for comp in set(self._component_map().values()):
                # C=identity at rest. In the log-conformation formulation the corresponding
                # Psi=0 is already the default, so nothing has to be set there.
                self.set_initial_condition(name + "_" + comp, 0 if comp == "xy" else 1)

    def define_scaling(self):
        if self.spatial_error_estimators:
            for comp in sorted(set(self._component_map().values())):
                self.add_spatial_error_estimator(grad(var(self.field_name + "_" + comp)))

    # ------------------------------------------------------------- residuals

    def define_residuals(self):
        comps = self._component_map()
        name = self.field_name
        identity = identity_matrix(3)

        # An unsolved out-of-plane entry means Psi_zz=0 in the log-conformation formulation and
        # C_zz=1 in the conformation one; both are the rest state of that component.
        absent = 0 if self.formulation == "log-conf" else 1
        field = self._assemble(lambda c: var(name + "_" + c), absent_diagonal=absent)
        field_test = self._assemble(lambda c: testfunction(name + "_" + c))

        u = var(self.velocity_name)
        wind = self.wind if self.wind is not None else u
        gradu = grad(u)

        # Advection and the time derivative are both built component by component: grad() of a
        # matrix does not give the rank-3 object that would be needed here. Each component is a
        # plain scalar field, for which both operations are well defined.
        advection = self._assemble(lambda c: dot(wind, grad(var(name + "_" + c))))
        dt_field = self._assemble(lambda c: partial_t(var(name + "_" + c)))

        if self.formulation == "log-conf":
            residual, C, trC = self._log_conformation_residual(field, gradu, dt_field, advection, identity)
        else:
            residual, C, trC = self._conformation_residual(field, gradu, dt_field, advection, identity)

        if self.time_scheme is not None:
            residual = time_scheme(self.time_scheme, residual)
        self.add_residual(weak(residual, field_test))

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
            for (i, j), c in sorted(comps.items()):
                if i <= j:
                    # In the conformation formulation these are the unknowns themselves and are
                    # output anyway; adding them again would clash with the field names.
                    if self.formulation == "log-conf":
                        self.add_local_function("conformation_" + c, C[i, j])
                    self.add_local_function("polymer_stress_" + c, stress[i, j])
            self.add_local_function("conformation_trace", trC)
            # The first normal stress difference, the quantity most rheological measurements report.
            self.add_local_function("polymer_N1", stress[0, 0] - stress[1, 1])

    # Plain conformation formulation: the upper-convected derivative written out. With pyoomph's
    # convention grad(u)[i,j]=d(u_i)/d(x_j), the upper-convected derivative of C is
    # dt(C) + (u.grad)C - grad(u)*C - C*grad(u)^t.
    def _conformation_residual(self, C: Expression, gradu: Expression, dt_field: Expression, advection: Expression, identity: Expression):
        trC = trace(C)
        stretch = matproduct(gradu, C) + matproduct(C, transpose(gradu))
        relax = self.model.relaxation_matrix(C, trC, identity) / self.relaxation_time
        residual = dt_field + advection - stretch + relax
        return residual, C, trC

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

    def _log_conformation_residual(self, psi: Expression, gradu: Expression, dt_field: Expression, advection: Expression, identity: Expression):
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
            eigenvalues = [exp(eigPsi[i, i]) for i in range(3)]
        else:
            # In Cartesian coordinates only the in-plane block is diagonalized. The third
            # eigendirection has to be put back by hand as R[2,2]=1, not the zero that
            # fill_to_max_vector_dim would leave there.
            R = self._se(matrix([[R2[0, 0], R2[0, 1], 0], [R2[1, 0], R2[1, 1], 0], [0, 0, 1]]))
            eigenvalues = [exp(eigPsi[0, 0]), exp(eigPsi[1, 1]), third]

        if axisym:
            decomposition = LogConfTensorDecompositionAxisymmetric(epsilon=self.eigen_epsilon, use_FD=self.use_FD,
                                                                   use_subexpression=self.use_subexpression)
        else:
            decomposition = LogConfTensorDecompositionCartesian2d(epsilon=self.eigen_epsilon,
                                                                  use_subexpression=self.use_subexpression)
        B, Omega = decomposition(R, gradu, self._diagonal(eigenvalues))

        relax = self.model.log_relaxation_matrix(C, Cinv, trC, identity) / self.relaxation_time
        rotation = matproduct(Omega, psi) - matproduct(psi, Omega)
        residual = dt_field + advection - rotation - 2 * B + relax
        return residual, C, trC

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
# Utilities
##############################################################################

def symmetric_2x2_matrix_log(M: Expression) -> Expression:
    r"""
    Analytic matrix logarithm of a symmetric positive definite 2x2 matrix.

    Uses :math:`\log\mathbf{M}=a\mathbf{I}+b\mathbf{M}` with
    :math:`b=(\log\Lambda_+-\log\Lambda_-)/(\Lambda_+-\Lambda_-)` and
    :math:`a=\log\Lambda_+-b\Lambda_+`, which follows from the fact that any function of a 2x2
    matrix is a linear combination of the identity and the matrix itself (Cayley-Hamilton).
    This is a purely symbolic expression, so it can be used e.g. to prescribe the
    log-conformation tensor on an inflow boundary. It is singular for degenerate eigenvalues.
    """
    t = M[0, 0] + M[1, 1]
    d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    root = square_root(t ** 2 / 4 - d)
    lp, lm = t / 2 + root, t / 2 - root
    b = (log(lp) - log(lm)) / (lp - lm)
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
    is x, the gradient direction y.
    """
    return matrix([[1 + 2 * weissenberg ** 2, weissenberg], [weissenberg, 1]], fill_to_max_vector_dim=False)
