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
"""
Residual-based stabilization of the Navier-Stokes equations: SUPG, PSPG, LSIC/grad-div, GLS, ASGS
and VMS, together with the inf-sup unstable equal-order velocity/pressure pairs they make usable.

:py:class:`StabilizedNavierStokes` subclasses
:py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations`, so every interface equation of
that module keeps working -- free surface, slip length, contact angle, connect-velocity, the
azimuthal stability machinery -- and the ALE/GCL handling comes for free. Those interface equations
that impose a *traction* additionally compensate the footprint that the stabilization leaves on the
natural boundary condition; see
:py:meth:`~pyoomph.equations.navier_stokes.StokesEquations.get_stabilization_traction` and
:py:attr:`StabilizedNavierStokes.natural_bc_correction`.

All added terms are proportional to the *strong* momentum residual and therefore vanish for the
exact solution: a consistent stabilization must not change the answer, only the conditioning. The
strong residual contains :math:`-\\nabla\\cdot(2\\mu\\mathbf{D}(\\vec{u}))`, which needs second
derivatives of the shape functions; pyoomph has those, so the scheme here is genuinely consistent
rather than dropping the term. On a quadratic velocity space dropping it costs three orders of
magnitude in the pressure error, so it is not an approximation one can make silently.

This module replaces the former ``pyoomph.equations.SUPG``, which added the stabilization
*alongside* the flow equations instead of subclassing them and whose strong residual omitted the
viscous term. Its ``ElementSizeForSUPG`` helper is gone too: the element length scale is available
directly as ``var("cartesian_element_length_h")``.
"""

from .. import *
from ..expressions import *
from .navier_stokes import NavierStokesEquations, StokesEquations
from .stabilization import (element_h as _element_h, inv_dt as _inv_dt, regularized_magnitude,
                            tau_advective_diffusive, _z_tezduyar, _maybe_sub)
from ..typings import *

if TYPE_CHECKING:
    from ..generic.codegen import FiniteElementCodeGenerator


_SPACE_ALIASES:dict[str,str] = {"C2C1": "TH", "C1C1": "C1", "C2C2": "C2", "TH": "TH", "CR": "CR",
                                "C1": "C1", "C2": "C2", "mini": "mini"}

#: Named combinations of the individual stabilization terms accepted by the ``stabilization``
#: argument of :py:class:`StabilizedNavierStokes`.
STABILIZATION_PRESETS:dict[str,set[str]] = {
    "none": set(),
    "SUPG": {"SUPG"},
    "PSPG": {"PSPG"},
    "LSIC": {"LSIC"},
    "SUPG+PSPG": {"SUPG", "PSPG"},
    "SUPGPSPGLSIC": {"SUPG", "PSPG", "LSIC"},
    "GLS": {"SUPG", "PSPG", "LSIC", "GLSVISC"},
    "ASGS": {"SUPG", "PSPG", "LSIC", "ASGSVISC"},
    "VMS": {"SUPG", "PSPG", "LSIC", "REYNOLDS"},
}


class StabilizedNavierStokes(NavierStokesEquations):
    """
    .. _StabilizedNavierStokes:

    Navier-Stokes equations with switchable residual-based stabilization. A subclass of
    :ref:`NavierStokesEquations <NavierStokesEquations>`, so it accepts all of its arguments as well.

    The added terms are, per element interior,

    .. math:: \\text{SUPG}\\quad &\\sum_K (\\tau_M\\,\\vec{a}\\cdot\\nabla\\vec{v},\\; \\vec{R}_M)_K
    .. math:: \\text{PSPG}\\quad &\\sum_K (\\tau_M/\\rho\\; \\nabla q,\\; \\vec{R}_M)_K
    .. math:: \\text{LSIC}\\quad &\\sum_K (\\tau_C\\rho\\,\\nabla\\cdot\\vec{u},\\; \\nabla\\cdot\\vec{v})_K

    with :math:`\\vec{a}=\\vec{u}-\\vec{u}_\\text{mesh}` and the strong momentum residual

    .. math:: \\vec{R}_M = \\rho\\left(\\partial_t\\vec{u} + \\vec{a}\\cdot\\nabla\\vec{u}\\right)
              + \\nabla p - \\nabla\\cdot(2\\mu\\mathbf{D}(\\vec{u})) - \\vec{f} - \\rho\\vec{g}\\,.

    GLS and ASGS add the viscous part of the perturbation operator (with the operator's own sign and
    with the adjoint sign, respectively), VMS adds the fine-scale Reynolds stress.

    Args:
        space: ``"C2C1"``/``"TH"`` (Taylor-Hood, inf-sup stable), ``"C1C1"``/``"C1"`` or
            ``"C2C2"``/``"C2"`` (equal order, inf-sup *unstable* -- these need PSPG or the pressure
            checkerboards and the system becomes effectively singular), also ``"CR"`` and
            ``"mini"``. An alias for the ``mode`` argument of the base class.
        viscous_form: ``"stress"`` uses :math:`2\\mu\\,\\text{sym}(\\nabla\\vec{u})` and hence the
            natural condition :math:`\\vec{n}\\cdot(2\\mu\\mathbf{D}) - p\\vec{n} = \\vec{t}`;
            ``"laplace"`` uses :math:`\\mu\\nabla\\vec{u}` and hence
            :math:`\\mu\\partial_n\\vec{u} - p\\vec{n} = \\vec{t}`. The two are identical in the bulk
            for a divergence-free field and constant viscosity but *not* on a Neumann boundary: with
            the stress form a developed channel profile is not a solution of the traction-free
            outflow, with the Laplace form it is. A free surface must use the stress form, since the
            traction there has to be the true stress.
        stabilization: a key of :py:data:`STABILIZATION_PRESETS` or an iterable of the individual
            flags ``"SUPG"``, ``"PSPG"``, ``"LSIC"``, ``"GLSVISC"``, ``"ASGSVISC"``, ``"REYNOLDS"``.
        tau_formula: ``"shakib"`` (Bazilevs/Codina-Shakib, the inverse square root of the sum of
            squares), ``"codina"`` (inverse of the sum) or ``"tezduyar"``.
        tauC_formula: ``"codina"`` (:math:`h^2/(c_C\\tau_M)`) or ``"tezduyar"``.
        include_viscous_in_residual: keep :math:`-\\nabla\\cdot(2\\mu\\mathbf{D})` in the strong
            residual. Required for consistency on C2 velocities; on C1 velocities the term is
            elementwise zero anyway, so leaving it on costs nothing. Only set this to False to
            reproduce a formulation that has no second derivatives available.
        constant_viscosity: assume :math:`\\nabla\\mu=0` when forming
            :math:`\\nabla\\cdot(2\\mu\\mathbf{D})`. Set to False for a variable viscosity, which
            makes GiNaC differentiate through it.
        transient_tau: include the :math:`1/\\Delta t` term in :math:`\\tau`. ``"auto"`` uses the
            BDF1 weight, which pyoomph zeroes in a steady solve, so the term switches itself off
            there.
        natural_bc_correction: which parts of the stabilization's footprint on the natural boundary
            condition the traction boundary conditions should subtract, so that they impose the
            physical traction rather than the physical traction plus that footprint. ``True`` for
            all of them, ``False`` (the default) for none, or an iterable of ``"SUPG"``, ``"LSIC"``,
            ``"REYNOLDS"``.

            It defaults to off because it is a trade-off rather than a free improvement. Measured on
            C1/C1 Poiseuille with the exact traction prescribed at the outflow (so that the parabola
            is not representable and the footprint is genuinely nonzero), switching the LSIC
            correction on reduces the pressure error *at* that boundary by a uniform 21 % but
            degrades the global pressure convergence from O(h^1.9) to O(h^1.6); the SUPG part
            contributes almost nothing either way. On a static free surface the whole thing is a
            wash. Switch it on when the traction on a particular boundary is the quantity of
            interest -- an imposed load, a measured force, a coupling to another domain -- and leave
            it off when the field in the interior is.
        C_I: coefficient of the viscous term in :math:`\\tau_M`; :math:`\\tau_M\\to h^2/(C_I\\nu)` in
            the Stokes limit. The default 4 is what the textbook formula writes but is on the
            diffusive side; 36 converges measurably better.
        c_t: coefficient of the transient term in :math:`\\tau_M`.
        c_C: coefficient in :math:`\\tau_C = h^2/(c_C\\tau_M)`.
        c_r: coefficient of the linear-drag rate in :math:`\\tau_M`, see
            :py:meth:`StabilizedNavierStokes.linear_drag_rate`.
        velocity_eps: :math:`|\\vec{u}|` is regularized as
            :math:`\\sqrt{\\vec{u}\\cdot\\vec{u}+(\\varepsilon U)^2}` so that :math:`\\tau` stays
            differentiable at rest. Given *relative* to the velocity scale, so that it remains
            meaningful in a dimensional problem.
        stab_factor: global prefactor on both :math:`\\tau`'s, for sensitivity studies.
    """

    # Which intermediate quantities get their own named temporary in the generated C. Measured:
    # tau_M is the one that matters (unwrapping it alone costs 50% of the assembly time -- it is the
    # deepest tree and multiplies three or four separate weak terms, so its derivative would be
    # expanded once per term). Wrapping R_M buys no runtime at all despite appearing just as often,
    # but shrinks the generated C by ~20% on VMS, and |u| is redundant once tau_M is wrapped.
    # Class attributes rather than constructor arguments because this is a code-generation knob.
    _wrap_R = True
    _wrap_tau = True
    _wrap_U = True

    def __init__(self, *,
                 space: str = "C2C1",
                 viscous_form: Literal["stress","laplace"] = "stress",
                 stabilization: "str | Iterable[str]" = "none",
                 tau_formula: Literal["shakib","codina","tezduyar"] = "shakib",
                 tauC_formula: Literal["codina","tezduyar"] = "codina",
                 include_viscous_in_residual: bool = True,
                 constant_viscosity: bool = True,
                 transient_tau: "Literal['auto'] | bool" = "auto",
                 natural_bc_correction: "bool | Iterable[str]" = False,
                 C_I: float = 4.0, c_t: float = 2.0, c_C: float = 4.0, c_r: float = 1.0,
                 velocity_eps: ExpressionOrNum = 1e-10,
                 stab_factor: ExpressionOrNum = 1,
                 **kwargs:Any):
        if space not in _SPACE_ALIASES:
            raise ValueError(f"unknown space '{space}', available: {sorted(_SPACE_ALIASES)}")
        kwargs.setdefault("mode", _SPACE_ALIASES[space])
        super().__init__(**kwargs)
        self.space = space
        self.viscous_form = viscous_form
        self.tau_formula:Literal["shakib","codina","tezduyar"] = tau_formula
        self.tauC_formula = tauC_formula
        self.include_viscous_in_residual = include_viscous_in_residual
        self.constant_viscosity = constant_viscosity
        self.transient_tau:"Literal['auto'] | bool" = transient_tau
        if isinstance(natural_bc_correction, bool):
            self.natural_bc_correction:set[str] = {"SUPG","LSIC","REYNOLDS"} if natural_bc_correction else set()
        else:
            self.natural_bc_correction = set(natural_bc_correction)
        unknown_corr = self.natural_bc_correction - {"SUPG","LSIC","REYNOLDS"}
        if unknown_corr:
            raise ValueError(f"unknown natural_bc_correction terms {sorted(unknown_corr)}")
        self.C_I, self.c_t, self.c_C, self.c_r = C_I, c_t, c_C, c_r
        self.velocity_eps = velocity_eps
        self.stab_factor = stab_factor

        if isinstance(stabilization, str):
            if stabilization not in STABILIZATION_PRESETS:
                raise ValueError(f"unknown stabilization preset '{stabilization}', "
                                 f"available: {sorted(STABILIZATION_PRESETS)}")
            self.stab = set(STABILIZATION_PRESETS[stabilization])
            self.stab_name = stabilization
        else:
            self.stab = set(stabilization)
            self.stab_name = "+".join(sorted(self.stab)) if self.stab else "none"
        unknown = self.stab - {"SUPG","PSPG","LSIC","GLSVISC","ASGSVISC","REYNOLDS"}
        if unknown:
            raise ValueError(f"unknown stabilization terms {sorted(unknown)}")

        if (self.stab & {"GLSVISC", "ASGSVISC"}) and not include_viscous_in_residual:
            # the viscous part of the perturbation operator is second order too: if second
            # derivatives are unavailable for R_M they are unavailable for the test operator as well
            raise ValueError("GLS/ASGS need include_viscous_in_residual=True")
        if viscous_form not in ("stress", "laplace"):
            raise ValueError(f"unknown viscous_form '{viscous_form}'")

    # -- Galerkin part: only the stress tensor differs from the base class ------------------------

    def define_stress_tensor(self)->Expression:
        if self.viscous_form == "stress":
            return super().define_stress_tensor()
        # Laplace form. -p I is kept exactly as the base class writes it, including pressure_factor
        # and pressure_sign_flip, so that only the viscous part changes.
        if self.stress_tensor is not None:
            return self.stress_tensor
        u, p = var(self.velocity_name), var(self.pressure_name)
        return (convert_to_expression(self.dynamic_viscosity) * grad(u)
                - identity_matrix() * self.pressure_factor * p * (-1 if self.pressure_sign_flip else 1))

    # -- building blocks for the strong residual ---------------------------------------------------

    def element_h(self)->Expression:
        """
        Isotropic element length :math:`V^{1/d}`, measured in *Cartesian* space.

        Deliberately ``"cartesian_element_length_h"`` and not ``"element_length_h"``: in an
        axisymmetric problem the latter is the revolved volume, so :math:`\\tau` would grow like
        :math:`r^{1/3}` away from the axis instead of tracking the actual cell size.
        """
        return _element_h()

    def convective_velocity(self)->Expression:
        """
        :math:`\\vec{u}-\\vec{u}_\\text{mesh}`. On a moving (ALE) mesh it is the relative velocity
        that is advected, so it is that one which must set both the streamline direction of SUPG and
        the cell Reynolds number in :math:`\\tau`. Using :math:`\\vec{u}` there is O(1) wrong on a
        fast-moving free surface.
        """
        u = var(self.velocity_name)
        if self.get_combined_equations()._assert_codegen()._coordinates_as_dofs: #type:ignore
            return u - mesh_velocity()
        return u

    def velocity_magnitude(self)->Expression:
        """
        Regularized :math:`|\\vec{u}-\\vec{u}_\\text{mesh}|`, so that :math:`\\tau` stays
        differentiable at rest. The regularization is *relative* to the velocity scale: a bare
        number would be added to :math:`\\vec{u}\\cdot\\vec{u}`, which carries m^2/s^2 in a
        dimensional problem, and pyoomph rejects that outright rather than guessing a scale.
        """
        a = self.convective_velocity()
        eps = self.velocity_eps * scale_factor(self.velocity_name)
        return _maybe_sub(self._wrap_U, regularized_magnitude(a, eps))

    def div_viscous_stress(self)->Expression:
        """
        Divergence of the viscous stress, for the strong residual. Uses second derivatives of the
        shape functions, so it is an element-wise (broken) quantity -- which is exactly what the
        element-wise stabilization integrals ask for.
        """
        u = var(self.velocity_name)
        mu = convert_to_expression(self.dynamic_viscosity)
        if self.constant_viscosity:
            # written out instead of div(2*mu*sym(grad(u))) so GiNaC need not differentiate mu
            if self.viscous_form == "stress":
                return mu * (div(grad(u)) + grad(div(u)))
            return mu * div(grad(u))
        if self.viscous_form == "stress":
            return div(2 * mu * sym(grad(u)))
        return div(mu * grad(u))

    def strong_momentum_residual(self)->Expression:
        """
        The strong momentum residual :math:`\\vec{R}_M`.

        Deliberately mirrors term by term what the Galerkin part actually assembles, including
        ``dt_factor``, ``nonlinear_factor``, ``pressure_factor``, the bulk force and gravity: a
        stabilization built on a *different* equation than the one being solved is inconsistent by
        construction.
        """
        u, p = var(self.velocity_name), var(self.pressure_name)
        rho = self.mass_density
        assert rho is not None
        R = rho * material_derivative(u, u, dt_factor=self.dt_factor,
                                      advection_factor=self.nonlinear_factor)
        R = R + self.pressure_factor * grad(p) * (-1 if self.pressure_sign_flip else 1)
        if self.include_viscous_in_residual:
            R = R - self.div_viscous_stress()
        if self.bulkforce is not None:
            R = R - self.bulkforce
        if self.gravity is not None:
            R = R - rho * self.gravity
        return R

    # -- tau ---------------------------------------------------------------------------------------

    def inv_dt(self)->Expression:
        """
        :math:`1/\\Delta t` as the time stepper sees it.

        Written as the BDF1 weight rather than as an explicit ``1/dt`` so that a *steady* solve,
        where pyoomph zeroes the weights, simply drops the transient term from :math:`\\tau` instead
        of dividing by an infinite time step. The weight itself is the *nondimensional*
        :math:`1/\\Delta t`, so it has to be divided by the temporal scale -- without that,
        :math:`\\tau` mixes 1/s^2 with a pure number and pyoomph rejects the expression in any
        dimensional problem.
        """
        return _inv_dt(self.transient_tau)

    def linear_drag_rate(self)->ExpressionOrNum:
        """
        Rate (1/s) of any term of the momentum equation that is *linear in the velocity itself*, and
        which therefore belongs in :math:`\\tau` alongside :math:`1/\\Delta t`, :math:`|a|/h` and
        :math:`\\nu/h^2`.

        Currently that is the Hele-Shaw drag :math:`-12\\mu\\vec{u}/\\delta^2`, i.e. a rate of
        :math:`12\\nu/\\delta^2`. Leaving it out is not a small error: on a 20 um Hele-Shaw cell it
        is roughly 24 times the in-plane viscous rate :math:`\\nu/h^2`, so :math:`\\tau` came out
        about two orders of magnitude too large and PSPG then suppressed a Marangoni-driven flow by
        four orders of magnitude. See ``dev_docs/stabilized_scalar_transport.md``.

        Override this to add a Darcy drag or any other linear sink of the same kind. Note that a
        drag folded into ``bulkforce`` by hand cannot be detected here -- ``bulkforce`` is an
        arbitrary vector -- so it has to be declared through this method.
        """
        if self.hele_shaw_thickness is None:
            return 0
        rho = self.mass_density
        assert rho is not None
        return 12 * self.dynamic_viscosity / (rho * self.hele_shaw_thickness ** 2)

    def _tau_M_raw(self)->Expression:
        """:math:`\\tau_M` without ``stab_factor``. :math:`\\tau_C` is built from it, so the
        prefactor has to be applied once at the end of each -- applying it inside would make it
        cancel out of :math:`\\tau_C=h^2/(c_C\\tau_M)` instead of scaling it."""
        rho = self.mass_density
        assert rho is not None
        return tau_advective_diffusive(self.element_h(), self.velocity_magnitude(),
                                       self.dynamic_viscosity / rho, self.inv_dt(),
                                       self.tau_formula, self.C_I, self.c_t,
                                       reaction=self.linear_drag_rate(), c_r=self.c_r)

    def tau_M(self)->Expression:
        """The momentum stabilization parameter :math:`\\tau_M`, in units of time."""
        return _maybe_sub(self._wrap_tau, self.stab_factor * self._tau_M_raw())

    def tau_C(self)->Expression:
        """The LSIC/grad-div parameter :math:`\\tau_C`, in units of a kinematic viscosity."""
        h, U = self.element_h(), self.velocity_magnitude()
        if self.tauC_formula == "codina":
            # tau_C = 1/(tau_M tr(G)), with tr(G) ~ c_C/h^2 for an isotropic element
            tau = h ** 2 / (self.c_C * self._tau_M_raw())
        elif self.tauC_formula == "tezduyar":
            rho = self.mass_density
            assert rho is not None
            nu = self.dynamic_viscosity / rho
            tau = h * U / 2 * _z_tezduyar(U * h / (2 * nu))
        else:
            raise ValueError(f"unknown tauC_formula '{self.tauC_formula}'")
        return _maybe_sub(self._wrap_tau, self.stab_factor * tau)

    # -- the footprint on the natural boundary condition -------------------------------------------

    def get_stabilization_traction(self,normal:Expression,bulk_domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        """
        The traction that the bulk stabilization terms deposit on a boundary, see
        :py:meth:`~pyoomph.equations.navier_stokes.StokesEquations.get_stabilization_traction`.

        Every stabilization term written against ``grad(v)`` contributes a surface integral when it
        is integrated by parts, so the natural condition of the stabilized formulation reads
        :math:`\\vec{n}\\cdot\\vec{\\vec{\\sigma}}+\\vec{t}_\\text{stab}=\\vec{t}` with

        .. math:: \\vec{t}_\\text{stab} = \\tau_M(\\vec{a}\\cdot\\vec{n})\\vec{R}_M
                  + \\tau_C\\rho\\,(\\nabla\\cdot\\vec{u})\\,\\vec{n}
                  - \\frac{\\tau_M^2}{\\rho}(\\vec{R}_M\\cdot\\vec{n})\\vec{R}_M

        from SUPG, LSIC and the VMS Reynolds term respectively. Each is proportional to a residual
        that vanishes for the exact solution, so subtracting them keeps the formulation consistent
        while making the prescribed traction the physical one on a finite mesh.

        The GLS/ASGS viscous perturbation is not in the list: it multiplies second derivatives of
        the test function rather than ``grad(v)``, so its effect on the boundary is not expressible
        as a traction.

        The PSPG term also has a surface footprint, :math:`(\\tau_M/\\rho)q\\,\\vec{R}_M\\cdot\\vec{n}`
        in the continuity equation, and it is deliberately *not* compensated anywhere: with
        :math:`q=1` the PSPG term drops out of the discrete continuity equation altogether, so
        global mass conservation is exact as it stands, and adding a boundary flux would break it.
        :py:class:`StabilizationBoundaryFlux` offers it for experiments.

        Which of the three are actually returned is selected by ``natural_bc_correction``, which
        defaults to *none*. Correcting the SUPG footprint measurably reduces the error at the
        boundary but degrades the global pressure convergence, because the uncorrected outflow term
        is part of what makes SUPG stable there.

        Returns zero if nothing is selected or no corresponding stabilization term is active.
        """
        active = self.stab & self.natural_bc_correction
        if not active:
            return Expression(0)
        # Everything strong must be evaluated in the bulk: on an interface grad() is the *surface*
        # gradient, which would silently drop exactly the normal derivatives that matter here.
        ed = (lambda e: e) if bulk_domain is None else (lambda e: evaluate_in_domain(e, bulk_domain))
        rho = self.mass_density
        assert rho is not None
        res = 0 * normal
        if active & {"SUPG", "REYNOLDS"}:
            R = _maybe_sub(self._wrap_R, ed(self.strong_momentum_residual()))
            tauM = ed(self.tau_M())
            if "SUPG" in active:
                res = res + tauM * dot(ed(self.convective_velocity()), normal) * R
            if "REYNOLDS" in active:
                res = res - tauM ** 2 / rho * dot(R, normal) * R
        if "LSIC" in active:
            res = res + ed(self.tau_C()) * rho * ed(div(var(self.velocity_name))) * normal
        return res

    # -- residuals ---------------------------------------------------------------------------------

    def define_residuals(self):
        super().define_residuals()          # Galerkin part, including the ALE/GCL machinery
        if not self.stab:
            return

        u, w = var_and_test(self.velocity_name)
        q = testfunction(self.pressure_name)
        a = self.convective_velocity()
        rho = self.mass_density
        assert rho is not None
        mu = self.dynamic_viscosity

        # No companion *surface* integral is added here. That is not an omission: the perturbed test
        # function multiplies R_M, which is zero for the exact solution, so the natural BC left over
        # by the Galerkin integration by parts is untouched. What these terms do leave behind on a
        # finite mesh is get_stabilization_traction(), which the traction BCs subtract.
        R = _maybe_sub(self._wrap_R, self.strong_momentum_residual())
        tauM = self.tau_M()

        if "SUPG" in self.stab:
            # (a.grad w)_i = a_j dw_i/dx_j = matproduct(grad(w),a), since grad(w)[i,j] = dw_i/dx_j
            self.add_weak(tauM * R, matproduct(grad(w), a))

        if "PSPG" in self.stab:
            self.add_weak(tauM / rho * R, grad(q))

        if "LSIC" in self.stab:
            self.add_weak(self.tau_C() * rho * div(u), div(w))

        if "GLSVISC" in self.stab or "ASGSVISC" in self.stab:
            # perturbation operator L(v) = rho a.grad v + grad q -/+ div(2 mu D(v)): GLS takes L
            # itself, ASGS its adjoint, which differs only in the sign of the viscous part.
            sgn = -1 if "GLSVISC" in self.stab else +1
            Lvisc = mu * (div(grad(w)) + grad(div(w))) if self.viscous_form == "stress" \
                else mu * div(grad(w))
            self.add_weak(tauM / rho * R, sgn * Lvisc)

        if "REYNOLDS" in self.stab:
            # VMS fine-scale Reynolds stress with u' = -tau_M R / rho:
            #   -(grad v, rho u' (x) u')  =  -(grad v, tau_M^2 R (x) R / rho)
            self.add_weak(-tauM ** 2 / rho * dyadic(R, R), grad(w))

    def describe(self)->str:
        """A short one-line description of the configuration, for test output."""
        return (f"{self.space}/{self.viscous_form}/{self.stab_name}/tau={self.tau_formula}"
                f"{'' if self.include_viscous_in_residual else '/NO-VISC-IN-R'}")


class ImposedTraction(InterfaceEquations):
    """
    Imposes an arbitrary traction :math:`\\langle\\vec{t},\\vec{v}\\rangle` on a boundary.

    This is *the* Neumann term of the momentum equation: it replaces the surface term
    :math:`\\vec{n}\\cdot\\vec{\\vec{\\sigma}}` that was dropped when the stress was integrated by
    parts. Leaving this equation away is the "do nothing" / free-traction boundary, i.e.
    :math:`\\vec{n}\\cdot\\vec{\\vec{\\sigma}}=0` -- and what that means depends on the
    ``viscous_form`` of the flow equations. Unlike
    :py:class:`~pyoomph.equations.navier_stokes.NavierStokesNormalTraction` the traction may have a
    tangential component.

    Args:
        traction: The traction vector to impose.
    """
    required_parent_type = StokesEquations

    def __init__(self, traction:Expression):
        super().__init__()
        self.traction = traction

    def define_residuals(self):
        peqs = self.get_parent_equations(StokesEquations)
        assert isinstance(peqs, StokesEquations)
        utest = testfunction(peqs.velocity_name)
        self.add_weak(-self.traction, utest)
        self.add_weak(-peqs.get_stabilization_traction(var("normal"), self.get_parent_domain()), utest)


class StabilizationBoundaryFlux(InterfaceEquations):
    """
    The optional PSPG boundary term :math:`-\\langle\\tau_M/\\rho\\,q,\\;\\vec{n}\\cdot\\vec{R}_M\\rangle`.

    Adding it turns the PSPG term :math:`+(\\tau_M/\\rho\\,\\nabla q,\\vec{R}_M)` into the
    integrated-by-parts pressure-Poisson form :math:`-(\\tau_M/\\rho\\,q,\\nabla\\cdot\\vec{R}_M)`.
    It is not part of the standard method and it is not needed for consistency, since
    :math:`\\vec{R}_M` vanishes for the exact solution either way. It is provided because it is the
    term one reaches for when the near-boundary pressure looks wrong, and because it makes explicit
    what the PSPG term does to the discrete mass balance at a boundary -- note that on a closed
    boundary it would *break* the exact global mass conservation that PSPG otherwise has.
    """
    required_parent_type = StabilizedNavierStokes

    def define_residuals(self):
        parent = self.get_parent_equations(StabilizedNavierStokes)
        assert isinstance(parent, StabilizedNavierStokes)
        pdom = self.get_parent_domain()
        q = testfunction(parent.pressure_name)
        R = evaluate_in_domain(parent.strong_momentum_residual(), pdom)
        tau = evaluate_in_domain(parent.tau_M(), pdom)
        rho = parent.mass_density
        assert rho is not None
        self.add_weak(-tau / rho * dot(var("normal"), R), q)


class BackflowStabilization(InterfaceEquations):
    """
    Removes the energy influx through an open ("do nothing") boundary where fluid enters.

    Testing the Galerkin form with :math:`\\vec{v}=\\vec{u}` gives the kinetic-energy balance

    .. math:: \\frac{d}{dt}\\text{KE} = -\\oint \\rho\\,(\\vec{u}\\cdot\\vec{n})\\,|\\vec{u}|^2/2
              - \\text{dissipation} + \\oint \\vec{t}\\cdot\\vec{u}\\,,

    so wherever fluid *enters* through an open boundary (:math:`\\vec{u}\\cdot\\vec{n}<0`) the
    convective term pumps energy in without bound. No bulk stabilization addresses this: SUPG, PSPG
    and LSIC are element-interior terms while the influx is a surface term. The cure is the surface
    term

    .. math:: -\\frac{\\beta}{2}\\oint\\rho\\,(\\vec{u}\\cdot\\vec{n})_-\\,(\\vec{u}\\cdot\\vec{v})\\,,
              \\qquad (a)_- = \\min(a,0)

    whose :math:`\\vec{v}=\\vec{u}` contribution removes exactly :math:`\\beta` times the
    destabilizing influx.

    Unlike everything else in this module this term is **not consistent**: it does not vanish for
    the exact solution wherever there is genuine backflow. Use the smallest ``beta`` that survives.

    Args:
        beta: Fraction of the backflow energy influx to remove; 1 removes all of it.
    """
    required_parent_type = StokesEquations

    def __init__(self, beta: ExpressionOrNum = 1):
        super().__init__()
        self.beta = beta

    def define_residuals(self):
        parent = self.get_parent_equations(StokesEquations)
        assert isinstance(parent, StokesEquations)
        rho = parent.mass_density
        if rho is None:
            raise RuntimeError("BackflowStabilization requires a mass_density on the parent equations")
        u, w = var_and_test(parent.velocity_name)
        un = dot(u, var("normal"))
        self.add_weak(-self.beta / 2 * rho * minimum(un, 0) * u, w)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
