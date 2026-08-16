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
Shared machinery for residual-based stabilization, and the SUPG/GLS/ASGS stabilization of the
*scalar transport* equations (advection-diffusion, mixture composition, temperature).

The free functions at the top -- :py:func:`element_h`, :py:func:`inv_dt`,
:py:func:`regularized_magnitude`, :py:func:`tau_advective_diffusive` -- are the pieces that the
momentum stabilization in :py:mod:`pyoomph.equations.stabilized_ns` and the scalar transport
stabilization here have in common. They live in this module rather than being written twice, so
that a fix to :math:`\\tau` cannot land in one of the two and not the other.

:py:class:`ScalarTransportEquations` is the common base class of
:py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`,
:py:class:`~pyoomph.equations.multi_component.CompositionAdvectionDiffusionEquations` and
:py:class:`~pyoomph.equations.multi_component.TemperatureConductionEquation`, each of which supplies
its own strong residual. All of them take a single ``stabilization`` keyword argument, which is off
by default; see :py:class:`ScalarTransportStabilization`.

It is a base class rather than a mixin because ``Equations`` is a nanobind type, and nanobind
permits exactly one base -- ``class Foo(Mixin, Equations)`` fails at class-creation time with
"nb_type_init(): invalid number of bases!".

Everything except the discontinuity capturing term is proportional to the *strong* residual of the
transport equation and therefore vanishes for the exact solution: a consistent stabilization must
not change the answer, only the conditioning.
"""

from ..generic import Equations
from ..expressions import *
from ..expressions.generic import ExpressionOrNum
from ..typings import *

if TYPE_CHECKING:
    from ..generic.codegen import FiniteElementCodeGenerator


# =================================================================================================
#  Pieces shared with the momentum stabilization in stabilized_ns.py
# =================================================================================================

def element_h() -> Expression:
    """
    Isotropic element length :math:`V^{1/d}`, measured in *Cartesian* space.

    Deliberately ``"cartesian_element_length_h"`` and not ``"element_length_h"``: in an axisymmetric
    problem the latter is the revolved volume, so :math:`\\tau` would grow like :math:`r^{1/3}` away
    from the axis instead of tracking the actual cell size.
    """
    return var("cartesian_element_length_h")


def inv_dt(transient: "Literal['auto'] | bool" = "auto") -> Expression:
    """
    :math:`1/\\Delta t` as the time stepper sees it, or zero if ``transient`` is False.

    Written as the BDF1 weight rather than as an explicit ``1/dt`` so that a *steady* solve, where
    pyoomph zeroes the weights, simply drops the transient term from :math:`\\tau` instead of
    dividing by an infinite time step. The weight itself is the *nondimensional* :math:`1/\\Delta t`,
    so it has to be divided by the temporal scale -- without that, :math:`\\tau` mixes 1/s^2 with a
    pure number and pyoomph rejects the expression in any dimensional problem.
    """
    if transient is False:
        return Expression(0)
    return timestepper_weight(1, 0, "BDF1") / scale_factor("temporal")


def regularized_magnitude(v: Expression, eps_abs: ExpressionOrNum) -> Expression:
    """
    :math:`\\sqrt{\\vec{v}\\cdot\\vec{v}+\\varepsilon^2}`, differentiable at :math:`\\vec{v}=0`.

    ``eps_abs`` must carry the same units as ``v``; the callers form it as a relative constant times
    a scale factor. A bare number here is the units trap of
    ``dev_docs/stabilized_navier_stokes.md`` section 5 and pyoomph rejects it outright in a
    dimensional problem.
    """
    return square_root(dot(v, v) + eps_abs ** 2)


def _z_tezduyar(Re_h: Expression) -> Expression:
    """Tezduyar's switch between the diffusive (Re_h<3) and the convective limit."""
    return minimum(Re_h / 3, 1)


def _maybe_sub(wrap: bool, expr: Expression) -> Expression:
    return subexpression(expr) if wrap else expr


def tau_advective_diffusive(h: Expression, U: Expression, diffusivity: ExpressionOrNum,
                            idt: Expression, formula: Literal["shakib", "codina", "tezduyar"] = "shakib",
                            C_I: float = 4.0, c_t: float = 2.0,
                            reaction: ExpressionOrNum = 0, c_r: float = 1.0) -> Expression:
    """
    The stabilization parameter of a transient advection-diffusion-*reaction* operator, in units of
    *time*.

    Identical in form for momentum and for a transported scalar -- only ``diffusivity`` differs
    (:math:`\\nu=\\mu/\\rho` there, :math:`D` or :math:`k/(\\rho c_p)` here), which is why it is one
    function and not two.

    :math:`\\tau` is the inverse of a sum of *rates*, one per mechanism that can remove a
    perturbation: :math:`1/\\Delta t`, :math:`|\\vec{a}|/h`, :math:`D/h^2` and the reaction rate.
    Leaving a mechanism out makes :math:`\\tau` too large by whatever that rate contributes, and
    :math:`\\tau` too large is not a mild error -- see the Hele-Shaw measurement in
    ``dev_docs/stabilized_scalar_transport.md``.

    Args:
        h: element length scale, see :py:func:`element_h`.
        U: regularized magnitude of the advecting velocity.
        diffusivity: a *kinematic* diffusivity, i.e. m^2/s.
        idt: :math:`1/\\Delta t`, see :py:func:`inv_dt`.
        formula: ``"shakib"`` (inverse square root of the sum of squares), ``"codina"`` (inverse of
            the sum) or ``"tezduyar"``.
        C_I: coefficient of the diffusive term; :math:`\\tau\\to h^2/(C_I D)` in the diffusive limit.
        c_t: coefficient of the transient term.
        reaction: rate (1/s) of any term *linear in the transported quantity itself*, e.g. the
            Hele-Shaw drag :math:`12\\nu/\\delta^2` or a Darcy drag. Zero by default.
        c_r: coefficient of that reaction term.
    """
    idt = c_t * idt
    react = c_r * reaction
    if formula == "shakib":
        # absolute() is a no-op on a sum of squares, but it is what makes the square root survive an
        # azimuthal/normal-mode expansion. GiNaC splits a fractional power into polar form unless the
        # basis reports info_flags::nonnegative (power::real_part), and it cannot deduce that here:
        # mul::info implements positive/negative but not nonnegative, so 4*U^2/h^2 is indeterminate.
        # Without this, the m!=0 code came out as |X|^(-1/2)*(cos/sinh of imag_part(atan2(0,X))) and
        # the generated element failed to load with "undefined symbol: imag_part".
        return 1 / square_root(absolute(idt ** 2 + (2 * U / h) ** 2 + (C_I * diffusivity / h ** 2) ** 2
                                        + react ** 2))
    elif formula == "codina":
        return 1 / (idt + 2 * U / h + C_I * diffusivity / h ** 2 + react)
    elif formula == "tezduyar":
        return 1 / (idt + 2 * U / (h * _z_tezduyar(U * h / (2 * diffusivity))) + react)
    raise ValueError(f"unknown tau_formula '{formula}'")


# =================================================================================================
#  Scalar transport stabilization
# =================================================================================================

#: The individual terms accepted by :py:class:`ScalarTransportStabilization`.
SCALAR_STABILIZATION_TERMS = {"SUPG", "GLSDIFF", "ASGSDIFF", "DC"}

#: Named combinations of the individual terms, accepted wherever a ``stabilization`` argument is.
SCALAR_STABILIZATION_PRESETS: dict[str, set[str]] = {
    "none": set(),
    "SUPG": {"SUPG"},
    "GLS": {"SUPG", "GLSDIFF"},
    "ASGS": {"SUPG", "ASGSDIFF"},
    "DC": {"DC"},
    "SUPG+DC": {"SUPG", "DC"},
    "GLS+DC": {"SUPG", "GLSDIFF", "DC"},
}


class ScalarTransportStabilization:
    """
    Settings for the residual-based stabilization of a scalar transport equation.

    Pass an instance -- or, for the common case, just the ``terms`` string, which is promoted to one
    -- as the ``stabilization`` argument of
    :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`,
    :py:class:`~pyoomph.equations.multi_component.CompositionAdvectionDiffusionEquations` or the
    temperature equations. ``None`` or ``"none"`` means no stabilization at all, which is the
    default everywhere.

    The added terms per element interior are, with :math:`v` the test function of the transported
    field, :math:`\\vec{a}=\\vec{u}-\\vec{u}_\\text{mesh}` the advecting velocity relative to the
    mesh and :math:`R` the strong residual of the transport equation,

    .. math:: \\text{SUPG}\\quad &\\sum_K (\\tau\\,\\vec{a}\\cdot\\nabla v,\\; R)_K
    .. math:: \\text{GLSDIFF/ASGSDIFF}\\quad &\\mp\\sum_K (\\tau\\,D\\,\\nabla^2 v,\\; R)_K
    .. math:: \\text{DC}\\quad &\\sum_K (\\hat\\rho\\,\\nu_\\text{dc}\\,\\mathbf{P}\\nabla c,\\;\\nabla v)_K

    The first two are proportional to :math:`R` and hence vanish for the exact solution. The
    discontinuity capturing term does not: it is an artificial diffusivity whose *magnitude* is
    driven by :math:`R`, so it decreases under refinement but is not zero on a representable
    solution. It is also the only term written against :math:`\\nabla v`, so it is the only one that
    changes the natural boundary condition of the transported field. Both are why it is off by
    default.

    Args:
        terms: a key of :py:data:`SCALAR_STABILIZATION_PRESETS`, an iterable of the individual flags
            ``"SUPG"``, ``"GLSDIFF"``, ``"ASGSDIFF"``, ``"DC"``, or ``None``/``"none"``.
        tau_formula: ``"shakib"``, ``"codina"`` or ``"tezduyar"``, see
            :py:func:`tau_advective_diffusive`. The classical nodally exact one-dimensional choice
            :math:`\\tau=\\frac{h}{2|a|}(\\coth\\mathrm{Pe}-1/\\mathrm{Pe})` is deliberately *not*
            offered: it is 0/0 as :math:`\\mathrm{Pe}\\to0` and would need a series expansion around
            that limit, which pyoomph's symbolic layer does not have.
        C_I: coefficient of the diffusive term in :math:`\\tau`. The default 4 is inherited from
            :py:class:`~pyoomph.equations.stabilized_ns.StabilizedNavierStokes`, where it was
            measured to be on the diffusive side; it has *not* been measured for scalar transport.
        c_t: coefficient of the transient term in :math:`\\tau`.
        stab_factor: global prefactor on :math:`\\tau` and :math:`\\nu_\\text{dc}`, for sensitivity
            studies. Setting it to zero must reproduce the unstabilized residual exactly.
        transient_tau: include the :math:`1/\\Delta t` term in :math:`\\tau`. ``"auto"`` uses the
            BDF1 weight, which pyoomph zeroes in a steady solve, so the term switches itself off
            there.
        include_diffusion_in_residual: keep the second-derivative diffusive term
            :math:`\\nabla\\cdot(D\\nabla c)` in the strong residual.

            **Switch it off on a mesh of linear simplices.** On an affine element map the second
            derivatives of C1 shape functions vanish identically, so the term contributes nothing
            and computing it is pure cost -- measured, dropping it changes the residual by 0 on
            triangles and 0 on tets, and saves about 1.37x on Jacobian assembly (the whole
            second-derivative shape machinery stops being generated). This is the case where
            turning it off is free rather than an approximation.

            Measured relative change of the residual when dropping it elsewhere:

            ===========================  ========
            C1 triangles / tets           0
            C1 quads, undistorted         1.8e-17
            C1 quads, bilinearly warped   2.4e-02
            C2 triangles                  1.4e-01
            C2 tets                       1.6e-01
            ===========================  ========

            The undistorted-quad entry is real but fragile, and not a reason to switch this on for a
            quad mesh: a bilinear Q1 function has :math:`\\partial_{xx}=\\partial_{yy}=0` only while
            the elements stay rectangles, which any mesh distortion -- and every moving mesh -- ends.
            The safe rule is the simplex one.

            It must also be off on wedges, pyramids and 0d domains, where second derivatives are
            unavailable at all.

            The momentum counterpart is
            :py:attr:`~pyoomph.equations.stabilized_ns.StabilizedNavierStokes.include_viscous_in_residual`.
        conservative_residual: whether the *advective* part of the strong residual is written in
            conservative form :math:`\\nabla\\cdot(\\vec{a}c)` or convective form
            :math:`\\vec{a}\\cdot\\nabla c`. ``"auto"`` mirrors whatever the equation actually
            assembles. The two differ by :math:`c\\,\\nabla\\cdot\\vec{a}`, which is not zero for a
            variable density or a compressible wind.
        velocity_eps: :math:`|\\vec{a}|` is regularized as
            :math:`\\sqrt{\\vec{a}\\cdot\\vec{a}+(\\varepsilon U)^2}` so that :math:`\\tau` stays
            differentiable at rest. Given *relative* to the velocity scale, so that it remains
            meaningful in a dimensional problem.
        velocity_scale: name of the scale used for that regularization.
        dc_factor: prefactor on the discontinuity capturing diffusivity.
        dc_form: ``"crosswind"`` adds the artificial diffusivity only perpendicular to the
            streamline, so it does not spoil the streamline accuracy SUPG buys; ``"isotropic"`` adds
            it in all directions.
        dc_eps: relative regularization of :math:`|\\nabla c|` and :math:`|R|` in
            :math:`\\nu_\\text{dc}`.
        dc_subtract_supg: use Codina's capped form, i.e. subtract the diffusivity SUPG already
            supplies and floor at zero, see :py:meth:`ScalarTransportEquations.dc_diffusivity`. On by
            default: the uncapped ratio is unbounded and does not switch itself off on a smooth
            solution, which makes it unusable at ``dc_factor`` = 1.
        natural_bc_correction: which parts of the stabilization's footprint on the natural boundary
            condition the flux boundary conditions should subtract, so that they impose the physical
            flux rather than the physical flux plus that footprint. ``True`` for all of them,
            ``False`` (the default) for none, or an iterable of ``"SUPG"``, ``"DC"``. See
            :py:meth:`ScalarTransportStabilizationMixin.get_stabilization_flux`.
    """

    def __init__(self, terms: "str | Iterable[str] | None" = "SUPG", *,
                 tau_formula: Literal["shakib", "codina", "tezduyar"] = "shakib",
                 C_I: float = 4.0, c_t: float = 2.0,
                 stab_factor: ExpressionOrNum = 1,
                 transient_tau: "Literal['auto'] | bool" = "auto",
                 include_diffusion_in_residual: bool = True,
                 conservative_residual: "Literal['auto'] | bool" = "auto",
                 velocity_eps: ExpressionOrNum = 1e-10,
                 velocity_scale: str = "velocity",
                 dc_factor: ExpressionOrNum = 1,
                 dc_form: Literal["crosswind", "isotropic"] = "crosswind",
                 dc_eps: ExpressionOrNum = 1e-10,
                 dc_subtract_supg: bool = True,
                 natural_bc_correction: "bool | Iterable[str]" = False):
        if terms is None:
            self.terms: set[str] = set()
            self.name = "none"
        elif isinstance(terms, str):
            if terms not in SCALAR_STABILIZATION_PRESETS:
                raise ValueError(f"unknown stabilization preset '{terms}', "
                                 f"available: {sorted(SCALAR_STABILIZATION_PRESETS)}")
            self.terms = set(SCALAR_STABILIZATION_PRESETS[terms])
            self.name = terms
        else:
            self.terms = set(terms)
            self.name = "+".join(sorted(self.terms)) if self.terms else "none"
        unknown = self.terms - SCALAR_STABILIZATION_TERMS
        if unknown:
            raise ValueError(f"unknown stabilization terms {sorted(unknown)}, "
                             f"available: {sorted(SCALAR_STABILIZATION_TERMS)}")
        if {"GLSDIFF", "ASGSDIFF"} <= self.terms:
            raise ValueError("GLSDIFF and ASGSDIFF are the two signs of the same term, pick one")

        if cast(str, tau_formula) not in ("shakib", "codina", "tezduyar"):
            raise ValueError(f"unknown tau_formula '{tau_formula}'")
        self.tau_formula: Literal["shakib", "codina", "tezduyar"] = tau_formula
        self.C_I, self.c_t = C_I, c_t
        self.stab_factor = stab_factor
        self.transient_tau: "Literal['auto'] | bool" = transient_tau
        self.include_diffusion_in_residual = include_diffusion_in_residual
        self.conservative_residual = conservative_residual
        self.velocity_eps = velocity_eps
        self.velocity_scale = velocity_scale
        self.dc_factor = dc_factor
        if cast(str, dc_form) not in ("crosswind", "isotropic"):
            raise ValueError(f"unknown dc_form '{dc_form}'")
        self.dc_form: Literal["crosswind", "isotropic"] = dc_form
        self.dc_eps = dc_eps
        self.dc_subtract_supg = dc_subtract_supg

        if isinstance(natural_bc_correction, bool):
            self.natural_bc_correction: set[str] = {"SUPG", "DC"} if natural_bc_correction else set()
        else:
            self.natural_bc_correction = set(natural_bc_correction)
        unknown_corr = self.natural_bc_correction - {"SUPG", "DC"}
        if unknown_corr:
            raise ValueError(f"unknown natural_bc_correction terms {sorted(unknown_corr)}")

        if (self.terms & {"GLSDIFF", "ASGSDIFF"}) and not include_diffusion_in_residual:
            # the diffusive part of the perturbation operator is second order too: if second
            # derivatives are unavailable for R they are unavailable for the test operator as well
            raise ValueError("GLSDIFF/ASGSDIFF need include_diffusion_in_residual=True")

    def __repr__(self) -> str:
        return f"{self.name}/tau={self.tau_formula}"


def _to_stabilization(stabilization: "str | Iterable[str] | ScalarTransportStabilization | None"
                      ) -> ScalarTransportStabilization:
    if isinstance(stabilization, ScalarTransportStabilization):
        return stabilization
    return ScalarTransportStabilization(stabilization)


class ScalarTransportEquations(Equations):
    """
    Base class of the scalar transport equations, adding optional SUPG/GLS/ASGS and discontinuity
    capturing.

    The concrete equation supplies five small hooks -- :py:meth:`stabilized_fieldnames`,
    :py:meth:`stabilization_wind`, :py:meth:`stabilization_diffusivity`,
    :py:meth:`stabilization_residual_scale` and :py:meth:`strong_residual` -- calls
    :py:meth:`_init_stabilization` in its ``__init__`` and
    :py:meth:`add_stabilization_residuals` at the end of its ``define_residuals``. Everything else,
    including the whole "is it switched on" bookkeeping, lives here.

    This class is inert until ``stabilization`` is set to something: with the default
    :py:meth:`add_stabilization_residuals` returns immediately and
    :py:meth:`get_stabilization_flux` returns ``Expression(0)``, so an unstabilized problem
    generates exactly the code it did before.
    """

    # Which intermediate quantities get their own named temporary in the generated C. Measured for
    # the momentum version in dev_docs/stabilized_navier_stokes.md section 6: tau is the one that
    # matters (unwrapping it alone costs 50% of the assembly time -- it is the deepest tree and
    # multiplies several separate weak terms, so its derivative would be expanded once per term),
    # while wrapping the strong residual buys no runtime but shrinks the generated C.
    # Class attributes rather than constructor arguments because this is a code-generation knob.
    _wrap_R = True
    _wrap_tau = True

    # -- to be implemented by the concrete equation ------------------------------------------------

    def stabilized_fieldnames(self) -> list[str]:
        """The names of the fields whose residual rows get the stabilization terms."""
        raise NotImplementedError

    def stabilization_wind(self) -> ExpressionOrNum:
        """The advecting velocity, *before* subtracting the mesh velocity."""
        raise NotImplementedError

    def stabilization_diffusivity(self, fieldname: str) -> ExpressionOrNum:
        """A *kinematic* diffusivity (m^2/s) for this field, used in :math:`\\tau`."""
        raise NotImplementedError

    def stabilization_residual_scale(self, fieldname: str) -> ExpressionOrNum:
        """
        The factor :math:`\\hat\\rho` multiplying the time derivative in the assembled equation:
        1 for a plain advection-diffusion equation, :math:`\\rho` for a mass fraction,
        :math:`\\rho c_p` for the temperature. It is what makes the added terms carry the same units
        as the Galerkin ones.
        """
        raise NotImplementedError

    def strong_residual(self, fieldname: str) -> Expression:
        """
        The strong residual of the transport equation for this field, mirroring term by term what
        the Galerkin part actually assembles. A stabilization built on a *different* equation than
        the one being solved is inconsistent by construction.
        """
        raise NotImplementedError

    # -- provided ----------------------------------------------------------------------------------

    def _init_stabilization(self, stabilization: "str | Iterable[str] | ScalarTransportStabilization | None",
                            velocity_scale: str | None = None):
        """Called from the concrete class's ``__init__``. Sets ``self.stab_cfg`` and ``self.stab``."""
        self.stab_cfg = _to_stabilization(stabilization)
        if velocity_scale is not None and self.stab_cfg.velocity_scale == "velocity":
            # the equation knows better than the default which scale its wind is measured against
            self.stab_cfg.velocity_scale = velocity_scale
        self.stab: set[str] = self.stab_cfg.terms

    def _stabilization_is_advective(self) -> bool:
        """
        Whether the SUPG/DC terms have anything to act on at all.

        With a zero wind on a static mesh the streamline weight is identically zero, so the terms
        would only bloat the generated code. This also keeps ``scale_factor("velocity")`` out of
        purely diffusive domains, where it need not be defined at all.
        """
        if not is_zero(self.stabilization_wind()):
            return True
        return bool(self.get_combined_equations()._assert_codegen()._coordinates_as_dofs)  # type:ignore

    def convective_velocity(self) -> Expression:
        """
        :math:`\\vec{a}=\\vec{u}-\\vec{u}_\\text{mesh}`. On a moving (ALE) mesh it is the relative
        velocity that is advected, so it is that one which must set both the streamline direction
        and the cell Peclet number in :math:`\\tau`.
        """
        a = convert_to_expression(self.stabilization_wind())
        if self.get_combined_equations()._assert_codegen()._coordinates_as_dofs:  # type:ignore
            a = a - mesh_velocity()
        return a

    def stabilization_velocity_magnitude(self) -> Expression:
        """
        Regularized :math:`|\\vec{a}|`, see :py:func:`regularized_magnitude`.

        Exactly zero, and in particular *not* touching the velocity scale, when there is nothing
        being advected: a purely diffusive domain need not define ``scale_factor("velocity")`` at
        all, and asking for it there fails at code generation rather than at construction.
        """
        if not self._stabilization_is_advective():
            return Expression(0)
        eps = self.stab_cfg.velocity_eps * scale_factor(self.stab_cfg.velocity_scale)
        return regularized_magnitude(self.convective_velocity(), eps)

    def stabilization_element_h(self) -> Expression:
        """The element length scale entering :math:`\\tau`. An override point."""
        return element_h()

    def tau(self, fieldname: str) -> Expression:
        """The stabilization parameter for this field, in units of time."""
        cfg = self.stab_cfg
        D = self.stabilization_diffusivity(fieldname)
        if cfg.tau_formula == "tezduyar" and is_zero(D):
            raise RuntimeError(f"tau_formula='tezduyar' divides by the diffusivity, but the "
                               f"diffusivity of '{fieldname}' is zero. Use 'shakib' or 'codina'.")
        t = tau_advective_diffusive(self.stabilization_element_h(),
                                    self.stabilization_velocity_magnitude(), D,
                                    inv_dt(cfg.transient_tau), cfg.tau_formula, cfg.C_I, cfg.c_t)
        return _maybe_sub(self._wrap_tau, cfg.stab_factor * t)

    def dc_gradient(self, fieldname: str) -> Expression:
        """The gradient that the discontinuity capturing term diffuses along."""
        g = grad(var(fieldname))
        if self.stab_cfg.dc_form == "isotropic":
            return g
        # Crosswind: project out the streamline component, so that the added diffusivity acts only
        # perpendicular to the flow and does not undo the streamline accuracy SUPG provides.
        ahat = self.convective_velocity() / self.stabilization_velocity_magnitude()
        return g - dot(g, ahat) * ahat

    def dc_diffusivity(self, fieldname: str, R: Expression) -> Expression:
        """
        The discontinuity capturing artificial diffusivity (m^2/s),

        .. math:: \\nu_\\text{dc} = \\max\\left(0,\\;
                  C\\,\\frac{h}{2}\\frac{|R|}{\\hat\\rho\\,|\\nabla c|} - \\tau|\\vec{a}|^2\\right)

        i.e. Codina's form: the raw ratio *minus* the diffusivity SUPG already supplies, floored at
        zero. The subtraction is what switches the term off where SUPG is already doing the job;
        without it the ratio is unbounded and stays active on a perfectly smooth solution, since
        :math:`|R|/(\\hat\\rho|\\nabla c|)` is of order :math:`|\\vec{a}|` there, which is the full
        first-order-upwind diffusivity. Set ``dc_subtract_supg=False`` for the raw ratio.

        **This term needs tuning and is off by default for good reason.** On an intense-Marangoni
        instability at mesh Peclet ~17 the first Newton solve diverges outright at ``dc_factor`` 1,
        0.1 and 0.03, and only runs at 0.01. That threshold is a property of the term's
        nonlinearity, not of its size or its regularization: it is unchanged by the cap above, by
        ``dc_eps`` anywhere from 1e-10 to 1e-1, by ``dc_form``, by a Newton relaxation factor of 0.5
        and by a globally convergent Newton. The cap does change the answer where the term does run
        -- 6% in kinetic energy at ``dc_factor`` = 0.01 -- it just does not move that threshold.
        Start at ``dc_factor`` = 0.01 and raise it only as far as Newton tolerates.

        Both magnitudes are regularized square roots rather than ``absolute_value``: the derivative
        of :math:`|x|` is 0/0 at the origin, and both :math:`R` and :math:`\\nabla c` are exactly
        zero on a uniform initial condition. The same trap is recorded for the viscoelastic
        stabilization, where it put NaNs in the Jacobian and sent the first Newton step to 1e105.
        The regularizations are *relative to a scale*, so that they stay meaningful in a dimensional
        problem.
        """
        cfg = self.stab_cfg
        rho_hat = self.stabilization_residual_scale(fieldname)
        g = grad(var(fieldname))
        geps = cfg.dc_eps * scale_factor(fieldname) / scale_factor("spatial")
        Reps = cfg.dc_eps * scale_factor(fieldname) / scale_factor("temporal") * rho_hat
        gmag = regularized_magnitude(g, geps)
        Rmag = square_root(R * R + Reps ** 2)
        nu = cfg.dc_factor * self.stabilization_element_h() / 2 * Rmag / (rho_hat * gmag)
        if cfg.dc_subtract_supg:
            U = self.stabilization_velocity_magnitude()
            nu = maximum(nu - self.tau(fieldname) * U ** 2, 0)
        return _maybe_sub(self._wrap_tau, nu)

    def add_stabilization_residuals(self, ts: "Callable[[Any],Any] | None" = None):
        """
        Adds the selected stabilization terms. Call this at the *end* of ``define_residuals``.

        No companion *surface* integral is added. That is not an omission for SUPG and GLS/ASGS: the
        perturbed test function multiplies the strong residual, which is zero for the exact
        solution, so the natural boundary condition left over by the Galerkin integration by parts
        is untouched. What these terms do leave behind on a finite mesh is
        :py:meth:`get_stabilization_flux`, which the flux boundary conditions subtract if asked to.

        Args:
            ts: the equation's own ``time_scheme`` wrapper, applied to the trial-side factor exactly
                as the Galerkin terms of that equation apply it. ``None`` for equations that do not
                use one.
        """
        if not self.stab:
            return
        if ts is None:
            ts = lambda e: e
        advective = self._stabilization_is_advective()
        if not advective and not (self.stab & {"GLSDIFF", "ASGSDIFF"}):
            return
        a = self.convective_velocity() if advective else None

        for fn in self.stabilized_fieldnames():
            v = testfunction(fn)
            R = _maybe_sub(self._wrap_R, self.strong_residual(fn))
            tau = self.tau(fn)

            if "SUPG" in self.stab and advective:
                assert a is not None
                self.add_residual(weak(ts(tau * R), dot(a, grad(v))))

            if self.stab & {"GLSDIFF", "ASGSDIFF"}:
                # perturbation operator P(v) = a.grad v - D grad^2 v: GLS takes P itself, ASGS its
                # adjoint, which differs only in the sign of the diffusive part.
                sgn = -1 if "GLSDIFF" in self.stab else +1
                self.add_residual(weak(ts(tau * R),
                                       sgn * self.stabilization_diffusivity(fn) * div(grad(v))))

            if "DC" in self.stab and advective:
                rho_hat = self.stabilization_residual_scale(fn)
                self.add_residual(weak(ts(rho_hat * self.dc_diffusivity(fn, R) * self.dc_gradient(fn)),
                                       grad(v)))

    def get_stabilization_flux(self, fieldname: str, normal: Expression,
                               bulk_domain: "str | FiniteElementCodeGenerator | None" = None) -> Expression:
        """
        The flux that the bulk stabilization terms deposit on a boundary of the transported field.

        Every stabilization term written against :math:`\\nabla v` contributes a surface integral
        when it is integrated by parts, so the natural condition of the stabilized formulation reads
        :math:`-\\vec{n}\\cdot\\vec{J}+S=\\text{(imposed flux)}` with

        .. math:: S = \\tau\\,(\\vec{a}\\cdot\\vec{n})\\,R
                      + \\hat\\rho\\,\\nu_\\text{dc}\\,\\vec{n}\\cdot\\mathbf{P}\\nabla c

        from SUPG and from the discontinuity capturing term respectively. Callers subtract it, i.e.
        they add ``-weak(S, testfunction(fieldname))``, so that the flux they prescribe is the
        physical one on a finite mesh. The SUPG part is proportional to a residual that vanishes for
        the exact solution, so subtracting it keeps the formulation consistent.

        The GLS/ASGS perturbation is not in the list: it multiplies second derivatives of the test
        function rather than :math:`\\nabla v`, so its effect on the boundary is not expressible as
        a flux.

        Which of the two are actually returned is selected by ``natural_bc_correction``, which
        defaults to none. Switching it on is a trade-off rather than a free improvement: it makes
        the *imposed flux at that boundary* the physical one, which matters when the interfacial
        flux itself is the quantity of interest -- an evaporation rate, a latent heat, a coupling to
        another domain -- and it is not worth it when the field in the interior is.

        Returns zero if nothing is selected or no corresponding stabilization term is active.
        """
        active = self.stab & self.stab_cfg.natural_bc_correction
        if not active or fieldname not in self.stabilized_fieldnames():
            return Expression(0)
        if not self._stabilization_is_advective():
            return Expression(0)
        # Everything strong must be evaluated in the bulk: on an interface grad() is the *surface*
        # gradient, which would silently drop exactly the normal derivatives that matter here.
        ed = (lambda e: e) if bulk_domain is None else (lambda e: evaluate_in_domain(e, bulk_domain))
        res = Expression(0)
        R = _maybe_sub(self._wrap_R, ed(self.strong_residual(fieldname)))
        if "SUPG" in active:
            res = res + ed(self.tau(fieldname)) * dot(ed(self.convective_velocity()), normal) * R
        if "DC" in active:
            res = res + ed(self.stabilization_residual_scale(fieldname)
                           * self.dc_diffusivity(fieldname, R)) * dot(ed(self.dc_gradient(fieldname)), normal)
        return res

    def describe_stabilization(self) -> str:
        """A short one-line description of the configuration, for test output."""
        return repr(self.stab_cfg)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
