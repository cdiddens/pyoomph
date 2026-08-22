#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

r"""
Transport of surfactants along a deforming interface.

A surfactant lives on the interface, is carried tangentially by the fluid and normally by the
interface itself, is diluted when the interface stretches, and -- if it is soluble -- exchanges with
the adjacent bulk. This module solves

.. math:: \partial_t\Gamma\big|_n + \nabla_S\!\cdot\!(\Gamma\vec u) = \nabla_S\!\cdot\!(D_S\nabla_S\Gamma) + S

which is Stone's equation, in any of three discrete forms.

**The conservative form is the default, and it is the reason this module exists.** An insoluble
surfactant cannot go anywhere: :math:`\int_S\Gamma\,\mathrm{d}S` is a conserved quantity and a
simulation that loses a part of it per time step is wrong in a way no mesh refinement fixes. The
form used previously -- :math:`\langle\partial_t\Gamma,v\rangle+\langle\nabla_S\!\cdot\!(\Gamma\vec
u_P),v\rangle` with an L2-projected advection velocity -- does exactly that, because the discrete
rate of change of the surface metric is not the discrete :math:`\nabla_S\!\cdot\!\vec w`, and the
ALE correction hidden inside :py:func:`~pyoomph.expressions.generic.partial_t` is non-conservative
advection. Measured on a closed interface with prescribed motion, it drifts by 7e-4 per unit time at
:math:`\Delta t=0.05` with BDF2, from tangential mesh sliding alone -- with the geometry held fixed.
The drift is :math:`\mathcal{O}(\Delta t^p)`, so it tracks the time-stepping order and never goes
away. Writing the same equation as the time derivative of the *whole integral* plus a flux,

.. math:: \frac{\mathrm{d}}{\mathrm{d}t}\int_S \Gamma v\,\mathrm{d}S
          -\int_S \Gamma(\vec u-\vec w)\cdot\nabla_S v\,\mathrm{d}S
          +\int_S D_S\nabla_S\Gamma\cdot\nabla_S v\,\mathrm{d}S = \int_S S\,v\,\mathrm{d}S

makes :math:`v=1` a telescoping difference of the discrete integral, so the amount is conserved to
the Newton tolerance -- measured 1e-14 -- on any mesh, at any time step, in any coordinate system.
It is also never less accurate: for pure normal interface motion it is *exact*, where the old form
is second order, and it is ten times more accurate under mass transfer. This mirrors what
``GCL=True`` already does for the bulk composition equations in
:py:mod:`~pyoomph.equations.multi_component`.

**The relative velocity needs no tangential projector, and the normal must not be smoothed.** In the
conservative form the advecting velocity appears only against :math:`\nabla_S v`, and on an interface
:py:func:`~pyoomph.expressions.generic.grad` is the *surface* gradient, which is exactly tangential
to the element. Projecting :math:`\vec u-\vec w` onto the tangent plane therefore changes nothing --
measured, to five digits. Substituting a smoothed, L2-projected normal for the element's own makes
it ten times *worse* under mass transfer: :math:`\vec u-\vec w` is then almost purely normal with
magnitude :math:`j/\rho`, and a smoothed normal is no longer orthogonal to the element tangent, so
the projection leaves a spurious tangential slip of order :math:`(j/\rho)\times` the smoothing error.
The conservative form contains no normal at all, which is the point.

**Positivity needs a different space, not a different form.** No continuous formulation here bounds
:math:`\Gamma`: on a front compressed onto the symmetry axis every one of them undershoots by the same
amount, and artificial diffusion is a smearing knob that is not even monotone in its coefficient.
``form="dg_upwind"`` with the default piecewise-constant space is the one that *is* bound-preserving --
implicit in time it is an M-matrix system, so :math:`\Gamma` cannot go negative at all. Measured on the
apex problem, the integral of the negative part is exactly zero where the continuous form reaches 0.13,
and the amount is still conserved to 1e-15. It costs accuracy, being first order: on that problem
:math:`\int\Gamma^2\,\mathrm{d}S` comes out 43 % low against the exact 91.3, where the continuous form
gets 91.1. ``space="DL"`` is far sharper but overshoots by a factor 2.5 and would need a limiter, so it
is not the default. The DG form needs a one-dimensional interface -- a curve in a 2d or axisymmetric
problem -- and refuses ``--distribute``; both are checked with a message rather than left to fail in the
mesh layer. It is also stiffer than the continuous forms and wants a smaller first time step when it is
coupled to a real flow.

**What happens at the ends of the interface.** Integrating the advection by parts creates a term
:math:`\oint_{\partial S}\Gamma(\vec u-\vec w)\cdot\vec m\,v\,\mathrm{d}l` at the interface's own
boundary -- a contact line, a symmetry axis, a corner. Leaving it out *is* the natural
zero-total-flux condition, and it is what makes the conservation exact; it is also the physically
right default, since an insoluble surfactant cannot leave the interface. Add
:py:class:`SurfactantEndFlux` where a nonzero flux is wanted. At an end point on the symmetry axis of
an axisymmetric problem nothing is needed either way: the measure of a point domain carries
:math:`2\pi r`, which is zero there. The legacy form behaves differently here -- it does not
integrate by parts, so it retains a genuine advective outflow at a contact line, which is what
:py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations` patches over by pinning the
projected advection velocity to the mesh velocity.
"""

from __future__ import annotations

from ..generic import Equations, InterfaceEquations
from ..expressions import *
from .generic import interface_transport_velocities
from .stabilization import ScalarTransportStabilization, element_h, regularized_magnitude
from ..typings import *

# Imported at runtime rather than under TYPE_CHECKING: sphinx evaluates the annotations below and
# cannot resolve them otherwise. materials.generic reaches back into this package only from inside
# functions, so there is no import cycle.
from ..materials.generic import BaseInterfaceProperties, SurfactantProperties, \
    MixtureLiquidProperties

#: Schemes accepted for the transient term. Kept as an alias because it is the published name; the
#: definition now lives beside :py:data:`~pyoomph.expressions.generic.TimeSteppingScheme`, since the
#: electrostatics module needs the same set and importing it from here would be backwards.
TransientSchemeEnum = IntegralTimeSteppingScheme

#: The transport forms :py:class:`SurfactantTransportEquations` can assemble.
SURFACTANT_TRANSPORT_FORMS = {"conservative", "legacy", "strong", "dg_upwind"}

#: Stabilization presets that only make sense on an interface, on top of the ones in
#: :py:data:`~pyoomph.equations.stabilization.SCALAR_STABILIZATION_PRESETS`.
SURFACTANT_STABILIZATIONS = {"artificial", "limited"}


class SurfactantTransportEquations(InterfaceEquations):
    r"""
    Transport of one or more surfactants along an interface.

    Add it to a free surface, either standalone::

        eqs += NavierStokesFreeSurface(surface_tension=sigma) @ "interface"
        eqs += SurfactantTransportEquations("Gamma", diffusivity=1e-9*meter**2/second) @ "interface"

    or, in a multi-component problem, by letting
    :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` drive it,
    which it does by default from the surfactants registered on the interface properties.

    Args:
        surfactants: Which surfactants to transport. ``None`` takes them from the parent's interface
            properties. Otherwise a name, a sequence of names, a ``dict`` mapping name to initial
            surface concentration, or :py:class:`~pyoomph.materials.generic.SurfactantProperties`
            objects -- so the class is usable without the materials machinery at all.
        diffusivity: Surface diffusivity, either one value for all or a ``dict`` per surfactant.
            ``None`` asks the interface properties.
        initial_concentration: Initial surface concentration, one value or a ``dict``. ``None`` asks
            the interface properties.
        form: ``"conservative"`` (default) assembles the time derivative of the whole integral plus a
            flux, which conserves the total amount exactly. ``"legacy"`` reproduces the older
            non-conservative form bit-for-bit, for reproducing published runs. ``"strong"`` is the
            same equation without the L2 projection of the advection velocity; it was measured to be
            worse than either of the other two and is here only as a diagnostic.
        variable: ``"direct"`` solves for :math:`\Gamma`. ``"log"`` solves for :math:`\psi` with
            :math:`\Gamma=\Gamma_\text{scale}\exp\psi`, which is positive at every point of every
            element by construction and still exactly conservative -- the constant test function
            kills the flux term whatever :math:`\Gamma` depends on. It costs Newton robustness: the
            residual is genuinely nonlinear, and a :math:`\Gamma` spanning more than about three
            decades needs a smaller time step.
        stabilization: ``None`` (default), any preset of
            :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`, or one of
            ``"artificial"`` (isotropic :math:`\nu=C h|\vec a|`) and ``"limited"`` (residual-driven,
            capped at the first-order upwind value :math:`h|\vec a|/2`). All of them are written
            against :math:`\nabla_S v` and therefore leave the exact conservation untouched. Note
            that on a one-dimensional interface -- any 2d or axisymmetric problem -- streamline
            diffusion *is* isotropic diffusion, and a crosswind term is identically zero.
        stab_factor: Prefactor of the artificial diffusivity.
        dc_factor: Prefactor of the residual-driven diffusivity of ``"limited"``. It has to be small:
            the term is strongly nonlinear and Newton stops converging above roughly 0.3.
        scheme: Time stepping scheme of the transient term. The default ``"BDF2_degr"`` degrades to a
            lower order on the first step, where there is no history yet. Plain ``"BDF2"`` here makes
            the *whole run* first order, which is easy to miss.
        space: Finite element space of the surfactant fields. ``None`` takes the interface velocity
            space of the parent flow equations, falling back to ``"C2"``.
        fluid_velocity: The velocity that carries the surfactant tangentially. ``None`` means
            ``var("velocity")``.
        interface_velocity: The velocity the interface itself moves with, i.e. what the surfactant
            follows normally. ``None`` means :py:func:`~pyoomph.expressions.generic.mesh_velocity`.
            It is deliberately *not* the fluid velocity: under evaporation the two differ by
            :math:`j/\rho` in the normal direction, and the surfactant stays with the interface.
        adsorption: Net ad-/desorption molar flux, positive towards the interface. ``None`` takes
            ``interface_props.surfactant_adsorption_rate``, which the isotherms in
            :py:mod:`~pyoomph.materials.surfactant_isotherms` fill in.
        bulk_coupling: Whether the adsorbed amount is removed from the bulk composition. ``"auto"``
            does it whenever the surfactant is also a component of the adjacent bulk mixture.
        field_prefix: Prefix of the field names. Do not change it lightly: the isotherms and every
            surface-tension law are written against ``surfconc_<name>``.
        concentration_scale: The named scale (or an expression) the surface concentration is
            nondimensionalised by.
        log_reference: With ``variable="log"``, the reference concentration in
            :math:`\Gamma=\Gamma_\text{ref}\exp\psi`. ``None`` uses the scale above -- which then
            *must* have been set with e.g. ``problem.set_scaling(surface_concentration=1*micro*mol/meter**2)``,
            since named scales default to 1 and an exponential cannot carry units.
        dt_factor: Prefactor of the transient term, e.g. ``0`` for a quasi-steady surfactant.
        advection_velocity_name: Name of the projected advection velocity field, ``"legacy"`` only.
        advection_velocity_space: Space of that field, ``"legacy"`` only.
        dg_alpha: Penalty coefficient of the interior-penalty surface diffusion, ``"dg_upwind"`` only.
            Ignored at order 0, where the penalty is not a stabilization but the two-point flux itself
            and its coefficient is fixed by consistency.
    """

    def __init__(self, surfactants: "str | Sequence[str] | dict[str, ExpressionOrNum] | SurfactantProperties | dict[SurfactantProperties, ExpressionOrNum] | None" = None, *,
                 diffusivity: "ExpressionOrNum | dict[str, ExpressionOrNum] | None" = None,
                 initial_concentration: "ExpressionOrNum | dict[str, ExpressionOrNum] | None" = None,
                 form: Literal["conservative", "legacy", "strong", "dg_upwind"] = "conservative",
                 variable: Literal["direct", "log"] = "direct",
                 stabilization: "str | Iterable[str] | ScalarTransportStabilization | None" = None,
                 stab_factor: ExpressionOrNum = 1,
                 dc_factor: ExpressionOrNum = 0.1,
                 scheme: TransientSchemeEnum = "BDF2_degr",
                 space: "FiniteElementSpaceEnum | None" = None,
                 fluid_velocity: ExpressionNumOrNone = None,
                 interface_velocity: ExpressionNumOrNone = None,
                 adsorption: "ExpressionOrNum | dict[str, ExpressionOrNum] | None" = None,
                 bulk_coupling: "Literal['auto'] | bool" = "auto",
                 field_prefix: str = "surfconc_",
                 concentration_scale: "ExpressionOrNum | str" = "surface_concentration",
                 log_reference: ExpressionNumOrNone = None,
                 dt_factor: ExpressionOrNum = 1,
                 advection_velocity_name: str = "_uinterf_proj",
                 advection_velocity_space: FiniteElementSpaceEnum = "C2",
                 dg_alpha: ExpressionOrNum = 1):
        super().__init__()
        if form not in SURFACTANT_TRANSPORT_FORMS:
            raise ValueError("unknown surfactant transport form '" + str(form) + "', available: "
                             + str(sorted(SURFACTANT_TRANSPORT_FORMS)))
        if variable not in ("direct", "log"):
            raise ValueError("unknown surfactant variable '" + str(variable) + "', use 'direct' or 'log'")
        self.form: Literal["conservative", "legacy", "strong", "dg_upwind"] = form
        if form == "dg_upwind":
            if space is None:
                # Piecewise constant is the whole point: with an upwind flux and an implicit step the
                # system is an M-matrix, so Gamma cannot go negative at all. A higher order needs a
                # limiter to keep that.
                space = "D0"
            elif not is_DG_space(space, allow_DL_and_D0=True):
                raise ValueError("form='dg_upwind' needs a discontinuous space, not '" + str(space)
                                 + "'. Use 'D0' (bound-preserving), or 'DL'/'D1'/'D2' with a limiter.")
            # Must be set before the problem is set up: the '_internal_facets_' child of this domain is
            # created from this flag in EquationTree._fill_dummy_equations, long before any residual is
            # assembled, and cannot be added afterwards.
            self.requires_interior_facet_terms = True
        self.variable: Literal["direct", "log"] = variable
        self.stabilization = stabilization
        self.stab_factor = stab_factor
        self.dc_factor = dc_factor
        self.scheme: TransientSchemeEnum = scheme
        self.space: "FiniteElementSpaceEnum | None" = space
        self.fluid_velocity = fluid_velocity
        self.interface_velocity = interface_velocity
        self.adsorption = adsorption
        self.bulk_coupling = bulk_coupling
        self.field_prefix = field_prefix
        self.dt_factor = dt_factor
        self.advection_velocity_name = advection_velocity_name
        self.advection_velocity_space: FiniteElementSpaceEnum = advection_velocity_space
        self.dg_alpha = dg_alpha
        self.concentration_scale: "ExpressionOrNum | str" = concentration_scale
        self.log_reference = log_reference
        self._diffusivity = diffusivity
        self._initial_concentration = initial_concentration
        self._surfactants_arg = surfactants
        # Set by _bind_to when a host (this object itself, or MultiComponentNavierStokesInterface)
        # drives the assembly, so that the accessors below can reach the materials plumbing.
        self._interface_props: "BaseInterfaceProperties | None" = None
        self._bulk_props: Any = None

    # ------------------------------------------------------------------ plumbing

    def _bind(self, interface_props: "BaseInterfaceProperties | None" = None, bulk_props: Any = None):
        """Attach the materials objects the host has, if any. Called before every assembly pass."""
        self._interface_props = interface_props
        self._bulk_props = bulk_props
        return self

    def _resolve_interface_props(self, host: Equations) -> "BaseInterfaceProperties | None":
        """The interface properties, if this problem has any. Everything material-derived is
        optional: the class must also work with nothing but a diffusivity and a number."""
        if self._interface_props is not None:
            return self._interface_props
        props = getattr(host, "interface_props", None)
        if props is not None:
            self._interface_props = cast("BaseInterfaceProperties", props)
            return self._interface_props
        # Standalone next to e.g. a NavierStokesFreeSurface: ask the equations we share the
        # interface with, since that is where a MultiComponentNavierStokesInterface would carry them.
        siblings = host.get_equation_of_type(Equations, always_as_list=True) if host is not None else []
        for sib in siblings:
            cand = getattr(sib, "interface_props", None)
            if cand is not None:
                self._interface_props = cast("BaseInterfaceProperties", cand)
                return self._interface_props
        return None

    def surfactant_names(self, host: Equations) -> list[str]:
        """The surfactants to transport, always sorted.

        ``sorted`` is not cosmetic: the names come from a ``set`` on the interface property class, so
        without it the nodal index of the surfactant fields would depend on the hash seed as soon as
        there is more than one of them.
        """
        arg = self._surfactants_arg
        if arg is None:
            props = self._resolve_interface_props(host)
            if props is None:
                raise RuntimeError("SurfactantTransportEquations was given no surfactants and found no "
                                   "interface properties to take them from. Pass e.g. "
                                   "SurfactantTransportEquations('my_surfactant', diffusivity=...).")
            return sorted(sp.name for sp in getattr(props, "_surfactants", {}).keys())
        if isinstance(arg, SurfactantProperties):
            return [arg.name]
        if isinstance(arg, str):
            return [arg]
        if isinstance(arg, dict):
            return sorted(k.name if isinstance(k, SurfactantProperties) else str(k) for k in arg.keys())
        return sorted(a.name if isinstance(a, SurfactantProperties) else str(a) for a in arg)

    def _per_surfactant(self, spec: Any, name: str, default: Any = None) -> Any:
        if spec is None:
            return default
        if isinstance(spec, dict):
            for k, v in spec.items():
                if (k.name if isinstance(k, SurfactantProperties) else str(k)) == name:
                    return v
            return default
        return spec

    def field_name(self, name: str) -> str:
        """Name of the *unknown*. With ``variable="log"`` this is not the concentration field."""
        if self.variable == "log":
            return "_log" + self.field_prefix + name
        return self.field_prefix + name

    def concentration_name(self, name: str) -> str:
        """Name the concentration is *scaled* under. Follows ``field_prefix``."""
        return self.field_prefix + name

    def public_name(self, name: str) -> str:
        """The name everything outside this module writes Gamma as.

        The isotherms in :py:mod:`~pyoomph.materials.surfactant_isotherms` and every surface tension
        law hard-code ``surfconc_<name>``. Whenever the solved field is called something else -- a
        log variable, or the solid-liquid prefix -- the concentration is bound to this name by
        substitution so that none of them has to know.
        """
        return "surfconc_" + name

    def log_reference_for(self, name: str) -> Expression:
        r"""The :math:`\Gamma_\text{ref}` of :math:`\Gamma=\Gamma_\text{ref}\exp\psi`.

        It has to carry the units of a surface concentration, because the exponential cannot. The
        named scale is the natural choice, but named scales default to 1, and no surfactant script in
        the tree currently sets ``surface_concentration`` -- with the direct variable that is only a
        conditioning matter, here it makes ``Gamma`` dimensionless and every surface tension law
        built from it fails on units. Hence the explicit ``log_reference`` escape hatch.
        """
        if self.log_reference is not None:
            return convert_to_expression(self.log_reference)
        return scale_factor(self.concentration_name(name))

    def concentration(self, name: str) -> Expression:
        r""":math:`\Gamma` as an expression, whichever variable is being solved for."""
        if self.variable == "log":
            return self.log_reference_for(name) * exp(var(self.field_name(name)))
        return var(self.field_name(name))

    def diffusivity(self, host: Equations, name: str) -> ExpressionOrNum:
        D = self._per_surfactant(self._diffusivity, name)
        if D is not None:
            return D
        props = self._resolve_interface_props(host)
        getter = getattr(props, "get_surface_diffusivity", None) if props is not None else None
        if getter is None:
            return 0
        D = getter(name)
        # get_surface_diffusivity legitimately returns None when nobody ever set one. A surfactant
        # without surface diffusion is a perfectly good model, so this is not an error - the old code
        # asserted here instead, which turned "you forgot a number" into an unexplained crash.
        return 0 if D is None else D

    def initial_concentration(self, host: Equations, name: str) -> ExpressionNumOrNone:
        ic = self._per_surfactant(self._initial_concentration, name)
        if ic is not None:
            return ic
        if isinstance(self._surfactants_arg, dict):
            for k, v in self._surfactants_arg.items():
                if (k.name if isinstance(k, SurfactantProperties) else str(k)) == name:
                    return v
        props = self._resolve_interface_props(host)
        for sp, amount in getattr(props, "_surfactants", {}).items():
            if sp.name == name:
                return amount
        return None

    def adsorption_rate(self, host: Equations, name: str) -> ExpressionNumOrNone:
        rate = self._per_surfactant(self.adsorption, name)
        if rate is not None:
            return rate
        props = self._resolve_interface_props(host)
        rates = getattr(props, "surfactant_adsorption_rate", None) if props is not None else None
        return None if rates is None else rates.get(name)

    def _space(self, host: Equations) -> FiniteElementSpaceEnum:
        if self.space is not None:
            return self.space
        # The surfactant lives where the velocity that advects it lives, so that the advection term
        # does not silently interpolate between spaces. Without flow equations in the parent - a
        # prescribed-motion testbed, say - there is nothing to match and C2 is the sensible default.
        from .navier_stokes import NavierStokesEquations
        pe = host.get_parent_equations(of_type=NavierStokesEquations)  # type:ignore[attr-defined]
        if pe is not None:
            return cast(FiniteElementSpaceEnum, pe.get_velocity_space_from_mode(for_interface=True))
        return "C2"

    def _velocities(self, host: Equations) -> tuple[Expression, Expression]:
        """The fluid velocity and the velocity of the interface itself, see
        :py:func:`~pyoomph.equations.generic.interface_transport_velocities`. Shared with the surface
        charge equations, which assemble the same conservative surface transport."""
        return interface_transport_velocities(cast(InterfaceEquations, host),
                                              self.fluid_velocity, self.interface_velocity)

    def uses_projected_advection_velocity(self, host: Equations) -> bool:
        """Whether ``_uinterf_proj`` exists, i.e. whether the legacy form is in use."""
        return self.form in ("legacy",) and len(self.surfactant_names(host)) > 0

    # ------------------------------------------------------------------ assembly

    def _check_dg_is_available(self, host: Equations) -> None:
        """Refuse the two configurations where the interface skeleton does not exist.

        Both fail anyway, but several frames deeper and with a message about facets rather than about
        surfactants -- and the MPI one names skeleton-of-skeleton, which is not what happened.
        """
        edim = host.get_element_dimension()
        if edim != 1:
            raise RuntimeError(
                "form='dg_upwind' needs interior facets between neighbouring interface elements, and "
                "pyoomph can only enumerate those for a one-dimensional interface -- a curve in a 2d or "
                "axisymmetric problem. This interface has dimension " + str(edim) + " (a surface in 3d), "
                "for which InterfaceMesh::fill_internal_facet_buffers throws. Use form='conservative' "
                "with variable='log' if you need positivity in 3d.")
        problem = host.get_current_code_generator().get_problem()
        if problem is not None and problem.is_distributed():
            raise RuntimeError(
                "form='dg_upwind' is not supported on a distributed (--distribute) problem: the "
                "interior-facet skeleton of an interface has no halo scheme, because its facets sit "
                "between face elements that are built on the fly rather than numbered before the "
                "distribution. Run without --distribute, or use form='conservative'.")

    def _define_fields_on(self, host: Equations) -> None:
        names = self.surfactant_names(host)
        if not names:
            return
        if self.form == "dg_upwind":
            self._check_dg_is_available(host)
        space = self._space(host)
        for name in names:
            fname = self.field_name(name)
            if self.variable == "log":
                # exp() of a dimensional quantity is not expressible in GiNaC's unit system, so the
                # unknown is the dimensionless logarithm and the scale is carried outside it.
                host.define_scalar_field(fname, space, scale=1,
                                         testscale=scale_factor("temporal") / scale_factor(self.concentration_name(name)))
            else:
                host.define_scalar_field(fname, space)
            if fname != self.public_name(name):
                G = self.concentration(name)
                host.define_field_by_substitution(self.public_name(name), G)
                host.add_local_function(self.public_name(name), G)
        if self.form == "legacy":
            host.define_vector_field(self.advection_velocity_name, self.advection_velocity_space)

    def _setup_scaling_on(self, host: Equations) -> None:
        names = self.surfactant_names(host)
        if not names:
            return
        scales: dict[str, "ExpressionOrNum | str"] = {}
        tscales: dict[str, "ExpressionOrNum | str"] = {}
        for name in names:
            cname = self.concentration_name(name)
            scales[cname] = self.concentration_scale
            if self.variable == "direct":
                tscales[cname] = scale_factor("temporal") / scale_factor(cname)
        if self.form == "legacy":
            scales[self.advection_velocity_name] = "velocity"
            tscales[self.advection_velocity_name] = 1 / scale_factor("velocity")
        host.set_scaling(scales)
        host.set_test_scaling(tscales)

    def _define_residuals_on(self, host: Equations) -> None:
        names = self.surfactant_names(host)
        if not names:
            return
        u, w = self._velocities(host)

        ui: Expression = Expression(0)
        if self.form in ("legacy", "strong"):
            # The advection velocity of the non-conservative forms: the tangential part of the fluid
            # velocity plus the normal part of the *interface* velocity, so that the surfactant stays
            # with the interface rather than with the liquid when the two differ by evaporation.
            n = host.get_normal()  # type:ignore[attr-defined]
            nn = dyadic(n, n)
            ui_expr = (u - matproduct(nn, u)) + dot(w, n) * n
            if self.form == "legacy":
                ui, ui_test = var_and_test(self.advection_velocity_name)
                host.add_residual(weak(ui - ui_expr, ui_test))
            else:
                ui = ui_expr

        for name in names:
            G = self.concentration(name)
            G_test = testfunction(self.field_name(name))
            D = self.diffusivity(host, name)
            ic = self.initial_concentration(host, name)
            if ic is not None:
                host.set_initial_condition(self.field_name(name),
                                           log(ic / self.log_reference_for(name)) if self.variable == "log" else ic,
                                           degraded_start="auto")

            if self.form in ("conservative", "dg_upwind"):
                # d/dt of the whole integral, so that the change of the surface metric is taken into
                # account by the same finite difference that advances the field. Testing with v=1
                # then telescopes exactly, which is what makes the amount conserved to the solver
                # tolerance rather than to the order of the time stepping.
                host.add_residual(self.dt_factor * time_derivative_of_integral(weak(G, G_test), scheme=self.scheme))
                # Only the slip relative to the mesh advects. No projection onto the tangent plane:
                # grad() on an interface is the surface gradient, so grad(G_test) is already exactly
                # orthogonal to the element normal and the normal part of (u-w) cannot contribute.
                host.add_residual(-weak(G * (u - w), grad(G_test)))
                if self.form == "dg_upwind":
                    # The numerical flux on the facets between neighbouring interface elements. On a
                    # curve a facet is a point, and var("normal") there is the in-surface conormal --
                    # the unit tangent of the curve signed outward from the near element -- not the
                    # interface normal, which stays reachable as var("normal", domain="..").
                    #
                    # Everything that is a *field* has to be restricted to one side with '+'/'-'.
                    # Unlike a bulk DG field, which a facet element carries as external data, a field
                    # owned by an interface lives in that interface element's internal data and is
                    # simply not visible on the facet itself: an unrestricted var(), and equally
                    # jump(...,at_facet=True), fails at code generation with "not defined in the
                    # equation or any parents".
                    m = var("normal")
                    # avg() rather than an unrestricted u-w, because a field is not visible on the
                    # facet itself (see above). The advecting velocity is continuous across a facet,
                    # so averaging it is the same expression as taking either side.
                    adotm = dot(avg(u - w), m)
                    # F = (a.m){{G}} + |a.m|/2 [[G]], i.e. (a.m) times the upwind value: for a.m > 0
                    # it collapses to (a.m)G+, for a.m < 0 to (a.m)G-. Single-valued, so summing the
                    # element residuals against v = 1 cancels it pairwise and the amount is conserved.
                    F = adotm * avg(G) + absolute(adotm) / 2 * jump(G)
                    host.add_interior_facet_residual(weak(F, jump(G_test)))
                    self._add_dg_diffusion(host, name, D, G, G_test, m)
            else:
                host.add_residual(self.dt_factor * weak(partial_t(G), G_test))
                host.add_residual(weak(div(G * ui), G_test))

            if not is_zero(D):
                host.add_residual(D * weak(grad(G), grad(G_test)))

            rate = self.adsorption_rate(host, name)
            if rate is not None and not is_zero(rate):
                rate = subexpression(rate)
                host.add_residual(-weak(rate, G_test))
                self._add_bulk_coupling(host, name, rate)

        self._add_stabilization(host, names)

    def _add_dg_diffusion(self, host: Equations, name: str, D: ExpressionOrNum,
                          G: Expression, G_test: Expression, m: Expression) -> None:
        r"""Surface diffusion across the facets, as an interior penalty.

        Without this there is no surface diffusion at all on a discontinuous space: the element term
        ``D*weak(grad(G),grad(v))`` couples nothing across a facet, and at order 0 it is *identically
        zero*, so a diffusivity would be accepted and silently ignored. Same symmetric interior penalty
        as :py:class:`~pyoomph.equations.poisson.PoissonEquation` uses in the bulk.

        At order 0 the two consistency terms vanish on their own (``grad`` of an elementwise constant
        is zero) and what remains, :math:`D[[\Gamma]][[v]]/\overline{h}`, is exactly the two-point
        finite-volume flux -- consistent rather than merely stabilizing, which is why the penalty
        coefficient is 1 there and not a tunable. It also keeps the system an M-matrix, so switching
        diffusion on does not cost the boundedness the upwind flux buys.
        """
        if is_zero(D):
            return
        order = get_order_of_space(self._space(host))
        stab = 1 if order == 0 else self.dg_alpha * (order + 1) * order
        h = var("cartesian_element_length_h")
        facet = -weak(jump(D * G) * m, avg(grad(G_test)))
        facet += -weak(avg(D * grad(G)), jump(G_test) * m)
        facet += weak(stab / avg(h) * jump(D * G) * m, jump(G_test) * m)
        host.add_interior_facet_residual(facet)

    def _add_bulk_coupling(self, host: Equations, name: str, rate: ExpressionOrNum) -> None:
        """Take out of the adjacent bulk exactly what arrives on the interface."""
        if self.bulk_coupling is False:
            return
        props = self._bulk_props
        if props is None:
            from .navier_stokes import NavierStokesEquations
            try:
                pe = host.get_parent_equations(of_type=NavierStokesEquations)  # type:ignore[attr-defined]
            except Exception:
                pe = None
            props = getattr(pe, "fluid_props", None)
        if not isinstance(props, MixtureLiquidProperties) or name not in props.components:
            if self.bulk_coupling is True:
                raise RuntimeError("bulk_coupling was requested for surfactant '" + name + "', but it is "
                                   "not a component of the adjacent bulk mixture.")
            return
        # The rate is molar; the bulk composition equations are in mass fractions.
        host.add_residual(weak(rate * props.pure_properties[name].molar_mass,
                               testfunction("massfrac_" + name)))

    def _add_stabilization(self, host: Equations, names: list[str]) -> None:
        if self.stabilization is None or self.stabilization == "none":
            return
        u, w = self._velocities(host)
        a = u - w
        h = element_h()
        amag = regularized_magnitude(a, 1e-10 * scale_factor("velocity"))
        for name in names:
            G = self.concentration(name)
            G_test = testfunction(self.field_name(name))
            if self.stabilization == "artificial":
                nu = self.stab_factor * h * amag
            elif self.stabilization == "limited":
                # Residual-driven diffusivity capped at the full first-order upwind value: smooth
                # solution -> R small -> nu ~ 0 -> plain Galerkin; at a front nu saturates at the
                # monotone value. Written against grad(G_test), so it is a diffusive flux and the
                # conservation of the conservative form survives it untouched.
                Gs = scale_factor(self.concentration_name(name))
                R = self.dt_factor * partial_t(G, ALE=False) + div(G * a)
                gmag = regularized_magnitude(grad(G), 1e-8 * Gs / scale_factor("spatial"))
                Rmag = square_root(R * R + (1e-8 * Gs / scale_factor("temporal")) ** 2)
                nu = minimum(self.dc_factor * h / 2 * Rmag / gmag, self.stab_factor * h * amag / 2)
            else:
                raise ValueError("SurfactantTransportEquations does not implement the stabilization '"
                                 + str(self.stabilization) + "'. Available: "
                                 + str(sorted(SURFACTANT_STABILIZATIONS)))
            host.add_residual(nu * weak(grad(G), grad(G_test)))

    # ------------------------------------------------------------------ standalone use

    def define_fields(self) -> None:
        self._define_fields_on(self)

    def define_scaling(self) -> None:
        super().define_scaling()
        self._setup_scaling_on(self)

    def define_residuals(self) -> None:
        self._define_residuals_on(self)


class SurfactantsAtSolidInterface(SurfactantTransportEquations):
    """
    Surfactants adsorbed on the solid-liquid interface, i.e. on the wetted patch of the substrate.

    Same transport as on a free surface, with two differences: nothing advects them tangentially
    (the liquid does not slip along the solid, so there is no surface flow to carry them), and the
    field prefix has to differ from the free surface's, because the two interfaces share their nodes
    at the contact line and would otherwise collide.

    Args:
        ls_properties: The liquid-solid interface properties, which supply the surfactants, their
            surface diffusivities and the ad-/desorption rates.
        out_surface_tension: Whether to publish the surface tension as a local expression for output.
    """

    def __init__(self, ls_properties: Any, out_surface_tension: bool = True, **kwargs: Any) -> None:
        # "_surfconcS_": we cannot use "surfconc_", since it would coincide with the liquid-gas
        # surfactants at the contact line.
        kwargs.setdefault("field_prefix", "_surfconcS_")
        kwargs.setdefault("space", "C2")
        kwargs.setdefault("fluid_velocity", 0)
        super().__init__(**kwargs)
        self.ls_properties = ls_properties
        self.out_surface_tension = out_surface_tension
        self._bind(ls_properties)

    @property
    def prefix(self) -> str:
        """Backwards-compatible alias of ``field_prefix``."""
        return self.field_prefix

    @prefix.setter
    def prefix(self, value: str) -> None:
        self.field_prefix = value

    def sanity_check(self):
        # Imported here rather than at module scope: multi_component imports this module, so naming
        # it up top would close the cycle.
        from .multi_component import CompositionAdvectionDiffusionEquations
        if self.get_parent_equations(of_type=CompositionAdvectionDiffusionEquations) is None:
            raise RuntimeError("SurfactantsAtSolidInterface must be attached to a domain carrying "
                               "CompositionAdvectionDiffusionEquations, since it takes the bulk "
                               "composition it exchanges with from there.")
        return super().sanity_check()

    def identify_surfactants_in_bulk(self) -> list[str]:
        """The bulk components that are surfactants and also live in the adjacent liquid."""
        from .multi_component import CompositionAdvectionDiffusionEquations
        parent = self.get_parent_equations(of_type=CompositionAdvectionDiffusionEquations)
        assert isinstance(parent, CompositionAdvectionDiffusionEquations)
        res: list[str] = []
        for cname in sorted(parent.fluid_props.components):
            c = parent.fluid_props.get_pure_component(cname)
            if isinstance(c, SurfactantProperties) and cname in self.ls_properties.get_liquid_properties().components:
                res.append(cname)
        return res

    def get_surfactant_field_name(self, cname: str) -> str:
        return self.field_name(cname)

    def surfactant_names(self, host: Equations) -> list[str]:
        if self._surfactants_arg is None:
            return self.identify_surfactants_in_bulk()
        return super().surfactant_names(host)

    def _define_residuals_on(self, host: Equations) -> None:
        if self.out_surface_tension:
            host.add_local_function("surface_tension", self.ls_properties.surface_tension)
        super()._define_residuals_on(host)


class SurfactantEndFlux(InterfaceEquations):
    r"""
    An imposed surfactant flux at an end point of an interface -- a contact line, a corner, or an
    edge where the interface stops.

    The conservative form of :py:class:`SurfactantTransportEquations` integrates the advection by
    parts, so its natural end condition is *zero total flux*: nothing leaves the interface, which is
    what makes the total amount exactly conserved and is the right default for an insoluble
    surfactant. Add this class where that is not what is wanted.

    A positive flux means surfactant *leaving* the interface through this end point.

    **Units: a flux per unit length of the end point**, i.e. mol/(m s), in every coordinate system --
    including a two-dimensional Cartesian one, where the end point is a point and its integration
    measure is the dimensionless 1. That is not an accident of this class: a test scale in pyoomph
    gains a factor 1/spatial per domain level, so the surfactant's test function, which belongs to the
    interface, already carries the length the point measure does not. The two cancel, and the result
    is independent of ``Problem.set_scaling(spatial=...)`` -- verified over four orders of magnitude,
    see ``tests/test_surfactant_transport.py``. Passing a rate in mol/s instead is rejected by the
    unit check of a dimensional problem, but would pass silently in a nondimensional one.

    Note that in an axisymmetric problem the measure of a point domain carries :math:`2\pi r`, so a
    flux imposed at an end point sitting on the symmetry axis contributes nothing at all -- correctly
    so, since the ring it lives on has zero circumference. Pass ``coordinate_system=cartesian`` if a
    plain point value is wanted there instead.

    Args:
        fluxes: The outward flux per surfactant and per unit end-point length, as ``name=expression``.
        coordinate_system: Override the coordinate system of the point integral.
    """
    required_parent_type = SurfactantTransportEquations

    def __init__(self, *, coordinate_system: OptionalCoordinateSystem = None, **fluxes: ExpressionOrNum):
        super().__init__()
        self.fluxes = fluxes
        self.coordinate_system = coordinate_system

    def define_residuals(self) -> None:
        parent = self.get_parent_equations(of_type=SurfactantTransportEquations)
        assert isinstance(parent, SurfactantTransportEquations)
        known = set(parent.surfactant_names(parent))
        for name, flux in self.fluxes.items():
            if name not in known:
                raise RuntimeError("SurfactantEndFlux was given a flux for '" + name + "', which the parent "
                                   "SurfactantTransportEquations does not transport. It has: " + str(sorted(known)))
            self.add_residual(weak(flux, testfunction(parent.field_name(name)),
                                   coordinate_system=self.coordinate_system))


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
