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
 
 
from .._deprecation import deprecated_kwargs as _deprecated_kwargs, deprecated_attribute_alias as _deprecated_attribute_alias
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..generic import Equations, InterfaceEquations
from ..generic.codegen import sorted_field_kwargs
from ..equations.generic import InitialCondition, SpatialErrorEstimator, FiniteElementSpaceEnum
from ..expressions import *  # Import grad et al
from .navier_stokes import NavierStokesEquations #type:ignore
from .stabilization import ScalarTransportEquations,ScalarTransportStabilization
from .salt_transport import SaltTransportEquations
from ..materials.generic import *
from ..typings import *
from ..materials.mass_transfer import MassTransferModelBase
from .generic import get_interface_field_connection_space
from .surfactants import SurfactantTransportEquations

if TYPE_CHECKING:
    from ..generic.codegen import EquationTree



def CompositionInitialCondition(fluid_props:AnyFluidProperties,isothermal:bool,initial_temperature:ExpressionNumOrNone=None):
    # sorted throughout this module: required_adv_diff_fields is a set, so its iteration order is
    # randomized per process by PYTHONHASHSEED. Anything derived from it that ends up in a field
    # definition, a residual or a generated expression must not inherit that.
    req_adv_diff = sorted(fluid_props.required_adv_diff_fields)
    ic = fluid_props.initial_condition
    icsettings = {"massfrac_" + n: ic["massfrac_" + n] for n in req_adv_diff if "massfrac_" + n in ic.keys()}
    if not isothermal:
        icT0 = ic.get("temperature")
        if icT0 is None:
            if initial_temperature is None:
                raise RuntimeError(
                    "You must set an initial temperature either by the definition of the fluid (with Mixture(...,temperature=...) or pass it with the initial_temperature kwarg")
            else:
                icT0 = initial_temperature
        icsettings["temperature"] = icT0

    return InitialCondition(degraded_start="auto", IC_name="", **icsettings)


def _salt_equations_for(fluid_props:AnyFluidProperties,salts:"Literal['auto'] | bool",
                        space:FiniteElementSpaceEnum,wind:ExpressionOrNum,
                        advection_by_parts:bool,GCL:bool,dt_factor:ExpressionOrNum,
                        stabilization:"str | Iterable[str] | ScalarTransportStabilization | None",
                        scheme:TimeSteppingScheme,
                        salt_treatment:"Literal['dilute','component']"="dilute")->"Equations | None":
    """The salt transport of a material that carries salts, or None.

    ``salts="auto"`` is the default everywhere: a salted material that silently generated the same
    system as an unsalted one is exactly the trap this closes, and an unsalted one is unaffected
    because there is nothing to add.
    """
    carries=bool(getattr(fluid_props,"get_salts",lambda:{})())
    if not carries:
        if salts is True:
            raise RuntimeError("salts=True, but no salt is dissolved in this material. Use "+
                               "add_salt() on it, or leave salts at its default \"auto\".")
        return None
    if salt_treatment=="component":
        # The composition equations transport the salt themselves, mass fraction and all; what is
        # still needed is the concentration fields everything else is written against.
        from .salt_transport import SaltConcentrationsFromMassFractions
        return SaltConcentrationsFromMassFractions(fluid_props)
    if salts is False:
        # Not nothing: the surface tension law reads a concentration field, so the salt still has to
        # have a value even when nobody transports it. FrozenSaltConcentrations stands down if an
        # electrolyte model turns out to be on the domain, which is how salts=False stays the route
        # to Poisson-Nernst-Planck.
        from .salt_transport import FrozenSaltConcentrations
        return FrozenSaltConcentrations(fluid_props)
    return SaltTransportEquations(fluid_props=fluid_props,space=space,wind=wind,
                                  advection_by_parts=advection_by_parts,GCL=GCL,dt_factor=dt_factor,
                                  stabilization=stabilization,scheme=scheme)


def CompositionDiffusionEquations(fluid_props:AnyFluidProperties, space:FiniteElementSpaceEnum="C2", dt_factor:ExpressionOrNum=1, with_IC:bool=True, spatial_errors:float | None=None,isothermal:bool=True,initial_temperature:ExpressionNumOrNone=None,
                                  compo_stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None,
                                  thermal_stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None,
                                  salts:"Literal['auto'] | bool"="auto") -> Equations:
    """
    Adds diffusion equations for the mass fractions of the components in a multi-component system, but without any Navier-Stokes equations. Can be used e.g. for diffusion-limited species transport in a gas phase.

    Args:
        fluid_props: The fluid properties.
        space: The space for the mass fraction fields.
        dt_factor: Factor for the time derivative in the mass fraction fields.
        with_IC: Include an initial condition for the initial composition.
        spatial_errors: Add spatial error estimators automatically.
        isothermal: If set to ``False``, a temperature equation is included.
        initial_temperature: Initial condition for the temperature.
        compo_stabilization: Optional residual-based stabilization of the mass fraction transport. Without a wind there is nothing for SUPG to act on unless the mesh moves, so this is mainly of interest on an ALE mesh.
        thermal_stabilization: The same for the temperature.
        salts: Whether to transport the salts dissolved in ``fluid_props``, see :py:func:`CompositionFlowEquations`.

    Returns:
        A coupled set of equations for the mass fractions for the diffusive transport of the components in the mixture.
    """
    res:Equations = CompositionAdvectionDiffusionEquations(fluid_props, space=space, dt_factor=dt_factor, wind=0, stabilization=compo_stabilization)
    salt_eqs=_salt_equations_for(fluid_props,salts,space,0,False,False,dt_factor,compo_stabilization,"BDF2")
    if salt_eqs is not None:
        res = res + salt_eqs
    if not isothermal:
        res+=TemperatureConductionEquation(fluid_props,space=space,stabilization=thermal_stabilization)
    if with_IC:
        res += CompositionInitialCondition(fluid_props,isothermal,initial_temperature)
    if spatial_errors is not None:
        if spatial_errors is True:
            compo_fields = ["massfrac_" + n for n in sorted(fluid_props.required_adv_diff_fields)]
            res += SpatialErrorEstimator(*compo_fields, for_which="both")
        elif spatial_errors is not False:
            raise RuntimeError("TODO")
    return res


def CompositionFlowEquations(fluid_props:AnyFluidProperties, compo_space:FiniteElementSpaceEnum="C1", compo_dt_factor:ExpressionOrNum=1, ns_mode:Literal["TH","CR","mini","C1","C2","C2C1","C1C1","C2C2"]="TH", boussinesq:bool=False,
                             gravity:ExpressionNumOrNone=None, bulkforce:ExpressionNumOrNone=None, ns_dt_factor:ExpressionOrNum=1, ns_nl_factor:ExpressionNumOrNone=None, with_IC:bool=True,
                             hele_shaw_thickness:ExpressionNumOrNone=None, spatial_errors:float | None=None, isothermal:bool=True,initial_temperature:ExpressionNumOrNone=None,additional_advection:ExpressionOrNum=0,momentum_scheme:TimeSteppingScheme="BDF2",continuity_scheme:TimeSteppingScheme="BDF2",compo_scheme:TimeSteppingScheme="BDF2",integrate_advection_by_parts:bool=False,wrap_params_in_subexpressions=True,thermal_dt_factor:ExpressionOrNum=1,thermal_adv_factor:ExpressionOrNum=1,GCL:bool=False,
                             ns_stabilization:"str | Iterable[str] | None"=None, ns_stabilization_options:"dict[str,Any] | None"=None,
                             compo_stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None,
                             thermal_stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None,
                             salts:"Literal['auto'] | bool"="auto",salt_space:FiniteElementSpaceEnum | None=None,
                             salt_stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None,
                             salt_treatment:"Literal['dilute','component']"="dilute") -> Equations:
    """
    Assembles a system for multi-component flow with advection-diffusion equations for mass fraction fields of the mixture composition and the Navier-Stokes equations. Potentially, also a temperature field is included.

    Args:
        fluid_props: The fluid properties.
        compo_space: Space for the mass fraction fields
        compo_dt_factor: Factor for the time derivative of the mass fraction fields
        ns_mode: Which Navier-Stokes discretization to use, Taylor-Hood (``"TH"``) or Crouzeix-Raviart (``"CR"``) or MINI Elements (``"mini"``).
        boussinesq: Use Boussinesq approximation
        gravity: Gravity vector [in m/s^2].
        bulkforce: Additional bulk force term.
        ns_dt_factor: Factor for the time derivative of the Navier-Stokes equations.
        ns_nl_factor: Factor for the non-linear term in the Navier-Stokes equations.
        with_IC: Include the initial mixture composition (and temperature) as initial condition.
        hele_shaw_thickness: If set, we consider a Hele-Shaw flow with the given thickness. This modifies a few terms in the Navier-Stokes equations.
        spatial_errors: Add spatial error estimators automatically.
        isothermal: If set to false, a temperature field is included.
        initial_temperature: Temperature initial condition.
        additional_advection: Adds an additional advection term.
        momentum_scheme: Selects the time stepping scheme for the momentum equation.
        continuity_scheme: Selects the time stepping scheme for the continuity equation.
        compo_scheme: Selects the time stepping scheme for the composition equations.
        integrate_advection_by_parts: Integrate the advection terms of the composition equations by parts.
        wrap_params_in_subexpressions: If True, all material properties in the equations are wrapped in subexpressions.
        thermal_dt_factor: Factor for the time derivative of the temperature field.
        thermal_adv_factor: Factor for the advection term of the temperature field.
        GCL: If True, the Geometric Conservation Law is enforced in the ALE formulation of the Navier-Stokes equations and the composition equations.
        salts: Whether to transport the salts dissolved in ``fluid_props``, see
            :py:class:`~pyoomph.equations.salt_transport.SaltTransportEquations`. ``"auto"`` (the
            default) does it whenever the material carries any, which is what makes
            ``water.add_salt("NaCl", 1*milli*molar)`` actually mean something here. Pass ``False``
            for the electrohydrodynamic route, where
            :py:func:`~pyoomph.equations.electrostatics.PoissonNernstPlanck` solves for the ions and
            the potential instead -- the two must not both define the ion concentrations.
        salt_space: Space of the salt fields. Defaults to ``compo_space``.
        salt_stabilization: The same as ``compo_stabilization``, for the salt transport.
        salt_treatment: ``"dilute"`` (the default) transports a dissolved salt as a concentration
            that takes no part in the mass fractions -- right to a few percent up to about half
            molar. ``"component"`` upgrades the material in place so that the salt is an ordinary
            component with a mass fraction, a mole fraction and a share of the volume; the
            composition equations then transport it, and the evaporation interface condition it
            needs is the one they already write for any non-volatile component. See
            :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.treat_salts_as_components`.
        ns_stabilization: Residual-based stabilization of the flow, e.g. ``"SUPG+PSPG"``. Anything but ``None`` switches the flow equations to :py:class:`~pyoomph.equations.stabilized_ns.StabilizedNavierStokes`. This is what makes the inf-sup unstable equal-order pairs ``ns_mode="C1C1"``/``"C2C2"`` usable.
        ns_stabilization_options: Further arguments of :py:class:`~pyoomph.equations.stabilized_ns.StabilizedNavierStokes`, e.g. ``{"tau_formula":"codina","C_I":36}``.
        compo_stabilization: Residual-based stabilization of the mass fraction transport, see :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`.
        thermal_stabilization: The same for the temperature transport.

    The three stabilization switches are independent and all default to *off*, i.e. to exactly the
    system that was assembled before they existed. None of them perturbs the interface physics:
    every added term is an element-interior integral written against the test function of the
    stabilized field only, so the Marangoni stress, the kinematic boundary condition, mass transfer,
    latent heat, surfactant transport and the contact-angle conditions -- which test against
    ``velocity``, ``mesh`` and the interface fields -- are structurally untouched and see the
    stabilized fields only through their values. The one exception is the discontinuity capturing
    term ``"DC"``, which is diffusion-like and therefore does change the natural boundary condition
    of the transported field; it is off by default and its footprint is exposed as
    ``get_stabilization_flux``.

    Returns:
        A coupled set of equations describing the multi-component flow of the mixture
    """

    if salt_treatment=="component":
        # In place, and before anything is built from the material: the composition equations, the
        # interface and the vapour pressures all read the same object, and they have to agree about
        # what a mass fraction means.
        if not getattr(fluid_props,"get_salts",lambda:{})():
            raise RuntimeError("salt_treatment='component' but nothing is dissolved in this "+
                               "material. Dissolve a salt, or leave it at 'dilute'.")
        if getattr(fluid_props,"is_pure",False):
            raise RuntimeError("A salt as a composition field makes the solution a mixture, and a "+
                               "pure liquid cannot hold components. Build it with "+
                               "Mixture(solvent + <c>*get_salt(...), salt_treatment='component').")
        cast(Any,fluid_props).treat_salts_as_components()
    if GCL:
        if not integrate_advection_by_parts:
            integrate_advection_by_parts=True
            print("WARNING: For GCL, the advection term of the composition equations is integrated by parts automatically.")
    ns_common:dict[str,Any] = dict(fluid_props=fluid_props, boussinesq=boussinesq, gravity=gravity,
                                   bulkforce=bulkforce, dt_factor=ns_dt_factor, nonlinear_factor=ns_nl_factor,
                                   momentum_scheme=momentum_scheme, continuity_scheme=continuity_scheme,
                                   wrap_params_in_subexpressions=wrap_params_in_subexpressions,
                                   hele_shaw_thickness=hele_shaw_thickness, GCL=GCL)
    ns:Equations
    inf_sup_stable = ns_mode in {"TH","CR","SV","mini","C2C1"}
    if not inf_sup_stable and (ns_stabilization is None or "PSPG" not in ns_stabilization):
        # Not an error: mode="C1"/"C2" were reachable before and stay reachable. But without PSPG the
        # equal-order pairs are unsolvable rather than merely inaccurate, so say so once.
        print("WARNING: ns_mode='"+str(ns_mode)+"' is an equal-order velocity/pressure pair. It is "
              "inf-sup unstable and the pressure checkerboards unless ns_stabilization contains 'PSPG'.")
    if ns_stabilization is None and not ns_stabilization_options:
        # Deliberately the plain class, not StabilizedNavierStokes(stabilization="none"): the default
        # must generate literally the same code as before, not merely an equivalent system. The alias
        # lookup is the identity for TH/CR/mini, so nothing changes for the documented values.
        from .stabilized_ns import _SPACE_ALIASES
        ns = NavierStokesEquations(mode=cast(Any,_SPACE_ALIASES.get(ns_mode,ns_mode)), **ns_common)
    else:
        from .stabilized_ns import StabilizedNavierStokes
        if ns_stabilization is not None and "PSPG" in ns_stabilization and inf_sup_stable:
            print("WARNING: PSPG on the inf-sup stable "+str(ns_mode)+" pair. It is not needed there and "
                  "measurably degrades the pressure error and the divergence of the velocity.")
        ns = StabilizedNavierStokes(space=ns_mode, stabilization=ns_stabilization if ns_stabilization is not None else "none",
                                    **(ns_stabilization_options or {}), **ns_common)
    wind=var("velocity")+additional_advection

    cp = CompositionAdvectionDiffusionEquations(fluid_props=fluid_props, space=compo_space, dt_factor=compo_dt_factor,
                                                boussinesq=boussinesq, wind=wind,integrate_advection_by_parts=integrate_advection_by_parts,wrap_params_in_subexpressions=wrap_params_in_subexpressions,GCL=GCL,scheme=compo_scheme,stabilization=compo_stabilization)
    res = ns + cp
    salt_eqs=_salt_equations_for(fluid_props,salts,salt_space if salt_space is not None else compo_space,
                                 wind,integrate_advection_by_parts,GCL,compo_dt_factor,
                                 salt_stabilization,compo_scheme,salt_treatment)
    if salt_eqs is not None:
        res = res + salt_eqs
    if not isothermal:
        res+=TemperatureAdvectionConductionEquation(fluid_props,space=compo_space,wind=wind,adv_factor=thermal_adv_factor,dt_factor=thermal_dt_factor,stabilization=thermal_stabilization)
    if with_IC:
        res += CompositionInitialCondition(fluid_props,isothermal,initial_temperature)
    if spatial_errors is not None:
        if spatial_errors is True:
            compo_fields = ["massfrac_" + n for n in sorted(fluid_props.required_adv_diff_fields)]
            res += SpatialErrorEstimator(*compo_fields, for_which="both", velocity=1)
        elif isinstance(spatial_errors,dict):
            res += SpatialErrorEstimator(**spatial_errors)
        elif spatial_errors is not False:
            raise RuntimeError("TODO")

    return res


class CompositionAdvectionDiffusionEquations(ScalarTransportEquations):
    """
    Represents the advection-diffusion equation for a single component in a multi-component system. 
    The equation is given by:
        
        partial_t(massfrac) + div(velocity*massfrac) = div(D*grad(massfrac)) + reaction_rate

    where massfrac is the mass fraction of the component, velocity is the velocity field, D is the diffusion coefficient, and reaction_rate is the reaction rate.
    
    Args:
        fluid_props(AnyFluidProperties): The fluid properties. Default is None.
        space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. second order continuous Lagrangian elements.
        wind(ExpressionOrNum): The wind field. Default is 0.
        dt_factor(ExpressionOrNum): The temporal factor. Default is 1.
        boussinesq(bool): Whether to consider the Boussinesq approximation. Default is False.
        integrate_advection_by_parts(bool): Whether to integrate the advection term by parts. Default is False.
        wrap_params_in_subexpressions(bool): Whether to wrap the parameters in subexpressions using GiNaC. Default is True.
        GCL(bool): Whether to consider the Generalized Continuity Equation. Default is False.
        scheme(TimeSteppingScheme): The time stepping scheme. Default is "BDF2".
        stabilization: Optional residual-based stabilization (SUPG etc.) of the mass fraction transport, see :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`. ``None`` (the default) adds nothing at all.

    .. note::
        **Scaling to set on problem level.** The residual is a *mass* balance, so each mass fraction
        is tested with ``scale_factor("temporal")/scale_factor("mass_density")``. ``mass_density`` is
        no field of the system and must be set by ``problem.set_scaling(mass_density=...)``.
    """

    def __init__(self, fluid_props:AnyFluidProperties, *, space:FiniteElementSpaceEnum="C2", wind:ExpressionOrNum=var("velocity"), dt_factor:ExpressionOrNum=1, boussinesq:bool=False,integrate_advection_by_parts:bool=False,wrap_params_in_subexpressions:bool=True, GCL:bool=False, scheme:TimeSteppingScheme="BDF2", stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
        super().__init__()
        self.dt_factor = dt_factor
        self.space:FiniteElementSpaceEnum = space
        self.wind = wind
        self.fluid_props = fluid_props
        self.fieldnames:list[str] = []
        self.component_names:dict[str,str] = {}
        self.stop_on_zero_diffusive_flux = True
        self.boussinesq = boussinesq
        self.integrate_advection_by_parts=integrate_advection_by_parts
        for n in sorted(self.fluid_props.required_adv_diff_fields):
            self.component_names["massfrac_" + n] = n
            self.fieldnames.append("massfrac_" + n)
        self.requires_interior_facet_terms=is_DG_space(self.space)
        self.DG_alpha=1
        self.wrap_params_in_subexpressions=wrap_params_in_subexpressions
        self.GCL=GCL
        self.scheme:TimeSteppingScheme=scheme
        self._init_stabilization(stabilization)

    def optional_subexpression(self,expr):
        if self.wrap_params_in_subexpressions:
            return subexpression(expr)
        else:
            return expr

    def define_fields(self):
        #my_domain = self.get_my_domain()  # My domain. Make sure that all additional variables are expanded here!
        my_domain =None # Actually a bad idea: e.g. Marangoni fill calculate the gradient at the interface, i.e. grad(sigma) -> grad(passive_field) -> grad(active_field[bulk]) ! WRONG
        if self.fluid_props.is_pure:
            self.define_field_by_substitution("massfrac_" + self.fluid_props.name, 1, also_on_interface=True)
            self.define_testfunction_by_substitution("massfrac_" + self.fluid_props.name, Expression(0), also_on_interface=True)
            self.define_field_by_substitution("molefrac_" + self.fluid_props.name, 1, also_on_interface=True)
            cmol = self.optional_subexpression(self.fluid_props.mass_density / self.fluid_props.molar_mass)
            self.define_field_by_substitution("molarconc_" + self.fluid_props.name, cmol, also_on_interface=True)
        else:
            assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
            remaining = 1  # Remaining mass fraction for the passive one
            remaining_test = Expression(0)  # Remaining test function

            # Get the passive field and add a substituion variable and testfunction for it
            # var(<passive mass fraction>) = 1 - sum(var(<solved mass fractions>))
            # testfunction(<passive mass fraction>)=- sum(testfunction(<solved mass fractions>))
            for f in self.fieldnames:
                self.define_scalar_field(f, self.space)
                remaining -= var(f, domain=my_domain)
                remaining_test -= testfunction(f, domain=my_domain,dimensional=False) # Dimensions are already introduced
            assert self.fluid_props.passive_field is not None
            self.define_field_by_substitution("massfrac_" + self.fluid_props.passive_field, remaining,
                                              also_on_interface=True)
            self.define_testfunction_by_substitution("massfrac_" + self.fluid_props.passive_field, remaining_test,
                                                     also_on_interface=True)

            # Also add substitutions for the molar fractions
            sum_massfrac_by_molar_mass = 0  # Sum of massfraction/molar_mass
            for n, c in self.fluid_props.pure_properties.items():
                sum_massfrac_by_molar_mass += var("massfrac_" + n, domain=my_domain) / c.molar_mass
                self.define_field_by_substitution("molefrac_" + n,
                                                  (var("massfrac_" + n, domain=my_domain) / c.molar_mass) / var(
                                                      "_sum_massfrac_by_molar_mass", domain=my_domain),
                                                  also_on_interface=True)
                cmol = self.optional_subexpression(
                    var("massfrac_" + n, domain=my_domain) * evaluate_in_domain(self.fluid_props.mass_density,
                                                                                my_domain) / c.molar_mass)
                self.define_field_by_substitution("molarconc_" + n, cmol, also_on_interface=True)
            sum_massfrac_by_molar_mass = self.optional_subexpression(sum_massfrac_by_molar_mass)
            self.define_field_by_substitution("_sum_massfrac_by_molar_mass", sum_massfrac_by_molar_mass,
                                              also_on_interface=True)

    def get_diffusion_coefficient(self, f1:str, f2:str | None=None) -> ExpressionNumOrNone:
        assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
        if f2 is None:
            f2 = f1
        return self.fluid_props.get_diffusion_coefficient(f1, f2, default=0)

    def get_diffusive_mass_flux_expression_for(self, fn:str) -> ExpressionOrNum:
        assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
        return self.fluid_props.get_diffusive_mass_flux_for(fn)

    # ---- hooks of the stabilization base class ---------------------------------------------------

    def stabilized_fieldnames(self) -> list[str]:
        # Only the *solved* fields. The passive one is a substitution, and its test function is minus
        # the sum of these (define_fields), so writing against it would smear a term into every row.
        return self.fieldnames

    def stabilization_wind(self) -> ExpressionOrNum:
        return self.wind

    def stabilization_residual_scale(self, fieldname:str) -> ExpressionOrNum:
        # The residual of these equations is a *mass* balance, so everything carries rho
        return scale_factor("mass_density") if self.boussinesq else self.fluid_props.mass_density

    def stabilization_diffusivity(self, fieldname:str) -> ExpressionOrNum:
        # tau only needs a representative scalar diffusivity, i.e. the diagonal entry. The full
        # matrix (and thermophoresis) still enters the strong residual through the diffusive flux.
        D = self.get_diffusion_coefficient(self.component_names[fieldname])
        assert D is not None
        return D

    def strong_residual(self, fieldname:str) -> Expression:
        rho_factor = self.stabilization_residual_scale(fieldname)
        f = var(fieldname)
        conservative = self.stab_cfg.conservative_residual
        if conservative == "auto":
            conservative = self.integrate_advection_by_parts
        if conservative:
            if self.GCL:
                # The GCL branch assembles the time derivative of the whole integral, i.e. strongly
                # d_t(rho f) + div(rho f u_mesh), which together with -weak(rho (a-u_m) f, grad v)
                # gives the conservative form below.
                R = self.dt_factor * partial_t(rho_factor * f) + div(rho_factor * self.wind * f)
            else:
                R = rho_factor * self.dt_factor * partial_t(f) + div(rho_factor * self.wind * f)
        else:
            R = rho_factor * (self.dt_factor * partial_t(f) + dot(self.wind, grad(f)))
        if self.stab_cfg.include_diffusion_in_residual:
            # The weak diffusion term is -weak(Jdiff, grad(v)), so the strong counterpart is +div(J).
            # Jdiff already carries the full diffusion matrix, thermophoresis and the -rho factor.
            R = R + div(convert_to_expression(
                self.get_diffusive_mass_flux_expression_for(self.component_names[fieldname])))
        if isinstance(self.fluid_props, MixtureLiquidProperties):
            R = R - self.fluid_props.get_reaction_rate(self.component_names[fieldname])
        return R

    def define_scaling(self):
        for fn in self.fieldnames:
            self.set_test_scaling({fn: scale_factor("temporal") / scale_factor("mass_density")})
        if not self.fluid_props.is_pure:
            assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
            assert self.fluid_props.passive_field is not None
            self.set_test_scaling({"massfrac_"+self.fluid_props.passive_field:scale_factor("temporal") / scale_factor("mass_density")})


    def define_residuals(self):
        rho_ref = scale_factor("mass_density")
        rho = self.fluid_props.mass_density
        ts=lambda expr : time_scheme(self.scheme, expr)
        for fn in self.fieldnames:
            f, f_test = var_and_test(fn)
            Jdiff = self.get_diffusive_mass_flux_expression_for(self.component_names[fn])
            if self.stop_on_zero_diffusive_flux and is_zero(Jdiff):
                raise RuntimeError("component " + self.component_names[fn] + " has no diffusion terms!")
            # TODO: This is not correct yet
            if self.boussinesq:
                rho_factor = rho_ref
            else:
                rho_factor = rho
            if self.integrate_advection_by_parts:
                if self.GCL:
                    # dt_factor scales the transient term. Here that term is the derivative of the
                    # *whole* integral, so the factor multiplies it from outside instead of sitting
                    # inside the d/dt, where the history terms would carry it at their own time
                    # levels. This used to be a plain add_dweak_dt(), i.e. dt_factor was silently
                    # dropped in this branch alone while the two below both applied it.
                    self.add_residual(self.dt_factor*time_derivative_of_integral(
                        weak(rho_factor * f, f_test), scheme=self.scheme))
                    w=mesh_velocity(scheme=self.scheme)
                    self.add_residual(-weak(ts(rho_factor *(self.wind-w)*f),grad(f_test)))
                else:
                    res = ts(rho_factor * (self.dt_factor * partial_t(f)))
                    self.add_residual(weak(res, f_test))
                    self.add_residual(-weak(ts(rho_factor *self.wind*f),grad(f_test)))
            else:
                self.add_residual(weak(ts(rho_factor * (self.dt_factor * partial_t(f) + dot(self.wind, grad(f)))), f_test))

            self.add_residual(-weak(ts(Jdiff), grad(f_test)))
            if isinstance(self.fluid_props,MixtureLiquidProperties):
                reaction_rate=self.fluid_props.get_reaction_rate(self.component_names[fn])
                self.add_residual(-weak(ts(reaction_rate),f_test))
            if self.requires_interior_facet_terms:
                raise RuntimeError("TODO: DG implementation")
        self.add_stabilization_residuals(ts)


class CompositionAdvectionDiffusionFluxEquations(InterfaceEquations):
    """
    Represents the flux through the interface that naturally arises from the integration by parts of the diffusion term in the advection-diffusion equation.
        
    Args:
        **kwargs(ExpressionOrNum): The fluxes. The keys are the names of the components and the values are the mass fluxes. 
    """
            
        
        
    def __init__(self, **kwargs:ExpressionOrNum):
        super(CompositionAdvectionDiffusionFluxEquations, self).__init__()
        self.fluxes = sorted_field_kwargs(kwargs)

    def define_residuals(self):
        parent = self.get_parent_equations(CompositionAdvectionDiffusionEquations)
        for name, flux in self.fluxes.items():
            fname = "massfrac_" + name
            test = testfunction(fname)
            self.add_residual(weak(flux, test))
            if isinstance(parent, CompositionAdvectionDiffusionEquations):
                # A stabilized bulk deposits its own flux here; subtract it so that the flux imposed
                # is the physical one. Zero unless natural_bc_correction is switched on.
                self.add_residual(-weak(parent.get_stabilization_flux(fname, self.get_normal(),
                                                                      self.get_parent_domain()), test))


class MultiComponentNavierStokesInterface(InterfaceEquations):
    """
    Represents a multi-component free surface interface between two fluids with multiple components.
    It considers mass transfer by a mass transfer model and automatically connects the velocity if necessary.

    Args:
        interface_props(AnyFluidFluidInterface): The interface properties (e.g. surface tension).
        kinbc_name(str): The name of the kinematic boundary condition multiplier. Default is "_kin_bc". 
        velo_connect_prefix(str): The prefix for the velocity connection fields. Default is "_lagr_conn_".
        masstransfer_model(Union[MassTransferModelBase,Literal[False]]): The mass transfer model (e.g. UNIFAC). Default is None.
        static(Union[Literal["auto"],bool]): Whether the interface is static. Default is "auto".
        surface_tension_theta(float): The theta method to consider the surface tension (0: explicit, i.e. from last step, 1: fully implicit). Default is 1.
        total_mass_loss_factor_inside(ExpressionOrNum): Multiplicative factor for the total mass loss inside the domain. Default is 1.
        total_mass_loss_factor_outside(ExpressionOrNum): Multiplicative factor for the total mass loss outside the domain. Default is 1.
        surface_tension_projection_space(Optional[FiniteElementSpaceEnum]): The finite element space for the surface tension projection. Default is None.
        additional_normal_traction(ExpressionOrNum): Additional normal traction. Default is 0.
        surface_tension_gradient_directly(bool): Whether to consider the surface tension gradient directly. Default is False.
        use_highest_space_for_velo_connection(bool): Whether to use the highest space for the velocity connection. Default is False.
        kinematic_bc_coordsys(Optional[BaseCoordinateSystem]): The coordinate system for the kinematic boundary condition. Default is None.
        kinematic_bc_space: The finite element space for the kinematic boundary condition. Default is None, means auto-select.
        additional_masstransfer_scale(ExpressionOrNum): Additional mass transfer scale. Default is 1.
        additional_kin_bc_test_scale(ExpressionOrNum): Additional kinematic boundary condition test scale. Default is 1.
        static_normal_interface_motion(ExpressionOrNum): If solved on a static mesh, we can mimic the interface motion by moving it in normal direction with this rate. Default is 0.
        static_interface_motion_testfunction(ExpressionNumOrNone): If set, we solve that the total outflux is zero by adjusting this.
        project_interface_flux(bool): If set to True, the interface flux (kinematic BC) is projected and used for the kinematic BC. Default is False.
        surface_tension_factor(ExpressionOrNum): The surface tension factor. Multiplicative factor for the imposition of the surface tension. Default is 1.
        surfactant_transport: How the surfactants registered on the interface properties are transported. ``None`` (the default) uses :py:class:`~pyoomph.equations.surfactants.SurfactantTransportEquations` in its conservative form, which keeps the total amount of an insoluble surfactant exact rather than to the order of the time stepping. Pass a configured instance to change the form, the variable or the stabilization -- ``SurfactantTransportEquations(form="legacy")`` reproduces the pre-2026 behaviour bit for bit. ``False`` switches the surfactant equations off entirely, e.g. to supply your own.

    .. note::
        **Scaling to set on problem level.** The mass transfer rate is scaled with
        ``scale_factor("velocity")*scale_factor("mass_density")``, i.e. besides the velocity also
        ``mass_density`` -- no field of the system -- has to be set by
        ``problem.set_scaling(mass_density=...)``.
    """
            
        
    kinematic_bc_coordinate_sys = _deprecated_attribute_alias("kinematic_bc_coordinate_sys","kinematic_bc_coordsys")

    @_deprecated_kwargs(kinematic_bc_coordinate_sys="kinematic_bc_coordsys")
    def __init__(self, interface_props:AnyFluidFluidInterface, *, kinbc_name:str="_kin_bc", velo_connect_prefix:str="_lagr_conn_",
                 masstransfer_model:MassTransferModelBase | Literal[False] | None=None, static:Literal["auto"] | bool="auto", surface_tension_theta:float=1, total_mass_loss_factor_inside:ExpressionOrNum=1,total_mass_loss_factor_outside:ExpressionOrNum=1,
                 surface_tension_projection_space:FiniteElementSpaceEnum | None=None,additional_normal_traction:ExpressionOrNum=0,surface_tension_gradient_directly:bool=False,use_highest_space_for_velo_connection:bool=False,kinematic_bc_coordsys:BaseCoordinateSystem | None=None,kinematic_bc_space:FiniteElementSpaceEnum | None=None,additional_masstransfer_scale=1,additional_kin_bc_test_scale=1,static_normal_interface_motion:ExpressionOrNum=0,static_interface_motion_testfunction:ExpressionNumOrNone=None,project_interface_flux:bool=False,surface_tension_factor:ExpressionOrNum=1,surfactant_transport:"SurfactantTransportEquations | Literal[False] | None"=None):
        super(MultiComponentNavierStokesInterface, self).__init__()
        self.interface_props = interface_props
        self.kinbc_name = kinbc_name
        self.velo_connect_prefix = velo_connect_prefix
        self.surface_tension_theta = surface_tension_theta 
        if masstransfer_model is None:
            self.masstransfer_model = self.interface_props.get_mass_transfer_model()
        elif masstransfer_model == False:
            self.masstransfer_model = None
        else:
            self.masstransfer_model=masstransfer_model
        self.masstransfer_model
        self._has_opposite_flow = False
        self.static = static
        self.total_mass_loss_factor_inside = total_mass_loss_factor_inside
        self.total_mass_loss_factor_outside=total_mass_loss_factor_outside
        self.surface_tension_projection_space:FiniteElementSpaceEnum | None = surface_tension_projection_space
        self.surface_tension_gradient_directly=surface_tension_gradient_directly
        self.additional_normal_traction=additional_normal_traction
        self.surfactant_advect_velo_name="_uinterf_proj"
        self.surfactant_advect_velo_space:FiniteElementSpaceEnum="C2"
        self.use_highest_space_for_velo_connection=use_highest_space_for_velo_connection
        self.kinematic_bc_coordsys=kinematic_bc_coordsys
        self.kinematic_bc_space:FiniteElementSpaceEnum | None=kinematic_bc_space
        self.additional_masstransfer_scale=additional_masstransfer_scale
        self.additional_kin_bc_test_scale=additional_kin_bc_test_scale
        self.static_normal_interface_motion=static_normal_interface_motion
        self.static_interface_motion_testfunction=static_interface_motion_testfunction
        self.project_interface_flux=project_interface_flux
        self.surface_tension_factor=surface_tension_factor
        self.surfactant_transport=surfactant_transport
        self._surfactant_transport:"SurfactantTransportEquations | None"=None
        if isinstance(surfactant_transport,SurfactantTransportEquations) and surfactant_transport.requires_interior_facet_terms:
            # A DG surfactant needs the '_internal_facets_' child of *this* domain, and that child is
            # created from this flag in EquationTree._fill_dummy_equations - which runs long before any
            # residual is assembled, so it has to be set here in the constructor rather than when the
            # delegate first assembles something.
            self.requires_interior_facet_terms=True

    def _resolve_surfactant_transport(self)->"SurfactantTransportEquations | None":
        """Pick the surfactant handler and bind it to this interface's material properties.

        Called at the top of every assembly hook, since define_fields/define_scaling/define_residuals
        each run on their own and none of them may depend on another having gone first.
        """
        if self.surfactant_transport is False:
            self._surfactant_transport=None
            return None
        if self._surfactant_transport is None:
            if self.surfactant_transport is None:
                self._surfactant_transport=SurfactantTransportEquations(
                    advection_velocity_name=self.surfactant_advect_velo_name,
                    advection_velocity_space=self.surfactant_advect_velo_space)
            else:
                self._surfactant_transport=cast(SurfactantTransportEquations,self.surfactant_transport)
        self._surfactant_transport._bind(self.interface_props)
        return self._surfactant_transport

    def get_surfactant_transport(self)->"SurfactantTransportEquations | None":
        """The :py:class:`~pyoomph.equations.surfactants.SurfactantTransportEquations` in use, if any."""
        return self._resolve_surfactant_transport()

    def uses_projected_surfactant_velocity(self)->bool:
        """Whether the projected advection velocity field ``_uinterf_proj`` exists.

        It only does in the legacy transport form. The conservative form integrates the advection by
        parts, so its natural end condition is already zero flux and there is nothing for the contact
        line to constrain - see the module docstring of :py:mod:`~pyoomph.equations.surfactants`.
        """
        st=self._resolve_surfactant_transport()
        return st is not None and st.uses_projected_advection_velocity(self)

    def define_fields(self):
        # Add kinematic boundary condition multiplier
        nseqs=self.get_parent_equations(of_type=NavierStokesEquations)
        assert isinstance(nseqs,NavierStokesEquations)
        inside_space=nseqs.get_velocity_space_from_mode(for_interface=True)
        
#        if nseqs.mode=="mini"
        kinbc_space=inside_space
        static=self.static
        if static=="auto":
            pdom=self.get_current_code_generator().get_parent_domain()
            assert pdom is not None # An interface always has a parent (bulk) domain
            static=not pdom._coordinates_as_dofs

        if not static in {"auto",False,True}:
            raise RuntimeError("property static must be either 'auto', True or False")
        if self.kinematic_bc_space is None:
            if not static:
                pdom=self.get_current_code_generator().get_parent_domain()
                assert pdom is not None # An interface always has a parent (bulk) domain
                pos_space=pdom._coordinate_space
                if pos_space=="":
                    raise RuntimeError("Find out the coordinate space:"+str())
                if pos_space=="C2TB":
                    kinbc_space="C2"
                elif pos_space=="C1TB":
                    kinbc_space="C1"
                else:
                    kinbc_space=cast("FiniteElementSpaceEnum",pos_space)
        else:
            kinbc_space=self.kinematic_bc_space

        self.define_scalar_field(self.kinbc_name, kinbc_space )
        # If other side has a NavierStokes equation, add also velocity connection
        self._has_opposite_flow = False
        if self.get_opposite_side_of_interface(raise_error_if_none=False):

            opp = self.get_opposite_side_of_interface()
            oppblk = opp.get_parent_domain()
            if oppblk is not None:
                oppblkeq=oppblk.get_equations()            
                if oppblkeq is not None:
                    oppns=oppblkeq.get_equation_of_type(NavierStokesEquations)
                    if oppns is not None and isinstance(oppns,NavierStokesEquations):
                        outside_space=oppns.get_velocity_space_from_mode(for_interface=True)
                        conn_space=get_interface_field_connection_space(inside_space,outside_space,use_highest_space=self.use_highest_space_for_velo_connection,
                                                                       parent_space=str(self.get_parent_domain()._coordinate_space),
                                                                       parent_dim=int(self.get_parent_domain().dimension))
                        assert conn_space!=""
                        fields = ["velocity_x", "velocity_y", "velocity_z"]
                        fields = fields[0:self.get_nodal_dimension()]
                        if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                            fields+=["velocity_phi"]
                        for f in fields:
                            self.define_scalar_field(self.velo_connect_prefix + f, conn_space)  # TODO: Other velocity spaces?
                        self._has_opposite_flow = True

        facet_space=nseqs.get_velocity_space_from_mode(for_interface=True)

        # The surfactant fields, their advection velocity (legacy form only) and the transport
        # residuals all live in pyoomph.equations.surfactants now, so that the same code serves this
        # class and a standalone SurfactantTransportEquations on a plain free surface.
        st = self._resolve_surfactant_transport()
        if st is not None:
            if st.space is None:
                st.space = facet_space
            st._define_fields_on(self)

        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props) 
            self.masstransfer_model.define_fields(self)
            self.masstransfer_model._clean_up_for_code() 

        if self.surface_tension_projection_space is not None:
            self.define_scalar_field("_surf_tension", self.surface_tension_projection_space)
            
        if self.project_interface_flux:
            self.define_scalar_field("interface_flux", facet_space, scale=scale_factor("velocity"), testscale=1/scale_factor("velocity"))

        
        if self.get_opposite_side_of_interface(raise_error_if_none=False):
            opp = self.get_opposite_side_of_interface()
            oppblk = opp.get_parent_domain()
            if oppblk is not None:
                oppblkeq=oppblk.get_equations()            
                if oppblkeq is not None:
                    oppns=oppblkeq.get_equation_of_type(NavierStokesEquations)
                    if oppns is not None and isinstance(oppns,NavierStokesEquations):        
                        aziinfo=self.get_azimuthal_r0_info()
                        csys=self.get_coordinate_system()
                        myfields = ["velocity_x", "velocity_y", "velocity_z"]
                        myfields = myfields[0:self.get_nodal_dimension()]
                        if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                            myfields+=["velocity_phi"]
                        for f in myfields:
                            for i in [0,1,2]:
                                if f in aziinfo[i]:
                                    aziinfo[i].add(self.velo_connect_prefix + f)
                                else:
                                    if self.velo_connect_prefix + f in aziinfo[i]:
                                        aziinfo[i].remove(self.velo_connect_prefix + f)


    def define_scaling(self):
        super(MultiComponentNavierStokesInterface, self).define_scaling()
        # The surfactant scalings used to be set here *and* again further down, the second call
        # overwriting the first with the same values. They are set once now, by the transport class.
        scals:dict[str,str | ExpressionOrNum] = {}
        scals["mass_transfer_rate"] = scale_factor("velocity") * scale_factor("mass_density")*self.additional_masstransfer_scale
        self.set_scaling(scals)
        self._add_named_numerical_factor(surface_tension_term=test_scale_factor("velocity")/scale_factor("spatial"))

        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props) 
            self.masstransfer_model.setup_scaling(self)
            self.masstransfer_model._clean_up_for_code()

        static=self.static
        if static=="auto":
            static=not self.get_current_code_generator()._coordinates_as_dofs

        if not static in {"auto",False,True}:
            raise RuntimeError("property static must be either 'auto', True or False")

        if static:
            self.set_scaling({self.kinbc_name: 1 / test_scale_factor("velocity")})
            self.set_test_scaling({self.kinbc_name: self.additional_kin_bc_test_scale / scale_factor("velocity")})
        else:
            self.set_scaling({self.kinbc_name: 1 / test_scale_factor("mesh")})
            self.set_test_scaling({self.kinbc_name: 1 / scale_factor("velocity")})

        st = self._resolve_surfactant_transport()
        if st is not None:
            st._setup_scaling_on(self)

        if self._has_opposite_flow:
            fields = ["velocity_x", "velocity_y", "velocity_z"]
            fields = fields[0:self.get_nodal_dimension()]
            if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                fields+=["velocity_phi"]
            vcscales = {}
            vctscales = {}
            for f in fields:
                vcscales[self.velo_connect_prefix + f] = 1 / test_scale_factor("velocity")
                vctscales[self.velo_connect_prefix + f] = 1 / scale_factor("velocity")
            self.set_scaling(vcscales)
            self.set_test_scaling(vctscales)

        if self.surface_tension_projection_space:
            self.set_scaling(_surf_tension=scale_factor("spatial") / test_scale_factor("velocity"))
            self.set_test_scaling(_surf_tension=1 / scale_factor("_surf_tension"))

    def define_residuals(self):
        u, u_test = var_and_test("velocity")
        R, R_test = var_and_test("mesh")
        l, l_test = var_and_test(self.kinbc_name)
        n = self.get_normal()

        inner_bulk_eqs = self.get_parent_domain().get_equations()
        ns_inner = inner_bulk_eqs.get_equation_of_type(NavierStokesEquations)
        assert isinstance(ns_inner,NavierStokesEquations)
        assert ns_inner.fluid_props is not None
        rho_inner = ns_inner.fluid_props.mass_density

        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props) 
            partial_mass_transfer_rates = self.masstransfer_model.get_all_masstransfer_rates()
            self.masstransfer_model.define_residuals(self)
            total_mass_transfer_rate = (sum([j for _, j in partial_mass_transfer_rates.items()]))
            self.masstransfer_model._clean_up_for_code() 
        else:
            total_mass_transfer_rate = 0
            partial_mass_transfer_rates:dict[str,Expression] = {}

        # Kinematic boundary condition
        actual_total_transfer_by_rho_inner = dot(mesh_velocity(scheme=ns_inner.momentum_scheme)+self.static_normal_interface_motion*n - u, n)
        kin_bc =  actual_total_transfer_by_rho_inner + self.total_mass_loss_factor_inside *total_mass_transfer_rate / rho_inner
        if self.project_interface_flux:
            iflux,ifluxtest=var_and_test("interface_flux")
            self.add_weak(iflux-kin_bc,ifluxtest)
            kin_bc=iflux
        static = self.static
        if static == "auto":
            static = not self.get_current_code_generator()._coordinates_as_dofs


        self.add_residual(weak(kin_bc, l_test,coordsys=self.kinematic_bc_coordsys))
        if static:
            self.add_residual(-weak(l, dot(n, u_test),coordsys=self.kinematic_bc_coordsys))
            if self.static_interface_motion_testfunction is not None:
                self.add_residual(weak(kin_bc,self.static_interface_motion_testfunction,coordsys=self.kinematic_bc_coordsys))
        else:
            self.add_residual(weak(l, dot(n, R_test),coordsys=self.kinematic_bc_coordsys))

        # dynamic boundary condition
        surf_tens = self.interface_props.surface_tension

        if surf_tens is None:
            raise RuntimeError("No surface tension set in the interface properties " + str(self.interface_props))
        

        if self.surface_tension_gradient_directly:
            if not static:
                raise RuntimeError("Cannot use surface_tension_gradient_directly=True if not static")

        if self.surface_tension_projection_space is not None:            
            surf_tens_proj, surf_tens_proj_test = var_and_test("_surf_tension")
            self.add_residual(weak(surf_tens_proj - surf_tens, surf_tens_proj_test))

            if self.surface_tension_theta != 1:
                surf_tens_proj = evaluate_in_past(surf_tens_proj, 1 - self.surface_tension_theta)
            if self.surface_tension_gradient_directly:
                self.add_residual(-weak( grad(self.surface_tension_factor*surf_tens_proj), u_test))
            else:
                self.add_residual(weak(self.surface_tension_factor*surf_tens_proj, div(u_test)))
        else:
            if self.surface_tension_gradient_directly:
                raise RuntimeError("Can only use surface_tension_gradient_directly if surface_tension_projection_space is set")
            if self.surface_tension_theta != 1:
                # theta states outright which time level the surface tension is taken at, so it
                # *replaces* the momentum scheme's own treatment of this term instead of composing
                # with it. Wrapping the already lagged sigma in time_scheme() again made theta=0
                # collapse back onto theta=1 exactly -- identical trajectories over 218 steps of an
                # intense-Marangoni instability, right down to the divergence time -- while theta=0.5
                # survived, i.e. only the full one-step lag was absorbed. The projected branch above
                # has never had the extra wrapper and theta has always worked there.
                self.add_residual(weak(self.surface_tension_factor
                                       * evaluate_in_past(surf_tens, 1 - self.surface_tension_theta),
                                       div(u_test)))
            else:
                self.add_residual(weak(time_scheme(ns_inner.momentum_scheme,self.surface_tension_factor*surf_tens), div(u_test)))

        
        self.add_residual(weak(self.additional_normal_traction,dot(n,u_test)))

        # A stabilized bulk leaves its own traction on this boundary (see
        # StokesEquations.get_stabilization_traction); subtract it so that the free surface balances
        # the *physical* stress against the surface tension, i.e. so that a flow stabilization cannot
        # perturb the Marangoni/mass-transfer balance. This is what NavierStokesFreeSurface does; the
        # class here is not a subclass of it, so it has to be repeated. Zero without stabilization.
        self.add_residual(-weak(ns_inner.get_stabilization_traction(n,self.get_parent_domain()),u_test))

        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props)
            vap_recoil=self.masstransfer_model.get_vapor_recoil_pressure()
            self.masstransfer_model._clean_up_for_code()
            self.add_residual(weak(time_scheme(ns_inner.momentum_scheme, vap_recoil),dot(n,u_test)))

        #total_mass_flux = actual_total_transfer_by_rho_inner * rho_inner
        if self.masstransfer_model is not None:
            # Component dynamics inside
            for name in sorted(ns_inner.fluid_props.required_adv_diff_fields):
                fname = "massfrac_" + name
                wi, wi_test = var_and_test(fname)
                # Both of them are fine# TODO: But at pinned contact lines, we have to see
                advdiffu_inner=inner_bulk_eqs.get_equation_of_type(CompositionAdvectionDiffusionEquations)
                assert isinstance(advdiffu_inner,CompositionAdvectionDiffusionEquations)
                assert partial_mass_transfer_rates is not None
                if advdiffu_inner.integrate_advection_by_parts:
                    #flux = wi * rho_inner*dot(var("velocity")-0*partial_t(var("mesh")),var("normal"))  -wi * total_mass_transfer_rate + partial_mass_transfer_rates.get(name, 0)
                    flux=partial_mass_transfer_rates.get(name, 0)
                else:
                    flux = -wi * total_mass_transfer_rate + partial_mass_transfer_rates.get(name, 0)
                # flux = wi * total_mass_flux + partial_mass_transfer_rates.get(name, 0)
                self.add_residual(weak(time_scheme(advdiffu_inner.scheme, flux), wi_test))
                # The branch above concerns the *Galerkin* advection's own surface term and is
                # orthogonal to the stabilization footprint, so this correction is the same either
                # way. Zero unless natural_bc_correction is switched on.
                self.add_residual(-weak(advdiffu_inner.get_stabilization_flux(fname,n,self.get_parent_domain()),wi_test))

            # Salt dynamics inside. A salt does not evaporate, so its interface flux is the j_i = 0
            # case of the loop above -- but *not* nothing: the liquid keeps flowing through the
            # receding surface while the salt does not, and without this term the salt would be
            # carried out with the vapour instead of piling up. Divided by rho because a salt field
            # is a molar concentration, whereas the mass fractions above are per unit mass.
            salt_inner = inner_bulk_eqs.get_equation_of_type(SaltTransportEquations)
            if isinstance(salt_inner,SaltTransportEquations):
                # Its own scheme, and not advdiffu_inner's: a pure liquid has no mass fraction
                # fields at all, so the loop above may not have run even once.
                for salt in salt_inner.salts:
                    fname = salt_inner.fieldname_of(salt.name)
                    cs, cs_test = var_and_test(fname)
                    if salt_inner.GCL:
                        # The conservative form advects with the velocity relative to the mesh, so
                        # its natural condition already *is* zero flux through the moving surface.
                        flux = 0
                    elif salt_inner.advection_by_parts:
                        # Integrating the advection by parts leaves its own surface term behind, so
                        # this branch differs from the one below by exactly c*u.n -- which, with the
                        # kinematic condition (u-u_mesh).n = j/rho, is what turns -c*j/rho into
                        # +c*u_mesh.n. Both say the same thing: nothing but the solvent leaves.
                        flux = cs * dot(mesh_velocity(scheme=salt_inner.scheme), n)
                    else:
                        # Natural condition is zero *diffusive* flux, so what is missing is the
                        # liquid flowing through the surface while the salt does not.
                        flux = -cs * total_mass_transfer_rate / rho_inner
                    self.add_residual(weak(time_scheme(salt_inner.scheme, flux), cs_test))
                    self.add_residual(-weak(salt_inner.get_stabilization_flux(fname,n,self.get_parent_domain()),cs_test))

            # Component dynamics outside if necessary
            if self.get_opposite_side_of_interface(raise_error_if_none=False):
                opp = self.get_opposite_side_of_interface()
                oppblk=opp.get_parent_domain()
                if oppblk is not None:
                    oppblkeq = oppblk.get_equations()
                    if oppblkeq.get_equation_of_type(CompositionAdvectionDiffusionEquations):
                        outadvdiffu = oppblkeq.get_equation_of_type(CompositionAdvectionDiffusionEquations)
                        assert isinstance(outadvdiffu,CompositionAdvectionDiffusionEquations)
                        # total_mass_flux = actual_total_transfer_by_rho_inner * rho_inner
                        for name in sorted(outadvdiffu.fluid_props.required_adv_diff_fields):
                            fname = "massfrac_" + name
                            wi, wi_test = var_and_test(fname, domain=opp)
                            # Both of them are fine# TODO: But at pinned contact lines, we have to see
                            flux = partial_mass_transfer_rates.get(name, 0)
                            if self._has_opposite_flow:
                                if outadvdiffu.integrate_advection_by_parts:
                                    raise RuntimeError("TODO")
                                flux += -wi * total_mass_transfer_rate
                                # flux += wi * total_mass_flux
                            self.add_residual(-weak(flux, wi_test))
                            # Same correction for the outer bulk, whose outward normal is -n
                            self.add_residual(-weak(outadvdiffu.get_stabilization_flux(fname,-n,oppblk),wi_test))
                        if outadvdiffu.fluid_props.passive_field is not None and outadvdiffu.fluid_props.passive_field in partial_mass_transfer_rates.keys() and not self._has_opposite_flow:
                            raise RuntimeError("The exterior phase only solves component diffusion and the component '"+outadvdiffu.fluid_props.passive_field+"' is passive (not solved for explicitly). Yet, we have a mass transfer of this component. This is problematic. Please set passive_field in the exterior phase to some non-volatile component (usually, it is e.g. air, must be the dominant part of the exterior mixture) or solve the flow in the exterior phase by using CompositionFlowEquations instead of CompositionDiffusionEquations")

            # Thermal effects
            tins=self.get_parent_equations(TemperatureConductionEquation)
            if isinstance(tins,TemperatureConductionEquation):
                self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props)
                latent_flux=self.masstransfer_model.get_latent_heat_flux()
                self.masstransfer_model._clean_up_for_code()
                _,T_test=var_and_test("temperature")
                self.add_residual(weak(latent_flux,T_test))
                self.add_residual(-weak(tins.get_stabilization_flux("temperature",n,self.get_parent_domain()),T_test))


        # Connect the velocity with the opposite side
        if self._has_opposite_flow:
            fields = ["velocity_x", "velocity_y", "velocity_z"]
            fields = fields[0:self.get_nodal_dimension()]
            if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                fields+=["velocity_phi"]
            pdom=self.get_opposite_side_of_interface().get_parent_domain()
            assert pdom is not None
            ns_outer = pdom.get_equations().get_equation_of_type(NavierStokesEquations)
            assert isinstance(ns_outer,NavierStokesEquations)
            assert ns_outer.fluid_props is not None
            rho_outer = ns_outer.fluid_props.mass_density
            rho_outer = evaluate_in_domain(rho_outer, self.get_opposite_side_of_interface())
            velojump_normal = subexpression(self.total_mass_loss_factor_inside*total_mass_transfer_rate / rho_inner - self.total_mass_loss_factor_outside*total_mass_transfer_rate / rho_outer)
            for i, f in enumerate(fields):
                l, l_test = var_and_test(self.velo_connect_prefix + f)
                inside, inside_test = var_and_test(f)
                outside, outside_test = var_and_test(f, domain=self.get_opposite_side_of_interface())
                self.add_residual(weak(inside - outside - velojump_normal * n[i], l_test))  # TODO: Possibly nodal connection?
                self.add_residual(weak(l, inside_test))
                self.add_residual(-weak(l, outside_test))

        st = self._resolve_surfactant_transport()
        if st is not None:
            # The interface velocity, not the fluid velocity, is what the surfactant follows in the
            # normal direction: under evaporation the two differ by j/rho and the surfactant stays
            # with the interface.
            st._bind(self.interface_props, ns_inner.fluid_props)
            st._define_residuals_on(self)

    def before_assigning_equations_postorder(self, mesh:AnyMesh):
        # Pin kinematic boundary condition where necessary
        static = self.static
        if static == "auto":
            static = not self.get_current_code_generator()._coordinates_as_dofs
        assert isinstance(mesh,InterfaceMesh)
        if self.kinematic_bc_space is None or self.kinematic_bc_space not in ["DL","D0"]:
            self.pin_redundant_lagrange_multipliers(mesh, self.kinbc_name, "velocity" if static else "mesh")

        # Pin velo connection where necessary
        if self._has_opposite_flow:
            fields = ["velocity_x", "velocity_y", "velocity_z"]
            fields = fields[0:self.get_nodal_dimension()]
            if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                fields+=["velocity_phi"]
            for f in fields:
                lname = self.velo_connect_prefix + f
                self.pin_redundant_lagrange_multipliers(mesh, lname, f, opposite_interface=f)

    def with_balanced_end(self,*boundaries:str):
        """
        Adds the :py:class:`MultiComponentNavierStokesInterfaceBalancedEnd` equations to the given boundaries of the interface.
        Thereby, the Neumann force is balanced at the specified end points.
        Use this when you neither want to fix the position nor the contact angle at specific end points
        """
        res:"Equations | EquationTree"=self
        for b in boundaries:
            res+=MultiComponentNavierStokesInterfaceBalancedEnd() @ b
        return res


class MultiComponentNavierStokesInterfaceBalancedEnd(InterfaceEquations):
    """
    The :py:class:`MultiComponentNavierStokesInterface` without ``surface_tension_gradient_directly`` uses the surface divergence theorem to integrate the surface tension term by parts. This results in a Neumann term at the end points of the interface, which is not balanced by default. This Neumann term can be used e.g. for contact angles, but if you do not want to impose any, you can cancel out the Neumann term by adding this equation to the end points of the interface. This will balance the surface tension at the end points, resulting in a free contact angle.
    """
    required_parent_type = MultiComponentNavierStokesInterface

    def define_residuals(self):
        inter_eqs=self.get_parent_equations(MultiComponentNavierStokesInterface)
        assert isinstance(inter_eqs,MultiComponentNavierStokesInterface)
        if inter_eqs.surface_tension_gradient_directly:
            return # Only when we used the surface divergence theorem, the Neumann terms are there
        ns=inter_eqs.get_parent_equations(NavierStokesEquations)
        assert isinstance(ns,NavierStokesEquations)
        n=self.get_normal() # Outward pointing tangent at the interface boundary
        u_test=testfunction(ns.velocity_name)
        # Must be the very same expression as the one entering weak(...,div(u_test)) in the parent, otherwise it does not cancel
        if inter_eqs.surface_tension_projection_space is not None:
            # The scale of _surf_tension is stored symbolically as spatial/test_scale(velocity) and is
            # re-expanded in whatever domain the field is used in. Test scales gain a factor 1/spatial per
            # domain level, so here at the end of the interface it would come out one length too large.
            sigma=var("_surf_tension",domain="..")
            if inter_eqs.surface_tension_theta!=1:
                sigma=evaluate_in_past(sigma,1-inter_eqs.surface_tension_theta)
            sigma*=test_scale_factor("velocity")/test_scale_factor("velocity",domain="..")
            self.add_weak(-inter_eqs.surface_tension_factor*sigma,dot(n,u_test))
        else:
            sigma=inter_eqs.interface_props.surface_tension
            if sigma is None:
                raise RuntimeError("No surface tension set in the interface properties " + str(inter_eqs.interface_props))
            # Same reason as in NavierStokesContactAngle: the surface tension belongs to the interface,
            # and a discontinuous surfactant inside it is invisible from this co-dimension-2 domain.
            sigma=evaluate_in_domain(0+sigma,"..")
            if inter_eqs.surface_tension_theta!=1:
                # Same split as the parent: with theta != 1 the lag replaces the momentum scheme's
                # treatment rather than being wrapped in it. It has to stay the very same expression
                # here or the end-point Neumann term no longer cancels.
                self.add_weak(-inter_eqs.surface_tension_factor
                              * evaluate_in_past(sigma,1-inter_eqs.surface_tension_theta),dot(n,u_test))
            else:
                self.add_weak(-time_scheme(ns.momentum_scheme,inter_eqs.surface_tension_factor*sigma),dot(n,u_test))


class CompositionDiffusionInfinityEquations(InterfaceEquations):
    """
        Represents the condition at infinity for the advection-diffusion equation, using the assumption that in the far field the mass fraction behaves as: 
        
        .. math::
            w + R \\dfrac{\\partial w}{\\partial r} = w_{\\infty}
    
        Hence, works only correctly in axisymmetric or 3D.
        For 2D, we use:

            .. math::
                w + \\dfrac{R}{\\log(R/L)} \\dfrac{\\partial w}{\\partial r} = w_{\\infty}

        Due to the logarithmic solution behavior in 2D, the farfield length L must be provided here.

        We furthermore only assume diagonal diffusion here
        
        Additionally, advection in normal (radial) direction should be considered i.e. using:

            .. math::
                u_r(r \\to \\infty) = u_R \\left(\\frac{R}{r}\\right)^2
        
        Args:
            origin(ExpressionOrNum): The origin of the system. Default is vector([0]).
            farfield_length(Optional[ExpressionNumOrNone]): The farfield length (only required for 2D). Default is None.
            **infinity_values(ExpressionOrNum): The values at infinity. The keys are the names of the components and the values are the mass fractions.
    """

    def __init__(self, origin:ExpressionOrNum=vector([0]), farfield_length:ExpressionNumOrNone=None, **infinity_values:ExpressionOrNum):
        super(CompositionDiffusionInfinityEquations, self).__init__()
        self.inftyvals = sorted_field_kwargs(infinity_values)
        self.origin = origin
        self.farfield_length = farfield_length

    def define_residuals(self):
        n = self.get_normal()
        d = var("coordinate") - self.origin
        parent = self.get_parent_equations(CompositionAdvectionDiffusionEquations)
        assert isinstance(parent,CompositionAdvectionDiffusionEquations)
        rho = parent.fluid_props.mass_density

        req_adv_diff = sorted(parent.fluid_props.required_adv_diff_fields)
        ic = parent.fluid_props.initial_condition
        inftyvals = { n: ic["massfrac_" + n] for n in req_adv_diff if "massfrac_" + n in ic.keys()}
        for k, v in self.inftyvals.items():
            if k.startswith("massfrac_"):
                k = k.lstrip("massfrac_")
            inftyvals[k] = v


        real_dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
        for fn, val in inftyvals.items():
            if val is False:
                continue
            if fn.startswith("massfrac_"):
                fn = fn.lstrip("massfrac_")
            D = parent.get_diffusion_coefficient(fn)
            assert D is not None
            y, y_test = var_and_test("massfrac_" + fn)
            R = square_root(dot(d, d))
            # Subtract the footprint a stabilized bulk leaves here. Zero by default.
            self.add_residual(-weak(parent.get_stabilization_flux("massfrac_"+fn,n,self.get_parent_domain()),y_test))

            
            
            if real_dim==1:
                if self.farfield_length is None:
                    raise RuntimeError("For 1D far-field monopole conditions, a farfield_length must be provided")
                dist_to_ff=dot(n,d)-self.farfield_length
                self.add_residual(weak(-rho*D * (y - val) /dist_to_ff,y_test))
            else:
                if real_dim==3:
                    coordsys_dim_factor = 1
                elif real_dim==2:
                    if self.farfield_length is None:
                        raise RuntimeError("For 2D far-field monopole conditions, a farfield_length must be provided")
                    coordsys_dim_factor = -1/log(R/self.farfield_length)
                else:
                    raise RuntimeError("Far-field monopole conditions are only implemented for real_dim in {1,2,3}, not "+str(real_dim))
                self.add_residual(weak(rho*D * coordsys_dim_factor * (y - val) * dot(n, d) / (dot(d, d)) , y_test) )



class TemperatureConductionEquation(ScalarTransportEquations):
    """
    Represents the temperature conduction equation of the form:

            rho * cp * partial_t(T) = div(k * grad(T))

    Args:
        material(AnyMaterialProperties): The material properties.
        space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. quadratic continuous Lagrangian elements.
        rho_override(ExpressionNumOrNone): The mass density. Default is None.
        cp_override(ExpressionNumOrNone): The specific heat capacity. Default is None.
        lambda_override(ExpressionNumOrNone): The thermal conductivity. Default is None.
        dt_factor(ExpressionOrNum): The factor for the time derivative. Default is 1.
        GCL(bool): Write the transient term as the derivative of the whole integral of ``rho*cp*T``, and (in the advective subclass) advect with the velocity *relative to the mesh*: the conservative ALE form. On a mesh that follows a moving or evaporating boundary the enthalpy is then conserved to machine precision instead of to the order of the time stepping. **If rho or cp depend on the temperature this is a different model**, not merely a different discretization: the GCL form differentiates the product ``rho*cp*T`` in time, where the standard form multiplies ``rho*cp`` onto ``partial_t(T)``. Identical for constant properties. Default is False.
        gcl_scheme: Time stepping scheme of the ``GCL`` transient, from the set a derivative of an integral understands. The default ``"BDF2_degr"`` degrades to first order in the first step, where an initial condition has no history.
        stabilization: Optional residual-based stabilization (SUPG etc.), see :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`. ``None`` (the default) adds nothing at all. Without a wind there is nothing to stabilize on a static mesh, but on a moving (ALE) mesh the transport by the mesh motion is stabilized.

    .. note::
        **Scaling to set on problem level.** The temperature equation is tested with
        ``scale_factor("temporal")/(scale_factor("temperature")*scale_factor("rho_cp"))``. Besides
        the ``temperature`` field scale, the volumetric heat capacity ``rho_cp`` is therefore
        required, i.e. ``problem.set_scaling(rho_cp=...)``, and it is no field of the system.
    """
    def __init__(self,material:AnyMaterialProperties,space:FiniteElementSpaceEnum="C2",rho_override:ExpressionNumOrNone=None,cp_override:ExpressionNumOrNone=None,lambda_override:ExpressionNumOrNone=None,dt_factor:ExpressionOrNum=1,GCL:bool=False,gcl_scheme:"IntegralTimeSteppingScheme"="BDF2_degr",stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
        super(TemperatureConductionEquation, self).__init__()
        self.material=material
        self.space:FiniteElementSpaceEnum=space
        self.rho_override,self.cp_override,self.lambda_override=rho_override,cp_override,lambda_override
        self.dt_factor=dt_factor
        self.GCL=GCL
        self.gcl_scheme:"IntegralTimeSteppingScheme"=gcl_scheme
        self._init_stabilization(stabilization)

    def define_fields(self):
        #self.define_scalar_field("temperature",self.space,testscale=scale_factor("spatial")**2/(scale_factor("temperature")*scale_factor("thermal_conductivity")))
        self.define_scalar_field("temperature", self.space, testscale=scale_factor("temporal") / (scale_factor("temperature") * scale_factor("rho_cp")))

    def get_rho_cp_k(self) -> tuple[ExpressionOrNum,ExpressionOrNum,ExpressionOrNum]:
        """The three material coefficients, honouring the ``*_override`` arguments. Resolved in one
        place so that the strong residual cannot drift away from the Galerkin one."""
        rho=self.material.mass_density if self.rho_override is None else self.rho_override
        k=self.material.thermal_conductivity if self.lambda_override is None else self.lambda_override
        cp=self.material.specific_heat_capacity if self.cp_override is None else self.cp_override
        if rho is None:
            raise RuntimeError("No mass_density defined in "+str(self.material))
        if k is None:
            raise RuntimeError("No thermal_conductivity defined in "+str(self.material))
        if cp is None:
            raise RuntimeError("No specific_heat_capacity defined in "+str(self.material))
        return rho,cp,k

    # ---- hooks of the stabilization base class ---------------------------------------------------

    def stabilized_fieldnames(self) -> list[str]:
        return ["temperature"]

    def stabilization_wind(self) -> ExpressionOrNum:
        return 0   # pure conduction; the advective subclass overrides this

    def stabilization_residual_scale(self, fieldname:str) -> ExpressionOrNum:
        rho,cp,_=self.get_rho_cp_k()
        return rho*cp

    def stabilization_diffusivity(self, fieldname:str) -> ExpressionOrNum:
        rho,cp,k=self.get_rho_cp_k()
        return k/(rho*cp)   # the thermal diffusivity, i.e. m^2/s as tau requires

    def strong_residual(self, fieldname:str) -> Expression:
        T=var("temperature")
        rho,cp,k=self.get_rho_cp_k()
        if self.GCL:
            # The GCL transient is strongly d_t(rho cp T)|_E + div(rho cp T w), and the mesh part of
            # the flux term contributes exactly -div(rho cp T w), so the two cancel and what remains
            # is the Eulerian derivative of the *enthalpy density* -- not rho*cp times the derivative
            # of T, which is the same thing only for constant properties.
            R:Expression=self.dt_factor*partial_t(rho*cp*T)
        else:
            R=self.dt_factor*rho*cp*partial_t(T)
        if self.stab_cfg.include_diffusion_in_residual:
            R=R-div(convert_to_expression(k)*grad(T))
        return R

    def define_galerkin_residuals(self):
        """The Galerkin part alone. Subclasses extend *this*, not ``define_residuals``, so that the
        stabilization stays the last thing added and is added exactly once."""
        T,T_test=var_and_test("temperature")
        rho,cp,k=self.get_rho_cp_k()
        if self.GCL:
            # The derivative of the whole integral of the enthalpy density, so that the change of the
            # element volume is taken by the same finite difference that advances the field. rho and
            # cp sit INSIDE the integral: if either is a field, the conserved quantity is rho*cp*T
            # and not T. dt_factor multiplies from outside, or the history terms would carry it at
            # their own time levels.
            self.add_residual(self.dt_factor*time_derivative_of_integral(weak(rho*cp*T,T_test),
                                                                         scheme=self.gcl_scheme))
            # The advective counterpart of the mesh motion. The subclass adds the fluid part.
            w=mesh_velocity()
            if not is_zero(w):
                self.add_residual(weak(self.dt_factor*rho*cp*T*w,grad(T_test)))
        else:
            self.add_residual(weak(self.dt_factor*rho*cp*partial_t(T),T_test))
        self.add_residual(weak(k*grad(T),grad(T_test)))

    def define_residuals(self):
        self.define_galerkin_residuals()
        # No time_scheme wrapper here: unlike the composition equations these do not use one.
        self.add_stabilization_residuals()


class TemperatureHeatFlux(InterfaceEquations):
    """
    Represents the heat flux through the interface.

    This class requires the parent equations to be of type TemperatureConductionEquation, meaning that if TemperatureConductionEquation (or subclasses) are not defined in the parent domain, an error will be raised.

    Args:
        q(ExpressionOrNum): The heat flux.
    """
    required_parent_type = TemperatureConductionEquation

    def __init__(self,q:ExpressionOrNum):
        super(TemperatureHeatFlux, self).__init__()
        self.q=q

    def define_residuals(self):
        _,T_test=var_and_test("temperature")
        self.add_residual(-weak(self.q,T_test))
        parent=self.get_parent_equations(TemperatureConductionEquation)
        if isinstance(parent,TemperatureConductionEquation):
            # Subtract the footprint a stabilized bulk leaves here, so that the imposed heat flux is
            # the physical one. Zero unless natural_bc_correction is switched on.
            self.add_residual(-weak(parent.get_stabilization_flux("temperature",self.get_normal(),
                                                                  self.get_parent_domain()),T_test))


class TemperatureAdvectionConductionEquation(TemperatureConductionEquation):
    """
    Represents the temperature advection-conduction equation of the form:
    
            rho * cp * (partial_t(T) + u * grad(T)) = div(k * grad(T))
        
    where rho is the mass density, cp is the specific heat capacity, u is the velocity, and k is the thermal conductivity.
    grad and div represent the gradient and divergence operators, respectively.

    Args:
        material(AnyMaterialProperties): The material properties.
        space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. quadratic continuous Lagrangian elements.
        wind(ExpressionOrNum): The velocity. Default is var("velocity").
        rho_override(ExpressionNumOrNone): The mass density. Default is None.
        cp_override(ExpressionNumOrNone): The specific heat capacity. Default is None.
        lambda_override(ExpressionNumOrNone): The thermal conductivity. Default is None.
        dt_factor(ExpressionOrNum): Multiplicative factor for the time derivative. Default is 1.
        adv_factor(ExpressionOrNum): Multiplicative factor for the advection term. Default is 1.
        stabilization: Optional residual-based stabilization (SUPG etc.), see :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`. ``None`` (the default) adds nothing at all.

    .. note::
        **Scaling to set on problem level.** As for
        :py:class:`TemperatureConductionEquation`, the test scaling requires the non-field scale
        ``rho_cp``, i.e. ``problem.set_scaling(rho_cp=...)``.
    """

    def __init__(self,material:AnyMaterialProperties,space:FiniteElementSpaceEnum="C2",wind:ExpressionOrNum=var("velocity"),rho_override:ExpressionNumOrNone=None,cp_override:ExpressionNumOrNone=None,lambda_override:ExpressionNumOrNone=None,dt_factor:ExpressionOrNum=1,adv_factor:ExpressionOrNum=1,GCL:bool=False,gcl_scheme:"IntegralTimeSteppingScheme"="BDF2_degr",stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
        super(TemperatureAdvectionConductionEquation, self).__init__(material,space,rho_override=rho_override,cp_override=cp_override,lambda_override=lambda_override,dt_factor=dt_factor,GCL=GCL,gcl_scheme=gcl_scheme,stabilization=stabilization)
        self.wind=wind
        self.adv_factor=adv_factor

    def stabilization_wind(self) -> ExpressionOrNum:
        return self.wind

    def strong_residual(self,fieldname:str) -> Expression:
        rho,cp,_=self.get_rho_cp_k()
        T=var("temperature")
        base=super(TemperatureAdvectionConductionEquation,self).strong_residual(fieldname)
        if self.GCL:
            # Mirror the flux form actually assembled, so that the sum with the base class's
            # d_t(rho cp T) + div(rho cp T w) is d_t(rho cp T) + div(rho cp T u).
            return base+self.adv_factor*div(rho*cp*T*self.wind)
        return base+self.adv_factor*rho*cp*dot(self.wind,grad(T))

    def define_galerkin_residuals(self):
        super(TemperatureAdvectionConductionEquation, self).define_galerkin_residuals()
        T,T_test=var_and_test("temperature")
        rho,cp,_=self.get_rho_cp_k()
        if self.GCL:
            # The fluid part of the flux. The base class already contributed the mesh part with the
            # opposite sign, so the two together are -weak(rho cp T (adv_factor*u - dt_factor*w),
            # grad(v)), i.e. advection with the velocity relative to the mesh.
            self.add_residual(-weak(self.adv_factor*rho*cp*T*self.wind,grad(T_test)))
        else:
            self.add_residual(weak(self.adv_factor*rho*cp*dot(self.wind,grad(T)),T_test))


class TemperatureInfinityEquations(InterfaceEquations):
    """
    Represents the condition at infinity for the temperature conduction equation, using the assumption that in the far field the temperature behaves as:
        
        T(r->infty) = T_infty + R/r(T_R-T_infty) for some large R and r>>R
    
    Hence, works only correctly in axisymmetric or 3d.
    
    Args:
        far_temperature(ExpressionOrNum): The temperature at infinity.
        origin(ExpressionOrNum): The origin of the system. Default is vector([0]).
    """

    def __init__(self,far_temperature:ExpressionOrNum,origin:ExpressionOrNum=vector([0])):
        super(TemperatureInfinityEquations, self).__init__()
        self.far_temperature = far_temperature
        self.origin = origin

    def define_residuals(self):
        n = self.get_normal()
        d = var("coordinate") - self.origin
        parent = self.get_parent_equations(TemperatureConductionEquation)
        assert isinstance(parent,TemperatureConductionEquation)
        k=parent.material.thermal_conductivity
        T, T_test = var_and_test("temperature")
        self.add_residual(weak(k * (T - self.far_temperature) * dot(n, d) / dot(d, d), T_test))
        # Subtract the footprint a stabilized bulk leaves here. Zero by default.
        self.add_residual(-weak(parent.get_stabilization_flux("temperature",n,self.get_parent_domain()),T_test))


class ThinLayerThermalConductionEquation(InterfaceEquations):
    """
    Considers a thin plate (not resolved in depth direction) of some material in between.
    Can be added in between two domains with temperature equations. 
    If there is no domain at the opposite side, the outside_temperature must be set manually
        
    Args:
        material(AnyMaterialProperties): The material properties.
        thickness(ExpressionOrNum): The thickness of the layer.
        ALE(Union[Literal["auto"],bool]): Whether to use the Arbitrary Lagrangian-Eulerian (ALE) formulation. Default is "auto".
        outside_temperature(ExpressionNumOrNone): The temperature at the outside of the layer. Default is None.
    """
        
    def __init__(self,material:AnyMaterialProperties,thickness:ExpressionOrNum,*,ALE:Literal["auto"] | bool="auto",outside_temperature:ExpressionNumOrNone=None):
        super().__init__()
        self.material=material
        self.thickness=thickness
        # TODO: Not sure about ALE here. It would only act tangentially along the connecting interface
        # We just assume it remains all static now, also we would have to consider the motion if thickness changes        
        self.ALE=ALE
        self.outside_temperature=outside_temperature
        
    def define_residuals(self):
        Tin,Tin_test=var_and_test("temperature")
        Tout_test:Expression | None=None
        if self.outside_temperature is None:
            Tout,Tout_test=var_and_test("temperature",domain="|.")
        else:
            Tout=self.outside_temperature
        rho=self.material.mass_density
        cp=self.material.specific_heat_capacity
        k=self.material.thermal_conductivity
        # The normal thermal conduction equations of a resolved domain would add 
        #   self.add_weak(rho*cp*partial_t(T),T_test)
        #   self.add_weak(k*grad(T),grad(T_test))
        # In the thickness direction, we span T and T_test as follows:
        #   T = Tin*PsiI(s)+Tout*PsiO(s)
        #   T_test is set to { PsiI(s), PsiO(s) }
        # We take s to range from [0,1], with s=0 at A and s=1 at B
        # Hence, we can write PsiI(s) = 1-s and PsiO(s) = s
        # The integration of weak(.,.) is carried out in the thickness direction manually
        
        dz=self.thickness
        # thickness-direction weak terms of weak(PsiI,PsiI),weak(PsiO,PsiO),weak(PsiI,PsiO)
        PsiIPsiI,PsiIPsiO=1/3*dz, 1/6*dz
        PsiOPsiO,PsiOPsiI=PsiIPsiI,PsiIPsiO
        # The same for weak(partial_z(PsiI),partial_z(PsiI)), etc. We get a 1/dz, since integration gives dz and each partial_z gives 1/dz
        dzPsiIdzPsiI,dzPsiOdzPsiI=1/dz,-1/dz
        dzPsiOdzPsiO,dzPsiIdzPsiO=dzPsiIdzPsiI,dzPsiOdzPsiI
                
        # TODO: If rho or cp are functions of the temperature, they will be evaluated at Tin here. We could blend it
        self.add_weak(rho*cp*(PsiIPsiI*partial_t(Tin,ALE=self.ALE)+PsiOPsiI*partial_t(Tout,ALE=self.ALE)),Tin_test)
        if self.outside_temperature is None:
            # Tout_test was necessarily assigned above, since self.outside_temperature is still None here
            assert Tout_test is not None
            self.add_weak(rho*cp*(PsiIPsiO*partial_t(Tin,ALE=self.ALE)+PsiOPsiO*partial_t(Tout,ALE=self.ALE)),Tout_test)
        # TODO: If lambda is a functions of the temperature, it well be evaluated at Tin here. We could blend it
        self.add_weak(k*(dzPsiIdzPsiI*Tin+dzPsiOdzPsiI*Tout),Tin_test)
        if self.outside_temperature is None:
            assert Tout_test is not None
            self.add_weak(k*(dzPsiIdzPsiO*Tin+dzPsiOdzPsiO*Tout),Tout_test)
        
        
class BalanceGravityAtFarField(InterfaceEquations):
    """
        When flow of e.g. the gas domain of an evaporating droplet with gravity is considered, the boundary conditions at the far field may not be traction-free.
        Otherwise, the hydrostatic pressure will lead to unphysical in/outflow at the far boundaries from the top to the bottom.
        We can add this to balance for the gravity term at the far field.
    
    This class requires the parent equations to be of type NavierStokesEquation, meaning that if NavierStokesEquation (or subclasses) are not defined in the parent domain, an error will be raised.
    
    Args:
        gravity_vector(ExpressionOrNum): The gravity vector.
        reference_point(ExpressionOrNum): The reference point. Default is vector(0).
    """
    required_parent_type = NavierStokesEquations
    def __init__(self,gravity_vector:ExpressionOrNum,reference_point:ExpressionOrNum=vector(0)):
        super(BalanceGravityAtFarField, self).__init__()
        self.g_vec=gravity_vector
        self.x_ref=reference_point

    def define_residuals(self):
        utest=testfunction("velocity")
        nseq=self.get_parent_equations()
        assert isinstance(nseq,NavierStokesEquations)
        fluid_props=nseq.fluid_props
        assert fluid_props is not None
        #rho0=fluid_props.evaluate_at_condition(fluid_props.mass_density,fluid_props.initial_condition)
        rho=fluid_props.mass_density
        x=var("coordinate")
        n=var("normal")
        self.add_residual(weak(rho*dot(self.g_vec,x-self.x_ref),dot(n,utest)))
        # This is a traction boundary condition, so it has to subtract the traction a stabilized bulk
        # deposits here, exactly as NavierStokesNormalTraction and StokesFlowRadialFarField do. The
        # full vector, not only its normal part: the boundary is do-nothing tangentially.
        self.add_residual(-weak(nseq.get_stabilization_traction(n,self.get_parent_domain()),utest))



# SurfactantsAtSolidInterface moved to pyoomph.equations.surfactants, so that the surfactants on the
# substrate and the ones on the free surface share one transport implementation. Re-exported here,
# since that is where every existing script imports it from.
from .surfactants import SurfactantsAtSolidInterface


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
