#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
 
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..generic import Equations, InterfaceEquations
from ..equations.generic import InitialCondition, SpatialErrorEstimator, FiniteElementSpaceEnum
from ..expressions import *  # Import grad et al
from .navier_stokes import NavierStokesEquations , NavierStokesSlipLength, NoSlipBC, PFEMOptions #type:ignore
from ..materials.generic import *
from .SUPG import ElementSizeForSUPG
from .generic import get_interface_field_connection_space



def CompositionInitialCondition(fluid_props:AnyFluidProperties,isothermal:bool,initial_temperature:ExpressionNumOrNone=None):
    req_adv_diff = fluid_props.required_adv_diff_fields
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

    return InitialCondition(**icsettings)


def CompositionDiffusionEquations(fluid_props:AnyFluidProperties, space:FiniteElementSpaceEnum="C2", dt_factor:ExpressionOrNum=1, with_IC:bool=True, spatial_errors:Optional[float]=None,isothermal:bool=True,initial_temperature:ExpressionNumOrNone=None) -> Equations:
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
    
    Returns:
        A coupled set of equations for the mass fractions for the diffusive transport of the components in the mixture.
    """
    res = CompositionAdvectionDiffusionEquations(fluid_props, space=space, dt_factor=dt_factor, wind=0)
    if not isothermal:
        res+=TemperatureConductionEquation(fluid_props,space=space)
    if with_IC:
        res += CompositionInitialCondition(fluid_props,isothermal,initial_temperature)
    if spatial_errors is not None:
        if spatial_errors is True:
            compo_fields = {"massfrac_" + n: 1.0 for n in fluid_props.required_adv_diff_fields}
            res += SpatialErrorEstimator(**compo_fields)
        elif spatial_errors is not False:
            raise RuntimeError("TODO")
    return res


def CompositionFlowEquations(fluid_props:AnyFluidProperties, compo_space:FiniteElementSpaceEnum="C1", compo_dt_factor:ExpressionOrNum=1, ns_mode:Literal["TH","CR"]="TH", boussinesq:bool=False,
                             gravity:ExpressionNumOrNone=None, bulkforce:ExpressionNumOrNone=None, ns_dt_factor:ExpressionOrNum=1, ns_nl_factor:ExpressionNumOrNone=None, with_IC:bool=True,
                             hele_shaw_thickness:ExpressionNumOrNone=None, spatial_errors:Optional[float]=None, useCompoSUPG:bool=False,isothermal:bool=True,initial_temperature:ExpressionNumOrNone=None,additional_advection:ExpressionOrNum=0,momentum_scheme:TimeSteppingScheme="BDF2",continuity_scheme:TimeSteppingScheme="BDF2",wrong_strain:bool=False,integrate_advection_by_parts:bool=False,PFEM:Union[PFEMOptions, bool]=False,wrap_params_in_subexpressions=True,thermal_dt_factor:ExpressionOrNum=1,thermal_adv_factor:ExpressionOrNum=1) -> Equations:
    """
    Assembles a system for multi-component flow with advection-diffusion equations for mass fraction fields of the mixture composition and the Navier-Stokes equations. Potentially, also a temperature field is included.

    Args:
        fluid_props: The fluid properties.
        compo_space: Space for the mass fraction fields
        compo_dt_factor: Factor for the time derivative of the mass fraction fields
        ns_mode: Which Navier-Stokes discretization to use, Taylor-Hood (``"TH"``) or Crouzeix-Raviart (``"CR"``).
        boussinesq: Use Boussinesq approximation
        gravity: Gravity vector [in m/s^2].
        bulkforce: Additional bulk force term.
        ns_dt_factor: Factor for the time derivative of the Navier-Stokes equations.
        ns_nl_factor: Factor for the non-linear term in the Navier-Stokes equations.
        with_IC: Include the initial mixture composition (and temperature) as initial condition.
        hele_shaw_thickness: If set, we consider a Hele-Shaw flow with the given thickness. This modifies a few terms in the Navier-Stokes equations.
        spatial_errors: Add spatial error estimators automatically.
        useCompoSUPG: Use SUPG for the composition advection.
        isothermal: If set to false, a temperature field is included.
        initial_temperature: Temperature initial condition.
        additional_advection: Adds an additional advection term.
        momentum_scheme: Selects the time stepping scheme for the momentum equation.
        continuity_scheme: Selects the time stepping scheme for the continuity equation.
        wrong_strain: Simplifies the strain by a simple Laplacian. Do not use when you e.g. imposed tractions, e.g. Marangoni forces.
        integrate_advection_by_parts: Integrate the advection terms of the composition equations by parts.
        PFEM: Options for the experimental Particle-Finite-Element-Method approach.
        wrap_params_in_subexpressions: If True, all material properties in the equations are wrapped in subexpressions.
        thermal_dt_factor: Factor for the time derivative of the temperature field.
        thermal_adv_factor: Factor for the advection term of the temperature field.

    Returns:
        A coupled set of equations describing the multi-component flow of the mixture 
    """
    if ns_nl_factor is None:
        if hele_shaw_thickness is None:
            ns_nl_factor = 1
        else:
            ns_nl_factor = 6 / 5
    if hele_shaw_thickness is not None:
        hsdamp = -12 * subexpression(fluid_props.dynamic_viscosity) * var("velocity") / subexpression(
            hele_shaw_thickness) ** 2
        bulkforce = hsdamp if bulkforce is None else bulkforce + hsdamp

    ns = NavierStokesEquations(fluid_props=fluid_props, mode=ns_mode, boussinesq=boussinesq, gravity=gravity,
                               bulkforce=bulkforce, dt_factor=ns_dt_factor, nonlinear_factor=ns_nl_factor,momentum_scheme=momentum_scheme,continuity_scheme=continuity_scheme,wrong_strain=wrong_strain,PFEM=PFEM,wrap_params_in_subexpressions=wrap_params_in_subexpressions)
    wind=var("velocity")+additional_advection
    cp = CompositionAdvectionDiffusionEquations(fluid_props=fluid_props, space=compo_space, dt_factor=compo_dt_factor,
                                                boussinesq=boussinesq, useSUPG=useCompoSUPG,wind=wind,integrate_advection_by_parts=integrate_advection_by_parts,wrap_params_in_subexpressions=wrap_params_in_subexpressions)
    res = ns + cp
    if not isothermal:
        res+=TemperatureAdvectionConductionEquation(fluid_props,space=compo_space,wind=wind,adv_factor=thermal_adv_factor,dt_factor=thermal_dt_factor)
    if useCompoSUPG:
        res += ElementSizeForSUPG()
    if with_IC:
        res += CompositionInitialCondition(fluid_props,isothermal,initial_temperature)
    if spatial_errors is not None:
        if spatial_errors is True:
            compo_fields = {"massfrac_" + n: 1.0 for n in fluid_props.required_adv_diff_fields}
            res += SpatialErrorEstimator(velocity=1, **compo_fields)
        elif isinstance(spatial_errors,dict):
            res += SpatialErrorEstimator(**spatial_errors)
        elif spatial_errors is not False:
            raise RuntimeError("TODO")

    return res


class CompositionAdvectionDiffusionEquations(Equations):
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
        useSUPG(bool): Whether to use the SUPG method. Default is False.
        integrate_advection_by_parts(bool): Whether to integrate the advection term by parts. Default is False.
        wrap_params_in_subexpressions(bool): Whether to wrap the parameters in subexpressions using GiNaC. Default is True.
    """
        
    def __init__(self, fluid_props:AnyFluidProperties, *, space:FiniteElementSpaceEnum="C2", wind:ExpressionOrNum=var("velocity"), dt_factor:ExpressionOrNum=1, boussinesq:bool=False, useSUPG:bool=False,integrate_advection_by_parts:bool=False,wrap_params_in_subexpressions:bool=True):
        super().__init__()
        self.dt_factor = dt_factor
        self.space:FiniteElementSpaceEnum = space
        self.wind = wind
        self.fluid_props = fluid_props
        self.fieldnames:List[str] = []
        self.component_names:Dict[str,str] = {}
        self.stop_on_zero_diffusive_flux = True
        self.boussinesq = boussinesq
        self.integrate_advection_by_parts=integrate_advection_by_parts
        for n in sorted(self.fluid_props.required_adv_diff_fields):
            self.component_names["massfrac_" + n] = n
            self.fieldnames.append("massfrac_" + n)
        self.useSUPG = useSUPG
        self.requires_interior_facet_terms=is_DG_space(self.space)
        self.DG_alpha=1
        self.wrap_params_in_subexpressions=wrap_params_in_subexpressions

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

    def get_diffusion_coefficient(self, f1:str, f2:Optional[str]=None) -> ExpressionNumOrNone:
        assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
        if f2 is None:
            f2 = f1
        return self.fluid_props.get_diffusion_coefficient(f1, f2, default=0)

    def get_diffusive_mass_flux_expression_for(self, fn:str) -> ExpressionOrNum:
        assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
        return self.fluid_props.get_diffusive_mass_flux_for(fn)

    def define_scaling(self):
        for fn in self.fieldnames:
            self.set_test_scaling(**{fn: scale_factor("temporal") / scale_factor("mass_density")})
        if not self.fluid_props.is_pure:
            assert isinstance(self.fluid_props,(MixtureLiquidProperties,MixtureGasProperties))
            assert self.fluid_props.passive_field is not None
            self.set_test_scaling(**{"massfrac_"+self.fluid_props.passive_field:scale_factor("temporal") / scale_factor("mass_density")})


    def get_supg_tau(self, field:str) -> Expression:
        elsize_eqs = self.get_combined_equations().get_equation_of_type(ElementSizeForSUPG)
        if elsize_eqs is None or (isinstance(elsize_eqs, list) and len(elsize_eqs) == 0):
            raise RuntimeError("SUPG only works if combined with a ElementSizeForSUPG Equation")
        assert isinstance(elsize_eqs,ElementSizeForSUPG)
        elemsize = var(elsize_eqs.varname)
        urel = self.wind - eval_flag("moving_mesh") * mesh_velocity()
        usqr = subexpression(dot(urel, urel))
        dt = subexpression(var("time") - evaluate_in_past(var("time")) + 1e-20 * scale_factor("temporal"))
        ht = subexpression(square_root(elemsize, self.get_element_dimension()) + 1e-20 * scale_factor("spatial"))
        k = 1 if self.space == "C1" else 2
        Ck = 60 * (2 ** (k - 2))
        sigma_BDF = 2
        if len(self.component_names) > 1:
            raise RuntimeError("SUPG does not work yet for ternary or higher systems")
        Dc=self.get_diffusion_coefficient(field)
        if Dc is None:
            return Expression(0)
        D = subexpression(Dc)
        tau = subexpression(1 / square_root((sigma_BDF / dt) ** 2 + usqr / ht ** 2 + Ck * (D / ht ** 2) ** 2))
        return tau

    def define_residuals(self):
        rho_ref = scale_factor("mass_density")
        rho = self.fluid_props.mass_density
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
                if self.useSUPG:
                    raise RuntimeError("TODO")
                res = rho_factor * (self.dt_factor * partial_t(f))
                self.add_residual(-weak(rho_factor *self.wind*f,grad(f_test)))
            else:
                res = rho_factor * (self.dt_factor * partial_t(f) + dot(self.wind, grad(f)))
            if self.useSUPG:
                res = subexpression(res)  # XXX Does not work here!
                self.add_residual(weak(self.get_supg_tau(self.component_names[fn]) * self.wind * res, grad(f_test)))
            self.add_residual(weak(res, f_test))
            self.add_residual(-weak(Jdiff, grad(f_test)))
            if isinstance(self.fluid_props,MixtureLiquidProperties):
                reaction_rate=self.fluid_props.get_reaction_rate(self.component_names[fn])
                self.add_residual(-weak(reaction_rate,f_test))
            if self.requires_interior_facet_terms:
                raise RuntimeError("TODO: DG implementation")


class CompositionAdvectionDiffusionFluxEquations(InterfaceEquations):
    """
    Represents the flux through the interface that naturally arises from the integration by parts of the diffusion term in the advection-diffusion equation.
        
    Args:
        **kwargs(ExpressionOrNum): The fluxes. The keys are the names of the components and the values are the mass fluxes. 
    """
            
        
        
    def __init__(self, **kwargs:ExpressionOrNum):
        super(CompositionAdvectionDiffusionFluxEquations, self).__init__()
        self.fluxes = kwargs.copy()

    def define_residuals(self):
        for name, flux in self.fluxes.items():
            fname = "massfrac_" + name
            test = testfunction(fname)
            self.add_residual(weak(flux, test))


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
        kinematic_bc_coordinate_sys(Optional[BaseCoordinateSystem]): The coordinate system for the kinematic boundary condition. Default is None.
        additional_masstransfer_scale(ExpressionOrNum): Additional mass transfer scale. Default is 1.
        additional_kin_bc_test_scale(ExpressionOrNum): Additional kinematic boundary condition test scale. Default is 1.
        static_normal_interface_motion(ExpressionOrNum): If solved on a static mesh, we can mimic the interface motion by moving it in normal direction with this rate. Default is 0.
        static_interface_motion_testfunction(ExpressionNumOrNone): If set, we solve that the total outflux is zero by adjusting this.
        project_interface_flux(bool): If set to True, the interface flux (kinematic BC) is projected and used for the kinematic BC. Default is False.
        surface_tension_factor(ExpressionOrNum): The surface tension factor. Multiplicative factor for the imposition of the surface tension. Default is 1.
    """
            
        
    from ..materials.mass_transfer import MassTransferModelBase
    def __init__(self, interface_props:AnyFluidFluidInterface, *, kinbc_name:str="_kin_bc", velo_connect_prefix:str="_lagr_conn_",
                 masstransfer_model:Optional[Union[MassTransferModelBase,Literal[False]]]=None, static:Union[Literal["auto"],bool]="auto", surface_tension_theta:float=1, total_mass_loss_factor_inside:ExpressionOrNum=1,total_mass_loss_factor_outside:ExpressionOrNum=1,
                 surface_tension_projection_space:Optional[FiniteElementSpaceEnum]=None,additional_normal_traction:ExpressionOrNum=0,surface_tension_gradient_directly:bool=False,use_highest_space_for_velo_connection:bool=False,kinematic_bc_coordinate_sys:Optional[BaseCoordinateSystem]=None,additional_masstransfer_scale=1,additional_kin_bc_test_scale=1,static_normal_interface_motion:ExpressionOrNum=0,static_interface_motion_testfunction:ExpressionNumOrNone=None,project_interface_flux:bool=False,surface_tension_factor:ExpressionOrNum=1):
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
        self.surface_tension_projection_space:Optional[FiniteElementSpaceEnum] = surface_tension_projection_space
        self.surface_tension_gradient_directly=surface_tension_gradient_directly
        self.additional_normal_traction=additional_normal_traction
        self.surfactant_advect_velo_name="_uinterf_proj"
        self.surfactant_advect_velo_space:FiniteElementSpaceEnum="C2"
        self.use_highest_space_for_velo_connection=use_highest_space_for_velo_connection
        self.kinematic_bc_coordinate_sys=kinematic_bc_coordinate_sys
        self.additional_masstransfer_scale=additional_masstransfer_scale
        self.additional_kin_bc_test_scale=additional_kin_bc_test_scale
        self.static_normal_interface_motion=static_normal_interface_motion
        self.static_interface_motion_testfunction=static_interface_motion_testfunction
        self.project_interface_flux=project_interface_flux
        self.surface_tension_factor=surface_tension_factor

    def define_fields(self):
        # Add kinematic boundary condition multiplier
        nseqs=self.get_parent_equations(of_type=NavierStokesEquations)
        assert isinstance(nseqs,NavierStokesEquations)
        inside_space=nseqs.get_velocity_space_from_mode(for_interface=True)
        kinbc_space=inside_space
#        if nseqs.mode=="mini"        
        if not nseqs.PFEM_options or not nseqs.PFEM_options.active:            
            static=self.static
            if static=="auto":
                static=not self.get_current_code_generator().get_parent_domain()._coordinates_as_dofs

            if not static in {"auto",False,True}:
                raise RuntimeError("property static must be either 'auto', True or False")
            if not static:
                pos_space=self.get_current_code_generator().get_parent_domain()._coordinate_space
                if pos_space=="":
                    raise RuntimeError("Find out the coordinate space:"+str())
                kinbc_space=pos_space
        
            self.define_scalar_field(self.kinbc_name, inside_space )
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
                        conn_space=get_interface_field_connection_space(inside_space,outside_space,use_highest_space=self.use_highest_space_for_velo_connection)
                        assert conn_space!=""
                        fields = ["velocity_x", "velocity_y", "velocity_z"]
                        fields = fields[0:self.get_nodal_dimension()]
                        if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
                            fields+=["velocity_phi"]
                        for f in fields:
                            self.define_scalar_field(self.velo_connect_prefix + f, conn_space)  # TODO: Other velocity spaces?
                        self._has_opposite_flow = True

        has_surfactants = False
        surfsI = self.interface_props.surfactants
        if surfsI is None:
            surfs:Set[str]=set()
        elif isinstance(surfsI, str):
            surfs = {surfsI}
        else:
            surfs=surfsI
        for s in surfs:
            self.define_scalar_field("surfconc_" + s, "C2")
            has_surfactants = True
        if has_surfactants:
            self.define_vector_field(self.surfactant_advect_velo_name, self.surfactant_advect_velo_space)

        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props) 
            self.masstransfer_model.define_fields(self)
            self.masstransfer_model._clean_up_for_code() 

        if self.surface_tension_projection_space is not None:
            self.define_scalar_field("_surf_tension", self.surface_tension_projection_space)
            
        if self.project_interface_flux:
            self.define_scalar_field("interface_flux", "C2",scale=scale_factor("velocity"),testscale=1/scale_factor("velocity"))

        
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
        scals:Dict[str,Union[str,ExpressionOrNum]] = {"surfconc_" + s.name: "surface_concentration" for s in self.interface_props._surfactants.keys()} 
        if len(scals) > 0:
            scals[self.surfactant_advect_velo_name] = "velocity"
            tscals = {"surfconc_" + s.name: scale_factor("temporal") / scale_factor("surfconc_" + s.name) for s in
                      self.interface_props._surfactants.keys()} 
            self.set_test_scaling(**tscals)
        scals["mass_transfer_rate"] = scale_factor("velocity") * scale_factor("mass_density")*self.additional_masstransfer_scale
        self.set_scaling(**scals)
        self.add_named_numerical_factor(surface_tension_term=test_scale_factor("velocity")/scale_factor("spatial"))

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
            self.set_scaling(**{self.kinbc_name: 1 / test_scale_factor("velocity")})
            self.set_test_scaling(**{self.kinbc_name: self.additional_kin_bc_test_scale / scale_factor("velocity")})
        else:
            self.set_scaling(**{self.kinbc_name: 1 / test_scale_factor("mesh")})
            self.set_test_scaling(**{self.kinbc_name: 1 / scale_factor("velocity")})

        has_surfactants = False
        surfscales = {self.surfactant_advect_velo_name: "velocity"}
        tsurfscales = {self.surfactant_advect_velo_name: 1 / scale_factor("velocity")}
        if self.interface_props.surfactants is not None:
            for s in self.interface_props.surfactants:
                surfscales["surfconc_" + s] = "surface_concentration"
                tsurfscales["surfconc_" + s] = scale_factor("temporal") / scale_factor("surface_concentration")
                has_surfactants = True
        if has_surfactants:
            self.set_scaling(**surfscales)
            self.set_test_scaling(**tsurfscales)

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
            self.set_scaling(**vcscales)
            self.set_test_scaling(**vctscales)

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
            partial_mass_transfer_rates:Dict[str,Expression] = {}

        # Kinematic boundary condition
        actual_total_transfer_by_rho_inner = dot(mesh_velocity()+self.static_normal_interface_motion*n - u, n)
        kin_bc =  actual_total_transfer_by_rho_inner + self.total_mass_loss_factor_inside *total_mass_transfer_rate / rho_inner
        if self.project_interface_flux:
            iflux,ifluxtest=var_and_test("interface_flux")
            self.add_weak(iflux-kin_bc,ifluxtest)
            kin_bc=iflux
        static = self.static
        if static == "auto":
            static = not self.get_current_code_generator()._coordinates_as_dofs


        if not ns_inner.PFEM_options or not ns_inner.PFEM_options.active:
            self.add_residual(weak(kin_bc, l_test,coordinate_system=self.kinematic_bc_coordinate_sys))        
            if static:
                self.add_residual(-weak(l, dot(n, u_test),coordinate_system=self.kinematic_bc_coordinate_sys))
                if self.static_interface_motion_testfunction is not None:
                    self.add_residual(weak(kin_bc,self.static_interface_motion_testfunction,coordinate_system=self.kinematic_bc_coordinate_sys))
            else:
                self.add_residual(weak(l, dot(n, R_test),coordinate_system=self.kinematic_bc_coordinate_sys))

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
            if self.surface_tension_theta != 1:
                surf_tens = evaluate_in_past(surf_tens, 1 - self.surface_tension_theta)            
            if self.surface_tension_gradient_directly:
                raise RuntimeError("Can only use surface_tension_gradient_directly if surface_tension_projection_space is set")
            else:
                self.add_residual(weak(self.surface_tension_factor*surf_tens, div(u_test)))

        
        self.add_residual(weak(self.additional_normal_traction,dot(n,u_test)))
        if self.masstransfer_model is not None:
            self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props)
            vap_recoil=self.masstransfer_model.get_vapor_recoil_pressure()
            self.masstransfer_model._clean_up_for_code()
            self.add_residual(weak(vap_recoil,dot(n,u_test)))

        #total_mass_flux = actual_total_transfer_by_rho_inner * rho_inner
        if self.masstransfer_model is not None:
            # Component dynamics inside
            for name in ns_inner.fluid_props.required_adv_diff_fields:
                fname = "massfrac_" + name
                wi, wi_test = var_and_test(fname)
                # Both of them are fine# TODO: But at pinned contact lines, we have to see
                advdiffu_inner=inner_bulk_eqs.get_equation_of_type(CompositionAdvectionDiffusionEquations)
                assert isinstance(advdiffu_inner,CompositionAdvectionDiffusionEquations)
                assert partial_mass_transfer_rates is not None
                if advdiffu_inner.integrate_advection_by_parts:
                    flux = wi * rho_inner*dot(var("velocity")-0*partial_t(var("mesh")),var("normal"))  -wi * total_mass_transfer_rate + partial_mass_transfer_rates.get(name, 0)
                else:
                    flux = -wi * total_mass_transfer_rate + partial_mass_transfer_rates.get(name, 0)
                # flux = wi * total_mass_flux + partial_mass_transfer_rates.get(name, 0)
                self.add_residual(weak(flux, wi_test))

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
                        for name in outadvdiffu.fluid_props.required_adv_diff_fields:
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
                        if outadvdiffu.fluid_props.passive_field in partial_mass_transfer_rates.keys() and not self._has_opposite_flow:
                            raise RuntimeError("The exterior phase only solves component diffusion and the component '"+outadvdiffu.fluid_props.passive_field+"' is passive (not solved for explicitly). Yet, we have a mass transfer of this component. This is problematic. Please set passive_field in the exterior phase to some non-volatile component (usually, it is e.g. air, must be the dominant part of the exterior mixture) or solve the flow in the exterior phase by using CompositionFlowEquations instead of CompositionDiffusionEquations")

            # Thermal effects
            tins=self.get_parent_equations(TemperatureConductionEquation)
            if isinstance(tins,TemperatureConductionEquation):
                self.masstransfer_model._setup_for_code(self.get_current_code_generator(),self.interface_props)
                latent_flux=self.masstransfer_model.get_latent_heat_flux()
                self.masstransfer_model._clean_up_for_code()
                _,T_test=var_and_test("temperature")
                self.add_residual(weak(latent_flux,T_test))


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

        if self.interface_props.surfactants is not None and len(self.interface_props.surfactants) > 0:
            nn = dyadic(n, n)
            ut_proj = u - nn @ u
            un_proj = dot(mesh_velocity(), n) * n
            ui, ui_test = var_and_test(self.surfactant_advect_velo_name)
            self.add_residual(weak(ui - (ut_proj + un_proj), ui_test))

            for sprops, amount in self.interface_props._surfactants.items(): 
                self.set_initial_condition("surfconc_" + sprops.name, amount, degraded_start="auto")
                assert isinstance(self.interface_props,LiquidGasInterfaceProperties)
                D = self.interface_props.get_surface_diffusivity(sprops.name)
                assert D is not None
                G, G_test = var_and_test("surfconc_" + sprops.name)
                self.add_residual(weak(partial_t(G), G_test))
                self.add_residual(D * weak(grad(G), grad(G_test)))
                self.add_residual(weak(div(G * ui), G_test))
                if self.interface_props.surfactant_adsorption_rate.get(sprops.name) is not None:
                    assert isinstance(ns_inner.fluid_props,MixtureLiquidProperties)
                    if sprops.name in ns_inner.fluid_props.components:
                        rate = subexpression(self.interface_props.surfactant_adsorption_rate[sprops.name])
                        self.add_residual(-weak(rate, G_test))
                        w_test = testfunction("massfrac_" + sprops.name)
                        # This is not accurate: We will lose partial mass of the liquid (but it won't contribute to any interface motion)
                        self.add_residual(weak(rate * ns_inner.fluid_props.pure_properties[sprops.name].molar_mass, w_test))

    def before_assigning_equations_postorder(self, mesh:AnyMesh):
        nsinner=self.get_parent_equations(NavierStokesEquations)
        assert isinstance(nsinner,NavierStokesEquations)
        if nsinner.PFEM_options and nsinner.PFEM_options.active:
            return 
        # Pin kinematic boundary condition where necessary
        static = self.static
        if static == "auto":
            static = not self.get_current_code_generator()._coordinates_as_dofs
        assert isinstance(mesh,InterfaceMesh)
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
        self.inftyvals = {**infinity_values}
        self.origin = origin
        self.farfield_length = farfield_length

    def define_residuals(self):
        n = self.get_normal()
        d = var("coordinate") - self.origin
        parent = self.get_parent_equations(CompositionAdvectionDiffusionEquations)
        assert isinstance(parent,CompositionAdvectionDiffusionEquations)
        rho = parent.fluid_props.mass_density

        req_adv_diff = parent.fluid_props.required_adv_diff_fields
        ic = parent.fluid_props.initial_condition
        inftyvals = { n: ic["massfrac_" + n] for n in req_adv_diff if "massfrac_" + n in ic.keys()}
        for k, v in self.inftyvals.items():
            if k.startswith("massfrac_"):
                k = k.lstrip("massfrac_")
            inftyvals[k] = v

        for fn, val in inftyvals.items():
            if val is False:
                continue
            if fn.startswith("massfrac_"):
                fn = fn.lstrip("massfrac_")
            D = parent.get_diffusion_coefficient(fn)
            assert D is not None
            y, y_test = var_and_test("massfrac_" + fn)
            R = square_root(dot(d, d))
            # 2D-axisymmetric (i.e. 3D) case
            if isinstance(self.get_coordinate_system(),AxisymmetricCoordinateSystem) or isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem) or (isinstance(self.get_coordinate_system(),CartesianCoordinateSystem) and self.get_nodal_dimension() == 3):
                self.add_residual(weak(rho * D * (y - val) * dot(n, d) / dot(d, d), y_test))
            # 2D case
            else:
                if self.farfield_length is None:
                    raise RuntimeError("For 2D CompositionDiffusionInfinityEquations, farfield_length must be provided")
                self.add_residual(weak(- rho * D * (y - val) * dot(n, d) / (dot(d, d) * log(R/self.farfield_length)), y_test))


class TemperatureConductionEquation(Equations):
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
    """
    def __init__(self,material:AnyMaterialProperties,space:FiniteElementSpaceEnum="C2",rho_override:ExpressionNumOrNone=None,cp_override:ExpressionNumOrNone=None,lambda_override:ExpressionNumOrNone=None,dt_factor:ExpressionOrNum=1):
        super(TemperatureConductionEquation, self).__init__()
        self.material=material
        self.space:FiniteElementSpaceEnum=space
        self.rho_override,self.cp_override,self.lambda_override=rho_override,cp_override,lambda_override
        self.dt_factor=dt_factor

    def define_fields(self):
        #self.define_scalar_field("temperature",self.space,testscale=scale_factor("spatial")**2/(scale_factor("temperature")*scale_factor("thermal_conductivity")))
        self.define_scalar_field("temperature", self.space, testscale=scale_factor("temporal") / (scale_factor("temperature") * scale_factor("rho_cp")))

    def define_residuals(self):
        T,T_test=var_and_test("temperature")
        rho=self.material.mass_density if self.rho_override is None else self.rho_override
        k=self.material.thermal_conductivity if self.lambda_override is None else self.lambda_override
        cp=self.material.specific_heat_capacity if self.cp_override is None else self.cp_override
        if rho is None:
            raise RuntimeError("No mass_density defined in "+str(self.material))
        if k is None:
            raise RuntimeError("No thermal_conductivity defined in "+str(self.material))
        if cp is None:
            raise RuntimeError("No specific_heat_capacity defined in "+str(self.material))
        self.add_residual(weak(self.dt_factor*rho*cp*partial_t(T),T_test))
        self.add_residual(weak(k*grad(T),grad(T_test)))


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
    """

    def __init__(self,material:AnyMaterialProperties,space:FiniteElementSpaceEnum="C2",wind:ExpressionOrNum=var("velocity"),rho_override:ExpressionNumOrNone=None,cp_override:ExpressionNumOrNone=None,lambda_override:ExpressionNumOrNone=None,dt_factor:ExpressionOrNum=1,adv_factor:ExpressionOrNum=1):
        super(TemperatureAdvectionConductionEquation, self).__init__(material,space,rho_override=rho_override,cp_override=cp_override,lambda_override=lambda_override,dt_factor=dt_factor)
        self.wind=wind
        self.adv_factor=adv_factor
        

    def define_residuals(self):
        super(TemperatureAdvectionConductionEquation, self).define_residuals()
        T,T_test=var_and_test("temperature")
        u=self.wind
        rho = self.material.mass_density if self.rho_override is None else self.rho_override
        cp = self.material.specific_heat_capacity if self.cp_override is None else self.cp_override
        self.add_residual(weak(self.adv_factor*rho*cp*dot(u,grad(T)),T_test))


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
        
    def __init__(self,material:AnyMaterialProperties,thickness:ExpressionOrNum,*,ALE:Union[Literal["auto"],bool]="auto",outside_temperature:ExpressionNumOrNone=None):
        super().__init__()
        self.material=material
        self.thickness=thickness
        # TODO: Not sure about ALE here. It would only act tangentially along the connecting interface
        # We just assume it remains all static now, also we would have to consider the motion if thickness changes        
        self.ALE=ALE
        self.outside_temperature=outside_temperature
        
    def define_residuals(self):
        Tin,Tin_test=var_and_test("temperature")
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
            self.add_weak(rho*cp*(PsiIPsiO*partial_t(Tin,ALE=self.ALE)+PsiOPsiO*partial_t(Tout,ALE=self.ALE)),Tout_test)        
        # TODO: If lambda is a functions of the temperature, it well be evaluated at Tin here. We could blend it
        self.add_weak(k*(dzPsiIdzPsiI*Tin+dzPsiOdzPsiI*Tout),Tin_test)
        if self.outside_temperature is None:
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



class SurfactantsAtSolidInterface(InterfaceEquations):
    """
    Represents the handling of surfactants at the solid-liquid-gas interface.
        
    This class requires the parent equations to be of type CompositionAdvectionDiffusionEquations, meaning that if CompositionAdvectionDiffusionEquations (or subclasses) are not defined in the parent domain, an error will be raised.    
        
    Args:
        ls_properties(LiquidSolidInterfaceProperties): The liquid-solid interface properties.
        out_surface_tension(bool): Whether to output the surface tension as a local expression. Default is True.
    """
    required_parent_type=CompositionAdvectionDiffusionEquations
    def __init__(self,ls_properties:LiquidSolidInterfaceProperties,out_surface_tension:bool=True) -> None:
        super().__init__()
        self.space:FiniteElementSpaceEnum="C2"
        self.prefix="_surfconcS_" # we cannot use surfconc_, since it would coincide with the LG-surfactants at the contact line
        self.ls_properties=ls_properties
        self.out_surface_tension=out_surface_tension # Do we output the surface tension (as local expression)

    def identify_surfactants_in_bulk(self) -> List[str]:
        parent=self.get_parent_equations(of_type=CompositionAdvectionDiffusionEquations)
        assert isinstance(parent,CompositionAdvectionDiffusionEquations)
        res:List[str]=[]
        for cname in parent.fluid_props.components:
            c=parent.fluid_props.get_pure_component(cname)
            if isinstance(c,SurfactantProperties) and cname in self.ls_properties.get_liquid_properties().components:
                res.append(cname)
        return res

    def get_surfactant_field_name(self,cname:str) -> str:
        return self.prefix+cname

    def define_fields(self) -> None:
        surfs=self.identify_surfactants_in_bulk()
        for s in surfs:
            fieldname=self.get_surfactant_field_name(s)
            self.define_scalar_field(fieldname,self.space,scale="surface_concentration",testscale=scale_factor("temporal")/scale_factor(fieldname))
            self.define_field_by_substitution("surfconc_"+s,var(fieldname)) # Bind as substitution for e.g. isotherms
            self.add_local_function("surfconc_"+s,var(fieldname)) # And also as local expression for output/plotting

    def define_residuals(self) -> None:
        surfs=self.identify_surfactants_in_bulk()
        parent=cast(CompositionAdvectionDiffusionEquations,self.get_parent_equations(of_type=CompositionAdvectionDiffusionEquations))

        if self.out_surface_tension:
#            expd=self.expand_expression_for_debugging(self.ls_properties.surface_tension)
            self.add_local_function("surface_tension",self.ls_properties.surface_tension)
                    
        for s in surfs:
            fieldname=self.get_surfactant_field_name(s)
            G,Gtest=var_and_test(fieldname)
            transfer=self.ls_properties.surfactant_adsorption_rate.get(s,0)
            transfer=subexpression(transfer)
            self.add_residual(weak(partial_t(G)-transfer,Gtest))            
            DS=self.ls_properties.get_surface_diffusivity(s)
            if DS is not None:
                self.add_residual(weak(DS*grad(G),grad(Gtest)))
            liq_c=parent.fluid_props.get_pure_component(s)
            w_test = testfunction("massfrac_" + s)
            assert liq_c is not None
            self.add_residual(weak(transfer * liq_c.molar_mass, w_test))