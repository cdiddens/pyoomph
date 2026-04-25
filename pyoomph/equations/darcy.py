#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
 
 
 
from ..materials.generic import AnyFluidProperties
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..generic import Equations,InterfaceEquations,CombinedEquations
from ..equations.generic import SpatialErrorEstimator,InitialCondition,ConnectFieldsAtInterface
from ..expressions import *  # Import grad et al
from ..expressions.units import meter
from ..typings import *

if TYPE_CHECKING:
    from ..materials.generic import *


class DarcyEquation(Equations):
    """
    Represents the Darcy equation for porous media flow. The governing equations are given by:
        
        d(rho_f*eta)/dt + div(rho_f*eta*u) = 0
        eta*u = -kappa/mu * grad(p)

    where rho_f is the fluid density, eta is the porosity, u is the velocity, kappa is the permeability, mu is the dynamic viscosity and p is the porous pressure.
    Optionally, the velocity field can be not solved for, in which case it is computed from the porous pressure, such that the equation for pressure is:
        
        div(rho_f*kappa/mu*grad(p)) = d(rho_f*eta)/dt
    
    Args:
        fluid_props: The fluid properties of the fluid in the porous media.
        permeability: The permeability of the porous media. Default is 1e-15 m^2.
        porosity: The porosity of the porous media. Default is 0.3.
        solve_also_velocity: If True, the velocity field is also solved for. Default is False.
    """
    def __init__(self,fluid_props:AnyFluidProperties,permeability:ExpressionOrNum=1e-15*meter**2,porosity:ExpressionOrNum=0.3,solve_also_velocity:bool=False):
        super(DarcyEquation, self).__init__()
        self.pname="porous_pressure"
        self.space:FiniteElementSpaceEnum="C2"
        self.permeability=permeability
        self.fluid_props=fluid_props
        self.porosity=porosity
        self.solve_also_velocity=solve_also_velocity

    def define_fields(self):
        #Create a field on second order space
        self.define_scalar_field(self.pname,self.space,testscale=scale_factor("spatial")**2/(scale_factor(self.pname)*scale_factor("temporal")))
        if self.solve_also_velocity:
            self.define_vector_field("velocity","C2",testscale=1/scale_factor("velocity"))
        else:
            self.define_field_by_substitution("velocity",-self.permeability/(self.fluid_props.dynamic_viscosity*self.porosity)*grad(var(self.pname)))

    def define_residuals(self):
        pp,pptest=var_and_test(self.pname)
        factor=self.fluid_props.mass_density*self.permeability/self.fluid_props.dynamic_viscosity
        self.add_residual(weak(factor*grad(pp),grad(pptest)))
        self.add_residual(weak(partial_t(self.fluid_props.mass_density*self.porosity), pptest))
        if self.solve_also_velocity:
            u,utest=var_and_test("velocity")
            self.add_residual(weak(u+self.permeability/(self.fluid_props.dynamic_viscosity*self.porosity)*grad(pp), utest))


# Porous connection: impose the droplet pressure to the pores & enforce the z-velocity of the droplet to be the porous velocity
class PorousNavierStokesConnection(InterfaceEquations):
    """
        Imposes the droplet pressure to the porous domain and enforces the z-velocity of the droplet to be the porous velocity.

        This class requires the parent equations to be of type DarcyEquation, meaning that if DarcyEquation (or subclasses) are not defined in the parent domain, an error will be raised.
    """

    required_parent_type = DarcyEquation # Must be attached to a domain where DarcyEquation is solved on

    def define_fields(self):
        #Define Lagrange multipliers
        self.define_scalar_field("_lagr_p","C2",scale=1/test_scale_factor("porous_pressure"),testscale=1/scale_factor("porous_pressure"))
        self.define_scalar_field("_lagr_u", "C2", scale=1 / test_scale_factor("velocity",domain=self.get_opposite_side_of_interface()),testscale=1 / scale_factor("velocity"))

    def define_residuals(self):
        # Impose the pressure from the droplet to the porous domain
        lp,lptest=var_and_test("_lagr_p")
        pp,pptest=var_and_test("porous_pressure")
        pd=var("pressure",domain=self.get_opposite_side_of_interface()) # droplet pressure
        self.add_residual(weak(lp,pptest))
        self.add_residual(weak(pp-pd, lptest))

        n=var("normal")
        lu, lutest = var_and_test("_lagr_u")
        ud, udtest = var_and_test("velocity",domain=self.get_opposite_side_of_interface())
        pp = var("porous_pressure", domain=self.get_parent_domain())
        parent=self.get_parent_equations()
        assert isinstance(parent,DarcyEquation)
        mu=parent.fluid_props.dynamic_viscosity
        permeability=parent.permeability
        porous_velo=-permeability/mu*grad(pp)
        self.add_residual(weak(lu, dot(n,udtest)))
        self.add_residual(weak(dot(ud- porous_velo,n) , lutest))

    def before_assigning_equations_postorder(self, mesh:AnyMesh):
        # Pin redundant multipliers, since at the contact line, velocity_y=0 is strongly set
        assert isinstance(mesh,InterfaceMesh)
        self.pin_redundant_lagrange_multipliers(mesh, "_lagr_u", [],opposite_interface=["velocity"])


# Porous front: Move the mesh along with the porous velocity
class PorousFront(InterfaceEquations):
    """
        Move the mesh along with the porous velocity.

        This class requires the parent equations to be of type DarcyEquation, meaning that if DarcyEquation (or subclasses) are not defined in the parent domain, an error will be raised.
    """
    required_parent_type = DarcyEquation
    def __init__(self):
        super(PorousFront, self).__init__()

    def define_fields(self):
        # Lagrange multiplier
        self.define_scalar_field("_lagr_front","C2",scale=1/test_scale_factor("mesh"),testscale=scale_factor("temporal")/scale_factor("spatial"))

    def define_residuals(self):
        l,ltest=var_and_test("_lagr_front")
        R,Rtest=var_and_test("mesh")
        n=var("normal")
        pp = var("porous_pressure", domain=self.get_parent_domain())
        self.add_residual(weak(l,dot(Rtest,n)))
        peqs=self.get_parent_equations()
        assert isinstance(peqs,DarcyEquation)
        mu = peqs.fluid_props.dynamic_viscosity
        permeability = peqs.permeability
        eta=peqs.porosity
        porous_velo = -permeability /(mu*eta) * grad(pp)
        #porous_velo=evaluate_in_past(porous_velo)
        self.add_residual(weak(dot(mesh_velocity()-porous_velo,n), ltest))

    def before_assigning_equations_postorder(self, mesh:AnyMesh):
        assert isinstance(mesh,InterfaceMesh)
        self.pin_redundant_lagrange_multipliers(mesh, "_lagr_front", ["mesh_x","mesh_y"])



## The ones to use for multi-component flow

# Darcy + Advection diffusion
def CompositionDarcyEquations(fluid_props:AnyFluidProperties,compo_space:FiniteElementSpaceEnum="C2",permeability:ExpressionOrNum=1e-15*meter**2,porosity:ExpressionOrNum=0.3,with_IC:bool=True,spatial_errors:Optional[float]=None,isothermal:bool=True,initial_temperature:Optional[ExpressionOrNum]=None,thermal_overrides:Optional[Dict[str,ExpressionOrNum]]=None) -> CombinedEquations:
    from .multi_component import CompositionAdvectionDiffusionEquations,TemperatureAdvectionConductionEquation
    dc=DarcyEquation(fluid_props,permeability=permeability,porosity=porosity,solve_also_velocity=True)

    cp=CompositionAdvectionDiffusionEquations(fluid_props=fluid_props,space=compo_space)
    res=dc+cp
    if not isothermal:
        # TODO: Advect with the superficial or interstitial velocity?
        u_advect=porosity*var("velocity") # I think it is more reasonable that the temperature is transported with the slower one...
        #u_advect=var("velocity")
        if thermal_overrides is None:
            thermal_kwargs={}
        else:
            thermal_kwargs={"rho_override":thermal_overrides.get("mass_density",None),"cp_override":thermal_overrides.get("specific_heat_capacity",None),"lambda_override":thermal_overrides.get("thermal_conductivity",None)}
        res += TemperatureAdvectionConductionEquation(fluid_props, space=compo_space,wind=u_advect,**thermal_kwargs)
    if with_IC:
        req_adv_diff = fluid_props.required_adv_diff_fields
        ic = fluid_props.initial_condition
        icsettings = {"massfrac_"+n: ic["massfrac_" + n] for n in req_adv_diff if "massfrac_" + n in ic.keys()}
        if not isothermal:
            icT0=ic.get("temperature")
            if icT0 is None:
                if initial_temperature is None:
                    raise RuntimeError("You must set an initial temperature either by the definition of the fluid (with Mixture(...,temperature=...) or pass it with the initial_temperature kwarg")
                else:
                    icT0=initial_temperature
            icsettings["temperature"]=icT0
        res+=InitialCondition(**icsettings)

    if spatial_errors is not None:
        if spatial_errors is True:
            compo_fields = {"massfrac_" + n: 1.0 for n in fluid_props.required_adv_diff_fields}
            res += SpatialErrorEstimator(velocity=1, **compo_fields)
        elif spatial_errors is not False:
            raise RuntimeError("TODO")

    return res

# Connection to a Navier-Stokes multi-component domain (imposing continuity of the composition)
def CompositionPorousNavierStokesConnection(fluid_props:AnyFluidProperties):
    psinter=PorousNavierStokesConnection()
    connfields = ["massfrac_" + n for n in fluid_props.required_adv_diff_fields]
    psinter += ConnectFieldsAtInterface(connfields)
    return psinter