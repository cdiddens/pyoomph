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


from temperature_conduction import *	# To get the mesh from the previous example
from pyoomph.expressions.units import * # units for dimensions
from pyoomph.equations.ALE import * # moving mesh equations


class ThermalConductionEquation(Equations):
    def __init__(self,k,rho,c_p):
        super(ThermalConductionEquation,self).__init__()
        # store conducitiviy, mass density and spec. heat capacity
        self.k,self.rho,self.c_p=k,rho,c_p

    def define_fields(self):
        # Note the testscale here: We want to nondimensionalize the entire equation by the scale "thermal_equation"
        # which will be set at the problem level
        self.define_scalar_field("T","C2",testscale=1/scale_factor("thermal_equation"))

    def define_residuals(self):
        T,T_test=var_and_test("T")
        self.add_residual(weak(self.rho*self.c_p*partial_t(T),T_test)+weak(self.k*grad(T),grad(T_test)))


class IceFrontSpeed(InterfaceEquations):
    required_parent_type=ThermalConductionEquation	# Must have ThermalConductionEquation on the inside bulk
    required_opposite_parent_type = ThermalConductionEquation # and ThermalConductionEquation on the outside bulk

    def __init__(self,latent_heat):
        super(IceFrontSpeed, self).__init__()
        self.latent_heat=latent_heat

    def define_fields(self):
        self.define_scalar_field("_lagr_interf_speed","C2",scale=1/test_scale_factor("mesh"),testscale=scale_factor("temporal")/scale_factor("spatial"))

    def define_residuals(self):
        n=var("normal")
        x,xtest=var_and_test("mesh")
        l,ltest=var_and_test("_lagr_interf_speed")
        k_in=self.get_parent_equations().k		# conductivity of the inside domain
        rho_in=self.get_parent_equations().rho	# density of the inside domain
        k_out=self.get_opposite_parent_equations().k # conductivity of the outside domain
        T_bulk_in=var("T",domain=self.get_parent_domain())	# temperature in the inside bulk
        T_bulk_out = var("T", domain=self.get_opposite_parent_domain()) # temperature in the outside bulk
        speed=dot(k_in*grad(T_bulk_in)-k_out*grad(T_bulk_out),n)/(rho_in*self.latent_heat)
        self.add_residual(weak(dot(mesh_velocity(),n)-speed,ltest))
        self.add_residual(weak(l,dot(xtest,n)))


class IceFrontProblem(Problem):
    def __init__(self):
        super(IceFrontProblem,self).__init__()

        # properties of the ice
        self.rho_ice=915*kilogram/(meter**3) # mass density
        self.k_ice=2.22*watt/(meter*kelvin) # thermal conductivity
        self.cp_ice=2.050*kilo*joule/(kilogram*kelvin)	# spec. heat capacity

        # properties of the liquid
        self.rho_liq=999.87*kilogram/(meter**3)
        self.k_liq=0.5610*watt/(meter*kelvin)
        self.cp_liq=4.22*kilo*joule/(kilogram*kelvin)

        self.T_eq=0*celsius # Melting point
        self.latent_heat= 334 *joule/gram # Latent heat of melting/solidification

        self.L=1*milli*meter # domain length
        self.front_start_fraction=0.3 # initial relative position of the front
        self.T_left=-1*celsius # left and right temperatures
        self.T_right=1*celsius


    def define_problem(self):
        # Mesh: a dimensional size and xI is set, also the domains are renamed
        self.add_mesh(TwoDomainMesh1d(L=self.L,xI=self.L*self.front_start_fraction,left_domain_name="ice",right_domain_name="liquid"))
        self.set_scaling(spatial=self.L,temporal=100*second) # Nondimensionalize space and time by these quantities
        self.set_scaling(T=kelvin) # Temperature scale
        # Now, we define the scale "thermal_equation", by what both thermal equations will be divided
        # We take the conduction term of the ice as reference here
        self.set_scaling(thermal_equation=scale_factor("T")*self.k_ice/scale_factor("spatial")**2)

        # Create similar equations on both domains
        # wrap the domain name and the corresponding properties
        domain_props=[["ice",self.k_ice,self.rho_ice,self.cp_ice,self.T_left],
                      ["liquid",self.k_liq,self.rho_liq,self.cp_liq,self.T_right]]
        for (domain_name,k,rho,cp,T_init) in domain_props: # iterate over the entries
            eqs=TextFileOutput() # Output
            eqs+=ThermalConductionEquation(k,rho,cp) # thermal transport eq
            eqs+=LaplaceSmoothedMesh()  # mesh motion
            eqs+=InitialCondition(T=T_init) # initial condition
            eqs+=SpatialErrorEstimator(T=1) # spatial adaptivity
            eqs+=DirichletBC(T=self.T_eq)@"interface" # melting point at the interface
            self.add_equations(eqs@domain_name) # add the equations

        # Dirichlet conditions
        self.add_equations(DirichletBC(T=self.T_left,mesh_x=0)@"ice/left")
        self.add_equations(DirichletBC(T=self.T_right,mesh_x=self.L)@"liquid/right")

        # Interface equations
        interf_eqs=IceFrontSpeed(self.latent_heat) # Front speed equation
        interf_eqs+=ConnectMeshAtInterface() # Connect the mesh at xI

        # We could also add it on "liquid/interface", but then we must use -self.latent_heat in the IceFrontSpeed
        self.add_equations(interf_eqs@"ice/interface")


if __name__=="__main__":
    with IceFrontProblem() as problem:
        problem.run(1000*second,startstep=0.00001*second,outstep=True,temporal_error=1,spatial_adapt=1,maxstep=2*second)



