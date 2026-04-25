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


from temperature_conduction_propagation import * # Get some equations from the previous example
from pyoomph.equations.navier_stokes import * # We also need a Navier-Stokes equation

# We are lazy and use the Gmsh approach here instead of adding the elements by hand
class TwoDomainMeshAxi(GmshTemplate):
    def __init__(self,R1,R2,H,resolution):
        super(TwoDomainMeshAxi, self).__init__()
        self.R1,self.R2,self.H=R1,R2,H
        self.default_resolution=resolution

    def define_geometry(self):
        p00, p0H=self.point(0,0), self.point(0,self.H)
        pR10, pR1H=self.point(self.R1,0), self.point(self.R1,self.H)
        pR20, pR2H = self.point(self.R2, 0), self.point(self.R2, self.H)
        self.create_lines(pR10,"ice_bottom",p00,"axisymm",p0H,"ice_top",pR1H,"interface",pR10,"liquid_bottom",pR20,"liquid_side",pR2H,"liquid_top",pR1H)
        self.plane_surface("ice_bottom","axisymm","interface","ice_top",name="ice")
        self.plane_surface("liquid_bottom", "liquid_side", "interface", "liquid_top", name="liquid")


# Augment the conduction equation by advection for the liquid phase
class ThermalAdvectionConductionEquation(ThermalConductionEquation):
    def __init__(self,k,rho,c_p,wind=var("velocity")):
        super(ThermalAdvectionConductionEquation, self).__init__(k=k,rho=rho,c_p=c_p)
        self.wind=wind

    def define_residuals(self):
        super(ThermalAdvectionConductionEquation, self).define_residuals() # define the conduction equation
        T,Ttest=var_and_test("T")
        self.add_residual(weak(self.rho*self.c_p*dot(self.wind,grad(T)),Ttest)) # Just add the advection term


# We inherit from the IceFrontProblem to take over the physical parameters. The problem will be quite different:
class IceConvectionProblem(IceFrontProblem):
    def __init__(self):
        super(IceConvectionProblem, self).__init__() # this will set all properties from the parent class
        self.L=4*centi*meter # L is now the cylinder height
        self.R1=1*centi*meter # radius of the ice cylinder
        self.R2=3*centi*meter # outer radius of the liquid cylinder

        self.T_ice=-1*celsius # initial ice cylinder temperature
        self.T_liq=6*celsius

        self.mu_liq=1*milli*pascal*second
        # Water density for buoyancy calculations
        Trel = (var("T") - self.T_eq) / kelvin # bind the relative temperature (measured in Kelvin)
        # Fit for the density anomaly
        self.rho_grav = (0.999849 + 5.77393e-05 * Trel - 7.18258e-06 * Trel ** 2) * gram / (centi * meter )** 3
        self.gravity=9.81*meter/second**2 * vector(0,-1) # gravity direction and strength

        self.resolution=0.05 # mesh resolution

    def define_problem(self):
        # Two-dimensional mesh
        self.add_mesh(TwoDomainMeshAxi(self.R1,self.R2,self.L,self.resolution))
        self.set_coordinate_system("axisymmetric") # axisymmetric coordinate system

        # Similar to the previous problem, scales for nondimensionalization
        self.set_scaling(spatial=self.R1,temporal=1*second)
        self.set_scaling(T=kelvin)
        self.set_scaling(thermal_equation=scale_factor("T") * self.k_ice / scale_factor("spatial") ** 2)
        self.set_scaling(velocity=scale_factor("spatial")/scale_factor("temporal"))
        self.set_scaling(pressure=self.mu_liq*scale_factor("velocity")/scale_factor("spatial"))

        # Equations for the ice domain
        ice_eqs=MeshFileOutput() # Output
        ice_eqs+=ThermalConductionEquation(self.k_ice,self.rho_ice,self.cp_ice) # thermal conduction
        ice_eqs +=InitialCondition(T=self.T_ice) # initially at ice temperature
        ice_eqs+=PseudoElasticMesh() # Mesh motion
        ice_eqs+=DirichletBC(mesh_x=0)@"axisymm" # fix mesh at axis of symmetry
        ice_eqs += DirichletBC(mesh_y=0) @ "ice_bottom" # and at the bottom
        ice_eqs += DirichletBC(mesh_y=self.L) @ "ice_top" # and the top
        ice_eqs += DirichletBC(T=self.T_eq)@"interface" # melting temperature at interface

        # Equations for the liquid domain
        liq_eqs=MeshFileOutput() # output
        liq_eqs+=ThermalAdvectionConductionEquation(self.k_liq,self.rho_liq,self.cp_liq) # thermal conduction + advection
        # Navier-Stokes including Boussinesq-like gravity term
        liq_eqs+=NavierStokesEquations(mass_density=self.rho_liq,dynamic_viscosity=self.mu_liq,bulkforce=self.gravity*self.rho_grav)
        liq_eqs+=InitialCondition(T=self.T_liq) # Liquid temperature as initial condition
        liq_eqs+=PseudoElasticMesh() # Mesh motion
        liq_eqs+=DirichletBC(mesh_y=0,velocity_x=0,velocity_y=0)@"liquid_bottom" # no-slip and fixed mesh at all boundaries
        liq_eqs += DirichletBC(mesh_y=self.L,velocity_x=0,velocity_y=0) @ "liquid_top"
        liq_eqs += DirichletBC(mesh_x=self.R2,T=self.T_liq,velocity_x=0,velocity_y=0)@"liquid_side"
        liq_eqs+=DirichletBC(T=self.T_eq,velocity_x=0,velocity_y=0)@"interface" # here the mesh is not fixed, but the temperature is
        liq_eqs += DirichletBC(pressure=0) @ "liquid_top/liquid_side" # For pure DirichletBCs, we must fix one pressure degree

        # Since we know that the mesh mainly moves in y-direction, we can speed up the calculation by removing the motion in y-direction
        ice_eqs += DirichletBC(mesh_y=True)
        liq_eqs += DirichletBC(mesh_y=True)

        # Interface: Connect the mesh position and impose the front motion
        interf_eqs=ConnectMeshAtInterface()
        interf_eqs+=IceFrontSpeed(self.latent_heat)

        # Add it to the ice side of the interface
        ice_eqs+=interf_eqs@"interface"

        # and add all equations to the problem
        self.add_equations(ice_eqs@"ice"+liq_eqs@"liquid")


if __name__=="__main__":
    with IceConvectionProblem() as problem:
        problem.run(400*second,outstep=True,startstep=1*second,maxstep=20*second,temporal_error=1)


