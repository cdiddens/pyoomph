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


from pyoomph import *
from pyoomph.equations.multi_component import *
from pyoomph.equations.ALE import *
from pyoomph.materials import *
from pyoomph.materials.mass_transfer import *
import pyoomph.materials.default_materials 

class CapillaryEvaporationProblem(Problem):
    def __init__(self):
        super().__init__()
        # Filled height in the capillary
        self.L=20*milli*meter
        # Capillary radius
        self.R=0.5*milli*meter        
        # Temperature of the system
        self.temperature=20*celsius
        # Initial Liquid mixture composition (glycerol/water)
        self.mixture=Mixture(20*percent*get_pure_liquid("glycerol")+get_pure_liquid("water"))
        # Gas phase composition (air with 20% relative humidity)
        self.gas=Mixture(get_pure_gas("air")+100*percent*get_pure_gas("water"),quantity="relative_humidity",temperature=20*celsius)
        # Gravity and ambient pressure
        self.g=9.81*meter/second**2
        self.ambient_pressure=1*atm
        # Whether we use the GCL or not
        self.use_GCL=True
        
    def define_problem(self):
        self+=LineMesh(size=self.L,N=1000)
        
        # Get the interface properties
        interf=self.mixture | self.gas
        c_water=self.mixture.get_vapor_mass_concentration("water",at_mixture_composition=False)
        # get the water mole fraction in the gas phase, calculate the relative humidity and use it to calculate the far-field water vapor concentration
        xWater=self.gas.evaluate_at_condition(self.gas.get_mole_fraction_field("water"),"IC", temperature=self.temperature,pressure=self.ambient_pressure)
        psat=self.mixture.evaluate_at_condition(self.mixture.get_vapor_pressure_for("water",pure=True),temperature=self.temperature)
        xSat=psat/self.ambient_pressure
        RH=xWater/xSat        
        # and use it to calulate the far-field water vapor concentration
        c_infty=self.mixture.get_vapor_mass_concentration("water",relative_humidity_for_far_field=RH,temperature=self.temperature)
        # Get the diffusion coefficient of water in air at the given temperature
        D_vap=self.gas.get_diffusion_coefficient("water")(temperature=self.temperature)
        # And the evaporation rate        
        j_water=4*D_vap*(c_water-c_infty)/self.R        
        
        # The evaporation rate at the initial condition is used to define the velocity scale for the problem
        j_water0=self.mixture.evaluate_at_condition(j_water,"IC",temperature=self.temperature)
        rho0=self.mixture.evaluate_at_condition(self.mixture.mass_density, "IC", temperature=self.temperature)
        Uscale=j_water0/rho0
       
        # Set reasonable scaling for the problem
        self.set_scaling(spatial=self.L,velocity=Uscale,pressure=rho0*self.g*self.L)
        self.define_named_var(temperature=self.temperature)
        self.mixture.set_reference_scaling_to_problem(self,temperature=self.temperature)
        
        # Flow and composition in the bulk, optionally using the GCL
        eqs=CompositionFlowEquations(self.mixture,gravity=self.g*vector(-1),GCL=self.use_GCL)
        # output and a fixed position at the left side (bottom of the capillary) where the liquid evaporates
        eqs+=LaplaceSmoothedMesh()
        eqs+=TextFileOutput()
        eqs+=DirichletBC(mesh_x=0)@"left" # Open part of the capillary, evaporating from here
        
        # Use a prescribed mass transfer model to impose the evaporation rate at the interface
        mdl=interf.set_mass_transfer_model(PrescribedMassTransfer(water=j_water))
        mdl.projection_space="C2" # Project the evaporation rate onto a continuous quadratic space (here, the space does not matter)
        eqs+=MultiComponentNavierStokesInterface(interf,static=True)@"left"
        
        # The right side of the capillary is allowed to move, but no evaporation is allowed there
        interf_no_evap=self.mixture | self.gas
        interf_no_evap.set_mass_transfer_model(None)
        eqs+=MultiComponentNavierStokesInterface(interf_no_evap)@"right"
        
        # Get the total mass of glycerol
        eqs+=IntegralObservables(M_glycerol=var("massfrac_glycerol")*self.mixture.mass_density)
        eqs+=IntegralObservableOutput("mass_evolution")
        # Get the filled height and it's velocity
        eqs+=IntegralObservables(y=self.L-var("mesh_x"),u=-mesh_velocity()[0])@"right"
        eqs+=IntegralObservableOutput("top_interface")@"right"
        
        
        
        # Refine the region near the evaporating interface to better resolve the gradients in the solution
        eqs+=RefineToLevel(4)@"left"
        
        self+=eqs@"domain"
        
if __name__=="__main__":
    with CapillaryEvaporationProblem() as problem:
        problem.DTSF_max_increase_factor=1.25
        problem.DTSF_min_decrease_factor=0.75
        problem.run(48*hour,outstep=True,startstep=0.001*second,temporal_error=1)