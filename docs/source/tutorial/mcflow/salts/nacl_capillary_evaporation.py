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



from pyoomph import *
from pyoomph.equations.multi_component import *
from pyoomph.equations.ALE import *
from pyoomph.materials import *
from pyoomph.materials.mass_transfer import *
import pyoomph.materials.default_materials
import pyoomph.materials.ions      # registers the ions and the salts

class SaltyCapillaryEvaporationProblem(Problem):
    def __init__(self,salt_treatment="dilute"):
        super().__init__()
        # "dilute": the salt is a concentration riding along, "component": it is a mass fraction
        self.salt_treatment=salt_treatment
        # Filled height in the capillary and capillary radius
        self.L=20*milli*meter
        self.R=0.5*milli*meter
        # Temperature of the system
        self.temperature=20*celsius
        # Initial salt concentration and relative humidity of the surrounding air
        self.c0=1*molar
        self.RH_ambient=80*percent
        # Gravity and ambient pressure
        self.g=9.81*meter/second**2
        self.ambient_pressure=1*atm

    def define_problem(self):
        # The very same brine, described in the two possible ways
        self.mixture=Mixture(get_pure_liquid("water")+self.c0*get_salt("NaCl"),
                             temperature=self.temperature,salt_treatment=self.salt_treatment)
        # AIOMFAC gives the water activity above the brine, i.e. the lowered vapor pressure
        self.mixture.set_activity_coefficients_by_unifac("AIOMFAC")
        self.gas=Mixture(get_pure_gas("air")+self.RH_ambient*get_pure_gas("water"),
                         quantity="relative_humidity",temperature=self.temperature)

        self+=LineMesh(size=self.L,N=400)

        # Get the interface properties and the local (composition-dependent) vapor concentration
        interf=self.mixture | self.gas
        c_water=self.mixture.get_vapor_mass_concentration("water",at_mixture_composition=False)
        # Get the water mole fraction in the gas phase and convert it to a relative humidity. Note
        # that the saturation pressure must be the one of *pure* water, not the one above the brine
        xWater=self.gas.evaluate_at_condition(self.gas.get_mole_fraction_field("water"),"IC",temperature=self.temperature,pressure=self.ambient_pressure)
        psat=self.mixture.evaluate_at_condition(self.mixture.get_vapor_pressure_for("water",pure=True),temperature=self.temperature)
        RH=xWater/(psat/self.ambient_pressure)
        # and use it to calculate the far-field water vapor concentration
        c_infty=self.mixture.get_vapor_mass_concentration("water",relative_humidity_for_far_field=RH,temperature=self.temperature)
        # Get the diffusion coefficient of water in air at the given temperature and the evaporation rate
        D_vap=self.gas.get_diffusion_coefficient("water")(temperature=self.temperature)
        j_water=4*D_vap*(c_water-c_infty)/self.R

        # The evaporation rate at the initial condition sets the velocity scale of the problem
        j_water0=self.mixture.evaluate_at_condition(j_water,"IC",temperature=self.temperature)
        rho0=self.mixture.evaluate_at_condition(self.mixture.mass_density,"IC",temperature=self.temperature)
        Uscale=j_water0/rho0

        self.set_scaling(spatial=self.L,velocity=Uscale,pressure=rho0*self.g*self.L)
        self.define_named_var(temperature=self.temperature)
        self.mixture.set_reference_scaling_to_problem(self,temperature=self.temperature)
        self.set_scaling(ion_concentration=self.c0)

        # Flow, composition and salt in the bulk, in the conservative (GCL) form of the equations
        eqs=CompositionFlowEquations(self.mixture,gravity=self.g*vector(-1),GCL=True,
                                     salt_treatment=self.salt_treatment)
        eqs+=LaplaceSmoothedMesh()
        # c_NaCl is a solved field when salt_treatment="dilute" and a substituted one otherwise.
        # Writing it and the mass density out under names of our own makes the two outputs directly
        # comparable, whichever of the two descriptions produced them
        eqs+=LocalExpressions(c_salt=var("c_NaCl"),density=self.mixture.mass_density)
        eqs+=TextFileOutput()
        eqs+=DirichletBC(mesh_x=0)@"left" # Open part of the capillary, evaporating from here

        # Use a prescribed mass transfer model to impose the evaporation rate at the interface
        mdl=interf.set_mass_transfer_model(PrescribedMassTransfer(water=j_water))
        mdl.projection_space="C2"
        eqs+=MultiComponentNavierStokesInterface(interf,static=True)@"left"

        # The right side of the capillary is allowed to move, but no evaporation is allowed there
        interf_no_evap=self.mixture | self.gas
        interf_no_evap.set_mass_transfer_model(None)
        eqs+=MultiComponentNavierStokesInterface(interf_no_evap)@"right"

        # Total amount of salt, total liquid mass and liquid volume
        eqs+=IntegralObservables(N_salt=var("c_NaCl")*pi*self.R**2,
                                 M_liquid=self.mixture.mass_density*pi*self.R**2,
                                 V=pi*self.R**2)
        eqs+=IntegralObservableOutput("bulk_evolution")
        # Filled height and its velocity
        eqs+=IntegralObservables(y=self.L-var("mesh_x"),u=-mesh_velocity()[0])@"right"
        eqs+=IntegralObservableOutput("top_interface")@"right"
        # And, at the evaporating end, the salt concentration, the surface tension and the water
        # activity, i.e. the vapor pressure lowering the salt is responsible for
        a_water=self.mixture.get_vapor_pressure_for("water")/psat
        eqs+=IntegralObservables(c_surf=var("c_NaCl"),sigma=interf.surface_tension,
                                 a_water=a_water,A=1)@"left"
        eqs+=IntegralObservableOutput("evaporating_end")@"left"

        # Refine the region near the evaporating interface to better resolve the gradients
        eqs+=RefineToLevel(4)@"left"

        self+=eqs@"domain"

if __name__=="__main__":
    for treatment in ["dilute","component"]:
        with SaltyCapillaryEvaporationProblem(treatment) as problem:
            problem.set_output_directory("nacl_capillary_"+treatment)
            problem.DTSF_max_increase_factor=1.25
            problem.DTSF_min_decrease_factor=0.75
            # A transient term divides a difference by dt and hence amplifies roundoff by 1/dt. The
            # temporal scale here is about 10 h, so a first step of 1 ms would be dt~3e-8 nondim and
            # the Newton residual could not fall below ~1e-8, i.e. below the default tolerance
            problem.run(48*hour,outstep=True,startstep=1*second,maxstep=1*hour,temporal_error=1)
