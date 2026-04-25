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


from pyoomph import *
from pyoomph.equations.multi_component import *
from pyoomph.expressions.utils import * # for the random perturbation
from pyoomph.materials import *
import pyoomph.materials.default_materials # Alternatively, define the materials by hand


class MarangoniHeleShawProblem(Problem):
    def __init__(self):
        super(MarangoniHeleShawProblem, self).__init__()
        # domain size: Gas size is the same as domain_length
        self.domain_length,self.domain_width=0.5*milli*meter,0.5*milli*meter
        self.cell_thickness=20*micro*meter # Hele-Shaw plate distance
        self.Nx=18 # Num elements in x direction
        self.max_refinement_level=3 # max refinement level to refine near the interface
        self.temperature=20*celsius # Temperature : Required for some properties
        self.liquid_mixture = Mixture(get_pure_liquid("water") + 0.5 * get_pure_liquid("ethanol"))  # Default liquid mixture
        # The gas mixture must be adjusted: In the experiment, the evaporation happens in 3d space
        # Here, it is just two-dimensional so the Green's function of the Poisson equation for diffusion is not bounded!
        # Therefore, we pin the vapor concentration at the far right to this composition
        self.gas_mixture = Mixture(get_pure_gas("air") + 20*percent * get_pure_gas("ethanol") + 40*percent * get_pure_gas("water"),quantity="relative_humidity",temperature=self.temperature)
        self.interface_props=None # Interface properties, are determined automatically if not set

    def define_problem(self):
        # Spatial and temporal scales must be set by hand
        self.set_scaling(spatial=self.domain_length, temporal=1 * second)
        # Set remaining scales by the liquid properties
        self.liquid_mixture.set_reference_scaling_to_problem(self, temperature=self.temperature)
        # Adjust pressure and velocity a bit to the problem
        self.set_scaling(pressure=10 * pascal, velocity=1e-4 * meter / second)
        # define global constants "temperature" and "absolute_pressure". It might be required by the fluid properties
        self.define_named_var(temperature=self.temperature, absolute_pressure=1 * atm)

        # Mesh: All elements with center further away than 1*domain_length (measured in spatial scale) will be gas, otherwise liquid
        domain_func=lambda x,y: "gas" if x>1 else "liquid"
        mesh=RectangularQuadMesh(size=[2*self.domain_length,self.domain_width],N=[2*self.Nx,int(self.Nx*self.domain_width/self.domain_length)],name=domain_func)
        self.add_mesh(mesh)

        # We can either set the interface properties by hand, e.g. to modify the surface tension
        # if not, we must find it from the material library
        if self.interface_props is None:
            # To get the interface properties, we can just use the | operator
            self.interface_props=self.liquid_mixture | self.gas_mixture
            # When a particular liquid-gas interface is not defined, it will use a default interface
            # This one will use a reasonable mass transfer model and the default_surface_tension["gas"] of the liquid properties

        liq_eqs=MeshFileOutput()
        # Flow with Hele-Shaw confinement and use second order for the composition
        liq_eqs+=CompositionFlowEquations(self.liquid_mixture,hele_shaw_thickness=self.cell_thickness,compo_space="C2",spatial_errors=True)
        liq_eqs+=DirichletBC(velocity_y=0)@"bottom"
        liq_eqs += DirichletBC(velocity_y=0) @ "top"
        liq_eqs+=MultiComponentNavierStokesInterface(self.interface_props)@"gas_liquid"
        liq_eqs+=RefineToLevel()@"gas_liquid" # And refine it to max_refinement_level

        # Gas
        gas_eqs=MeshFileOutput()
        gas_eqs+=CompositionDiffusionEquations(self.gas_mixture) # just diffusion
        # And fix the far boundary to the initial condition by iterating over all advection diffusion fields for the mass fractions
        gas_eqs+=DirichletBC(**{"massfrac_"+c:True for c in self.gas_mixture.required_adv_diff_fields})@"right"

        self.add_equations(liq_eqs@"liquid"+gas_eqs@"gas")


if __name__=="__main__":
    with MarangoniHeleShawProblem() as problem:
        # Slightly perturb the interface
        # 10 random numbers with a small amplitude linearily interpolated on the interval 0:1
        randpert=DeterministicRandomField(min_x=[0],max_x=[1],amplitude=0.002,Nresolution=10)
        yn=var("coordinate_y")/problem.domain_width # normalized coordinate
        randpert=randpert(yn) # interpolated random fields
        # Perturb the interface composition slightly
        problem.additional_equations+=InitialCondition(massfrac_ethanol=problem.liquid_mixture.initial_condition["massfrac_ethanol"]+randpert)@"liquid/gas_liquid"
        problem.run(10*second,startstep=0.01*second,maxstep=0.5*second,outstep=True,temporal_error=1,spatial_adapt=1)





