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
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import *
from pyoomph.equations.electrostatics import *
from pyoomph.meshes.simplemeshes import LineMesh


class DoubleLayerRelaxationProblem(Problem):
    def __init__(self,c0=1*micro*molar,transfer=False):
        super().__init__()
        # Bulk concentration of a symmetric 1:1 electrolyte, and its diffusivity
        self.c0=c0
        self.D=1e-9*meter**2/second
        # Liquid film and gas gap, both 10 um
        self.Lliq=10*micro*meter
        self.Lgas=10*micro*meter
        # Applied voltage between the electrode in the liquid and the counter electrode in the gas
        self.V=5*volt
        # Temperature and the two permittivities
        self.temperature=20*celsius
        self.eps_liq=80
        self.eps_gas=1
        # Whether ions may adsorb on the liquid|gas interface, i.e. whether charge is transferred
        # from the liquid to the surface, or whether the interface is merely polarized
        self.transfer=transfer
        # Areal density of adsorption sites, as a multiple of c0*lambda_D -- i.e. of the amount of
        # ions that already sits within one Debye length of the surface. Below about 0.1 the
        # adsorbed charge is a correction to the field-driven layer; above about 1 it overturns it.
        self.site_density=2
        # Elements in the liquid: the Debye layer has to be resolved, everything else is easy
        self.Nliq=440
        self.Ngas=40

    def debye_length(self):
        """The Debye length of the bulk electrolyte, which sets the whole problem."""
        return debye_length(self.eps_liq*epsilon_0,self.c0,self.temperature)

    def debye_time(self):
        """lambda_D^2/D -- the intrinsic relaxation time of the diffuse layer."""
        return self.debye_length()**2/self.D

    def define_problem(self):
        lD=self.debye_length()
        # The potential scale must be the APPLIED voltage and not the thermal voltage. Both are
        # "correct" -- scales are cosmetic for the answer -- but 5 V is 200 thermal voltages, and a
        # nondimensional potential of 200 makes the Newton solver fail on the very first step.
        set_electrostatic_scaling(self,potential=self.V,temperature=self.temperature,
                                  ion_concentration=self.c0,length=lD,
                                  permittivity=self.eps_liq*epsilon_0)
        self.set_scaling(spatial=self.Lliq,temporal=self.debye_time())

        # One mesh, two domains: the liquid|gas interface is then created automatically as "liq_gas"
        self+=LineMesh(N=self.Nliq+self.Ngas,size=self.Lliq+self.Lgas,
                       left_name="electrode",right_name="counter",
                       name=lambda x: "liq" if x<1.0 else "gas")

        # The electrolyte: one cation and one anion, no flow. The natural boundary condition of
        # NernstPlanckEquations is a blocking wall, so the electrode needs nothing but its potential.
        ions=symmetric_electrolyte(self.c0,1,cation_diffusivity=self.D,anion_diffusivity=self.D)
        liq=PoissonNernstPlanck(ions,relative_permittivity=self.eps_liq,wind=0,
                                temperature=self.temperature)
        liq+=ElectrodeBC(self.V)@"electrode"

        # The gas is a plain dielectric: no ions, no conductivity, just Gauss's law
        gas=ElectricPotentialEquations(relative_permittivity=self.eps_gas)
        gas+=ElectrodeBC(0)@"counter"

        # The liquid|gas interface. Without transfer it only transmits the displacement field;
        # with transfer, cations adsorb on it and the charge they carry leaves the liquid.
        if self.transfer:
            # A Langmuir isotherm: dGamma/dt = k_a*c*(Gamma_max-Gamma) - k_d*Gamma, written as a
            # molar rate per ion. SurfaceChargeConservation turns it into z*F*rate of surface
            # charge and takes exactly the same number of moles out of the adjacent bulk.
            Gmax=self.site_density*self.c0*lD
            k_a=50/(self.debye_time()*self.c0)
            k_d=50/self.debye_time()
            Gamma=var("qs")/faraday_constant
            ifeqs=SurfaceChargeConservation(name="qs",bulk_currents=0,advection_velocity=0,
                    adsorption={"cation":k_a*var("c_cation")*(Gmax-Gamma)-k_d*Gamma})
            ifeqs+=ElectricPotentialConnection(surface_charge_density="qs")
        else:
            ifeqs=ElectricPotentialConnection()

        # The potential drop across the diffuse layer at the interface, i.e. the zeta potential:
        # the liquid is a conductor, so the applied voltage appears at the interface almost in full
        # and the interesting quantity is what is left over after subtracting it.
        ifeqs+=IntegralObservables(_area=1,zeta=var("phi")-self.V)
        # The adsorbed amount, and the total that has to stay constant (see below)
        adsorbed=var("qs")/faraday_constant if self.transfer else 0
        ifeqs+=IntegralObservables(adsorbed=adsorbed)
        liq+=ifeqs@"liq_gas"

        # Ion conservation: with blocking walls everywhere, the cations can only go onto the
        # surface, so the sum of "dissolved" here and "adsorbed" on the interface may not move at
        # all. The two live on different domains and therefore land in different output files; they
        # are added up when the results are read back.
        liq+=IntegralObservables(_volume=1,dissolved=var("c_cation"),
                                 charge=var("charge_density"))
        liq+=IntegralObservableOutput("bulk")
        liq+=IntegralObservableOutput("interface")@"liq_gas"
        liq+=TextFileOutput()
        gas+=TextFileOutput()

        self+=liq@"liq"
        self+=gas@"gas"


def solve(problem,outdir):
    """Run for twelve Debye times, with a fixed step of a tenth of one.

    Deliberately not adaptive: the whole point of the run is to resolve a single exponential, and
    an adaptive stepper is very good at stepping straight over it -- with ``temporal_error=1`` it
    reaches equilibrium inside the first output interval and there is nothing left to look at.
    """
    problem.set_output_directory(outdir)
    tD=problem.debye_time()
    problem.run(12*tD,outstep=0.1*tD,startstep=0.1*tD)


if __name__=="__main__":
    # The base case: the interface is polarized, but no charge crosses it
    with DoubleLayerRelaxationProblem() as problem:
        solve(problem,"dl_polarized")

    # The same cell, but now cations may adsorb on the interface. With few sites the adsorbed
    # charge is a correction; with many it overturns the field-driven layer altogether.
    for sites in [0.1,0.5,2]:
        with DoubleLayerRelaxationProblem(transfer=True) as problem:
            problem.site_density=sites
            solve(problem,"dl_transfer_{:g}".format(sites))

    # A concentration sweep, to see which of the three time scales the relaxation follows
    for c0 in [0.25*micro*molar,1*micro*molar,4*micro*molar,16*micro*molar]:
        with DoubleLayerRelaxationProblem(c0=c0) as problem:
            solve(problem,"dl_sweep_{:g}nM".format(float(c0/(nano*molar))))
