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


from pyoomph import *  # Import pyoomph
from pyoomph.expressions import *  # Import some additional things to express e.g. partial_t


# Anharmonic oscillator by first order system with different time stepping schemes
class AnharmonicOscillator(ODEEquations):
    def __init__(self, scheme):
        super(AnharmonicOscillator, self).__init__()
        self.scheme = scheme

    def define_fields(self):
        self.define_ode_variable("y")
        self.define_ode_variable("dot_y")

    def define_residuals(self):
        y = var("y")
        dot_y = var("dot_y")
        residual = (partial_t(dot_y) + y ** 3) * testfunction(dot_y)
        residual += (partial_t(y) - dot_y) * testfunction(y)
        # Here, we evaluate the chosen scheme just by applying time_scheme(scheme,...)
        self.add_residual(time_scheme(self.scheme, residual))


class AnharmonicOscillatorProblem(Problem):
    def __init__(self, scheme):  # Passing scheme here
        super(AnharmonicOscillatorProblem, self).__init__()
        self.scheme = scheme

    def define_problem(self):
        eqs = AnharmonicOscillator(scheme=self.scheme)

        t = var("time")  # Time variable
        eqs += InitialCondition(y=1,dot_y=0)

        # Calculate the total energy. We also use time_scheme here, e.g. the energy is evaluated by the same scheme as the time stepping
        y = var("y")
        total_energy = time_scheme(self.scheme, 1/2 * partial_t(y) ** 2 + 1/4 * y ** 4)
        eqs += ODEObservables(Etot=total_energy)  # Add the total energy as observable

        eqs += ODEFileOutput()
        self.add_equations(eqs @ "anharmonic_oscillator")


if __name__ == "__main__":
    for scheme in {"BDF1", "BDF2", "Newmark2", "MPT", "TPZ", "Simpson", "Boole"}:
        with AnharmonicOscillatorProblem(scheme) as problem:
            problem.set_output_directory("osci_timestepping_scheme_" + scheme)
            problem.run(endtime=100, numouts=200)
