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

 
from kuramoto_sivanshinsky import * # Import the previous example problem

class KSEBifurcationProblem(Problem):
    def __init__(self):
        super(KSEBifurcationProblem, self).__init__()
        self.periods,self.period_y_factor=2,1 # consider two full periods in both directions
        self.N_per_period=20 # elements per period and direction
        self.kc=1/square_root(2) # wave number of the pattern
        # Introduce parameters
        # for both linear and quadratic damping with some initial settings
        self.param_gamma=self.define_global_parameter(gamma=0.24) 
        self.param_delta=self.define_global_parameter(delta=0)


    def define_problem(self):
        kc, N = self.kc, self.N_per_period*self.periods
        Lx = self.periods * 4 * pi / kc # Calulate fitting mesh size
        Ly = self.period_y_factor*self.periods* 4 * square_root(1/3) * pi / kc
        mesh = RectangularQuadMesh(size=[Lx, Ly], N=[N, int(N* Ly / Lx)])
        self.add_mesh(mesh)

        eqs=MeshFileOutput()
        eqs+=DampedKuramotoSivashinskyEquation(gamma=self.param_gamma,delta=self.param_delta)

        # Register different initial conditions
        A=3
        x,y=var(["coordinate_x","coordinate_y"])
        eqs += InitialCondition(h=0,IC_name="flat")
        eqs += InitialCondition(h=2*A/9*(cos(kc*x)+2*cos(kc/2*x)*cos(kc*square_root(3)/2*y)),IC_name="hexdots")
        eqs += InitialCondition(h=-2 * A / 9 * (cos(kc * x) + 2 * cos(kc / 2 * x) * cos(kc * square_root(3) / 2 * y)),IC_name="hexholes")
        eqs += InitialCondition(h= A / 2 * cos(kc * x),IC_name="stripes")

        # And integral observables, in particular h_rms
        eqs += IntegralObservables(_area=1,_h_integral=var("h"),_h_sqr_integral=var("h")**2)
        eqs += IntegralObservables(h_avg=lambda _area,_h_integral : _h_integral/_area)
        eqs += IntegralObservables(h_rms=lambda _area, h_avg,_h_sqr_integral: square_root(_h_sqr_integral/_area - h_avg**2))

        self.add_equations(eqs@"domain")


# slepc eigensolver is more reliable here
import pyoomph.solvers.petsc # Requires petsc4py, slepc4py. Might not work in Windows

if __name__ == "__main__":
    with KSEBifurcationProblem() as problem:
        problem.initialise()
        problem.param_gamma.value=0.24
        problem.param_delta.value = 0.0
        problem.set_initial_condition(ic_name="hexdots") # set the hexdot initial condition
        problem.solve(timestep=10) # One transient step to converge towards the stationary solution
        problem.solve() # stationary solve

        problem.set_eigensolver("slepc") # Set the slepc eigensolver
        # Write eigenvalues to file
        eigenfile=open(os.path.join(problem.get_output_directory(),"hexdots.txt"),"w")
        def output_with_eigen():
            eigvals,eigvects=problem.solve_eigenproblem(6,shift=0) # solve for 6 eigenvalues with zero shift
            h_rms=problem.get_mesh("domain").evaluate_observable("h_rms") # get the root mean square
            line=[problem.param_gamma.value,h_rms,eigvals[0].real,eigvals[0].imag] # line to write
            eigenfile.write("\t".join(map(str,line))+"\n") # write to file
            eigenfile.flush()
            problem.output_at_increased_time() # and write the output

        # Arclength continuation
        output_with_eigen()
        ds=0.001
        while problem.param_gamma.value>0.23:
            ds=problem.arclength_continuation(problem.param_gamma,ds,max_ds=0.005)
            output_with_eigen()

