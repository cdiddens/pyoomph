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


from kuramoto_sivanshinsky_arclength_eigen import * # Import the previous example problem

if __name__ == "__main__":
    with KSEBifurcationProblem() as problem:
        # Output the zeroth eigenvector. Will only output if the eigenvalue/vector is calculated either by
        # solve_eigenproblem or by bifurcation tracking
        problem.additional_equations+=MeshFileOutput(eigenvector=0,eigenmode="real",filetrunk="eigen0_real")@"domain"

        problem.initialise()
        problem.param_gamma.value=0.24
        problem.param_delta.value = 0.0
        problem.set_initial_condition(ic_name="hexdots")
        problem.solve(timestep=10) # One transient step to converge towards the stationary solution
        problem.solve() # stationary solve

        # from the previous example we know that the fold bifurcation happens close to 0.28
        problem.param_gamma.value=0.28
        problem.solve() # solve at gamma=0.28

        # Activate bifurcation tracking
        problem.activate_bifurcation_tracking(problem.param_gamma,"fold")
        problem.solve()
        print("FOLD BIFURCATION HAPPENS AT",problem.param_gamma.value)

        hexfold_file = open(os.path.join(problem.get_output_directory(), "hexfold.txt"), "w")
        def output_with_params():
            h_rms = problem.get_mesh("domain").evaluate_observable("h_rms")  # get the root mean square
            line = [problem.param_gamma.value, problem.param_delta.value,h_rms]  # line to write
            hexfold_file.write("\t".join(map(str, line)) + "\n")  # write to file
            hexfold_file.flush()
            problem.output_at_increased_time()  # and write the output

        output_with_params()
        ds = 0.025
        while problem.param_delta.value < 0.5:
            ds = problem.arclength_continuation(problem.param_delta, ds, max_ds=0.025)
            output_with_params()
