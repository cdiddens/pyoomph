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


from kuramoto_sivanshinsky_bifurcation import *
from pyoomph.output.plotting import *


class KSEPlotter(MatplotlibPlotter):
    def define_plot(self):
        cb=self.add_colorbar("h",position="top center") # Add a color bar, but do not show it
        cb.invisible=True

        self.add_plot("domain/h",colorbar=cb) # plot the height field (or its eigenfunction)

        # Parameters as text
        self.add_text(r"$\gamma={:5.4f}, \delta={:5.4f}$".format(self.get_problem().param_gamma.value,self.get_problem().param_delta.value), position="bottom left",textsize=15)

        # Information text, based on whether we plot the eigenfunction or the normal solution
        if self.eigenvector is not None: # eigenfunction is plotted
            self.add_text("eigenvalue={:5.4f}".format(self.get_eigenvalue()),position="bottom right",textsize=15) # eigenvalue (will be 0 anyhow)
            if self.eigenmode=="abs":
                self.add_text("eigenfunction magnitude",textsize=20,position="top center") # title
            elif self.eigenmode=="real":
                self.add_text("eigenfunction (real)", textsize=20, position="top center") # title
            elif self.eigenmode == "imag":
                self.add_text("eigenfunction (imag.)", textsize=20, position="top center") # title
        else:
            self.add_text("height field", textsize=20, position="top center") # title


if __name__ == "__main__":
    with KSEBifurcationProblem() as problem:
        # Add a bunch of plotters
        problem.plotter=[KSEPlotter(problem)] # Plot the normal solution
        problem.plotter.append(KSEPlotter(problem,eigenvector=0,eigenmode="real",filetrunk="eigenreal_{:05d}")) # real part of the eigenfunction
        problem.plotter.append(KSEPlotter(problem, eigenvector=0, eigenmode="imag", filetrunk="eigenimag_{:05d}")) # imag. part
        problem.plotter.append(KSEPlotter(problem, eigenvector=0, eigenmode="abs", filetrunk="eigenabs_{:05d}")) # magnitude

        for p in problem.plotter:
            p.file_ext=["png","pdf"] # plot both png and pdf for all plotters

        # ...
        # the rest is the same
        # ...

        problem.N_per_period *= 2  # make it more accurate

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
