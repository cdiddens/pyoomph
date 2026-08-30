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
import numpy,io


class BranchSwitchODE(ODEEquations):
    def __init__(self,r):
        super(BranchSwitchODE, self).__init__()
        self.r=r

    def define_fields(self):
        self.define_ode_variable("x","y")

    def define_residuals(self):
        x,x_test=var_and_test("x")
        y,y_test=var_and_test("y")
        self.add_weak(partial_t(x)-(self.r*x+x**2-x**3),x_test)
        self.add_weak(partial_t(y)-((x-rational_num(3,2))*y-y**3),y_test)


class BranchSwitchProblem(Problem):
    def __init__(self):
        super(BranchSwitchProblem, self).__init__()
        self.r=self.define_global_parameter(r=-1)

    def define_problem(self):
        eqs=BranchSwitchODE(self.r)
        eqs+=InitialCondition(x=0,y=0)
        eqs+=ODEFileOutput()
        self.add_equations(eqs@"ode")


if __name__=="__main__":
    with BranchSwitchProblem() as problem:

        # Branch switching needs the analytically derived Hessian: the normal form is built from it
        problem.setup_for_stability_analysis(analytic_hessian=True)
        # Store the arclength tangent in the state files as well. Without it, a state reloaded in the
        # middle of a continuation cannot tell which way along the branch it was going.
        problem.continuation_data_in_states=True

        # Some utility functions to get the current state and write it to the output file. 
        def get_state():
            ode=problem.get_ode("ode")
            return problem.r.value,ode.get_value("x",as_float=True),ode.get_value("y",as_float=True)

        outfile=None
        def start_branch(name):
            global outfile
            outfile= problem.create_text_file_output(name+".txt",["r","x","y","lambda1","lambda2"])

        def write_state():
            eigenvals,_eigenvects=problem.solve_eigenproblem(2)
            r,x,y=get_state()
            outfile.add_row(r,x,y,eigenvals[0].real,eigenvals[1].real)
            problem.output()
            return numpy.amax(numpy.real(eigenvals))

        def write_bifurcation():
            r,x,y=get_state()
            outfile.add_row(r,x,y,0.0,0.0)

        # Some further utility functions to carry on along a branch until a bifurcation is found, or to continue until a certain r is reached
        def continue_to_bifurcation(ds,max_ds=0.05,maxsteps=500):
            """Arclength continuation until the largest eigenvalue changes sign. Returns the step size to carry on with."""
            sign=numpy.sign(write_state())
            for _ in range(maxsteps):
                ds=problem.arclength_continuation(problem.r,ds,max_ds=max_ds)
                if numpy.sign(write_state())!=sign:
                    return ds
            raise RuntimeError("no bifurcation found along this branch")

        def continue_further(ds,until_r,max_ds=0.05,maxsteps=500):
            for _ in range(maxsteps):
                if problem.r.value>=until_r:
                    return
                ds=problem.arclength_continuation(problem.r,ds,max_ds=max_ds)
                write_state()


        

        # Start on the trivial branch x=y=0 at r=-1, where it is stable, and increase r
        problem.r.value=-1
        problem.solve()
        start_branch("branch_trivial")
        ds=continue_to_bifurcation(0.047)

        # We have just stepped over the bifurcation, i.e. the trivial branch has become unstable here.
        # Before turning to the bifurcation itself, scan out the rest of the trivial branch. 
        near_transcritical=problem.backup_state() # Store the state 
        continue_further(ds,2) # scan the unstable branch
        problem.load_state(near_transcritical,quiet=False) # go back to the bifurcation

        # Converge exactly onto the bifurcation and ask what it is
        problem.activate_bifurcation_tracking("r")
        problem.solve()
        normal_form=problem.classify_bifurcation("r")
        print("Found a",normal_form["type"],"at r,x,y =",get_state())

        # It is a transcritical, so there is a second branch through it. Step onto it and follow it
        # towards decreasing r until the fold at r=-1/4. The step size returned by switch_branch is
        # small on purpose and grows again by itself during the continuation.
        start_branch("branch_nontrivial")
        write_bifurcation()
        ds=problem.switch_branch("r",normal_form=normal_form,direction=-1)
        print("Switched onto the second branch at r,x,y =",get_state())
        ds=continue_to_bifurcation(ds)

        # We are just past the fold. Remember where, so that we can come back after inspecting it        
        state_past_fold=problem.backup_state(quiet=True)

        problem.activate_bifurcation_tracking("r")
        problem.solve()
        normal_form=problem.classify_bifurcation("r")
        print("Found a",normal_form["type"],"at r,x,y =",get_state())
        # We cannot switch branches here, of course, since it is a fold
        problem.deactivate_bifurcation_tracking()

        # Back to just past the fold and carry on
        problem.load_state(state_past_fold,quiet=True)
        ds=continue_to_bifurcation(ds)

        # Beyond the pitchfork, y=0 has become unstable. Again, scan out that part of the branch first
        # and then come back to the bifurcation.
        near_pitchfork=problem.backup_state(quiet=True)
        continue_further(ds,2)
        problem.load_state(near_pitchfork,quiet=True)

        # The pitchfork. It must be located with the pitchfork tracker: the fold tracker used above
        # augments the system in a way that is itself singular at a symmetry-breaking bifurcation.
        problem.activate_bifurcation_tracking("r","pitchfork")
        problem.solve()
        # And it must be classified with assume="pitchfork": the y -> -y symmetry of the system makes
        # the quadratic coefficient b2 vanish identically, so all that is left for the classification
        # to measure is round-off, and the symmetry answers the question better than the numbers can.
        normal_form=problem.classify_bifurcation("r",assume="pitchfork")
        print("Found a",normal_form["type"],"at r,x,y =",get_state())

        # Both branches of the pitchfork sit at the same r and differ only in the sign of y, so the
        # direction argument is the only thing that tells them apart
        state_at_pitchfork=problem.backup_state(quiet=True)
        for direction,name in [(1,"branch_pitchfork_plus"),(-1,"branch_pitchfork_minus")]:
            problem.load_state(state_at_pitchfork,quiet=True)
            start_branch(name)
            write_bifurcation()
            ds=problem.switch_branch("r",normal_form=normal_form,direction=direction)
            print("Switched onto the branch with y =",get_state()[2])
            write_state()
            while problem.r.value<2:
                ds=problem.arclength_continuation(problem.r,ds,max_ds=0.1)
                r,x,y=get_state()
                write_state()
                print("r,x,y = {:.5f}, {:.5f}, {:.5f}, analytically y = {:+.5f}".format(r,x,y,direction*numpy.sqrt(x-1.5)))
