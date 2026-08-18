#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

"""A state dumped while a bifurcation tracker is active must be loadable afterwards.

That is exactly what the bifurcation GUI does: locate_bifurcation() activates the tracker, solves, and
records the point - which dumps a state - and only then deactivates. The dump therefore carried a
continuation tangent of the AUGMENTED length (2n+1 for a fold, 3n+2 for a Hopf/azimuthal), and
reloading the diagram later threw

    Mismatching size in the dof direction vector and the actual number of DoFs

out of Problem::set_dof_direction_arclength, which took the whole reload with it.

Small ODE, no mesh, no PETSc: this reproduces the mechanism without the crash the real problem hits.
"""
import os, sys
from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t

class Eqs(ODEEquations):
    def __init__(self, mu, b): super().__init__(); self.mu=mu; self.b=b
    def define_fields(self): self.define_ode_variable("u")
    def define_residuals(self):
        u,ut=var_and_test("u"); self.add_weak(partial_t(u)-(self.mu-u**2-self.b*u),ut)

class Prob(Problem):
    def define_problem(self):
        self+=(Eqs(self.get_global_parameter("mu"),self.get_global_parameter("b"))
               +InitialCondition(u=1.0))@"ode"


# Run under a __main__ guard: this file lives in tests/ and the suite is invoked as
# `python -m pytest *.py` (see citools/nightly_develop.sh), which hands pytest every .py in
# the directory by name -- and a file named on the command line is imported regardless of
# python_files. Without the guard the whole problem below is built and solved at COLLECTION
# time, and pyoomph's own argv parsing then reads pytest's first filename argument as the
# output directory, so it tries to mkdir over a source file and collection dies. Every other
# worker in this directory is guarded the same way.
def main():
    with Prob() as problem:
        problem.set_output_directory(sys.argv[1])
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.continuation_data_in_states=True
        problem.get_global_parameter("mu").value=1.0
        problem.get_global_parameter("b").value=0.4
        problem.quiet()
        problem.solve()
        # A real continuation tangent must exist, or there is nothing to store and nothing to prove.
        problem.arclength_continuation("mu",-0.2)
        print("plain ndof                :",problem.ndof(),
              " tangent len:",len(problem.get_arclength_dof_derivative_vector()))

        problem.solve_eigenproblem(1)
        problem.activate_bifurcation_tracking("mu","fold")
        problem.solve()
        # A locus: continue in the OTHER parameter with the tracker active. This is what leaves a non-empty
        # tangent of the augmented length, which is what actually got written into the dump.
        problem.arclength_continuation("b",0.05)
        print("augmented ndof            :",problem.ndof(),
              " tangent len:",len(problem.get_arclength_dof_derivative_vector()))
        assert len(problem.get_arclength_dof_derivative_vector())==problem.ndof()>1, \
            "the probe needs a NON-EMPTY augmented tangent to be testing anything"
        fn=os.path.join(sys.argv[1],"tracked.dump")
        problem.save_state(fn)          # <- the GUI does exactly this, while tracking
        problem.deactivate_bifurcation_tracking()
        print("back to plain ndof        :",problem.ndof())

        problem.load_state(fn)          # <- used to raise
        print("loaded a tracked dump     : OK")

        # And the ordinary (unaugmented) case must still round-trip its tangent. Restart from a regular
        # point: the dump just loaded sits ON the fold, where the plain Jacobian is singular.
        problem.get_global_parameter("mu").value=1.0
        problem.get_global_parameter("b").value=0.4
        problem.set_initial_condition()
        problem.solve()
        problem.arclength_continuation("mu",-0.1)
        n_before=len(problem.get_arclength_dof_derivative_vector())
        fn2=os.path.join(sys.argv[1],"plain.dump")
        problem.save_state(fn2)
        problem.reset_arc_length_parameters()
        problem.load_state(fn2)
        n_after=len(problem.get_arclength_dof_derivative_vector())
        print("plain dump round trip     : tangent {:d} -> {:d}".format(n_before,n_after))
        assert n_after==n_before>0, "an ordinary dump must still restore its continuation tangent"
        print("TRACKED STATE OK")
        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
