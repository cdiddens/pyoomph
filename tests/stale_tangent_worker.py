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

"""A dump whose stored continuation tangent no longer matches the problem must still load.

This is the branch that rescues a state file written BEFORE the save-side guard - the user's diagram
already contains an augmented tangent. Reproduced here by changing ndof between save and load, which
produces the same mismatch without needing a tracker.
"""
import os, sys
from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, var, grad, exp, partial_t
from pyoomph.equations.generic import SpatialErrorEstimator, IntegralObservables
from pyoomph.meshes.simplemeshes import LineMesh

class Bratu(Equations):
    def __init__(self, lam): super().__init__(); self.lam=lam
    def define_fields(self): self.define_scalar_field("u","C2")
    def define_residuals(self):
        u,v=var_and_test("u")
        self.add_weak(partial_t(u),v); self.add_weak(grad(u),grad(v)); self.add_weak(-self.lam*exp(u),v)

class Prob(Problem):
    def define_problem(self):
        self.add_mesh(LineMesh(N=20))
        eqs=Bratu(self.get_global_parameter("lam"))+InitialCondition(u=0)
        eqs+=DirichletBC(u=0)@"left"; eqs+=DirichletBC(u=0)@"right"
        eqs+=SpatialErrorEstimator(u=1)
        eqs+=IntegralObservables(_a=1,_ui=var("u"))
        eqs+=IntegralObservables(u_avg=lambda _a,_ui: _ui/_a)
        self+=eqs@"domain"


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
        problem.continuation_data_in_states=True
        problem.max_refinement_level=4
        problem.max_permitted_error=1e-7; problem.min_permitted_error=1e-9
        problem.get_global_parameter("lam").value=1.0
        problem.solve()
        for _ in range(3): problem.arclength_continuation("lam",0.05)
        n0=problem.ndof()
        fn=os.path.join(sys.argv[1],"coarse.dump")
        problem.save_state(fn)
        print("saved with ndof           :",n0," tangent:",len(problem.get_arclength_dof_derivative_vector()))
        problem.adapt()
        print("after adapt, ndof         :",problem.ndof())
        assert problem.ndof()!=n0, "the probe needs the dof count to change"
        problem.load_state(fn)          # stored tangent has n0 entries, problem now has more
        n_after=len(problem.get_arclength_dof_derivative_vector())
        print("loaded a stale-tangent dump: OK, ndof now {:d}, tangent {:d}".format(problem.ndof(),n_after))
        # The dump restores its own mesh, so by the time the parked tangent is applied the counts agree
        # again and it is RESTORED - not merely survived. Applying it before the renumbering was the bug.
        assert problem.ndof()==n0, "the dump restores its own mesh"
        assert n_after==n0, "the continuation tangent must be restored, not dropped: got "+str(n_after)
        print("STALE TANGENT OK")
        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
