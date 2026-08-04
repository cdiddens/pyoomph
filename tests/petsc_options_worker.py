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

# The subject of test_petsc_options_hygiene: a run that has SLEPc/MUMPS configured but solves with a
# different linear solver and never touches an eigenproblem. Has to be a separate process, because
# what is under test is only printed by PetscFinalize, i.e. at interpreter shutdown.

import sys

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var, partial_t, weak, testfunction


class _Decay(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("y")

    def define_residuals(self):
        y = var("y")
        self.add_residual(weak(partial_t(y) + y, testfunction(y)))


class _DecayProblem(Problem):
    def define_problem(self):
        self.add_equations((_Decay() + InitialCondition(y=1)) @ "ode")


if __name__ == "__main__":
    with _DecayProblem() as p:
        p.quiet()
        p.set_linear_solver("superlu")   # deliberately not PETSc: nothing here needs the options below
        p.set_eigensolver("slepc_mumps")
        try:
            p.get_eigen_solver()         # constructing it is what fills PETSc's options database
        except RuntimeError as e:
            # No MUMPS in this PETSc build -- then there is nothing for this test to be about.
            print("PYOOMPH_NO_MUMPS", e)
            sys.exit(0)
        p.run(1, numouts=1)
    print("PYOOMPH_WORKER_DONE")
