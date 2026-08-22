#  ========================================================================
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
#  ========================================================================

# Worker for tests/test_mpi_openmp_assembly.py: hybrid MPI + OpenMP.
#
# The comparison is made INSIDE one rank, between an assembly on one thread and one on several, and
# only the verdict is reported. Comparing two separate mpirun invocations instead would compare two
# MPI solves, and those are not bit-reproducible run to run (measured: ~1e-12 apart on the converged
# dofs of an identical pair of runs) - the assembly difference this is looking for is exactly zero,
# so it would drown.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import Problem, DirichletBC
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.generic.mpi import get_mpi_rank
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _Cavity(Problem):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def define_problem(self):
        self += RectangularQuadMesh(N=self.n)
        eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "left"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "right"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self += eqs @ "domain"


def _snapshot(problem):
    res, jac = problem.assemble_jacobian(with_residual=True)
    return [numpy.array(res), jac.data.copy(), jac.indices.copy(), jac.indptr.copy(),
            numpy.array(problem.get_residuals())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    args, _ = ap.parse_known_args()

    result = {"rank": get_mpi_rank()}
    try:
        problem = _Cavity(args.n)
        problem.set_output_directory(args.outdir)
        problem.initialise()
        # Deliberately NOT converged first: at a converged state the residual is ~0 and cancellation
        # would hide a difference that is present in the assembled entries.
        serial = _snapshot(problem)
        before = problem._get_parallel_assemblies_done()
        problem._set_num_assembly_threads(args.threads)
        threaded = _snapshot(problem)
        result["threaded_runs"] = int(problem._get_parallel_assemblies_done() - before)
        result["ndof"] = int(problem.ndof())
        result["identical"] = [bool(numpy.array_equal(a, b)) for a, b in zip(serial, threaded)]
        result["maxdiff"] = [float(numpy.max(numpy.abs(a - b))) if a.shape == b.shape and a.size else 0.0
                             for a, b in zip(serial, threaded)]
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    print("PYOOMPH_MPI_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
