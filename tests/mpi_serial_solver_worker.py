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

# Worker for test_mpi_serial_linear_solvers.py -- launched under mpirun, prints one
# PYOOMPH_MPI_RESULT <json> line per rank. NOT a test module itself (the name deliberately does not
# start with "test_", so pytest ignores it).
#
# It solves a nonlinear Poisson problem with a solver that is not MPI-parallel, which oomph-lib still
# routes through its distributed entry point (SuperLUSolver::solve branches on nproc(), not on
# whether the mesh was distributed). The backend therefore takes the gather-to-root path in
# GenericLinearSystemSolver.

import argparse
import json
import os
import resource
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy  # noqa: E402

from pyoomph import *  # noqa: E402,F403
from pyoomph.expressions import *  # noqa: E402,F403
from pyoomph.meshes.simplemeshes import RectangularQuadMesh  # noqa: E402
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc  # noqa: E402
from pyoomph.solvers.generic import GenericLinearSystemSolver, SolverError  # noqa: E402


class NonlinearPoisson(Equations):
    """grad^2 u = 3*exp(u): nonlinear enough to need several Newton steps, so a solution written
    back onto only part of the dof vector has somewhere to show."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(3 * exp(u), v))


class SolverTestProblem(Problem):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        eqs = NonlinearPoisson()
        for bnd in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ bnd
        self += eqs @ "domain"


def _install_failing_solver(base_idname, mode, delay):
    """Register a subclass of the requested backend that misbehaves on rank 0 only.

    The point of the tests using it is that a failure rank 0 alone can see must still end (or be
    retried on) every rank, rather than leaving the others in a collective.
    """
    # Import the module the backend lives in first: the registry is populated by importing, and which
    # solver modules an mpirun has imported by this point depends on the default cascade.
    import importlib
    importlib.import_module("pyoomph.solvers." + ("scipy" if base_idname in ("superlu", "umfpack") else base_idname))
    base = GenericLinearSystemSolver._registered_solvers[base_idname]

    @GenericLinearSystemSolver.register_solver()
    class _RootMisbehaves(base):  # type: ignore[valid-type,misc]
        idname = "_root_misbehaves"

        def solve_serial(self, op_flag, n, nnz, nrhs, values, rowind, colptr, b, ldb, transpose):
            if op_flag == 1 and get_mpi_rank() == 0:
                if delay > 0:
                    time.sleep(delay)
                if mode == "solvererror":
                    raise SolverError("simulated factorisation failure on rank 0")
                if mode == "valueerror":
                    raise ValueError("simulated non-solver failure on rank 0")
            return super().solve_serial(op_flag, n, nnz, nrhs, values, rowind, colptr, b, ldb, transpose)

    return "_root_misbehaves"


def _cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--fail-mode", default="none", choices=["none", "solvererror", "valueerror"])
    ap.add_argument("--root-solve-delay", type=float, default=0.0)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "solver": args.solver}
    try:
        solver = args.solver
        if args.fail_mode != "none" or args.root_solve_delay > 0:
            solver = _install_failing_solver(args.solver, args.fail_mode, args.root_solve_delay)
        with SolverTestProblem(args.N) as problem:
            problem.set_output_directory(args.outdir)
            problem.set_linear_solver(solver)
            problem.initialise()
            problem.solve()
            # A second solve from a perturbed state: several more Newton steps, and it is where a
            # stale factorisation or a half-written dof vector would show up.
            dofs = numpy.array(problem.get_current_dofs()[0], dtype=numpy.float64)
            problem.set_current_dofs(list(dofs * 0.5))
            cpu0, wall0 = _cpu_seconds(), time.time()
            problem.solve()
            payload["solve_cpu"] = round(_cpu_seconds() - cpu0, 3)
            payload["solve_wall"] = round(time.time() - wall0, 3)
            dofs = numpy.array(problem.get_current_dofs()[0], dtype=numpy.float64)
            res = numpy.array(problem.get_residuals(), dtype=numpy.float64)
            payload.update({
                "ndof": int(problem.ndof()),
                "len_dofs": int(len(dofs)),
                "finite": bool(numpy.all(numpy.isfinite(dofs))),
                # The l2 norm rather than an index-weighted checksum: --distribute renumbers the
                # equations, so anything depending on the dof ORDER is not comparable across regimes.
                "l2": float(numpy.linalg.norm(dofs)),
                "maxres": float(numpy.max(numpy.abs(res))),
            })
    except BaseException as e:  # noqa: BLE001
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-1500:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)
    if "error" in payload:
        sys.exit(1)


if __name__ == "__main__":
    main()
