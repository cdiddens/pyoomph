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

# Worker for test_mpi_custom_assembler_handoff.py. Launched under mpirun, prints one
# PYOOMPH_MPI_RESULT <json> line per rank. NOT a test module itself (the name deliberately does not
# start with "test_", so pytest ignores it).
#
# What is under test is the use_custom_residual_jacobian HANDOFF in src/problem.cpp -- the three
# branches of get_residuals / get_derivative_wrt_global_parameter / get_jacobian that take a
# Python-supplied system and copy it into oomph's DoubleVector / CRDoubleMatrix. A Python assembler
# returns the WHOLE system indexed 0..ndof-1, while the caller has already built both on the LINEAR
# SOLVER's distribution, which under mpirun is a row block -- and, without --distribute, not even
# the same partition of those rows as the dof distribution (oomph's SuperLUSolver::solve(Problem*)
# imposes a uniform split). Copying ndof entries into a vector that holds nrow_local() of them runs
# off the end of the local storage. See dev_docs/mpi_augmented_systems.md B2.
#
# This deliberately does NOT go through Problem.set_custom_assembler, which still refuses nproc>1
# (_require_single_rank, for the rest of that pipeline: the multi-assembly below it throws under
# MPI). The guard sits on set_custom_assembler, not on use_custom_residual_jacobian, and
# get_custom_residuals_jacobian is documented as overridable directly -- which is the whole reason
# the handoff can be fixed and tested on its own, ahead of the pipeline it serves.
#
# The custom path here returns exactly what the ORDINARY assembly would, so the two are
# numerically interchangeable and any difference in the converged answer is the handoff's. Three
# things are reported:
#
#   - "usqr": the integral of u^2 over the mesh after a Newton solve driven through the custom
#     handoff. Partition-independent (evaluate_integral_function skips halo elements and
#     MPI_Allreduce-sums), so it is comparable across serial, replicated and distributed runs,
#     which a dof vector is not -- distribute() renumbers.
#   - "usqr_ordinary": the same quantity with the handoff switched off, from the same process. This
#     is the strong assertion: it compares the two paths under identical everything else.
#   - "fresh_matrix_ok": whether Problem.assemble_jacobian() -- which hands in a FRESH
#     CRDoubleMatrix rather than one oomph has already distributed -- survives while the custom
#     path is active. That is the case that used to write a full row_start array into a zero-row
#     matrix.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.solvers.generic import GenericLinearSystemSolver


def _install_row_block_probe(base_idname):
    """A backend that records the row block the solver hands the assembly, then defers to its base.

    Without this the test could pass for the wrong reason. The DOF distribution is not distributed
    on a replicated mpirun, so a worker that only looked at _get_dof_distribution_info() would see
    nrow_local == ndof and conclude there was nothing to slice. The distribution that matters is the
    LINEAR SOLVER's: oomph's SuperLUSolver::solve(Problem*) builds
    LinearAlgebraDistribution(comm, ndof, true), a uniform split, as soon as nproc > 1. That is the
    row block B2 copied ndof entries into.
    """
    import importlib
    # Same module remap GenericLinearSystemSolver.factory_solver() uses: the registry is populated by
    # importing, and the module a backend lives in is not always named after it.
    _MODULE_OF = {"petsc_mumps": "petsc", "superlu": "scipy", "umfpack": "scipy"}
    importlib.import_module("pyoomph.solvers." + _MODULE_OF.get(base_idname, base_idname))
    base = GenericLinearSystemSolver._registered_solvers[base_idname]

    @GenericLinearSystemSolver.register_solver()
    class _RowBlockProbe(base):  # type: ignore[valid-type,misc]
        idname = "_row_block_probe"
        seen = []

        def solve_distributed(self, op_flag, allow_permutations, n, nnz_local, nrow_local, first_row,
                              values, col_index, row_start, b, nprow, npcol, doc, data, info):
            if op_flag == 1:
                _RowBlockProbe.seen.append((int(n), int(nrow_local), int(first_row)))
            return super().solve_distributed(op_flag, allow_permutations, n, nnz_local, nrow_local,
                                             first_row, values, col_index, row_start, b, nprow,
                                             npcol, doc, data, info)

    return _RowBlockProbe


class NonlinearPoisson(Equations):
    """laplace(u) = 3*exp(u). Nonlinear, so Newton takes several steps and a corrupted residual or
    Jacobian has somewhere to show rather than being hidden by a one-step solve."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(3 * exp(u), v))


class HandoffProblem(Problem):
    """Solves the above either normally or by handing the very same system back through the
    use_custom_residual_jacobian hook.

    ``custom`` is a plain flag rather than an installed CustomAssemblyBase on purpose: no custom
    assembler exists here, so nothing in this worker touches the multi-assembly or the augmented
    dof machinery. The only thing exercised is the C++ copy-in.
    """

    def __init__(self, N=6):
        super().__init__()
        self.N = N
        self.custom = False
        self.n_handoffs = 0

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        eqs = NonlinearPoisson()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"

    def set_custom_handoff(self, on):
        self.custom = on
        self.use_custom_residual_jacobian = on

    def get_custom_residuals_jacobian(self, info):
        # Assemble the ordinary way and hand the result straight back, so the custom path computes
        # the same system the normal one would. global_csr=True gathers the Jacobian onto every rank
        # and the residual always comes back globally indexed, which is the contract a Python
        # assembler is written against.
        self.use_custom_residual_jacobian = False
        try:
            if info.require_jacobian():
                R, J = self.assemble_jacobian(with_residual=True, global_csr=True)
            else:
                name = info.get_parameter_name()
                if name != "":
                    raise RuntimeError("this worker registers no global parameter")
                R = numpy.array(self.get_residuals(), dtype=numpy.float64)
                J = None
        finally:
            self.use_custom_residual_jacobian = True
        self.n_handoffs += 1
        R = numpy.ascontiguousarray(numpy.asarray(R, dtype=numpy.float64))
        info.set_custom_residuals(R)
        if J is not None:
            J = J.tocsr()
            info.set_custom_jacobian(
                numpy.ascontiguousarray(J.data.astype(numpy.float64)),
                numpy.ascontiguousarray(J.indices.astype(numpy.int32)),
                numpy.ascontiguousarray(J.indptr.astype(numpy.int32)),
            )


def _usqr(p):
    return float(p.get_mesh("domain").evaluate_all_observables()["usqr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--distribute", action="store_true")
    args, _ = ap.parse_known_args()

    payload: dict = {"rank": get_mpi_rank(), "nproc": max(get_mpi_nproc(), 1)}
    try:
        probe = _install_row_block_probe("petsc_mumps" if max(get_mpi_nproc(), 1) > 1 else "superlu")
        with HandoffProblem(args.N) as p:
            p.set_output_directory(args.outdir)
            p.set_linear_solver("_row_block_probe")
            p.initialise()

            # (a) the ordinary path, as the reference
            p.solve()
            payload["usqr_ordinary"] = _usqr(p)
            payload["ndof"] = int(p.ndof())
            _n, nrow_local, first_row, distributed = p._get_dof_distribution_info()
            payload["dof_nrow_local"] = int(nrow_local)
            payload["dof_first_row"] = int(first_row)
            payload["dof_distributed"] = bool(distributed)

            # (b) the same solve again, driven through the custom handoff, from the same state
            p.set_current_dofs(numpy.zeros(p.ndof()))
            p.set_custom_handoff(True)
            p.solve()
            payload["usqr"] = _usqr(p)
            payload["n_handoffs"] = int(p.n_handoffs)

            # (c) a FRESH CRDoubleMatrix while the custom path is active
            try:
                R, J = p.assemble_jacobian(with_residual=True, global_csr=True)
                payload["fresh_matrix_ok"] = True
                payload["fresh_nnz"] = int(J.nnz)
                payload["fresh_shape"] = [int(J.shape[0]), int(J.shape[1])]
                payload["fresh_res_len"] = int(len(R))
            except Exception as e:
                payload["fresh_matrix_ok"] = False
                payload["fresh_error"] = repr(e)

            # What the solver actually asked for. Under mpirun these must be real row blocks --
            # nrow_local < n -- or the run proved nothing about the slicing.
            payload["solver_row_blocks"] = sorted(set(probe.seen))

            p.set_custom_handoff(False)
        payload["ok"] = True
    except Exception as e:
        payload["ok"] = False
        payload["error"] = repr(e)
        payload["traceback"] = traceback.format_exc()
    # Written to the REAL stdout, not through sys.stdout. Importing pyoomph installs the MPI console
    # (pyoomph/generic/logging.py), whose default "condensed" mode mutes stdout on every rank but 0 --
    # so a plain print() here would report from one rank and the between-rank assertions would pass
    # by being vacuous. sys.__stdout__ is the unwrapped stream, which is what makes this a per-rank
    # result rather than rank 0's opinion.
    out = sys.__stdout__ or sys.stdout
    out.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
