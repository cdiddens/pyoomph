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

# Worker for test_mpi_multiassembly.py. Prints one PYOOMPH_MPI_RESULT line per rank. NOT a test
# module itself.
#
# The base-problem multi-assembly under MPI -- B1 of mpi_augmented_systems.md. It used to throw
# ("This likely does not work in parallel" / "...in distributed parallel") because the routine keeps
# this rank's element slice and nothing in it reduced over the ranks. It now hands the work to
# oomph's own parallel_sparse_assemble(), which does the off-processor row exchange, and returns this
# rank's ROW BLOCK of the base equations with global column indices.
#
# Reached through MultiAssembleRequest directly, with a dof augmentation installed only because the
# base-problem routine insists on one (it exists to assemble the base block OF an augmented system).
# No custom assembler is installed, so Problem.set_custom_assembler's nproc>1 refusal is not in the
# way: what is under test is the assembly, not the pipeline above it.
#
# Two kinds of comparison are possible and both are reported:
#
#   - REPLICATED (plain mpirun): the dof numbering is identical to serial, so the gathered row blocks
#     can be compared to the serial CSR entry by entry. That is the strong check, and the one that
#     would have caught a missing reduction immediately -- without one, each rank holds only the
#     contributions of its own element slice.
#   - --distribute: distribute() renumbers, so nothing indexed by dof is comparable. The quantities
#     reported instead are invariant under a permutation of the unknowns: the Frobenius norm of each
#     matrix, the 2-norm of each vector, and the total nnz.

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.generic.bifurcation_tools import MultiAssembleRequest


class Poisson(Equations):
    """laplace(u) = 3*exp(u) + lam*u, with lam a global parameter so dR/dp and dJ/dp are non-trivial."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(3 * exp(u) + self.lam * u, v))


class MAProblem(Problem):
    def __init__(self, N=6):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        self.lam = self.define_global_parameter(lam=0.5)
        eqs = Poisson(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"
        # dJdU asks for a Hessian-vector product, which needs the analytic Hessian generated at JIT
        # time; without it the assembly refuses rather than silently differencing.
        self.setup_for_stability_analysis(analytic_hessian=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--distribute", action="store_true")
    args, _ = ap.parse_known_args()

    payload: dict = {"rank": get_mpi_rank(), "nproc": max(get_mpi_nproc(), 1)}
    try:
        with MAProblem(args.N) as p:
            p.set_output_directory(args.outdir)
            p.initialise()
            p.solve()

            n_global = p.ndof()
            # The augmentation exists only to satisfy the base-problem routine's "you must have
            # augmented dofs" precondition; its own rows are never assembled here.
            aug = p._create_dof_augmentation()
            aug.add_vector(numpy.zeros(n_global))
            aug.add_scalar(0.0)
            p._add_augmented_dofs(aug)

            V = numpy.linspace(0.25, 1.75, n_global)  # a Hessian contraction direction
            req = MultiAssembleRequest(p)
            res = req.R().J().dRdp("lam").dJdp("lam").dJdU(V).assemble()
            R, J, dRdp, dJdp, HV = res

            payload["n"] = int(req.n)
            payload["nrow_local"] = int(req.nrow_local)
            payload["first_row"] = int(req.first_row)
            payload["shapes"] = [[int(M.shape[0]), int(M.shape[1])] for M in (J, dJdp, HV)]
            payload["vec_lengths"] = [int(len(R)), int(len(dRdp))]

            # Permutation-invariant summaries, for the --distribute comparison.
            def fro(M):
                return float(numpy.sqrt(float((M.data ** 2).sum())))

            payload["fro"] = {"J": fro(J), "dJdp": fro(dJdp), "HV": fro(HV)}
            payload["nnz"] = {"J": int(J.nnz), "dJdp": int(dJdp.nnz), "HV": int(HV.nnz)}
            payload["l2"] = {"R": float(numpy.linalg.norm(R)), "dRdp": float(numpy.linalg.norm(dRdp))}

            payload["J_sorted"] = bool(J.has_sorted_indices)

            # The raw row block goes to a FILE, one per rank, not into the result line. Every rank
            # writes the shared stdout pipe, and a line of a few thousand characters does not arrive
            # atomically -- two ranks' writes interleave and the JSON no longer parses. Measured at
            # np=3 before this was split out.
            numpy.savez(os.path.join(args.outdir, "block_rank%d.npz" % get_mpi_rank()),
                        indptr=J.indptr, indices=J.indices, data=J.data,
                        R=numpy.asarray(R, dtype=numpy.float64),
                        nrow_local=numpy.array([J.shape[0]]), first_row=numpy.array([req.first_row]),
                        n=numpy.array([req.n]))

            p._reset_augmented_dof_vector_to_nonaugmented()
        payload["ok"] = True
    except Exception as e:
        payload["ok"] = False
        payload["error"] = repr(e)
        payload["traceback"] = traceback.format_exc()[-2500:]
    out = sys.__stdout__ or sys.stdout
    out.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
