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

# Worker for test_mpi_bordered_solve.py: a tracker-shaped bordered system assembled and SOLVED under
# MPI, with no tracker involved.
#
# This is the step between the backend's own tests and the first real tracker. The unit tests build
# bordered systems over fabricated splits but never solve one; the MPI backend tests solve nothing at
# all, because solve() needs a Problem's linear solver. What is exercised here is the whole chain the
# way a tracker will use it:
#
#   a dof augmentation installed  ->  the base layout and the equation table come from it
#   the multi-assembly            ->  J and the Hessian-vector product as local row blocks (B1)
#   backend.block()/stack()       ->  the 2N+1 bordered system on the augmented layout
#   backend.solve()               ->  through solve_python_built_distributed, PETSc MPIAIJ + MUMPS
#
# The system is deliberately NOT a real fold: the scalar diagonal carries 1 instead of 0 so that it
# is nonsingular wherever the base state happens to be, since the point here is the plumbing and not
# the bifurcation. (A real tracker leaves that entry absent, which is what
# dev_docs/mpi_augmented_systems.md section 5 discusses as a MUMPS null-pivot hazard.)
#
# The solution is reported in NAIVE order -- un-permuted through the equation table and gathered --
# so it is directly comparable with the serial run, which no dof-indexed quantity otherwise is.

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
from pyoomph.generic.distributed_la import DistVector, RowLayout, get_la_backend


class Poisson(Equations):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(3 * exp(u) + self.lam * u, v))


class BorderedProblem(Problem):
    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        self.lam = self.define_global_parameter(lam=0.5)
        eqs = Poisson(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"
        self.setup_for_stability_analysis(analytic_hessian=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--distribute", action="store_true")
    args, _ = ap.parse_known_args()

    rank, nproc = get_mpi_rank(), max(get_mpi_nproc(), 1)
    payload: dict = {"rank": rank, "nproc": nproc}
    try:
        with BorderedProblem(args.N) as p:
            p.set_output_directory(args.outdir)
            p.initialise()
            p.solve()

            nbase_global = p.ndof()
            # NOT linspace over the dof index: distribute() renumbers the dofs, so a dof-indexed
            # vector means a different thing in each regime and nothing derived from it would be
            # comparable. This is the same trap the other workers document -- only
            # numbering-independent quantities survive a renumbering. A constant is numbering
            # independent and still exercises every block of the bordered system.
            V_global = numpy.full(nbase_global, 0.75)

            aug = p._create_dof_augmentation()
            aug.add_vector(V_global)
            aug.add_parameter("lam")
            p._add_augmented_dofs(aug)

            base = RowLayout.base(p)
            augmented = RowLayout.augmented(p)
            table = p._get_augmented_eqn_table()
            B = get_la_backend(p)
            payload["backend"] = B.name
            payload["base"] = [base.n, base.first_row, base.nrow_local, base.distributed]
            payload["aug"] = [augmented.n, augmented.first_row, augmented.nrow_local, augmented.distributed]

            # B1: the base blocks, as this rank's rows with global column indices.
            V = DistVector.from_global(V_global, base)
            req = MultiAssembleRequest(p)
            R, J, dRdp, HV = req.R().J().dRdp("lam").dJdU(V.to_global()).assemble()
            assert req.nrow_local == base.nrow_local, (req.nrow_local, base.nrow_local)
            assert req.first_row == base.first_row, (req.first_row, base.first_row)

            Jm = B.matrix(J, base, base.n)
            HVm = B.matrix(HV, base, base.n)
            vR = B.vector(R, base)
            vdRdp = B.vector(dRdp, base)

            groups = [False, False, True]  # [u | V | lam]
            # A border row needs the whole vector, by contract; V0 is replicated data anyway.
            V0 = DistVector(V_global, RowLayout.serial(base.n))

            A = B.block([[Jm,   None,       B.col(vdRdp)],
                         [HVm,  Jm,         B.col(Jm.matvec(V))],
                         [None, B.row(V0),  1.0]],
                        groups, base, augmented, table)
            rhs = B.stack([vR, Jm.matvec(V), 0.25], groups, base, augmented, table)
            payload["A_nnz_local"] = int(A.nnz)
            payload["A_fro"] = A.frobenius_norm()
            payload["rhs_norm"] = rhs.norm()

            x = B.solve(A, rhs)
            payload["x_norm"] = x.norm()
            # The residual of the solve, which is the only check that does not depend on the ordering.
            payload["solve_residual"] = (A.matvec(x) - rhs).norm() / max(rhs.norm(), 1e-300)

            # Un-permute to naive order and gather, so the answer is comparable with serial.
            naive = numpy.zeros(augmented.n)
            real_of = (lambda i: i) if len(table) == 0 else (lambda i: int(table[i]))
            local = numpy.zeros(augmented.n)
            for i in range(augmented.n):
                r = real_of(i)
                if augmented.owns(r):
                    local[i] = x.local[r - augmented.first_row]
            if nproc > 1 and augmented.distributed:
                from pyoomph.generic.mpi import get_mpi_sum
                # Each naive index is owned by exactly one rank, so the sum IS the gathered vector.
                naive = get_mpi_sum(local)
            else:
                # Replicated (or serial): every rank already holds the whole solution, and summing
                # would multiply it by nproc.
                naive = local
            numpy.savez(os.path.join(args.outdir, "solution_rank%d.npz" % rank),
                        naive=numpy.asarray(naive, dtype=numpy.float64))
            payload["naive_norm"] = float(numpy.linalg.norm(naive))

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
