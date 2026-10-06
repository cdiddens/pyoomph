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

# Worker for test_mpi_distributed_la.py. The parts of pyoomph/generic/distributed_la.py that
# tests/test_distributed_la.py cannot reach: the collectives, and the PETSc matrix.
#
# The unit tests fabricate a multi-rank split in one process, which covers all the layout and
# translation arithmetic and none of the communication. Here the layout is real, so the reductions,
# to_global(), global_value(), the PETSc matrix-vector product and the PETSc transpose are the things
# under test. The reference is always the same global system, built identically on every rank from a
# fixed seed, so no rank has to be told the answer.
#
# No Problem is involved: this module only needs a communicator. The solve is therefore not covered
# here -- it goes through a Problem's linear solver -- and is left to the tracker tests.

import argparse
import json
import sys
import traceback

import numpy
import scipy.sparse

from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.generic.distributed_la import DistVector, RowLayout, get_la_backend


def _uniform_cuts(n, nproc):
    """The same contiguous split on every rank, so the layouts agree by construction."""
    base, rem = divmod(n, nproc)
    cuts, first = [], 0
    for r in range(nproc):
        nloc = base + (1 if r < rem else 0)
        cuts.append((first, nloc))
        first += nloc
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args, _ = ap.parse_known_args()

    rank, nproc = get_mpi_rank(), max(get_mpi_nproc(), 1)
    payload: dict = {"rank": rank, "nproc": nproc}
    try:
        n = args.n
        rng = numpy.random.default_rng(4242)  # identical on every rank
        A = scipy.sparse.random(n, n, density=0.3, format="csr", random_state=777).tocsr()
        A = (A + scipy.sparse.eye(n, format="csr") * 5.0).tocsr()  # well conditioned, for the transpose
        x = rng.normal(size=n)
        y = rng.normal(size=n)

        first_row, nrow_local = _uniform_cuts(n, nproc)[rank]
        layout = RowLayout.block(n, first_row, nrow_local)
        layout.validate()  # collective: the blocks must tile [0,n)
        payload["layout"] = [layout.n, layout.first_row, layout.nrow_local, layout.distributed]

        B = get_la_backend(None)
        payload["backend"] = B.name

        vx = DistVector.from_global(x, layout)
        vy = DistVector.from_global(y, layout)

        # reductions, against the global reference
        payload["dot"] = vx.dot(vy)
        payload["dot_ref"] = float(numpy.dot(x, y))
        payload["norm"] = vx.norm()
        payload["norm_ref"] = float(numpy.linalg.norm(x))
        payload["max_abs"] = vx.max_abs()
        payload["max_abs_ref"] = float(numpy.max(numpy.abs(x)))

        # the named escapes
        payload["to_global_matches"] = bool(numpy.allclose(vx.to_global(), x))
        probe = [0, n // 3, n - 1]
        payload["global_value"] = [vx.global_value(g) for g in probe]
        payload["global_value_ref"] = [float(x[g]) for g in probe]

        # the matrix: this rank's row block, with global column indices
        M = B.matrix(A[layout.local_slice, :].tocsr(), layout, n)
        payload["matrix_class"] = type(M).__name__

        got = M.matvec(vx)
        payload["matvec_max_err"] = float(numpy.max(numpy.abs(got.local - (A @ x)[layout.local_slice])))
        payload["frobenius"] = M.frobenius_norm()
        payload["frobenius_ref"] = float(numpy.sqrt((A.data ** 2).sum()))

        # the transpose, checked by <A^T x, y> == <x, A y> rather than by comparing blocks, because a
        # transposed row block need not land on the same split
        MT = M.transpose()
        payload["transpose_layout"] = [MT.layout.n, MT.layout.first_row, MT.layout.nrow_local]
        vx_T = DistVector.from_global(x, MT.layout)
        ATx = MT.matvec(vx_T)
        payload["ATx_dot_y"] = ATx.dot(DistVector.from_global(y, MT.layout))
        payload["x_dot_Ay"] = vx.dot(M.matvec(vy))

        # transpose_onto, which is the one a bordered system can use: it must land on THIS layout,
        # not on whichever split a transpose naturally produces, so unlike M.transpose() above it CAN
        # be compared block by block against the global transpose. That is the whole point of it --
        # landing on the wrong partition gives a plausible matrix rather than an error.
        AT_local = B.transpose_onto(A[layout.local_slice, :].tocsr(), layout)
        ref = A.transpose().tocsr()[layout.local_slice, :].tocsr()
        payload["transpose_onto_shape"] = list(AT_local.shape)
        payload["transpose_onto_shape_ref"] = list(ref.shape)
        payload["transpose_onto_max_err"] = float(abs(AT_local - ref).max()) if ref.nnz else 0.0
        payload["transpose_onto_nnz"] = int(AT_local.nnz)
        payload["transpose_onto_nnz_ref"] = int(ref.nnz)
        payload["transpose_onto_sorted"] = bool(AT_local.has_sorted_indices)

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
