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

# Worker for test_mpi_global_meshdata.py: solves a small Poisson problem and reports a
# numbering-independent summary of the globally merged mesh data (see
# pyoomph/meshes/meshdatamerge.py), so a --distribute run can be compared to a serial one.
#
# Everything reported is invariant under node/element numbering, because the merged data is ordered
# rank by rank and a partition therefore has no reason to reproduce the serial order:
#   - counts,
#   - a digest over the SORTED node coordinates and over the sorted per-element coordinate sets.
#     Coordinates come straight from the node positions and are bit-identical across runs, so a
#     digest is exact here and catches a missing or a spurious merge immediately,
#   - field values as permutation-invariant statistics with a tolerance, since those go through the
#     linear solver and agree only to round-off.

import argparse
import hashlib
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class PoissonEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("d0field", "D0")

    def define_residuals(self):
        u, v = var_and_test("u")
        d0, d0t = var_and_test("d0field")
        x, y = var("coordinate_x"), var("coordinate_y")
        self.add_residual(weak(grad(u), grad(v)) - weak(1 + x * y, v) + weak(d0 - u, d0t))
        self.add_local_function("uplus", u + 10 * x)


class IfaceEqs(InterfaceEquations):
    def define_residuals(self):
        self.add_residual(weak(var("u") - 1, testfunction("u")))


class GlobalMeshDataProblem(Problem):
    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.write_states = False  # save_state does not support distributed meshes

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1])
        eqs = PoissonEqs() + DirichletBC(u=0) @ "left" + IfaceEqs() @ "top"
        self += eqs @ "domain"


def _digest(rows):
    """Order-independent digest of an array of coordinate rows."""
    if len(rows) == 0:
        return "empty"
    flat = numpy.asarray(rows, dtype=numpy.float64).round(10) + 0.0  # +0.0 turns -0.0 into 0.0
    order = numpy.lexsort(tuple(flat[:, i] for i in range(flat.shape[1] - 1, -1, -1)))
    return hashlib.sha1(flat[order].tobytes()).hexdigest()


def _summarize(cache, with_segments):
    co = cache.get_coordinates()
    res = {"nnode": int(co.shape[1]), "nelem": int(cache.elem_indices.shape[0]),
           "coord_digest": _digest(co.transpose()),
           "elem_types": sorted(set(int(t) for t in cache.elem_types))}
    # one element type per mesh here, so every column of elem_indices is used (no padding)
    assert len(res["elem_types"]) <= 1, "the worker assumes a uniform element type"
    fingerprints = []
    for row in cache.elem_indices:
        pts = sorted(tuple(co[:, int(i)]) for i in row)
        fingerprints.append(numpy.array(pts).flatten())
    res["elem_digest"] = _digest(fingerprints)
    for field in ("u", "uplus"):
        values = cache.get_data(field)
        if values is None:
            continue
        res[field + "_sum"] = float(numpy.sum(values))
        res[field + "_sqsum"] = float(numpy.sum(numpy.asarray(values) ** 2))
        res[field + "_max"] = float(numpy.amax(values))
    if cache.D0_data.size:
        res["d0_sum"] = float(numpy.sum(cache.D0_data))
    if with_segments:
        segments, _ = cache.get_interface_line_segments()
        res["segment_lengths"] = sorted(len(s) for s in segments)
    return res


def run_case(N=8, discontinuous=False, outdir="_global_meshdata", twice=False):
    """Returns the summary of the merged data (on rank 0) plus what the other ranks saw."""
    result = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "meshes": {}}
    with GlobalMeshDataProblem(N=N) as problem:
        problem.set_output_directory(outdir)
        problem.solve()
        for name in ("domain", "domain/top"):
            data = problem.get_cached_mesh_data(name, tesselate_tri=True, discontinuous=discontinuous,
                                                global_mesh=True)
            if twice:
                # The second request must be served from the cache on rank 0 without leaving the other
                # ranks in a gather that nobody joins
                again = problem.get_cached_mesh_data(name, tesselate_tri=True, discontinuous=discontinuous,
                                                     global_mesh=True)
                assert again is data, "the merged data was not cached"
            result["meshes"][name] = None if data is None else _summarize(data, with_segments=not discontinuous
                                                                         and name == "domain/top")
            result["distributed"] = bool(problem.get_mesh(name).is_mesh_distributed())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="_global_meshdata")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--discontinuous", action="store_true")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--distribute", action="store_true")  # consumed by pyoomph itself
    args, _ = parser.parse_known_args()
    try:
        out = run_case(N=args.size, discontinuous=args.discontinuous, outdir=args.outdir, twice=args.twice)
    except BaseException as e:
        out = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "error": str(e),
               "traceback": traceback.format_exc()}
    print("PYOOMPH_MPI_RESULT " + json.dumps(out), flush=True)
