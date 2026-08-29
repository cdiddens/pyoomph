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

# Worker launched by tests/test_mpi_boundary_membership.py as
#     mpirun -n N python mpi_boundary_membership_worker.py --kinds hex,tet --level L --outdir D --distribute
# Not a test module itself (the name deliberately does not start with "test_").
#
# Each rank builds the slab of tests/slab_mesh.py -- one element thick, top and bottom sharing the
# boundary name "wall", which is the configuration nodal boundary membership goes wrong on --
# distributes it, refines, and then prints PYOOMPH_MPI_RESULT <json> carrying, per boundary:
#   * the rounded positions of the nodes MARKED as being on it, and
#   * the rounded positions of the nodes the INTERFACE MESH on it actually owns.
# Neither list is conclusive on its own rank: a node reached only through a halo element may sit on a
# facet that lives on somebody else. The test unions them across ranks and compares there, and
# separately checks that no two ranks disagree about a node they both hold.

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph import *  # noqa: E402
from pyoomph.equations.poisson import PoissonEquation  # noqa: E402
from pyoomph.generic.mpi import get_mpi_rank  # noqa: E402
from slab_mesh import SlabTemplate  # noqa: E402

_BOUNDARIES = ["wall", "side"]


class _SlabProblem(Problem):
    def __init__(self, family, level, ncell):
        super().__init__()
        self._family, self._level, self._ncell = family, level, ncell

    def define_problem(self):
        self += SlabTemplate(N=self._ncell, family=self._family)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ _BOUNDARIES
        eqs += RefineToLevel(self._level)
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _key(nd):
    return "%.11g,%.11g,%.11g" % (nd.x(0), nd.x(1), nd.x(2))


def _run(family, level, outdir):
    # "with" is load-bearing: a Problem left un-released keeps its distributed state alive and the next
    # distribute() in the same process then dies. Same reasoning as mpi_curved_worker.
    with _SlabProblem(family, level, ncell=2) as problem:
        problem.set_output_directory(os.path.join(outdir, "out_" + family))
        problem.max_refinement_level = level + 2
        problem.initialise()
        problem.solve()
        # The refinement above happens BEFORE the mesh is distributed, so every element is still local,
        # every node decidable, and the repair finishes without needing to tell anyone. This one is the
        # point of the exercise: it creates over-marked nodes on an already-distributed mesh, where a
        # rank reaches some of them only through a halo element and has to be told what their owner
        # decided. Measured on the 4-rank hex slab: 55 of the 180 nodes a rank holds are undecidable
        # there. Without the cross-rank push the checks below fail.
        problem.refine_uniformly()
        mesh = problem.get_mesh("domain")
        marked, on_facets = {}, {}
        for bname in _BOUNDARIES:
            b = mesh.get_boundary_index(bname)
            marked[bname] = sorted(_key(nd) for nd in mesh.nodes() if nd.is_on_boundary(b))
            on_facets[bname] = sorted(_key(nd) for nd in problem.get_mesh("domain/" + bname).nodes())
        spurious, missing = mesh.check_boundary_node_membership()
        return {"nelem": mesh.nelement(), "marked": marked, "on_facets": on_facets,
                "selfcheck": [spurious, missing]}


def main():
    # parse_known_args: --distribute is pyoomph's own flag and is read straight off sys.argv by the
    # Problem, so it must survive here rather than being consumed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()
    for kind in args.kinds.split(","):
        payload = {"kind": kind, "rank": get_mpi_rank()}
        try:
            payload.update(_run(kind, args.level, args.outdir))
        except Exception as e:  # a failure in one case still yields a diagnosable result
            payload["error"] = repr(e)
            payload["traceback"] = traceback.format_exc()
        # Via a file rather than stdout: these payloads are tens of kB of node positions, and mpirun
        # truncates a long line (measured: cut at 4096 characters, mid-token).
        path = os.path.join(args.outdir, "result_%s_%d.json" % (kind, payload["rank"]))
        with open(path, "w") as f:
            json.dump(payload, f)
        print("PYOOMPH_MPI_RESULT " + path, flush=True)


if __name__ == "__main__":
    main()
