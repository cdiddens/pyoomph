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

# Worker launched by tests/test_mpi_curved_boundaries.py as
#     mpirun -n N python mpi_curved_worker.py --kinds quad,tri,... --level L --outdir D --distribute
# Not a test module itself (the name deliberately does not start with "test_").
#
# Each rank builds each curved mesh, distributes, refines and solves, then prints
#     PYOOMPH_MPI_RESULT <json>
# with its rank's worst deviation from the exact circle/sphere. The geometries are imported from
# test_curved_boundaries so that the serial and distributed suites cannot drift apart.

import argparse
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph import *  # noqa: E402
from pyoomph.equations.poisson import PoissonEquation  # noqa: E402
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc  # noqa: E402
from pyoomph.meshes.simplemeshes import CircularMesh, SphericalOctantMesh  # noqa: E402
from test_curved_boundaries import _GmshBallTemplate, _R, _TetBallTemplate, _TriDiskTemplate  # noqa: E402


def _build(kind):
    """Return (template, boundary name, spatial dimension) for a case id."""
    if kind == "quad":
        return CircularMesh(radius=_R, segments="all"), "circumference", 2
    if kind == "tri":
        return _TriDiskTemplate(), "circumference", 2
    if kind == "sphere":
        return SphericalOctantMesh(radius=_R), "shell", 3
    if kind == "tetball":
        return _TetBallTemplate(), "shell", 3
    if kind == "gmshball":
        return _GmshBallTemplate(), "shell", 3
    raise SystemExit("unknown case " + repr(kind))


class _CurvedProblem(Problem):
    def __init__(self, kind, level):
        super().__init__()
        self._kind, self._level = kind, level

    def define_problem(self):
        template, self._bname, self._ndim = _build(self._kind)
        self += template
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ self._bname
        eqs += RefineToLevel(self._level)
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _solve(kind, level, outdir, load_balance=False):
    # "with" is load-bearing here, not tidiness: a Problem left un-released keeps its distributed state
    # alive, and the NEXT Problem's distribute() in the same process then dies. Measured while writing
    # this -- whichever case ran second failed, regardless of which it was. box_cases.solve_case uses
    # the same pattern for the same reason.
    with _CurvedProblem(kind, level) as problem:
        problem.set_output_directory(os.path.join(outdir, "out_" + kind + ("_lb" if load_balance else "")))
        problem.max_refinement_level = level + 1
        # The supported way to load-balance: pyoomph does it inside the initial adaption. (Calling
        # Problem.load_balance() by hand afterwards dies in generate_interface_elements with "bulkmesh
        # was not set" -- identically with and without curved entities, so that is a separate,
        # pre-existing defect and not what this is testing. See dev_docs 22.2.)
        problem.call_load_balance_in_initial_adaption = load_balance
        problem.initialise()
        problem.solve()
        mesh = problem.get_mesh("domain")
        bidx = mesh.get_boundary_index(problem._bname)
        worst, count = 0.0, 0
        for node in mesh.nodes():
            if node.is_on_boundary(bidx):
                r = math.sqrt(sum(node.x(i) ** 2 for i in range(problem._ndim)))
                worst = max(worst, abs(r - _R))
                count += 1
        return {"nelem": mesh.nelement(), "bnodes": count, "worst": worst}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--load-balance", action="store_true")
    args, _ = parser.parse_known_args()
    for kind in args.kinds.split(","):
        payload = {"kind": kind, "rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
        try:
            payload.update(_solve(kind, args.level, args.outdir, args.load_balance))
        except Exception as e:  # a failure in one case still yields a diagnosable line
            payload["error"] = repr(e)
            payload["traceback"] = traceback.format_exc()
        print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
