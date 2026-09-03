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

# Worker launched by tests/test_mpi_dof_description.py as
#     mpirun -n N python mpi_dof_description_worker.py --outdir D [--distribute]
# Not a test module itself (the name deliberately does not start with "test_").
#
# Prints PYOOMPH_MPI_RESULT <path to json> with this rank's Problem.get_dof_description(): the type
# names, the per-dof type index and how many elements of each mesh of the tree this rank holds. The
# problem is deliberately one no rank holds all of - a moving mesh, a global (ODE) dof, an interface
# Lagrange multiplier on the top boundary and a second one on the single point where "top" meets
# "left", which by construction lives on exactly one rank while every other rank carries that mesh
# empty. That is the configuration the walk used to raise on.

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph import *  # noqa: E402
from pyoomph.equations.ALE import LaplaceSmoothedMesh  # noqa: E402
from pyoomph.equations.generic import EnforcedDirichlet, IntegralConstraint  # noqa: E402
from pyoomph.equations.poisson import PoissonEquation  # noqa: E402
from pyoomph.generic.mpi import get_mpi_rank  # noqa: E402
from pyoomph.meshes.mesh import ODEStorageMesh  # noqa: E402


class _DofDescriptionProblem(Problem):
    def define_problem(self):
        # A long thin strip, so that a partition into vertical slabs leaves the "left" and
        # "right" boundaries - and the point where "top" meets "left" - entirely on one rank
        # each, which is the case get_dof_description() used to raise on.
        self += RectangularQuadMesh(N=[24, 2], size=[12, 1])
        eqs = PoissonEquation(source=1, space="C2")
        eqs += LaplaceSmoothedMesh()
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ ["left", "right", "top", "bottom"]
        eqs += DirichletBC(u=0) @ "bottom"
        eqs += EnforcedDirichlet(u=1) @ "top"
        eqs += EnforcedDirichlet(u=1) @ "top/left"  # a single point: at most one rank holds it
        eqs += IntegralConstraint(u=0.25)
        self += eqs @ "domain"


def _element_counts(problem):
    """How many elements of each mesh of the tree this rank holds, keyed by the mesh's full name."""
    counts = {}

    def walk(m):
        counts[m.get_full_name()] = m.nelement()
        if not isinstance(m, ODEStorageMesh):
            for im in m._interfacemeshes.values():
                walk(im)

    for m in problem._meshdict.values():
        walk(m)
    return counts


def _run(outdir):
    rank = get_mpi_rank()
    payload = {"rank": rank}
    try:
        with _DofDescriptionProblem() as problem:
            problem.set_output_directory(os.path.join(outdir, "out"))
            problem.initialise()
            payload["nelement"] = _element_counts(problem)
            types, names = problem.get_dof_description()
            payload["names"] = list(names)
            payload["types"] = [int(t) for t in types]
            payload["ndof"] = int(problem.ndof())
            payload["distributed"] = bool(problem.is_distributed())
    except Exception as e:
        payload["error"] = str(e)
        payload["traceback"] = traceback.format_exc()
    # Through a file, not the line itself: the payload is one entry per dof and mpirun truncates a
    # long line at 4096 characters.
    path = os.path.join(outdir, "dofdescr_rank%d.json" % rank)
    with open(path, "w") as f:
        json.dump(payload, f)
    print("PYOOMPH_MPI_RESULT " + path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()  # --distribute is consumed by Problem.initialise()
    os.makedirs(args.outdir, exist_ok=True)
    _run(args.outdir)


# Guarded: the nightly runs `pytest *.py` from inside tests/, which hands pytest every helper
# script in this directory as an explicit argument. pytest force-imports those (neither
# collect_ignore_glob nor pytest_ignore_collect applies to a file named on the command line), so an
# unguarded main() ran argparse at collection time and killed the whole session with SystemExit: 2.
if __name__ == "__main__":
    main()
