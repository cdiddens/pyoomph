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

# Worker for tests/test_mpi_rank_zero_failures.py -- launched under mpirun.
# A single rank asks for a remesh, which is what a RemeshWhen criterion does when only its part of
# the mesh is distorted. force_remesh() is collective, so the request has to reach every rank.

import argparse
import sys
import traceback

from pyoomph import Problem, DirichletBC
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.generic.mpi import get_mpi_rank


class Pois(Problem):
    def define_problem(self):
        mesh = RectangularQuadMesh(N=6)
        mesh.remesher = Remesher2d(mesh)
        self.mesh_template = mesh
        self.add_mesh(mesh)
        eqs = PoissonEquation(source=1)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


# Everything that runs lives in main(), like the other mpi_*_worker.py files. The suite is
# invoked as `pytest *.py`, so pytest imports every file in this directory during collection --
# and argparse at module scope then exits with "the following arguments are required: --outdir",
# which surfaces as an INTERNALERROR that collects zero tests and aborts the entire run.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ask-on-rank", type=int, default=0,
                        help="the only rank asking for a remesh")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    p = Pois()
    try:
        with p:
            p.set_output_directory(args.outdir)
            p.quiet()
            p.initialise()
            p.solve()
            # Stand-in for the criterion firing only where the distorted elements happen to live.
            if get_mpi_rank() == args.ask_on_rank:
                p._domains_to_remesh.add(p.mesh_template)
            asked = len(p._domains_to_remesh)
            did = p.remesh_if_necessary()
            p.solve()  # collective: a rank that skipped the remesh would already be stuck by now
        print("PYOOMPH_MPI_RESULT rank=%d asked=%d remeshed=%s" % (get_mpi_rank(), asked, did))
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (get_mpi_rank(), type(e).__name__, e))
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(3)


if __name__ == "__main__":
    main()
