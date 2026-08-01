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

# Worker for tests/test_mpi_rank_zero_failures.py -- launched under mpirun.
# Only rank 0 writes the gmsh files; --fail-write makes that write raise, which is the situation
# the other ranks used to wait out forever.

import argparse
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True)
parser.add_argument("--fail-write", choices=["none", "geo", "msh"], default="none")
args, rest = parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import gmsh  # type:ignore

from pyoomph import Problem, DirichletBC
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_any

if args.fail_write != "none":
    _suffix = ".geo_unrolled" if args.fail_write == "geo" else ".msh"
    _orig_write = gmsh.write

    def _write(path, *a, **k):
        if path.endswith(_suffix) and get_mpi_rank() == 0:
            raise RuntimeError("simulated failure while writing " + path)
        return _orig_write(path, *a, **k)

    gmsh.write = _write


class Disc(GmshTemplate):
    def define_geometry(self):
        self.default_resolution = 0.1
        self.mesh_mode = "tris"
        c = self.point(0, 0)
        e = self.point(1, 0)
        n = self.point(0, 1)
        w = self.point(-1, 0)
        s = self.point(0, -1)
        self.circle_arc(e, n, center=c, name="wall")
        self.circle_arc(n, w, center=c, name="wall")
        self.circle_arc(w, s, center=c, name="wall")
        self.circle_arc(s, e, center=c, name="wall")
        self.plane_surface("wall", name="domain")


class Pois(Problem):
    def define_problem(self):
        self.add_mesh(Disc())
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "wall"
        self.add_equations(eqs @ "domain")


p = Pois()
try:
    with p:
        p.set_output_directory(args.outdir)
        p.quiet()
        p.initialise()
    # The primitive that keeps the ranks from disagreeing about whether to (re)generate the mesh,
    # which would call the collectives in generate_mesh_to_file() on a subset of them.
    print("PYOOMPH_MPI_ANY rank=%d one=%s none=%s" % (
        get_mpi_rank(), get_mpi_any(get_mpi_rank() == 0), get_mpi_any(False)))
    print("PYOOMPH_MPI_RESULT rank=%d completed" % get_mpi_rank())
except BaseException as e:  # noqa: BLE001
    print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (get_mpi_rank(), type(e).__name__, e))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(3)
