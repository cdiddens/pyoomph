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

# Worker for tests/test_mpi_remeshing.py -- launched under mpirun.
# A quarter disc whose curved boundary is rebuilt on remeshing, i.e. the geometry depends on the
# mesh being replaced. See dev_docs/distributed_remeshing.md.

import argparse
import hashlib
import sys
import traceback

from pyoomph import Problem, MeshFileOutput, Equations, ElementSpace
from pyoomph.equations.generic import ProjectExpression
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.generic.mpi import get_mpi_rank


def _digest(segments):
    """Identity of the reconstructed boundary, comparable between ranks and against a serial run.

    Rounded before hashing: the merged data is assembled from the very same node coordinates on every
    rank, so this is not a tolerance, only insurance against a repr difference."""
    h = hashlib.sha1()
    for seg in segments:
        h.update(b"|")
        for x, y in seg:
            h.update(("%.12f,%.12f;" % (float(x), float(y))).encode())
    return h.hexdigest()[:16]


class Disc(GmshTemplate):
    #: Raise inside define_geometry() on this rank only, to test that the others end with it.
    #:
    #: Deliberately raised *after* get_boundary_coordinates(), which is collective on a distributed
    #: mesh: a rank that never reaches that call leaves the others inside its merge, which is a hang
    #: nothing downstream can catch. This models what can actually be caught - user code that gets
    #: through the collectives and then produces something invalid on one rank.
    fail_on_rank = None

    def define_geometry(self):
        self.default_resolution = 0.1
        origin = self.point(0, 0)
        if self.is_first_time():
            right = self.point(1, 0)
            top = self.point(0, 1)
            self.circle_arc(right, top, center=origin, name="interface")
        else:
            # The point of the exercise: without the merge this only sees the local part of the arc,
            # and at three ranks rank 0 sees none of it at all.
            segments = self.get_boundary_coordinates("domain/interface", sort_along_axis="x+")
            print("PYOOMPH_MPI_BOUNDARY rank=%d nseg=%d npts=%s digest=%s" % (
                get_mpi_rank(), len(segments), ",".join(str(len(s)) for s in segments),
                _digest(segments)), flush=True)
            if self.fail_on_rank is not None and get_mpi_rank() == self.fail_on_rank:
                raise RuntimeError("simulated failure inside define_geometry")
            pts = [self.point(x, y) for x, y in segments[0]]
            self.spline(pts, name="interface")
            right, top = pts[-1], pts[0]
        self.create_lines(right, "substrate", origin, "axis", top)
        self.plane_surface("substrate", "axis", "interface", name="domain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--remesher2d", action="store_true",
                        help="use the automatic Remesher2d instead of recreation")
    parser.add_argument("--force", action="store_true",
                        help="set experimental_distributed_remeshing, i.e. run the unfinished path")
    parser.add_argument("--fail-define-geometry-on-rank", type=int, default=None,
                        help="raise inside define_geometry on this rank only")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    p = Problem()
    mesh = Disc()
    mesh.fail_on_rank = args.fail_define_geometry_on_rank
    if args.remesher2d:
        mesh.remesher = Remesher2d(mesh)
    try:
        with p:
            p.set_output_directory(args.outdir)
            p.quiet()
            p.experimental_distributed_remeshing = args.force
            p += mesh
            p += (MeshFileOutput() + ElementSpace("C2") + ProjectExpression(u=0)
                  + Equations() @ "interface") @ "domain"
            p.initialise()
            p.force_remesh()
            ndof, distributed = p.ndof(), bool(p.get_mesh("domain").is_mesh_distributed())
        print("PYOOMPH_MPI_RESULT rank=%d remeshed ndof=%d distributed=%s" % (
            get_mpi_rank(), ndof, distributed))
    except BaseException as e:  # noqa: BLE001
        # Flattened: the refusal message is multi-line, and only the first line would carry the
        # prefix the test greps for.
        print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (
            get_mpi_rank(), type(e).__name__, " | ".join(str(e).splitlines())))
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(3)


if __name__ == "__main__":
    main()
