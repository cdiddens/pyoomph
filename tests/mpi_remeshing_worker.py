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

import numpy

from pyoomph import Problem, MeshFileOutput, Equations, ElementSpace, var
from pyoomph.expressions import var_and_test, avg, weak
from pyoomph.equations.generic import ProjectExpression
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.meshes.zeta import AssignZetaCoordinatesByArclength
from pyoomph.generic.mpi import get_mpi_rank

#: Projected onto the mesh before remeshing, so that the transfer to the new mesh has something to
#: carry. Varies in both directions and is not a polynomial the C2 space reproduces exactly, so a
#: value that travelled the wrong way cannot coincide with the right one.
TRANSFERRED_FIELD = 1 + var("coordinate_x") ** 2 - 2 * var("coordinate_y") ** 3


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


def _field_summary(problem, name):
    """What the transferred field looks like on the WHOLE new mesh, or None off rank 0.

    Read through the globally merged mesh data, so that the numbers describe the same thing serially
    and distributed - a per-rank summary would only describe that rank's partition. Collective, so
    every rank has to get here. Statistics rather than a digest, because the transfer is arithmetic
    and the last bits are allowed to differ."""
    data = problem.get_cached_mesh_data(name, global_mesh=True)
    if data is None:
        return None
    u = numpy.asarray(data.get_data("u"))
    coords = data.get_coordinates()
    # Weighted by position as well: a value that landed on the wrong node leaves the plain sums
    # almost untouched if it merely swapped with another, but not these.
    return {"nnode": len(u), "usum": float(numpy.sum(u)), "usqsum": float(numpy.sum(u * u)),
            "umin": float(numpy.amin(u)), "umax": float(numpy.amax(u)),
            "uxsum": float(numpy.sum(u * coords[0])), "uysum": float(numpy.sum(u * coords[1]))}


class _SkeletonTrace(Equations):
    """A facet unknown on the interior-facet skeleton, plus the observables that describe it.

    Distributed, a facet whose two elements land on different ranks is owned by one of them and held
    as a halo by the other, so it is one unknown numbered once; a remesh destroys the whole skeleton
    and rebuilds it, which means the ownership has to be rebuilt with it. `meas` is what says whether
    it was: it is the total measure of the skeleton, and does not involve the solution at all, so a
    facet enumerated twice or dropped moves it by that facet's length."""

    def define_fields(self):
        self.define_scalar_field("lam", "DL")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        u = avg(var("u"))
        self.add_residual(weak(lam - u, lamtest))
        self.add_integral_function("meas", 1 * self.get_dx())
        self.add_integral_function("lamsum", lam * self.get_dx())
        self.add_integral_function("lamerr2", (lam - u) ** 2 * self.get_dx())
        # The trace is determined by the bulk solution, so a facet the transfer could not reach can
        # recover it rather than staying at zero -- the recommended default for a trace field.
        self.set_facet_recovery("lam", avg(var("u")))


class _NeedsSkeleton(Equations):
    """Only there to make the bulk mesh build its `_internal_facets_` subdomain."""

    def __init__(self):
        super().__init__()
        self.requires_interior_facet_terms = True


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


class Box(GmshTemplate):
    """A second domain that never asks whether it is being remeshed.

    force_remesh() therefore skips it (see MeshedMeshTemplate._remeshing_can_change_the_mesh), which
    is what makes a partial remesh - the case the re-distribution has to refuse, since the mesh it
    leaves alone is still partitioned from before."""

    def define_geometry(self):
        self.default_resolution = 0.25
        corners = [self.point(x, y) for x, y in ((-2, -2), (-1, -2), (-1, -1), (-2, -1))]
        self.create_lines(corners[0], "bottom", corners[1], "right", corners[2], "top", corners[3],
                          "left", corners[0])
        self.plane_surface("bottom", "right", "top", "left", name="box")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--remesher2d", action="store_true",
                        help="use the automatic Remesher2d instead of recreation")
    parser.add_argument("--force", action="store_true",
                        help="set experimental_distributed_remeshing, i.e. run the unfinished path")
    parser.add_argument("--fail-define-geometry-on-rank", type=int, default=None,
                        help="raise inside define_geometry on this rank only")
    parser.add_argument("--second-domain", action="store_true",
                        help="add a domain that is not remeshed, i.e. make it a partial remesh")
    parser.add_argument("--num-adapt", type=int, default=None,
                        help="pass num_adapt to force_remesh explicitly")
    parser.add_argument("--codim2", action="store_true",
                        help="put equations where two boundaries meet, i.e. on a codimension-2 interface")
    parser.add_argument("--facet-field", action="store_true",
                        help="declare an unknown on the interior-facet skeleton as well")
    parser.add_argument("--zeta", action="store_true",
                        help="parameterise the interface by arclength, i.e. transfer through a zeta chart")
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
            interface_eqs = Equations()
            if args.zeta:
                # Arclength is a property of the whole curve, so this only works if the ranks
                # parameterise the merged interface rather than their own piece of it. It also sends
                # the transfer down the zeta branch of nodal_interpolate_from instead of projection.
                interface_eqs = interface_eqs + AssignZetaCoordinatesByArclength(sort_along_axis="x+")
            if args.codim2:
                # Where the arc meets the axis: an interface of the interface, transferred by the
                # nearest-node matching that is not pooled across the ranks.
                interface_eqs = interface_eqs + Equations() @ "axis"
            domain_eqs = (MeshFileOutput() + ElementSpace("C2") + ProjectExpression(u=TRANSFERRED_FIELD)
                          + interface_eqs @ "interface")
            if args.facet_field:
                domain_eqs = domain_eqs + _NeedsSkeleton() + _SkeletonTrace() @ "_internal_facets_"
            p += domain_eqs @ "domain"
            if args.second_domain:
                p += Box()
                p += (MeshFileOutput() + ElementSpace("C2") + ProjectExpression(u=0)) @ "box"
            p.initialise()
            # Solve the projection, so that u carries the field before the remesh and nothing
            # re-imposes it afterwards: what ends up on the new mesh is what the transfer put there.
            p.solve()
            p.force_remesh(num_adapt=args.num_adapt)
            ndof, distributed = p.ndof(), bool(p.get_mesh("domain").is_mesh_distributed())
            transferred = _field_summary(p, "domain")
            skeleton = None
            if args.facet_field:
                # After the remesh AND after a solve on the new mesh, so the trace is the one the new
                # skeleton determines rather than whatever the transfer happened to place.
                p.solve()
                skeleton = {k: float(v) for k, v in
                            p.get_mesh("domain/_internal_facets_").evaluate_all_observables().items()}
        print("PYOOMPH_MPI_RESULT rank=%d remeshed ndof=%d distributed=%s" % (
            get_mpi_rank(), ndof, distributed))
        if transferred is not None:
            print("PYOOMPH_MPI_FIELD " + " ".join("%s=%.12g" % kv for kv in sorted(transferred.items())))
        if skeleton is not None:
            print("PYOOMPH_MPI_SKELETON " + " ".join("%s=%.12g" % kv for kv in sorted(skeleton.items())))
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
