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

"""Worker for tests/test_mpi_axisymm_reconnection.py -- launched under mpirun, with and without
``--distribute``.

The scenarios are the kinematic ones of tests/axisymm_reconnection_worker.py, imported from there
rather than restated: a prescribed dumbbell that pinches, two spheroids that coalesce, an interface
field carried across a pinch, and the overlap guard that rejects a Newton step. What is under test
here is not the geometry - the serial suites pin that - but that the whole pipeline stays in step
across the ranks:

* the detection reads the WHOLE interface (a partition of it is a different, truncated shape, and a
  waist can sit exactly on a partition cut), so it merges, decides on rank 0 and broadcasts the plan;
* every rank must end up with the *identical* plan, since ``define_geometry`` then rebuilds the
  geometry from it with no further communication;
* the overlap guard runs per Newton step and must return the same verdict everywhere, or one rank
  abandons the solve while the others carry on.

The digest is therefore read off the globally MERGED mesh data, so that the numbers describe the same
thing serially and distributed, and compared against a serial reference run of the very same worker.
"""

import argparse
import json
import os
import sys
import traceback

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from axisymm_reconnection_worker import ReconnectionProblem, dumbbell, two_blobs  # noqa: E402

from pyoomph.generic.mpi import get_mpi_rank  # noqa: E402
from pyoomph.meshes.axisymm_topology import revolved_volume  # noqa: E402
from pyoomph.meshes.ordering import sort_line_segments  # noqa: E402


# --------------------------------------------------------------------------------------
# Reporting, always off the merged mesh data
# --------------------------------------------------------------------------------------

def _merged(problem, name):
    """The whole mesh's data on rank 0, ``None`` elsewhere. Collective: every rank must reach it."""
    return problem.get_cached_mesh_data(name, nondimensional=True, tesselate_tri=False,
                                        global_mesh=True)


def _round(x, digits=9):
    # The mesh is generated from the same .geo on every rank, so these agree far beyond this; the
    # rounding only keeps a last-bit repr difference out of the JSON comparison.
    return float(numpy.round(float(x), digits))


def interface_digest(problem, iface_name, fieldname=None):
    """Fragment count, per-fragment volume and (optionally) an interface field, from the merged data.

    Sorted along +y and each polyline oriented bottom-to-top, so the answer does not depend on which
    rank contributed which piece - the merged element list is a concatenation over the ranks and the
    segment walk starts wherever that puts an endpoint.
    """
    data = _merged(problem, iface_name)
    if data is None:
        return None
    segs, _ = data.get_interface_line_segments()
    pts = data.get_coordinates()
    segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface_digest")
    out = {"n_fragments": len(segs), "volumes": [], "ends": [], "npoints": []}
    vals = None if fieldname is None else numpy.asarray(data.get_data(fieldname))
    fmin, fmax, fsum = [], [], []
    for s in segs:
        idx = list(s)
        if pts[1, idx[0]] > pts[1, idx[-1]]:
            idx = list(reversed(idx))
        poly = numpy.array([[pts[0, i], pts[1, i]] for i in idx], dtype=float)
        out["volumes"].append(_round(revolved_volume(_closed(poly)), 8))
        out["ends"].append([[_round(poly[0, 0]), _round(poly[0, 1])],
                            [_round(poly[-1, 0]), _round(poly[-1, 1])]])
        out["npoints"].append(len(idx))
        if vals is not None:
            v = vals[idx]
            fmin.append(_round(numpy.amin(v), 7))
            fmax.append(_round(numpy.amax(v), 7))
            fsum.append(_round(numpy.sum(v), 6))
    if vals is not None:
        out["field"] = fieldname
        out["fmin"], out["fmax"], out["fsum"] = fmin, fmax, fsum
    return out


def _closed(poly):
    """Close an interface polyline onto the axis so its revolved volume is defined."""
    extra = []
    if abs(poly[-1, 0]) > 0.0:
        extra.append([0.0, float(poly[-1, 1])])
    if abs(poly[0, 0]) > 0.0:
        extra.append([0.0, float(poly[0, 1])])
    return poly if not extra else numpy.vstack([poly, numpy.array(extra, dtype=float)])


def bulk_digest(problem, domain, fieldname="u"):
    """Node count and the field statistics of a whole bulk domain, from the merged data."""
    data = _merged(problem, domain)
    if data is None:
        return None
    coords = data.get_coordinates()
    u = numpy.asarray(data.get_data(fieldname))
    return {"nnode": int(len(u)), "nelem": int(len(data.elem_indices)),
            "umin": _round(numpy.amin(u), 7), "umax": _round(numpy.amax(u), 7),
            "usum": _round(numpy.sum(u), 6),
            "uxsum": _round(numpy.sum(u * coords[0]), 6),
            "uysum": _round(numpy.sum(u * coords[1]), 6),
            "n_nonfinite": int(numpy.sum(~numpy.isfinite(u)))}


def axis_digest(problem, axis_name):
    data = _merged(problem, axis_name)
    if data is None:
        return None
    segs, _ = data.get_interface_line_segments()
    pts = data.get_coordinates()
    segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="axis_digest")
    spans = []
    for s in segs:
        ys = [float(pts[1, i]) for i in s]
        spans.append([_round(min(ys)), _round(max(ys))])
    return sorted(spans)


def zeta_ok(problem, iface_name):
    """Whether the chart on this rank's share of the interface is still a chart.

    Local by design (see dev_docs/distributed_remeshing.md §3.4a): the test is that the elements THIS
    rank holds tile their own stretch of zeta without overlapping, which is as meaningful on a
    partition as on the whole interface. Reported per rank so that a rank whose share was left with a
    folded chart cannot hide behind rank 0's.
    """
    from pyoomph.meshes.zeta import _check_zeta_is_invertible
    mesh = problem.get_mesh(iface_name)
    bulk = mesh.get_bulk_mesh()
    bind = bulk.get_boundary_index(mesh.get_name())
    if not bulk.is_boundary_coordinate_defined(bind):
        return "none"
    try:
        _check_zeta_is_invertible(mesh, bind, "the MPI reconnection worker")
    except RuntimeError as e:  # noqa: BLE001
        return "BROKEN: " + str(e).splitlines()[0]
    return "ok"


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------

def _setup(p, outdir):
    p.set_output_directory(outdir)
    p.quiet()
    p.initialise()


def case_detect(outdir, args):
    """Detection only: no remesh, so this isolates the merged-and-broadcast plan itself.

    Every rank reports what it was given, and the test compares those to each other and to serial.
    """
    p = ReconnectionProblem([dumbbell(args.neck)], rmin=0.12, distmin=None, scale=0.5,
                            resolution=args.resolution, moving_mesh=False, check_motion=False)
    _setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    tmpl, rec = p.mesh_template, p.reconnection
    plan = rec._last_plan if tmpl.has_pending_surgery_plan("liquid/interface") else None
    per_rank = {"has_plan": bool(tmpl.has_pending_surgery_plan("liquid/interface")),
                "queued": bool(tmpl in p._domains_to_remesh),
                "armed": bool(rec._armed)}
    if plan is not None:
        entry = tmpl._pending_surgery_plans["liquid/interface"]
        extra = entry[1]
        per_rank["plan"] = {
            "kinds": [e.kind for e in plan.events],
            "z": [_round(e.z_center) for e in plan.events],
            "before": [_round(v, 8) for v in plan.fragment_volumes_before],
            "after": [_round(v, 8) for v in plan.fragment_volumes_after],
            "chain_points": [int(len(c.points)) for c in plan.new_chains],
            # The chart that carries the interface fields across the event. Hashed rather than
            # dumped: it is a few hundred rows, and what matters is that the ranks got the same one.
            "zeta_span": [[_round(float(c.zeta[0]), 8), _round(float(c.zeta[-1]), 8)]
                          for c in plan.new_chains],
            "table_shape": list(numpy.shape(extra["old_zeta_table"])),
            "table_sum": _round(float(numpy.sum(extra["old_zeta_table"])), 6),
            "axis_name": extra["axis_name"],
            "opposite_axis_name": extra["opposite_axis_name"],
        }
    return per_rank, None


def case_pinch(outdir, args):
    """The pinch-off itself, through the automatic remesh inside solve().

    That path is the one with the extra MPI net under it: the request is recorded during the solve,
    made unanimous by Problem._agree_on_domains_to_remesh() and carried out afterwards.
    """
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None,
                            resolution=args.resolution, moving_mesh=False, check_motion=False)
    _setup(p, outdir)
    p.solve()
    per_rank = {"zeta": zeta_ok(p, "liquid/interface"), "ndof": int(p.ndof()),
                "distributed": bool(p.get_mesh("liquid").is_mesh_distributed()),
                "has_plan": bool(p.mesh_template.has_pending_surgery_plan("liquid/interface"))}
    digest = {"interface": interface_digest(p, "liquid/interface"),
              "bulk": bulk_digest(p, "liquid"),
              "axis": axis_digest(p, "liquid/axis")}
    p.solve()  # the new mesh must be solvable, and must not re-detect
    per_rank["has_plan_after"] = bool(p.mesh_template.has_pending_surgery_plan("liquid/interface"))
    return per_rank, digest


def case_coalescence(outdir, args):
    distmin = 0.05
    p = ReconnectionProblem(two_blobs(0.4 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=False, check_motion=False,
                            overlap_reject_factor=None, size_via_callable=True)
    _setup(p, outdir)
    p.solve()
    per_rank = {"zeta": zeta_ok(p, "liquid/interface"), "ndof": int(p.ndof()),
                "distributed": bool(p.get_mesh("liquid").is_mesh_distributed())}
    digest = {"interface": interface_digest(p, "liquid/interface"),
              "bulk": bulk_digest(p, "liquid"),
              "axis": axis_digest(p, "liquid/axis")}
    p.solve()
    per_rank["has_plan_after"] = bool(p.mesh_template.has_pending_surgery_plan("liquid/interface"))
    return per_rank, digest


def case_transfer_pinch(outdir, args):
    """An interface field f = z and a bulk field u = 1 + z, carried through the pinch.

    The interface field is the one that needs the surgery's zeta chart written onto both meshes -
    which every rank does for its own nodes, from the broadcast plan. A second, ordinary remesh
    afterwards, because a one-off chart that is not taken back again makes exactly that one fail.
    """
    p = ReconnectionProblem([dumbbell(0.04)], rmin=0.12, distmin=None, resolution=args.resolution,
                            moving_mesh=False, check_motion=False, surface_field=True)
    _setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve()
    before = {"interface": interface_digest(p, "liquid/interface", "f"),
              "bulk": bulk_digest(p, "liquid")}
    p.force_remesh({p.mesh_template})
    after = {"interface": interface_digest(p, "liquid/interface", "f"),
             "bulk": bulk_digest(p, "liquid"), "axis": axis_digest(p, "liquid/axis")}
    per_rank = {"zeta": zeta_ok(p, "liquid/interface"), "ndof": int(p.ndof()),
                "distributed": bool(p.get_mesh("liquid").is_mesh_distributed())}
    p.solve()
    p.force_remesh({p.mesh_template})
    second = {"interface": interface_digest(p, "liquid/interface", "f"),
              "bulk": bulk_digest(p, "liquid")}
    per_rank["zeta_after_second"] = zeta_ok(p, "liquid/interface")
    return per_rank, {"before": before, "after": after, "after_second_remesh": second}


def case_guard(outdir, args):
    """The overlap guard: armed by a close gap, then asked to take a step straight through it.

    The per-Newton-step hook. Distributed, the two approaching tips are routinely on different ranks,
    so the verdict has to be computed on the merged interface and broadcast - a rejection seen by
    some ranks and not the others abandons the solve on those alone.
    """
    distmin = 0.05
    p = ReconnectionProblem(two_blobs(2.0 * distmin), rmin=None, distmin=distmin,
                            resolution=args.resolution, moving_mesh=True, check_motion=True)
    _setup(p, outdir)
    p.do_call_remeshing_when_necessary = False
    p.solve(timestep=1.0)
    per_rank = {"armed": bool(p.reconnection._armed),
                "limit": _round(p.reconnection.overlap_reject_factor * distmin)}

    # A modest step: the gap stays well above the limit, so the guard must let it through.
    p.get_global_parameter("Vz").value = 0.02
    try:
        p.solve(timestep=1.0)
        per_rank["harmless_rejected"] = False
    except BaseException as e:  # noqa: BLE001
        per_rank["harmless_rejected"] = True
        per_rank["harmless_exception"] = type(e).__name__
    per_rank["armed_after_harmless"] = bool(p.reconnection._armed)

    # And one that would drive the two tips straight through each other must be refused.
    p.get_global_parameter("Vz").value = 0.2
    try:
        p.solve(timestep=1.0)
        per_rank["guard_rejected"] = False
    except BaseException as e:  # noqa: BLE001
        per_rank["guard_rejected"] = True
        per_rank["guard_exception"] = type(e).__name__
    digest = {"interface": interface_digest(p, "liquid/interface")}
    return per_rank, digest


CASES = {"detect": case_detect, "pinch": case_pinch, "coalescence": case_coalescence,
         "transfer_pinch": case_transfer_pinch, "guard": case_guard}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--neck", type=float, default=0.04)
    parser.add_argument("--resolution", type=float, default=0.08)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    try:
        per_rank, digest = CASES[args.case](args.outdir, args)
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (
            get_mpi_rank(), type(e).__name__, " | ".join(str(e).splitlines())), flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(3)
    # One line per rank: what that rank ended up with. Everything in it must be identical on all of
    # them, which is the property the whole stage is about.
    print("PYOOMPH_MPI_RESULT rank=%d finished %s" % (
        get_mpi_rank(), json.dumps(per_rank, sort_keys=True)), flush=True)
    # And one line from rank 0 with the merged-mesh digest, compared against a serial run. Only from
    # rank 0: that is where the merged data exists at all, the others hold None throughout.
    if digest is not None and get_mpi_rank() == 0:
        print("PYOOMPH_MPI_DIGEST " + json.dumps(digest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
