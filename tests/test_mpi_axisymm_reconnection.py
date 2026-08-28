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

# AxisymmetricReconnection under mpirun, with and without --distribute (stage 7 of the axisymmetric
# pinch-off/coalescence rewrite; the geometry itself is pinned by the serial suites
# tests/test_axisymm_*).
#
# What can go wrong here is not a wrong number, it is the ranks disagreeing:
#
# * the detection is a morphological statement about the WHOLE interface. Read off a partition it is
#   a different, truncated shape, cut wherever the partition happens to run - and a waist sitting on
#   a partition boundary is not a corner case, it is what partitioning by element count does. So the
#   interface is merged, the plan is made on rank 0 and broadcast, and every rank must end up with
#   the IDENTICAL plan: define_geometry then rebuilds the geometry from it with no further
#   communication, and two ranks with different plans describe two different geometries.
# * the overlap guard runs on every Newton step and returns a bool. A rejection seen by some ranks
#   and not the others abandons the solve on those alone.
# * a remesh is collective throughout (gmsh, the transfer, the re-distribution), so a rank that does
#   not ask for it waits for the others for ever rather than failing.
#
# Every run is therefore under a timeout that fails rather than waits, and every assertion is about
# all the ranks agreeing - with each other and with a serial run of the very same worker.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_axisymm_reconnection_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    if shutil.which("gmsh") is None:
        return "gmsh not found"
    try:
        import shapely  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return "shapely is not installed (pip install pyoomph[topology]): " + str(e)
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(tmpdir, case, extra_args=(), nproc=2, timeout=600):
    """The worker under mpirun, in its own fresh directory."""
    os.makedirs(str(tmpdir), exist_ok=True)
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "0") == "1":
        # Off by default here: these machines have enough cores for the four ranks this module ever
        # asks for, and oversubscribing turns a deadlock into a machine that stops responding.
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--case", case,
            "--outdir", os.path.join(str(tmpdir), "out")] + list(extra_args)
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired as e:
        # text=True, so e.stdout is a str; None when nothing was captured before the kill.
        raise AssertionError(
            "mpirun -n %d (%s %s) did not finish within %d s -- the ranks disagreed about a "
            "collective instead of ending the run.\n--- stdout tail ---\n%s" % (
                nproc, case, " ".join(extra_args) or "no extra args", timeout,
                (e.stdout or "")[-3000:]))


def _run_serially(tmpdir, case, extra_args=(), timeout=600):
    """The same worker without mpirun: the reference every distributed run must reproduce."""
    os.makedirs(str(tmpdir), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--case", case,
           "--outdir", os.path.join(str(tmpdir), "out")] + list(extra_args)
    return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout)


def _ok(proc, what):
    assert proc.returncode == 0, "%s failed (rc=%d):\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
        what, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])
    return proc


def _marked(proc, marker):
    """Every worker line carrying ``marker``, from the marker onwards.

    Not ``startswith``: oomph-lib's C++ output reaches the same terminal through its own buffer, so a
    banner written without a newline (which is what the abandoned Newton solve of the guard case
    produces) ends up prepended to the next Python line. The Python side writes whole lines
    atomically, so everything from the marker on is intact; only what precedes it is foreign.
    """
    out = []
    for line in proc.stdout.splitlines():
        at = line.find(marker)
        if at >= 0:
            out.append(line[at + len(marker):])
    return out


def _per_rank(proc):
    """``{rank: state}`` from the one line each rank prints when it is done."""
    out = {}
    for rest in _marked(proc, "PYOOMPH_MPI_RESULT "):
        rank = int(rest.split("rank=")[1].split()[0])
        if " finished " in rest:
            out[rank] = json.loads(rest.split(" finished ", 1)[1])
        else:
            out[rank] = {"raised": rest.split(" raised ", 1)[1]}
    return out


def _digest(proc):
    for rest in _marked(proc, "PYOOMPH_MPI_DIGEST "):
        return json.loads(rest)
    return None


def _all_ranks_agree(proc, nproc, case):
    states = _per_rank(proc)
    assert len(states) == nproc, "only %d of %d ranks reported (%s):\n%s\n--- stderr tail ---\n%s" % (
        len(states), nproc, case, "\n".join(sorted(map(str, states))), proc.stderr[-2000:])
    for rank, state in sorted(states.items()):
        assert "raised" not in state, "rank %d raised: %s" % (rank, state["raised"])
        assert state == states[0], (
            "rank %d ended up in a different state than rank 0.\n  rank %d: %s\n  rank 0: %s" % (
                rank, rank, json.dumps(state, sort_keys=True), json.dumps(states[0], sort_keys=True)))
    return states[0]


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("distribute", [False, True])
def test_the_plan_is_the_same_on_every_rank_and_serially(tmp_path, nproc, distribute):
    """The surgery plan itself: merged, decided on rank 0, broadcast.

    Without --distribute every rank holds the whole interface and takes the local path unchanged,
    which is the control; with it the plan comes off the merged interface. Both are compared against
    a serial run, so this pins the plan and not merely that the ranks agree on something.

    The plan carries the zeta chart that will carry the interface fields across the event, so the
    table it was built from is summed into the comparison too - a chart that differed between the
    ranks would transfer the fields differently on each of them.
    """
    reference = _ok(_run_serially(tmp_path / "serial", "detect"), "the serial reference run")
    serial = _per_rank(reference)[0]
    assert serial["has_plan"] and serial["queued"], "the serial run did not plan a pinch-off: %s" % serial

    extra = ["--distribute"] if distribute else []
    proc = _ok(_run(tmp_path / ("mpi%d" % nproc), "detect", extra, nproc=nproc), "the distributed run")
    got = _all_ranks_agree(proc, nproc, "detect")
    assert got == serial, (
        "the plan differs from the serial one.\n  distributed: %s\n  serial:      %s" % (
            json.dumps(got, sort_keys=True), json.dumps(serial, sort_keys=True)))


@pytest.mark.parametrize("nproc", [2, 4])
def test_a_neck_above_rmin_is_not_planned_by_any_rank(tmp_path, nproc):
    """The negative control, and the one a partition could most easily get wrong.

    A neck of 0.3 is comfortably above rmin=0.12, but a rank holding only the waist sees an interface
    whose thinnest point is all there is. Detecting on the merged interface is what keeps every rank
    from planning an event that is not there.
    """
    proc = _ok(_run(tmp_path, "detect", ["--distribute", "--neck", "0.3"], nproc=nproc),
               "the distributed run")
    got = _all_ranks_agree(proc, nproc, "detect")
    assert not got["has_plan"] and not got["queued"], \
        "a pinch-off was planned for a neck well above rmin: %s" % got


# --------------------------------------------------------------------------------------
# The event remesh, end to end
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("case", ["pinch", "coalescence"])
def test_the_event_remesh_reproduces_the_serial_one(tmp_path, nproc, case):
    """Detection, surgery, transfer and re-distribution, against a serial run of the same worker.

    ``pinch`` goes through the automatic remesh inside solve(), i.e. through
    Problem._agree_on_domains_to_remesh() and the deferred _perform_pending_remesh(); ``coalescence``
    is the other topology, and the one whose new mesh has a thin bridge in it.

    The digest is read off the globally MERGED mesh data, so the numbers describe the whole mesh in
    both runs rather than one rank's partition. ndof is the sharpest single assertion: a mesh left
    replicated while the problem is still numbered as if it were distributed comes out nproc times
    too large.
    """
    reference = _ok(_run_serially(tmp_path / "serial", case), "the serial reference run")
    serial_state, serial_digest = _per_rank(reference)[0], _digest(reference)
    assert serial_digest is not None, "the serial run reported no digest:\n%s" % reference.stdout[-2000:]
    assert serial_digest["interface"]["n_fragments"] == (2 if case == "pinch" else 1), \
        "the serial run did not perform the %s: %s" % (case, serial_digest["interface"])

    proc = _ok(_run(tmp_path / ("mpi%d" % nproc), case, ["--distribute"], nproc=nproc),
               "the distributed run")
    got = _all_ranks_agree(proc, nproc, case)
    assert got["distributed"], "the rebuilt mesh was left replicated: %s" % got
    assert got["ndof"] == serial_state["ndof"], (
        "ndof is %d after the distributed remesh and %d serially -- a factor of %d would mean the "
        "replicated mesh was numbered as if it were distributed" % (
            got["ndof"], serial_state["ndof"], nproc))
    assert got["zeta"] == serial_state["zeta"], (
        "the interface chart came out %r distributed and %r serially" % (got["zeta"], serial_state["zeta"]))
    digest = _digest(proc)
    assert digest is not None, "no rank reported the digest:\n%s" % proc.stdout[-2000:]
    _compare_digest(digest, serial_digest, "the %s remesh" % case)


@pytest.mark.parametrize("nproc", [2, 4])
def test_an_interface_field_survives_the_distributed_event(tmp_path, nproc):
    """The zeta chart the surgery writes onto both interfaces, and the field it carries.

    The chart is the one thing the geometry alone cannot supply - old and new interface are not the
    same curve near the event - so it comes out of the broadcast plan and each rank writes it onto
    the nodes it holds. If the plan differed between the ranks, or if a rank wrote the chart onto
    only part of its share, the field would arrive scrambled on exactly that part.

    The ordinary remesh that follows is in the digest as well: the one-off chart has to have been
    taken back again, or that one fails with "Boundary coordinate along ... is defined on the old,
    but not the new mesh".
    """
    reference = _ok(_run_serially(tmp_path / "serial", "transfer_pinch"), "the serial reference run")
    serial_state, serial_digest = _per_rank(reference)[0], _digest(reference)
    assert serial_digest is not None
    before = serial_digest["before"]["interface"]
    assert before["n_fragments"] == 1 and serial_digest["after"]["interface"]["n_fragments"] == 2, \
        "the serial run did not pinch: %s -> %s" % (before, serial_digest["after"]["interface"])

    proc = _ok(_run(tmp_path / ("mpi%d" % nproc), "transfer_pinch", ["--distribute"], nproc=nproc),
               "the distributed run")
    got = _all_ranks_agree(proc, nproc, "transfer_pinch")
    assert got["ndof"] == serial_state["ndof"]
    digest = _digest(proc)
    assert digest is not None, "no rank reported the digest:\n%s" % proc.stdout[-2000:]
    for stage in ("before", "after", "after_second_remesh"):
        _compare_digest(digest[stage], serial_digest[stage], "the field " + stage)


@pytest.mark.parametrize("nproc", [2, 4])
def test_both_phases_of_a_shared_interface_come_through(tmp_path, nproc):
    """A pinch with a gas phase on the far side of the same interface.

    The interface is a boundary of BOTH domains, so the gas has its own interface mesh with its own
    zeta - transferred by the gas domain's interpolator, which this equation is never dispatched with.
    It reaches it through ``BaseMeshToMeshInterpolator.remesh_group``, and it does so on every rank
    from the same broadcast plan. The gas is also the side the pinch hands a stretch of axis that was
    liquid a moment ago, i.e. the one place where the bulk-locate path runs.

    Both domains come out of one template, which is what lets this be remeshed distributed at all:
    remeshing only some domains of a distributed problem is refused, since the ones left alone stay
    partitioned from before.
    """
    reference = _ok(_run_serially(tmp_path / "serial", "twophase"), "the serial reference run")
    serial_state, serial_digest = _per_rank(reference)[0], _digest(reference)
    assert serial_digest is not None
    assert serial_digest["interface"]["n_fragments"] == 2, "the serial run did not pinch"
    assert serial_digest["gas_interface"]["n_fragments"] == 2, \
        "the gas side of the interface did not follow the pinch"

    proc = _ok(_run(tmp_path / ("mpi%d" % nproc), "twophase", ["--distribute"], nproc=nproc),
               "the distributed run")
    got = _all_ranks_agree(proc, nproc, "twophase")
    assert got["ndof"] == serial_state["ndof"], (
        "ndof is %d after the distributed two-phase remesh and %d serially" % (
            got["ndof"], serial_state["ndof"]))
    _compare_digest(_digest(proc), serial_digest, "the two-phase pinch")


# --------------------------------------------------------------------------------------
# The per-Newton-step guard
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("distribute", [False, True])
def test_the_overlap_guard_rejects_on_every_rank_or_on_none(tmp_path, nproc, distribute):
    """The guard runs inside the Newton loop, so this is where a hang would show.

    Three things have to come out the same everywhere: that the guard armed at all, that it let the
    harmless step through, and that it refused the one driving the two tips through each other. The
    arming matters as much as the verdict - it is what decides whether a rank enters the merge on the
    next Newton step, so a rank that armed differently would be the one left in the collective.
    """
    reference = _ok(_run_serially(tmp_path / "serial", "guard"), "the serial reference run")
    serial = _per_rank(reference)[0]
    assert serial["armed"] and not serial["harmless_rejected"] and serial["guard_rejected"], \
        "the serial run did not exercise the guard: %s" % serial

    extra = ["--distribute"] if distribute else []
    proc = _ok(_run(tmp_path / ("mpi%d" % nproc), "guard", extra, nproc=nproc), "the distributed run")
    got = _all_ranks_agree(proc, nproc, "guard")
    for key in ("armed", "harmless_rejected", "armed_after_harmless", "guard_rejected"):
        assert got[key] == serial[key], (
            "%s is %r under mpirun and %r serially" % (key, got[key], serial[key]))


# --------------------------------------------------------------------------------------

#: Structural quantities: the mesh is generated from the same geometry description on every rank, so
#: these have to be equal, not close.
_EXACT = ("n_fragments", "npoints", "nnode", "nelem", "n_nonfinite", "field")

#: Sums over the transferred fields. Slack, and deliberately: the handful of nodes that no rank can
#: locate in the old mesh - the fresh cap of a pinch, the fresh bridge of a coalescence, which by
#: construction lie outside the old geometry everywhere - fall back to a nearest-node blend, and that
#: blend is rank-local. The destination mesh is still replicated when it runs, so each rank blends
#: from its own share of the old mesh and the owner's answer is the one that survives the
#: re-distribution. Measured here: 3 of ~400 nodes, moving the sums by up to 2e-4 relative and
#: varying with the number of ranks. A value that landed on the WRONG SIDE of the event would move
#: them by O(1), which is what this still catches - as do umin/umax/fmin/fmax, which are exact.
#: (Pre-existing, and not specific to a topological change; see dev_docs/distributed_remeshing.md.)
_SUM_TOL = 2e-3


def _near(got, expected, tol, where):
    """Compare two pieces of digest, descending into lists and dicts. Own recursion rather than
    pytest.approx, which refuses a nested sequence (the endpoint lists are one)."""
    if isinstance(expected, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(expected), \
            "%s: %r vs %r" % (where, got, expected)
        for k in sorted(expected):
            _near(got[k], expected[k], _SUM_TOL if str(k).endswith("sum") else tol, where + "/" + str(k))
    elif isinstance(expected, (list, tuple)):
        assert isinstance(got, (list, tuple)) and len(got) == len(expected), \
            "%s: %r vs %r" % (where, got, expected)
        for i, (g, e) in enumerate(zip(got, expected)):
            _near(g, e, tol, "%s[%d]" % (where, i))
    elif isinstance(expected, bool) or isinstance(got, bool) or expected is None or got is None:
        assert got == expected, "%s is %r, serially %r" % (where, got, expected)
    elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        assert got == pytest.approx(expected, rel=tol, abs=1e-9), \
            "%s is %r, serially %r" % (where, got, expected)
    else:
        assert got == expected, "%s is %r, serially %r" % (where, got, expected)


def _compare_digest(got, expected, what):
    """Structure exactly, sums to the tolerance the rank-local blend fallback leaves."""
    assert sorted(got) == sorted(expected), "%s: different keys %s vs %s" % (what, sorted(got), sorted(expected))
    for key in sorted(expected):
        g, e = got[key], expected[key]
        if isinstance(e, dict):
            assert (g is None) == (e is None), "%s/%s: one of the two is missing" % (what, key)
            for sub in sorted(e):
                if sub in _EXACT:
                    assert g[sub] == e[sub], "%s/%s/%s is %r, serially %r" % (what, key, sub, g[sub], e[sub])
                else:
                    _near(g[sub], e[sub], _SUM_TOL if sub.endswith("sum") else 1e-6,
                          "%s/%s/%s" % (what, key, sub))
        else:
            _near(g, e, 1e-6, "%s/%s" % (what, key))
