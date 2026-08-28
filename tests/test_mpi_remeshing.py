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

# Remeshing under --distribute, built stage by stage in dev_docs/distributed_remeshing.md. The
# Both remeshing paths work by now and are exercised here in the configuration they ship in; what is
# not built yet (the projection interpolator, remeshing a subset of the domains, adapting the new
# mesh) is refused by name, and the refusals are tested too, because what those paths would do
# instead of failing is produce a plausible wrong answer.
#
# What the whole thing used to do instead of saying that it could not:
#
# * at two ranks every rank rebuilt the geometry from its own partition of the boundary, and since
#   only rank 0's .msh file is kept, rank 0's truncated wedge silently became the mesh for all of
#   them. The run then finished with a mesh replicated on every rank while the problem still
#   numbered the equations as if it were distributed, i.e. ndof came out nproc times too large;
# * at three and four ranks the rank holding no element of the boundary raised inside
#   define_geometry() while the others walked into the barriers of generate_mesh_to_file(), and the
#   job hung. At three ranks that rank is rank 0.
#
# So the tests below cover, in order: the refusals that remain and the cross-rank agreement that
# replaced the hang (stage 0), the merged boundary every rank now rebuilds (stage 1), the
# re-partitioning of the rebuilt mesh and the two things it cannot do (stage 2), and the transferred
# field itself (stage 3). Every run is under a timeout that fails rather than waits, since the
# regressions being guarded against are deadlocks.

import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_remeshing_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    if shutil.which("gmsh") is None:
        return "gmsh not found"
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


def _run(tmpdir, extra_args, nproc=2, timeout=300):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir)] + extra_args
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
        raise AssertionError(
            "mpirun -n %d did not finish within %d s -- a failure on one rank deadlocked the "
            "others instead of ending the run (%s).\n--- stdout tail ---\n%s" % (
                nproc, timeout, " ".join(extra_args) or "no extra args", (e.stdout or "")[-3000:]))


def _run_serially(tmpdir, extra_args, timeout=300):
    """The same worker without mpirun, for the reference the distributed runs must reproduce."""
    cmd = [sys.executable, _WORKER, "--outdir", str(tmpdir)] + extra_args
    return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout)


def _reported(proc):
    return [l for l in proc.stdout.splitlines() if l.startswith("PYOOMPH_MPI_RESULT ")]


def _remesh_fields(proc):
    """{rank: {"ndof": ..., "distributed": ...}} from the ranks that got through the remesh."""
    out = {}
    for line in _reported(proc):
        if " remeshed " not in line:
            continue
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        out[int(fields["rank"])] = fields
    return out


def _field(proc):
    """The summary of the transferred field, from whichever process reported it (rank 0)."""
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_FIELD "):
            return {k: float(v) for k, v in (f.split("=", 1) for f in line.split()[1:])}
    return None


def _skeleton(proc):
    """The interior-facet skeleton's observables after the remesh, from whichever rank reported."""
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_SKELETON "):
            return {k: float(v) for k, v in (f.split("=", 1) for f in line.split()[1:])}
    return None


def _boundaries(proc):
    """What each rank got out of get_boundary_coordinates(), as {rank: line}."""
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_BOUNDARY "):
            fields = dict(f.split("=", 1) for f in line.split()[1:])
            out[int(fields["rank"])] = fields
    return out


@pytest.mark.parametrize("nproc", [2, 4])
def test_remeshing_without_distribute_still_works(tmp_path, nproc):
    """Control: every rank holds the whole mesh there, so nothing about remeshing changes."""
    proc = _run(tmp_path, [], nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    reported = _reported(proc)
    assert len(reported) == nproc, "not every rank finished:\n%s" % "\n".join(reported)
    ndofs = set()
    for line in reported:
        assert "remeshed" in line, "a rank did not remesh: %s" % line
        assert "distributed=False" in line, "the mesh should not be distributed here: %s" % line
        ndofs.add(line.split("ndof=")[1].split()[0])
    assert len(ndofs) == 1, "the ranks disagree on the size of the remeshed problem: %s" % ndofs


@pytest.mark.parametrize("failing_rank", [0, 2])
def test_failing_define_geometry_ends_the_job_instead_of_hanging(tmp_path, failing_rank):
    """One rank raises inside define_geometry while the others get through it.

    Everything the backend does afterwards is collective, so without the agreement in
    MeshedMeshTemplate._do_define_geometry the surviving ranks would still be sitting in the barriers
    of generate_mesh_to_file() and this test would time out rather than fail. Both a failing rank 0
    and a failing rank 2 are covered, since the agreement is symmetric rather than rooted at rank 0.

    The worker raises *after* get_boundary_coordinates() on purpose; see its `fail_on_rank`.
    """
    proc = _run(tmp_path, ["--distribute", "--fail-define-geometry-on-rank",
                           str(failing_rank)], nproc=3)
    assert proc.returncode != 0, "the run reported success although one rank could not remesh"
    reported = _reported(proc)
    assert len(reported) == 3, \
        "expected all three ranks to end, got:\n%s\n--- stderr tail ---\n%s" % (
            "\n".join(reported), proc.stderr[-2000:])
    assert sum("simulated failure inside define_geometry" in l for l in reported) == 1, \
        "the injected failure did not hit exactly one rank:\n%s" % "\n".join(reported)
    assert sum("Another MPI rank failed inside the define_geometry" in l for l in reported) == 2, \
        "the surviving ranks did not learn that another rank failed:\n%s" % "\n".join(reported)


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_boundary_coordinates_are_the_whole_boundary_on_every_rank(tmp_path, nproc):
    """Stage 1: get_boundary_coordinates() merges, so no rank rebuilds a truncated geometry.

    Before this, each rank saw only its own partition of the arc - at three ranks rank 0 saw none of
    it at all and the run hung. The digest is compared against a serial run of the same worker, so
    this pins the reconstructed geometry itself, not just that the ranks agree with each other.
    """
    reference = _run_serially(tmp_path / "serial", [])
    assert reference.returncode == 0, \
        "the serial reference run failed:\n%s" % reference.stdout[-2000:]
    ref = _boundaries(reference)
    assert len(ref) == 1, "the serial run did not report its boundary:\n%s" % reference.stdout[-2000:]
    expected = ref[0]

    proc = _run(tmp_path / "mpi", ["--distribute"], nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    got = _boundaries(proc)
    assert len(got) == nproc, \
        "not every rank reported a boundary, got ranks %s" % sorted(got)
    for rank, fields in sorted(got.items()):
        assert fields["digest"] == expected["digest"], (
            "rank %d rebuilt a different boundary than the serial run "
            "(%s points in %s segment(s) vs %s in %s)" % (
                rank, fields["npts"], fields["nseg"], expected["npts"], expected["nseg"]))


@pytest.mark.parametrize("nproc", [2, 3, 4])
def test_the_remeshed_problem_is_partitioned_again(tmp_path, nproc):
    """Stage 2: the rebuilt meshes are replicated, so the problem has to be distributed again.

    Without that, every rank holds the whole mesh while oomph-lib still numbers the equations as if
    it were distributed, and ndof comes out nproc times too large - which is exactly what the
    comparison against the serial run below catches.
    """
    reference = _run_serially(tmp_path / "serial", [])
    assert reference.returncode == 0, \
        "the serial reference run failed:\n%s" % reference.stdout[-2000:]
    serial = _remesh_fields(reference)[0]
    assert serial["distributed"] == "False", "the serial run reports a distributed mesh"

    proc = _run(tmp_path / "mpi", ["--distribute"], nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    got = _remesh_fields(proc)
    assert len(got) == nproc, "not every rank finished the remesh, got ranks %s" % sorted(got)
    for rank, fields in sorted(got.items()):
        assert fields["distributed"] == "True", \
            "rank %d kept a replicated mesh after remeshing" % rank
        assert fields["ndof"] == serial["ndof"], (
            "rank %d reports ndof=%s after remeshing, the serial run %s -- a factor of %s would mean "
            "the replicated mesh was numbered as if it were distributed" % (
                rank, fields["ndof"], serial["ndof"], nproc))


def test_remeshing_only_some_domains_is_refused(tmp_path):
    """A domain that is not remeshed stays partitioned, and oomph cannot distribute a mesh twice.

    Refused before the first mesh is replaced: raising once force_remesh() is under way leaves the
    problem half rebuilt, which does not even survive interpreter shutdown (it segfaulted at exit
    while this was checked at the re-distribution instead). Hence the check on the returncode: 3 is
    the worker reporting the exception, anything else means it died on the way out.
    """
    proc = _run(tmp_path, ["--distribute", "--second-domain"], nproc=2)
    assert proc.returncode == 3, \
        "expected the worker to report the refusal (3), got %d:\n--- stderr tail ---\n%s" % (
            proc.returncode, proc.stderr[-2000:])
    reported = _reported(proc)
    assert len(reported) == 2, "expected both ranks to raise:\n%s" % "\n".join(reported)
    for line in reported:
        assert "only some domains" in line and "box" in line, \
            "the refusal did not name the domain left behind: %s" % line


def test_adapting_the_remeshed_mesh_is_refused_when_asked_for_explicitly(tmp_path):
    """num_adapt>0 would leave the new mesh non-uniformly refined, which distribute() rejects.

    The default (num_adapt=None, i.e. max_refinement_level) is dropped to 0 with a printed note
    instead; only a value the caller named is refused, since that is a request about the result.
    """
    proc = _run(tmp_path, ["--distribute", "--num-adapt", "3"], nproc=2)
    assert proc.returncode == 3, \
        "expected the worker to report the refusal (3), got %d:\n--- stderr tail ---\n%s" % (
            proc.returncode, proc.stderr[-2000:])
    reported = _reported(proc)
    assert len(reported) == 2, "expected both ranks to raise:\n%s" % "\n".join(reported)
    for line in reported:
        assert "num_adapt=3" in line and "uniformly refined" in line, \
            "the refusal did not explain itself: %s" % line


@pytest.mark.parametrize("nproc", [2, 3, 4])
@pytest.mark.parametrize("variant", ["plain", "codim2", "zeta", "remesher2d"])
def test_the_transferred_field_matches_the_serial_one(tmp_path, nproc, variant):
    """The old solution reaches the new mesh even where it crossed a rank boundary.

    Each rank can only place the new nodes that fall into its own share of the old mesh. What it
    cannot place is not a failure - it is another rank's to place - so the ranks pool what each of
    them found instead of blending the rest from local nodes, which used to produce confident wrong
    values for a third of the mesh with only a warning to show for it.

The variants are the three transfer mechanisms, which fail in different ways:

    * ``plain`` - point location into the old mesh, pooled by "who found it";
    * ``codim2`` - equations where the arc meets the axis, i.e. an interface of an interface, matched
      by nearest node along the boundary. Every rank produces a match for every node there, however
      far its own nearest old node is, so pooling is "whose match is closest"; and it is the case
      where a rank holding no part of the old corner at all is ordinary;
    * ``zeta`` - the interface parameterised by arclength, which is a property of the whole curve and
      therefore has to be built on the merged interface rather than on this rank's piece of it. It
      also sends the transfer down the zeta branch of the point location;
    * ``remesher2d`` - the automatic remesher, which reconstructs the geometry from the boundaries of
      the mesh it replaces. Those are stitched together from every rank's share, since a partition is
      bounded partly by named boundaries and partly by the partition cut.

    Both halves matter: that nothing fell through to the blend at all, and that what arrived is what
    a serial run gets. The comparison is against merged global mesh data, so the numbers describe the
    whole mesh in both runs. The tolerance covers the projection solve, which is a linear solve on a
    partitioned matrix here and on a whole one serially, so its result already differs in the last
    few digits before the transfer even starts.
    """
    extra = [] if variant == "plain" else ["--" + variant]
    reference = _run_serially(tmp_path / "serial", extra)
    assert reference.returncode == 0, \
        "the serial reference run failed:\n%s" % reference.stdout[-2000:]
    serial = _field(reference)
    assert serial is not None and serial["nnode"] > 0, \
        "the serial run did not report the field:\n%s" % reference.stdout[-2000:]

    proc = _run(tmp_path / "mpi", ["--distribute"] + extra, nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    assert "could not be located" not in proc.stdout, (
        "a node fell through to the nearest-node blend, i.e. the ranks did not pool the transfer:\n%s"
        % "\n".join(l for l in proc.stdout.splitlines() if "could not be located" in l))

    got = _field(proc)
    assert got is not None, "no rank reported the field:\n%s" % proc.stdout[-2000:]
    assert got["nnode"] == serial["nnode"], \
        "the merged mesh has %g nodes, the serial one %g" % (got["nnode"], serial["nnode"])
    for key in ("usum", "usqsum", "umin", "umax", "uxsum", "uysum"):
        assert got[key] == pytest.approx(serial[key], rel=1e-6, abs=1e-9), \
            "%s is %.12g after the distributed remesh, %.12g serially" % (key, got[key], serial[key])


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("facet_space", ["DL", "D1"])
def test_a_facet_field_survives_the_distributed_remesh(tmp_path, nproc, facet_space):
    """An unknown on the interior-facet skeleton, through a remesh of a distributed problem.

    A remesh destroys the skeleton entirely and rebuilds it from the new bulk mesh, so which facets
    exist, which rank holds them and which of the holders OWNS each of them are all different
    afterwards. `meas` is the assertion that carries the weight: it is the total measure of the
    skeleton and does not involve the solution at all, so a facet that ended up enumerated on both
    of its holders inflates it by that facet's length and a dropped one deflates it - neither of which
    the solution would necessarily reveal.

    "D1" is the nodal case, and the one that reaches the DG block of the values pooled across ranks
    when a rank could not place an element itself (Mesh::share_interpolation_across_ranks): the DL/D0
    block sits BEHIND that one in the element's internal data.
    """
    args = ["--facet-field", "--facet-space", facet_space]
    reference = _run_serially(tmp_path / "serial", args, timeout=600)
    assert reference.returncode == 0, \
        "the serial reference run failed:\n%s" % reference.stdout[-2000:]
    serial = _skeleton(reference)
    assert serial is not None and serial["meas"] > 0, \
        "the serial run did not report the skeleton:\n%s" % reference.stdout[-2000:]

    proc = _run(tmp_path / "mpi", ["--distribute"] + args, nproc=nproc, timeout=900)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    fields = _remesh_fields(proc)
    assert len(fields) == nproc, "only %d of %d ranks got through the remesh" % (len(fields), nproc)
    serial_ndof = _remesh_fields(reference)[0]["ndof"]
    for rank, f in sorted(fields.items()):
        # Numbered once per holder instead of once per facet, this is larger than serial by whole
        # facets' worth of dofs.
        assert f["ndof"] == serial_ndof, \
            "rank %d has ndof %s after the remesh, %s serially -- the facet unknowns are numbered " \
            "more than once" % (rank, f["ndof"], serial_ndof)
    got = _skeleton(proc)
    assert got is not None, "no rank reported the skeleton:\n%s" % proc.stdout[-2000:]
    assert got["meas"] == pytest.approx(serial["meas"], rel=1e-9), \
        "the rebuilt skeleton measures %.12g, the serial one %.12g" % (got["meas"], serial["meas"])
    for key in ("lamsum", "lamerr2"):
        assert got[key] == pytest.approx(serial[key], rel=1e-6, abs=1e-12), \
            "%s is %.12g after the distributed remesh, %.12g serially" % (key, got[key], serial[key])
