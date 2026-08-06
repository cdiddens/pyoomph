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

# Remeshing under --distribute. See dev_docs/distributed_remeshing.md for the full picture; the
# short version is that it does not work yet and, before the refusal these tests pin down, it did
# not say so either:
#
# * at two ranks every rank rebuilt the geometry from its own partition of the boundary, and since
#   only rank 0's .msh file is kept, rank 0's truncated wedge silently became the mesh for all of
#   them. The run then finished with a mesh replicated on every rank while the problem still
#   numbered the equations as if it were distributed, i.e. ndof came out nproc times too large;
# * at three and four ranks the rank holding no element of the boundary raised inside
#   define_geometry() while the others walked into the barriers of generate_mesh_to_file(), and the
#   job hung. At three ranks that rank is rank 0.
#
# So: force_remesh() refuses on a distributed problem, and MeshedMeshTemplate._do_define_geometry()
# agrees across ranks on whether define_geometry() succeeded. The last test is the one that would
# hang without the latter, which is why every run here has a timeout that fails rather than waits.

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


def _boundaries(proc):
    """What each rank got out of get_boundary_coordinates(), as {rank: line}."""
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_BOUNDARY "):
            fields = dict(f.split("=", 1) for f in line.split()[1:])
            out[int(fields["rank"])] = fields
    return out


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("remesher", ["recreation", "remesher2d"])
def test_distributed_remeshing_is_refused_on_every_rank(tmp_path, nproc, remesher):
    args = ["--distribute"] + (["--remesher2d"] if remesher == "remesher2d" else [])
    proc = _run(tmp_path, args, nproc=nproc)
    assert proc.returncode != 0, \
        "the run reported success although remeshing distributed does not work:\n%s" % (
            "\n".join(_reported(proc)))
    reported = _reported(proc)
    assert len(reported) == nproc, \
        "expected all %d ranks to raise, got:\n%s\n--- stderr tail ---\n%s" % (
            nproc, "\n".join(reported), proc.stderr[-2000:])
    for line in reported:
        assert "not supported on a distributed" in line, "unhelpful failure on a rank: %s" % line
        # The message has to name the path that is broken, since the two fail for different reasons
        # and are fixed separately.
        expected = "Remesher2d:" if remesher == "remesher2d" else "RemesherViaRecreation:"
        assert expected in line, "the refusal did not name the remesher: %s" % line


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
    proc = _run(tmp_path, ["--distribute", "--force", "--fail-define-geometry-on-rank",
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

    proc = _run(tmp_path / "mpi", ["--distribute", "--force"], nproc=nproc)
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
