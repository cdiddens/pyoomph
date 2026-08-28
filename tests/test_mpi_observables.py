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

# Two things that describe the whole problem and used to be answered per rank.
#
# * ExtremumObservables. Mesh::evaluate_extremum samples the elements THIS rank holds, so every rank
#   reported the extremum of its own partition. In docs/source/tutorial/ale/rayleigh_plateau.py that
#   observable sets the time step (dt = 0.1 * r_min), so the ranks marched to different times: the run
#   did not fail, it quietly stopped being one simulation.
# * Problem.create_text_file_output. Every rank opened and wrote the same file, and the rows
#   interleaved mid-number - beads_on_string's minimum.txt came out as
#   "10.0\t0.8511691410.0\t0.85116914...".
#
# The observables here are functions of the coordinates rather than of a solved field on purpose: a
# projection solve is not bit-reproducible between a serial run and an mpirun, and what is under test
# is the reduction.

import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_observables_worker.py")

_FIELDS = ("lo", "hi", "lox", "loy", "hix", "hiy", "metric", "edge")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
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


def _run(tmpdir, extra_args, nproc=None, timeout=300):
    cmd = []
    if nproc is not None:
        cmd = ["mpirun", "-n", str(nproc)]
        if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
            cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir)] + extra_args
    env = dict(os.environ)
    if nproc is not None:
        # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
        # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
        ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
        os.makedirs(ompi_tmp, exist_ok=True)
        env["TMPDIR"] = ompi_tmp
    return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)


def _reported(proc):
    out = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("PYOOMPH_MPI_RESULT "):
            continue
        fields = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
        out[int(fields["rank"])] = fields
    return out


@pytest.mark.parametrize("nproc", [2, 3, 4])
@pytest.mark.parametrize("distribute", [False, True])
def test_extrema_are_taken_over_the_whole_mesh(tmp_path, nproc, distribute):
    """Every rank must report the extremum of the whole mesh, not of its own partition.

    Also covers the case the dimensional value has to survive: at four ranks a rank can hold no
    element of the "top" boundary at all, so the unit cannot be read off a local element there and
    comes from the registered expression instead.
    """
    reference = _run(tmp_path / "serial", [])
    assert reference.returncode == 0, "the serial reference run failed:\n%s" % reference.stdout[-2000:]
    serial = _reported(reference)[0]

    args = ["--distribute"] if distribute else []
    proc = _run(tmp_path / "mpi", args, nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    got = _reported(proc)
    assert len(got) == nproc, "not every rank reported, got ranks %s" % sorted(got)
    for rank, fields in sorted(got.items()):
        for key in _FIELDS:
            assert float(fields[key]) == pytest.approx(float(serial[key]), rel=1e-12, abs=1e-14), \
                "rank %d reports %s=%s, the serial run %s" % (rank, key, fields[key], serial[key])


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("distribute", [False, True])
def test_a_text_output_file_is_written_once(tmp_path, nproc, distribute):
    """One header and one set of rows, not one per rank interleaved into the same file."""
    args = ["--distribute"] if distribute else []
    proc = _run(tmp_path, args, nproc=nproc)
    assert proc.returncode == 0, \
        "the run failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            proc.stdout[-2000:], proc.stderr[-2000:])
    with open(os.path.join(str(tmp_path), "rows.txt")) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    assert lines[0].startswith("#"), "no header in the file:\n%s" % "\n".join(lines[:4])
    assert len(lines) == 4, \
        "expected one header and three rows, got %d lines - the ranks wrote over each other:\n%s" % (
            len(lines), "\n".join(lines[:8]))
    for i, line in enumerate(lines[1:]):
        assert line.split("\t") == [str(i), str(10.0 * i)], "row %d is garbled: %r" % (i, line)
