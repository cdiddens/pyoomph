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

# Periodic orbit tracking under a plain mpirun, and in particular PeriodicOrbit.output_orbit().
#
# output_orbit() writes the orbit into a subdirectory by moving the problem's output directory there
# and back again. _ODEFileOutput.change_output_directory() reacted to that by closing its file --
# unconditionally, although init() opens one on rank 0 alone. Every other rank raised
# AttributeError: 'NoneType' object has no attribute 'close' and unwound, while rank 0 walked on
# into the continuation and stopped in its next collective: the run HUNG rather than failed, which
# is why docs/source/tutorial/temporal/orbit/manual_orbit.py was reported as "stuck" under mpirun.
#
# There is no other orbit coverage under MPI, so this also pins the plain "does orbit tracking work
# on more than one rank" question: the period of the Stuart-Landau limit cycle is 2*pi exactly.
#
# And it is the gate for a second, unrelated deadlock, which it is worth knowing this file catches.
# The problem is ONE element, so on more than one rank the replicated element range leaves some ranks
# with nothing to assemble. A rank with no elements never asks for a symbolic sparsity mask and so
# never discovers that the orbit handler's block has none, while the rank holding the element does --
# and if that discovery returns instead of being voted on, one rank walks back into oomph-lib's
# assembly while the other enters MPI_Alltoall. See dev_docs/structural_assembly.md.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_orbit_output_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(nproc, tmpdir):
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, "--outdir", str(tmpdir)]
    # Own TMPDIR for the nested mpirun; see tests/test_mpi_bifurcation_tracking.py for why.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        # The runs take a few seconds; the bound is generous but not so generous that a
        # re-introduced deadlock takes a quarter of an hour to report itself.
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired as e:
        # The failure this test exists for presents as a hang, not as a non-zero exit: the ranks
        # that raise leave rank 0 waiting. So a timeout has to be a failure, not an inconclusive run.
        raise AssertionError(
            "mpirun did not finish within 300 s -- suspect ranks that died while rank 0 waited, or a\n"
            "rank that left a collective the others were still in (this problem has ONE element, so\n"
            "some ranks assemble nothing at all)."
            "\n--- stdout tail ---\n%s" % ((e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


@pytest.mark.parametrize("nproc", [2, 3])
def test_orbit_output_survives_mpirun(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path)
    ref = per_rank[0]
    for r in per_rank[1:]:
        for key in ("ndof", "T", "orbit_dir_exists", "orbit_rows", "base_rows_after"):
            assert r[key] == ref[key], "ranks disagree on %s: rank 0 %r vs rank %d %r" % (
                key, ref[key], r["rank"], r[key])
    # The orbit itself: the Stuart-Landau limit cycle has period 2*pi. Loose enough for the
    # discretisation (24 b-spline samples), tight enough that a wrong orbit fails.
    assert abs(ref["T"] - 2 * numpy.pi) < 1e-6, "period %r, expected 2*pi" % ref["T"]
    # output_orbit() wrote the subdirectory, one row per sample...
    assert ref["orbit_dir_exists"], "output_orbit() did not create its subdirectory"
    assert ref["orbit_rows"] == 24, "orbit output has %d rows, expected 24" % ref["orbit_rows"]
    # ...and the original file is writable again afterwards, i.e. it was reopened, not left closed.
    assert ref["base_rows_after"] >= 1, "the ODE output was not reopened after output_orbit()"
