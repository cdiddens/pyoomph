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

"""Hybrid MPI + OpenMP: threads INSIDE each rank.

The distributed assembly has its own element loop and its own scatter maps
(DistributedFrozenSparsity), so the bit-identity that tests/test_openmp_assembly.py establishes for
the serial loops says nothing about it. This runs the same comparison on each rank, in both the
distributed (--distribute) and the replicated mode.

Kept small on purpose: 2 ranks x 2 threads is 4 workers, which is what the development machines have.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_openmp_worker.py")


def _skip_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    from pyoomph import _pyoomph_core
    if not _pyoomph_core.has_openmp:
        return "this build has no OpenMP"
    return None


_SKIP = _skip_reason()
pytestmark = [pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)), pytest.mark.slow]


def _run(tmpdir, nproc, threads, distribute, timeout=900):
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER,
           "--outdir", str(tmpdir), "--threads", str(threads)]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics at all.
    env = dict(os.environ)
    session = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(session, exist_ok=True)
    env["TMPDIR"] = session
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError("mpirun did not finish within %d s -- suspect a deadlock in the threaded "
                             "distributed assembly.\n--- stdout tail ---\n%s"
                             % (timeout, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        raise AssertionError("no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n"
                             "--- stderr tail ---\n%s" % (proc.returncode, proc.stdout[-3000:],
                                                          proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (r["rank"], r["error"],
                                                                r.get("traceback", ""))
    return per_rank


_NAMES = ["residual (with Jacobian)", "Jacobian values", "Jacobian columns", "Jacobian row starts",
          "residual (on its own)"]


@pytest.mark.parametrize("distribute", [True, False], ids=["distributed", "replicated"])
def test_threaded_distributed_assembly_is_bit_identical(tmp_path, distribute):
    per_rank = _run(tmp_path, nproc=2, threads=2, distribute=distribute)
    for r in per_rank:
        assert r["threaded_runs"] > 0, (
            "rank %d never ran the threaded element loop, so its comparison proves nothing" % r["rank"])
        for name, same, diff in zip(_NAMES, r["identical"], r["maxdiff"]):
            assert same, "rank %d: %s differs by %g between 1 and 2 threads" % (r["rank"], name, diff)
