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

# The MPI half of tests/test_dof_ordering.py. Marked slow; run with --full.
#
# pyoomph::Problem::reorder_global_eqn_numbers is called from one place, and the claim made for that
# place (src/thirdparty/INFO_oomph-lib) is that ONE implementation serves both MPI modes:
#
#   REPLICATED (mpirun without --distribute): every rank holds the whole mesh, the equation numbers at
#     the hook are already the final global ones, and the permutation is the whole numbering. Every
#     rank must compute the same permutation, which it does because the mode and the mesh are the same
#     on all of them.
#   DISTRIBUTED (--distribute): the equation numbers at the hook are still rank-local (0..my_n-1, halo
#     data being Is_pinned), and synchronise_eqn_numbers() shifts them by this rank's base immediately
#     afterwards. A rank-local permutation therefore leaves each rank's GLOBAL range contiguous, which
#     the distributed assembly and the static-condensation row ownership both require. Nothing here
#     checks contiguity directly; what it checks is that the answer survives, which is what a broken
#     range would destroy.
#
# The "reverse" layout is used because it is a genuine bijection that moves nearly every dof, so it is
# the strongest null hypothesis available for the real layouts built on the same machinery.
#
# Comparisons are within one (nproc, distribute) configuration only. The per-rank checksums are sums
# over the nodes a rank holds, and replicated every rank holds all of them while distributed the halo
# nodes are counted more than once -- so the totals differ BETWEEN configurations by design, and only
# the reordered-vs-default pair within a configuration is meaningful.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_dof_ordering_worker.py")


def _mpi_reason():
    """None if an MPI run is possible here, else the reason to skip."""
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

_RTOL = 1e-8


def _run(nproc, tmpdir, mode, distribute, timeout=600):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    outdir = os.path.join(str(tmpdir), "run_%d_%s_%s" % (nproc, mode or "none", int(distribute)))
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, "--outdir", outdir, "--mode", mode]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a singleton MPI job owning an
    # Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies with exit 1 and
    # no diagnostics. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: a decision taken on one rank alone leaves the others in a collective for
        # ever, and that has to surface as a failure rather than as a suite that never returns.
        raise AssertionError(
            "mpirun did not finish within %d s (nproc=%d, mode=%r, distribute=%s).\n"
            "--- stdout tail ---\n%s" % (timeout, nproc, mode, distribute, (e.stdout or "")[-3000:]))
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
        assert "error" not in r, "rank %d failed:\n%s" % (r["rank"], r.get("traceback", r.get("error")))
    return per_rank


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("distribute", [False, True], ids=["replicated", "distributed"])
def test_reordering_does_not_change_the_answer(tmp_path, nproc, distribute):
    plain = _run(nproc, tmp_path, "", distribute)
    rev = _run(nproc, tmp_path, "reverse", distribute)

    for r in plain + rev:
        assert r["distributed"] is distribute, \
            "rank %d reports distributed=%s, expected %s" % (r["rank"], r["distributed"], distribute)

    # ndof is global and must be identical: a permutation may not change WHICH values are dofs.
    assert {r["ndof"] for r in plain} == {r["ndof"] for r in rev}

    # Newton must take the same route, on every rank.
    assert sorted(r["newton"] for r in plain) == sorted(r["newton"] for r in rev)

    for key in ("checksum", "sqsum"):
        a = sum(r[key] for r in plain)
        b = sum(r[key] for r in rev)
        assert abs(a - b) <= _RTOL * max(1.0, abs(a)), \
            "%s moved under reordering: %.12g -> %.12g (nproc=%d, distributed=%s)" \
            % (key, a, b, nproc, distribute)
