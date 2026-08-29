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

# Deflated solving and deflated continuation under MPI. Serial cover is tests/test_deflation.py and
# the reasoning is dev_docs/deflation.md.
#
# Deflation does NOT use the Python custom-assembler pipeline, which still refuses nproc>1: it scales
# the assembled residual by a scalar and rescales the Newton step, both on top of the ordinary
# assembly. So the two MPI regimes exercise very little new code -- but what they do exercise is easy
# to get wrong in a way that produces a plausible wrong answer:
#
#   1. plain `mpirun`, no --distribute. Every rank holds the whole mesh and the whole dof vector, so
#      the deflation factor needs no reduction at all -- but oomph row-SPLITS the linear algebra as
#      soon as nproc>1, so the dot product grad(log M).dU in the step rescale is over a row block and
#      DOES need one, against a slice of a full-length gradient. The dof numbering is unchanged here,
#      so this case must reproduce the serial run essentially exactly.
#   2. `--distribute`. Each rank owns a block of the dofs, the gradient IS that block, and both the
#      distances and the dot product are allreduces. Two things had to be arranged for it: the
#      solver's rows are asked to be the dof rows (Problem::preferred_linear_solver_distribution --
#      oomph's default there is a uniform split, a DIFFERENT partition of the same rows, and the two
#      vectors are then not comparable entry by entry), and the PETSc backend has to remember the
#      first_row it was given at factorise time, because oomph passes 0 on the back-substitution.
#
# Everything compared across the two is numbering-independent -- solution counts and mesh integrals
# -- because distribute() renumbers the dofs.
#
# The same seed explores the same sequence of perturbations in every configuration, and the branch
# counts and lengths are asserted across --distribute because of it: the perturbation is drawn as a
# random field over the node coordinates (Problem._deflation_random_perturbation), not from dof
# indices, so renumbering the dofs does not change it. While it was drawn from dof indices, deflated
# continuation found three branches serially and one under --distribute from the same seed.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_deflation_worker.py")


def _mpi_reason():
    """None if a distributed deflation run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:
        return "petsc4py not available"
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# The solutions themselves are converged Newton solutions of the same problem, so they agree far
# better than engineering accuracy; what moves them within their own tolerance is a different MUMPS
# ordering and a different sequence of perturbations reaching them. Measured spread across
# serial / np=2 / np=2 --distribute / np=3 --distribute is ~1e-10 on the integrals.
_OBS_ATOL = 1e-7
# Replicated mpirun does not renumber anything and follows exactly the serial search, so there the
# agreement is round-off.
_REPLICATED_ATOL = 1e-12


def _run(nproc, tmpdir, distribute, case="solve", size=8, timeout=1800):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    # No --oversubscribe, and nproc small enough to fit real cores: this project's machines are not
    # to be oversubscribed, and an oversubscribed run makes any timing meaningless anyway.
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER,
           "--outdir", str(tmpdir), "--case", case, "--size", str(size)]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and
    # owns an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies with
    # exit code 1 and no diagnostics. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose. The failure mode this suite exists for is a HANG, not a crash: the
        # deflation search is a sequence of Newton solves whose success decides what happens next, so
        # a rank that disagrees about convergence sits in the next collective for ever. That has to
        # surface as a failure, not as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s case=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, case, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


def _serial_reference(tmpdir, case="solve", size=8):
    sys.path.insert(0, _HERE)
    try:
        import mpi_deflation_worker
        return mpi_deflation_worker._CASES[case](
            N=size, outdir=os.path.join(str(tmpdir), "serial_" + case))
    finally:
        sys.path.remove(_HERE)


def _solution_set(r):
    return r["solutions"] if "solutions" in r else r["final"]


def _assert_ranks_agree(per_rank):
    """Every rank must report the same solution set.

    Under --distribute the ranks hold different dof blocks and reduce to get the deflation factor, so
    a reduction skipped on one rank shows up here -- though the more likely symptom of a rank that
    took a different branch is the timeout in _run(), because the next collective never completes.
    """
    ref = json.dumps(_solution_set(per_rank[0]), sort_keys=True)
    for r in per_rank[1:]:
        assert json.dumps(_solution_set(r), sort_keys=True) == ref, \
            "rank %d reports a different solution set from rank 0" % r["rank"]


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_deflated_solve(tmp_path, nproc, distribute):
    """All three solutions of the 2D pitchfork PDE, under MPI, against the serial run.

    The symmetric pair is what this really tests: +u1 and -u1 have the SAME integral of u^2 and are
    told apart only by the signed integral, and neither is reachable at all unless the deflation
    factor and the step rescale are both right. A run that loses them still "succeeds" and returns
    the trivial solution -- which is exactly what a wrong row offset in the dot product did.
    """
    ref = _serial_reference(tmp_path, case="solve")
    assert ref["nsolutions"] == 3, "the serial reference itself did not find all three solutions"
    per_rank = _run(nproc, tmp_path, distribute, case="solve")
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    assert per_rank[0]["distributed"] is distribute
    _assert_ranks_agree(per_rank)
    got = per_rank[0]
    atol = _REPLICATED_ATOL if not distribute else _OBS_ATOL
    assert got["nsolutions"] == ref["nsolutions"], \
        "found %d solutions, serial found %d" % (got["nsolutions"], ref["nsolutions"])
    for a, b in zip(ref["solutions"], got["solutions"]):
        for k in a:
            assert abs(a[k] - b[k]) <= atol, "%s is %r, serial %r" % (k, b[k], a[k])


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_deflated_continuation(tmp_path, nproc, distribute):
    """Deflated continuation across the pitchfork: three branches, same lengths, same end points.

    More constraining than the single deflated solve, because the branch bookkeeping between
    parameter steps -- which branches survived, which are new, which two were swapped, and which
    "new" solution is really an old one re-found -- is decided from Newton successes and dof
    distances on every rank.
    """
    ref = _serial_reference(tmp_path, case="continuation", size=6)
    assert ref["nbranches"] == 3, "the serial reference itself did not find all three branches"
    per_rank = _run(nproc, tmp_path, distribute, case="continuation", size=6)
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    got = per_rank[0]
    assert got["distributed"] is distribute
    _assert_ranks_agree(per_rank)
    assert got["nbranches"] == ref["nbranches"], \
        "found %d branches, serial found %d" % (got["nbranches"], ref["nbranches"])
    assert got["branch_lengths"] == ref["branch_lengths"], \
        "branch lengths %r, serial %r" % (got["branch_lengths"], ref["branch_lengths"])
    atol = _REPLICATED_ATOL if not distribute else _OBS_ATOL
    for a, b in zip(ref["final"], got["final"]):
        for k in a:
            assert abs(a[k] - b[k]) <= atol, "%s is %r, serial %r" % (k, b[k], a[k])
