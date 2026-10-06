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

# The DISTRIBUTED dof layout of a Python augmentation (DofAugmentations), which
# tests/test_python_augmentation_layout.py covers only in its replicated form.
#
# DofAugmentations now shares AugmentedDofDistributionHelper with the C++ trackers of
# src/bifurcation.cpp, so the layout under --distribute is theirs: rank d owns its base rows, then
# its rows of each vector block, with each scalar contributing one row on rank 0 ALONE, and the whole
# thing tiling [0, n_aug) in rank order. That last detail is what makes the ranks' owned counts
# differ, and it is the cheapest way to tell a genuinely distributed layout from the replicated one.
#
# Reached through the low-level bindings rather than Problem.set_custom_assembler, which still refuses
# nproc>1 for the rest of that pipeline (the multi-assembly under it throws under MPI -- B1 of
# mpi_augmented_systems.md, still open). Nothing is assembled while the augmentation is installed, so
# none of that is in the way: the dof bookkeeping is separable and is what is tested here.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_augmentation_layout_worker.py")


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    try:
        import pymetis  # type:ignore  # noqa: F401
    except Exception:
        return "pymetis not available (needed to partition the mesh for --distribute)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(nproc, tmpdir, distribute, timeout=900):
    cmd = []
    if nproc > 1:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir)]
    if distribute:
        cmd += ["--distribute"]
    # This pytest process is itself a singleton MPI job (importing pyoomph calls MPI_Init) and owns an
    # Open MPI session directory under TMPDIR; a nested mpirun collides with it. Give the child its own.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    assert len(per_rank) == nproc, (
        "reported from %d of %d ranks (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
        % (len(per_rank), nproc, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    for r in per_rank:
        assert r.get("ok"), "rank %s failed: %s\n%s" % (r.get("rank"), r.get("error"), r.get("traceback", ""))
    return sorted(per_rank, key=lambda r: r["rank"])


def _tiles(per_rank, key):
    """Assert the per-rank (first_row, nrow_local) of `key` tile [0, n) in rank order, and return n."""
    n = per_rank[0][key][0]
    expect = 0
    for r in per_rank:
        n_r, nrow_local, first_row, _dist = r[key]
        assert n_r == n, "rank %d thinks the %s layout has %d rows, rank 0 says %d" % (r["rank"], key, n_r, n)
        assert first_row == expect, (
            "the %s row blocks do not tile: rank %d starts at %d where %d was expected (%s)"
            % (key, r["rank"], first_row, expect, [x[key] for x in per_rank]))
        expect += nrow_local
    assert expect == n, "the %s row blocks cover %d of %d rows" % (key, expect, n)
    return n


@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_augmented_layout_tiles_and_puts_scalars_on_rank_zero(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path, distribute=True)
    base_n = _tiles(per_rank, "base")
    aug_n = _tiles(per_rank, "aug")
    assert aug_n == 2 * base_n + 1, "one vector block plus one scalar over %d base dofs is 2N+1, got %d" % (base_n, aug_n)

    for r in per_rank:
        assert r["aug"][3], "rank %d's augmented layout is not distributed" % r["rank"]
        base_local = r["base"][1]
        # rank d owns: its base rows, its rows of the one vector block, and the scalar iff d == 0
        expect_local = 2 * base_local + (1 if r["rank"] == 0 else 0)
        assert r["aug"][1] == expect_local, (
            "rank %d owns %d augmented rows, expected 2*%d%s = %d"
            % (r["rank"], r["aug"][1], base_local, " + 1 (the scalar)" if r["rank"] == 0 else "", expect_local))
        # split() must hand back THIS rank's rows of the block, plus the scalar
        assert r["split_lengths"] == [base_local, 1]

    # Only rank 0 carries the scalar, so the owned counts cannot all be equal. A replicated layout
    # that slipped through would satisfy the tiling check on one rank and fail here.
    assert len({r["aug"][1] for r in per_rank}) > 1 or nproc == 1, \
        "every rank owns the same number of augmented rows, which a distributed layout with a rank-0 scalar cannot produce"


@pytest.mark.parametrize("nproc", [2, 3])
def test_the_registered_guess_lands_on_the_owning_rows(tmp_path, nproc):
    """The guess is registered at full global length; each rank must keep its own slice of it.

    The worker registers arange(ndof), so a block filled from the wrong rows -- or copied wholesale
    onto every rank, which is what the replicated branch does -- shows up directly in the values.
    """
    per_rank = _run(nproc, tmp_path, distribute=True)
    for r in per_rank:
        first_row, nrow_local = r["base"][2], r["base"][1]
        assert r["block_values"] == [float(first_row + i) for i in range(nrow_local)], (
            "rank %d's vector block does not hold rows %d..%d of the registered guess"
            % (r["rank"], first_row, first_row + nrow_local - 1))
        # The scalar is stored, not distributed, so every rank reads the registered value.
        assert r["scalar_value"] == pytest.approx(7.25)


@pytest.mark.parametrize("nproc", [2, 3])
def test_base_layout_stays_visible_and_is_restored(tmp_path, nproc):
    """While augmented, the BASE layout must still be reachable, and teardown must put it back.

    Problem::augmented_dof_distribution_helper() is what makes the first half true for a Python
    augmentation; restore_base_distribution() the second. Under --distribute the restore is a pointer
    swap back, not a non-distributed rebuild -- which is what the old code did unconditionally and
    which would have flattened a distributed base layout.
    """
    per_rank = _run(nproc, tmp_path, distribute=True)
    for r in per_rank:
        assert r["aug_base"] == r["base"], (
            "rank %d: the base layout reads %s while augmented, but was %s before" % (r["rank"], r["aug_base"], r["base"]))
        assert r["restored"] == r["base"], (
            "rank %d: after teardown the layout is %s, not the original %s" % (r["rank"], r["restored"], r["base"]))


def test_replicated_augmentation_keeps_the_historical_layout(tmp_path):
    """Plain mpirun: every rank holds everything, non-distributed, exactly as before the move."""
    per_rank = _run(2, tmp_path, distribute=False)
    for r in per_rank:
        base_n = r["base"][0]
        assert r["base"] == [base_n, base_n, 0, False]
        assert r["aug"] == [2 * base_n + 1, 2 * base_n + 1, 0, False]
        assert r["split_lengths"] == [base_n, 1]
        assert r["block_values"] == [float(i) for i in range(base_n)]


@pytest.mark.parametrize("nproc,distribute", [(2, True), (3, True), (2, False)])
def test_the_base_state_is_untouched_by_an_augmentation(tmp_path, nproc, distribute):
    """Installing and tearing down an augmentation must not move the solution.

    The augmented dofs are appended to Dof_pt and the distribution is swapped; if any of that reached
    the base dofs, the mesh integral would move, and re-solving afterwards would land somewhere else.
    """
    per_rank = _run(nproc, tmp_path, distribute=distribute)
    ref = per_rank[0]["usqr"]
    for r in per_rank:
        assert r["usqr_after"] == pytest.approx(r["usqr"], rel=1e-14), "the augmentation moved the base state"
        assert r["usqr_resolved"] == pytest.approx(r["usqr"], rel=1e-10), "re-solving after teardown gave a different answer"
        assert r["usqr"] == pytest.approx(ref, rel=1e-12), "the ranks disagree about the base state"
