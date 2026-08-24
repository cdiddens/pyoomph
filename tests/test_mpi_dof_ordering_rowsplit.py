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
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

# Aligning the MPI row split with the dof layout's blocks. Marked slow; run with --full.
#
# A replicated MPI run (mpirun without --distribute) splits the Jacobian's rows uniformly and
# contiguously, by nobody's choice, so the cut points land inside whatever blocks the numbering has --
# one per rank boundary, essentially always. Two things care:
#
#   * a BLOCK PRECONDITIONER wants a node's unknowns on one rank, or the block size it is told about is
#     a lie on the ranks whose share begins mid-node;
#   * STATIC CONDENSATION cannot eliminate a component whose rows are split, because no rank holds it
#     in full -- and that is a correctness requirement, not a performance one.
#
# pyoomph::Problem::preferred_linear_solver_distribution (a pyoomph-added virtual in the vendored
# oomph-lib, consulted by create_new_linear_algebra_distribution and SuperLUSolver::solve) hands back
# a split whose cuts have been moved off the blocks. Condensation decides whenever it is on, since its
# claim is the stronger one; a dof layout is consulted only otherwise. The snapping itself is one
# helper, snap_cuts_to_blocks, shared by both.
#
# The headline result asserted here is
# test_element_layout_unblocks_replicated_cr_condensation. Crouzeix-Raviart condensation pairs the
# bubble velocity (nodal) with the DL pressure gradient modes (element-internal), and oomph numbers
# every nodal value before any internal one, so the two halves of a block sit hundreds of equations
# apart, no contiguous cut keeps one whole, and the plan builder refuses. An ElementBlockOrdering puts
# them next to each other and the refusal goes away. dev_docs/replicated_condensation_gather.md plans
# to serve the same case by gathering a straddling component's rows onto an elected owner; that plan
# rejected renumbering in its section 2.2, on the assumption that it would have to happen after
# assign_eqn_numbers() had returned. It does not -- see dev_docs/dof_ordering.md -- and this is the
# consequence. The gather is still the answer for selections that CANNOT be made contiguous, e.g. an
# interior-penalty DG one.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_dof_ordering_rowsplit_worker.py")


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

_RTOL = 1e-8


def _run(nproc, tmp_path, layout="none", condense=0, distribute=False, timeout=600):
    outdir = os.path.join(str(tmp_path), "r%d_%s_c%d_d%d" % (nproc, layout, condense, int(distribute)))
    cmd = [sys.executable, _WORKER, "--outdir", outdir, "--layout", layout, "--condense", str(condense)]
    if nproc > 1:
        cmd = ["mpirun", "-n", str(nproc)] + cmd
    if distribute:
        cmd += ["--distribute"]
    env = dict(os.environ)
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    ompi_tmp = os.path.join(str(tmp_path), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError("mpirun did not finish within %d s (nproc=%d, layout=%s, condense=%d, "
                             "distribute=%s) -- suspect a one-sided refusal.\n--- stdout tail ---\n%s"
                             % (timeout, nproc, layout, condense, distribute, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):])
                for l in proc.stdout.splitlines() if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        raise AssertionError("no results (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s"
                             % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    return per_rank


def _ok(per_rank):
    for r in per_rank:
        assert "error" not in r, "rank %d failed:\n%s" % (r["rank"], r.get("traceback", r["error"]))
    return per_rank


# ---------------------------------------------------------------------------------------------------
# The cuts
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("layout", ["nodal", "elem"])
def test_the_cuts_fall_between_the_blocks(tmp_path, nproc, layout):
    """What the split is for. Every rank computes the same cuts (they are a function of the numbering,
    which is identical on every rank replicated), and no interior cut point falls strictly inside a
    block."""
    per_rank = _ok(_run(nproc, tmp_path, layout=layout))
    for r in per_rank:
        assert r["nblocks"] > 0, "the layout produced no blocks at all"
        assert len(r["cuts"]) == nproc + 1, \
            "expected %d cut points, got %s (max block %d of ndof %d)" \
            % (nproc + 1, r["cuts"], r["maxblock"], r["ndof"])
        assert r["nstraddled"] == 0, "cuts %s cut through %d blocks" % (r["cuts"], r["nstraddled"])
        assert r["cuts"][0] == 0 and r["cuts"][-1] == r["ndof"]
        assert r["cuts"] == sorted(r["cuts"])
    ref = per_rank[0]["cuts"]
    for r in per_rank:
        assert r["cuts"] == ref, "the ranks disagree about the cuts: %s vs %s" % (r["cuts"], ref)


def test_no_layout_means_no_opinion(tmp_path):
    """Without a layout there are no blocks, so nothing is asked for and oomph's uniform split stands.
    This is the path every existing run takes and it must stay untouched."""
    for r in _ok(_run(4, tmp_path, layout="none")):
        assert r["nblocks"] == 0
        assert r["cuts"] == []


@pytest.mark.parametrize("nproc", [2, 4])
def test_a_distributed_run_asks_for_nothing(tmp_path, nproc):
    """Distributed, each rank's dofs are already one contiguous global range (synchronise_eqn_numbers)
    and the permutation was rank-local, so no block can straddle a rank and there is nothing to fix."""
    for r in _ok(_run(nproc, tmp_path, layout="elem", distribute=True)):
        assert r["distributed"] is True
        assert r["cuts"] == [], "a distributed run should state no preference, got %s" % r["cuts"]


# ---------------------------------------------------------------------------------------------------
# The payoff
# ---------------------------------------------------------------------------------------------------

def test_replicated_cr_condensation_is_refused_without_a_layout(tmp_path):
    """The state of affairs this work removes, asserted so that the next test is known to be testing
    something. The refusal is collective and names the dofs."""
    per_rank = _run(4, tmp_path, layout="none", condense=1)
    errs = [r for r in per_rank if "error" in r]
    assert errs, "expected the CR selection to be refused replicated, but it went through"
    assert "split across MPI ranks" in errs[0]["error"], errs[0]["error"]


@pytest.mark.parametrize("nproc", [2, 4])
def test_element_layout_unblocks_replicated_cr_condensation(tmp_path, nproc):
    """The headline: with the bubble velocity and the pressure gradient modes numbered together, the
    row cuts can be moved off the blocks, no component straddles, and the elimination runs -- to the
    same answer as the serial reference.

    _last_jacobian_was_condensed() is asserted because a silent decline would reproduce the reference
    perfectly while eliminating nothing."""
    serial = _ok(_run(1, tmp_path, layout="none", condense=1))[0]
    assert serial["condensed"], "the serial reference did not condense"

    per_rank = _ok(_run(nproc, tmp_path, layout="elem", condense=1))
    for r in per_rank:
        assert r["condensed"], "rank %d solved without condensing" % r["rank"]
        assert len(r["cond_cuts"]) == nproc + 1, \
            "condensation stated no row-cut preference: %s" % r["cond_cuts"]
        assert r["ndof"] == serial["ndof"]
        assert r["newton"] == serial["newton"]
        assert abs(r["checksum"] - serial["checksum"]) <= _RTOL * max(1.0, abs(serial["checksum"])), \
            "rank %d converged elsewhere: %.12g vs serial %.12g" % (r["rank"], r["checksum"], serial["checksum"])


@pytest.mark.parametrize("distribute", [False, True], ids=["replicated", "distributed"])
def test_the_layout_does_not_disturb_a_working_condensation(tmp_path, distribute):
    """Distributed condensation already worked; adding a layout must not change what it computes."""
    if not distribute:
        pytest.skip("replicated without a layout is the refusal asserted above")
    a = _ok(_run(4, tmp_path, layout="none", condense=1, distribute=True))
    b = _ok(_run(4, tmp_path, layout="elem", condense=1, distribute=True))
    assert all(r["condensed"] for r in a + b)
    ca, cb = sum(r["checksum"] for r in a), sum(r["checksum"] for r in b)
    assert abs(ca - cb) <= _RTOL * max(1.0, abs(ca)), "%.12g vs %.12g" % (ca, cb)
