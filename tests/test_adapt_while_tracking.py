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

# Adapting the mesh while a C++ bifurcation tracker is installed.
#
# dev_docs/mpi_augmented_systems.md section 4 carried "adapt() and arclength continuation while
# tracking" as refused, "Blocked by the history-dof refusals in Problem::get_dofs(t,...)/
# set_dofs(t,...)". Both halves of that were wrong. The history dof accessors have worked distributed
# since commit 2531e00, and the thing that was actually broken was not MPI-specific and not a refusal
# at all: adapting with a tracker installed dropped the tracker SERIALLY, and the resulting failure
# ("MAXIMUM NUMBER OF ITERATIONS (10) REACHED", from a Newton solve with no tracker sitting on a
# singular Jacobian) named neither adaptation nor bifurcation tracking.
#
# The per-case reasoning is in tests/adapt_while_tracking_worker.py. What this driver adds is the
# comparison across regimes: pytest runs serially, so each test launches the worker under mpirun and
# compares against a serial run of the same worker.
#
# Why the numbers are comparable at all: the tracked critical parameter and the base ndof are global
# quantities, so every rank must report the same ones, and serial, a replicated mpirun and
# --distribute must land on the same fold. Unlike the eigenfunction adaptation in
# tests/test_mpi_eigen_adapt.py, this adaptation is driven by the BASE state's Z2 error, and on this
# smooth 1D problem every element is refined at every level -- so the meshes agree exactly too, and
# the comparison is an equality rather than a tolerance on the physics.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "adapt_while_tracking_worker.py")

_CASES = ["adapt", "solve_adapt", "arclength", "arclength_adapt"]


def _mpi_reason():
    """None if a distributed tracked run is possible here, else the reason to skip."""
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

# Serial, replicated and --distribute all solve the same discrete augmented system with the same
# direct solver, and measured they agree to the last digit or two. The fold is a double root, so the
# Newton tolerance is not the answer's accuracy (see mpi_augmented_systems.md section 11); 1e-9 is
# well inside the agreement measured at 1, 2 and 3 ranks and well outside anything structural.
_CROSS_RTOL = 1e-9
# Between ranks of ONE run there is nothing left to differ -- the same collective produced them.
_RANK_RTOL = 1e-12


def _run(nproc, tmpdir, distribute, case, adapt_levels=1, timeout=900):
    """Launch the worker under mpirun (or in-process for nproc=1) and return per-rank results."""
    outdir = os.path.join(str(tmpdir), "%s_n%d_%s_a%d" % (
        case, nproc, "dist" if distribute else "repl", adapt_levels))
    cmd = ["mpirun", "-n", str(nproc)]
    # No --oversubscribe: this project's machines have the cores these ranks need, and an
    # oversubscribed run trades a deadlock for a machine that stops responding.
    cmd += [sys.executable, _WORKER, "--case", case, "--outdir", outdir,
            "--adapt-levels", str(adapt_levels),
            # Without this only rank 0 reaches stdout (the default MPI output mode is "condensed"),
            # and a per-rank comparison would silently compare one rank with itself.
            "--mpi-output=all"]
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
        # Bounded on purpose: a refinement decision taken on one rank and not another leaves the
        # others waiting in a collective rather than returning. That has to surface as a FAILURE,
        # not as a suite that never finishes.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s case=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, case, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        marker = "PYOOMPH_MPI_RESULT "
        if marker in line:
            per_rank.append(json.loads(line[line.index(marker) + len(marker):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


_SERIAL_CACHE = {}


def _serial(tmpdir, case, adapt_levels=1):
    """The serial reference, run as its own process and cached per case.

    Its own process rather than in-process: installing a C++ bifurcation handler and then adapting
    leaves enough state behind that sharing the pytest process between cases would make one case's
    result depend on which ran before it.
    """
    key = (case, adapt_levels)
    if key in _SERIAL_CACHE:
        return _SERIAL_CACHE[key]
    res = _run(1, tmpdir, False, case, adapt_levels=adapt_levels)[0]
    _SERIAL_CACHE[key] = res
    return res


@pytest.mark.parametrize("case", ["adapt", "solve_adapt"])
def test_tracker_survives_the_adaptation(tmp_path, case):
    """The tracker must be installed again BEFORE the next Newton solve, not after it.

    This is the assertion the whole suite exists for. Before the fix, `mode_at_end` was never
    reached: the run died inside the adapting solve, because the Newton solve that oomph runs
    immediately after an adaptation is the one that used to trigger the reactivation, and without a
    tracker it sits on the fold's singular Jacobian and converges linearly at the 1/4 residual ratio
    (0.0345, 0.00749, 0.00184, ...) until the iteration cap stops it.
    """
    res = _serial(tmp_path, case)
    assert res["mode_after_tracking"] == "fold"
    assert res["mode_at_end"] == "fold", "the tracker did not survive the adaptation"
    if case == "adapt":
        # A bare adapt() does no Newton solve, so nothing downstream can repair this: the request
        # has to have been honoured by the time adapt() returns.
        assert res["mode_right_after_adapt"] == "fold", \
            "adapt() returned with the tracker off and a reactivation still pending"
        assert res["pending_right_after_adapt"] is False
        assert res["nrefined"] > 0, "nothing was refined, so nothing is being tested"


def test_the_adaptation_refines_and_moves_the_fold(tmp_path):
    """The refined mesh must actually be a different problem, and a more accurate one.

    Guards against the suite passing on an adaptation that did nothing. The base mesh doubles
    (39 -> 79 dofs, i.e. 79 -> 159 augmented), and the fold moves by ~2e-6 -- the discretisation
    error of the coarse fold, not noise, and small enough to confirm it is the same fold.
    """
    res = _serial(tmp_path, "adapt")
    assert res["ndof_tracked_fine"] > res["ndof_tracked_coarse"]
    shift = abs(res["lam_c_fine"] - res["lam_c_coarse"])
    assert 1e-9 < shift < 1e-3, \
        "the fold moved by %.3g, which is neither a refinement nor the same fold" % shift


def test_tracked_arclength_without_adaptation_is_unaffected(tmp_path):
    """The control: a fold locus with spatial_adapt=0 worked before and must still work.

    Without this, a regression that broke tracked continuation outright would still leave the
    adaptation tests above passing.
    """
    res = _serial(tmp_path, "arclength")
    assert res["mode_at_end"] == "fold"
    locus = res["locus"]
    assert len(locus) == 3
    # lam_c(b) rises monotonically along this locus, away from the b=0 starting point.
    bs = [p[0] for p in locus]
    lams = [p[1] for p in locus]
    assert bs == sorted(bs) and lams == sorted(lams), "the locus doubled back: " + str(locus)
    assert lams[0] > res["lam_c_coarse"]
    # ndof must NOT have changed: this case adapts nothing, and a mesh that moved here would mean
    # something else triggered an adaptation.
    assert res["ndof_tracked_fine"] == res["ndof_tracked_coarse"]


def test_adapting_inside_an_arclength_step_is_refused(tmp_path):
    """Still refused, and the refusal has to name the workaround that actually works.

    Measured with the guard lifted: adapting inside the step changes ndof halfway through it, the
    arclength constraint stops meaning anything, and oomph rejects the step and halves Ds for ever
    (40+ rejections down to Ds = 1e-13) rather than failing. The message is asserted because its
    recommendation -- spatial_adapt=0 then solve(spatial_adapt=N) -- was itself broken until the
    reactivation fix, and a refusal pointing at a broken workaround is worse than no refusal.
    """
    res = _serial(tmp_path, "arclength_adapt")
    assert res["refusal"] is not None, "adapting inside a tracked arclength step was not refused"
    assert "spatial_adapt=0" in res["refusal"]
    assert "solve(spatial_adapt=1)" in res["refusal"]
    # The refusal must not have disturbed the tracker on its way out.
    assert res["mode_at_end"] == "fold"


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("nproc", [2, 3])
@pytest.mark.parametrize("distribute", [False, True], ids=["replicated", "distributed"])
def test_matches_serial_on_every_partition(tmp_path, case, nproc, distribute):
    """Every rank agrees with every other, and the whole run agrees with serial.

    The tracked fold and the base ndof are global, so a partition-dependent answer here would mean
    the augmented system is being assembled differently depending on who owns which row. Both
    regimes are covered because they fail differently: replicated keeps the whole mesh on every rank
    (so a disagreement is in the augmented algebra), while --distribute partitions it (so a
    disagreement can also be in the adaptation).
    """
    per_rank = _run(nproc, tmp_path, distribute, case)
    ref = _serial(tmp_path, case)

    for r in per_rank:
        assert r["mode_after_tracking"] == ref["mode_after_tracking"]
        assert r["mode_at_end"] == ref["mode_at_end"], \
            "rank %d ended with tracking %r, serial ended with %r" % (
                r["rank"], r["mode_at_end"], ref["mode_at_end"])
        assert r["ndof_tracked_coarse"] == ref["ndof_tracked_coarse"]
        assert r["lam_c_coarse"] == pytest.approx(ref["lam_c_coarse"], rel=_CROSS_RTOL)
        if "ndof_tracked_fine" in ref:
            assert r["ndof_tracked_fine"] == ref["ndof_tracked_fine"]
        if "lam_c_fine" in ref:
            assert r["lam_c_fine"] == pytest.approx(ref["lam_c_fine"], rel=_CROSS_RTOL), \
                "rank %d found the refined fold at %.12g, serial at %.12g" % (
                    r["rank"], r["lam_c_fine"], ref["lam_c_fine"])
        if "locus" in ref:
            for (b, lam), (bref, lamref) in zip(r["locus"], ref["locus"]):
                assert b == pytest.approx(bref, rel=_CROSS_RTOL)
                assert lam == pytest.approx(lamref, rel=_CROSS_RTOL)
        if "refusal" in ref:
            # The guard is a Python-level check on a global quantity, so it must fire identically on
            # every rank. One rank raising alone would leave the others in the next collective.
            assert (r["refusal"] is None) == (ref["refusal"] is None)

    # And the ranks of this one run against each other, tighter than against serial.
    first = per_rank[0]
    for r in per_rank[1:]:
        assert r["lam_c_coarse"] == pytest.approx(first["lam_c_coarse"], rel=_RANK_RTOL)
        if "lam_c_fine" in first:
            assert r["lam_c_fine"] == pytest.approx(first["lam_c_fine"], rel=_RANK_RTOL)


@pytest.mark.parametrize("nproc,distribute", [(1, False), (2, True), (3, True)],
                         ids=["serial", "distributed_n2", "distributed_n3"])
def test_multi_level_adaptation_while_tracking(tmp_path, nproc, distribute):
    """Three adaptation levels in one solve, i.e. the reactivation happening inside oomph's loop.

    The single-level cases above never make the augmented ndof change more than once, so they would
    not catch a reactivation that works the first time and leaves oomph's multi-level adapt loop
    holding a stale dof count. Three levels take the base mesh 39 -> 319 dofs (79 -> 639 augmented).

    The fold converges with the mesh, which is the independent check that it is the same fold being
    tracked throughout rather than whatever the refined system happens to solve to:
    3.5138328738 (coarse) -> 3.5138308542 (1 level) -> 3.5138307197 (3 levels).
    """
    per_rank = _run(nproc, tmp_path, distribute, "solve_adapt", adapt_levels=3)
    one_level = _serial(tmp_path, "solve_adapt", adapt_levels=1)
    for r in per_rank:
        assert r["mode_at_end"] == "fold", "the tracker did not survive three adaptation levels"
        assert r["ndof_tracked_fine"] > one_level["ndof_tracked_fine"]
        # Monotone convergence towards the continuum fold: the three-level answer must sit on the
        # same side of the one-level answer as the one-level answer sits of the coarse one.
        d1 = one_level["lam_c_fine"] - one_level["lam_c_coarse"]
        d3 = r["lam_c_fine"] - one_level["lam_c_fine"]
        assert d1 * d3 > 0, \
            "refining further moved the fold back: coarse %.12g, 1 level %.12g, 3 levels %.12g" % (
                one_level["lam_c_coarse"], one_level["lam_c_fine"], r["lam_c_fine"])
        assert abs(d3) < abs(d1), "the refinement is not converging: |d3|=%.3g >= |d1|=%.3g" % (
            abs(d3), abs(d1))
