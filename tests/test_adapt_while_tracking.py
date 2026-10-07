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


def _run(nproc, tmpdir, distribute, case, adapt_levels=1, reverse=False, inner_product=None,
         timeout=900):
    """Launch the worker under mpirun (or in-process for nproc=1) and return per-rank results."""
    outdir = os.path.join(str(tmpdir), "%s_n%d_%s_a%d%s_%s" % (
        case, nproc, "dist" if distribute else "repl", adapt_levels, "_rev" if reverse else "",
        inner_product or "none"))
    cmd = ["mpirun", "-n", str(nproc)]
    # No --oversubscribe: this project's machines have the cores these ranks need, and an
    # oversubscribed run trades a deadlock for a machine that stops responding.
    cmd += [sys.executable, _WORKER, "--case", case, "--outdir", outdir,
            "--adapt-levels", str(adapt_levels)] + (["--reverse"] if reverse else []) + (
            ["--inner-product", inner_product] if inner_product else []) + [
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


def _serial(tmpdir, case, adapt_levels=1, reverse=False, inner_product=None):
    """The serial reference, run as its own process and cached per case.

    Its own process rather than in-process: installing a C++ bifurcation handler and then adapting
    leaves enough state behind that sharing the pytest process between cases would make one case's
    result depend on which ran before it.
    """
    key = (case, adapt_levels, reverse, inner_product)
    if key in _SERIAL_CACHE:
        return _SERIAL_CACHE[key]
    res = _run(1, tmpdir, False, case, adapt_levels=adapt_levels, reverse=reverse,
               inner_product=inner_product)[0]
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


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_locus_survives_the_recommended_workaround(tmp_path, reverse):
    """Two locus steps, an adapting solve, two more: the path the arclength refusal recommends.

    This is the one reachable path on which the carried arclength tangent is discarded -- not by the
    dead "strip the tracker part" branch, but by activate_bifurcation_tracking's own
    reset_arc_length_parameters() when the reactivation runs. Measured, that loss costs nothing
    (dev_docs/mpi_augmented_systems.md section 14), so what is asserted here is the CONTRACT that has
    to hold either way, rather than the current behaviour: if someone implements the carry, this
    suite should keep passing.

    Both directions, because a lost tangent's classic failure is a continuation that walks the wrong
    way: the recomputed tangent must take its sign from ds and nothing else.
    """
    res = _serial(tmp_path, "locus_then_adapt", reverse=reverse)

    assert res["mode_at_end"] == "fold", "the tracker did not survive the adapting solve"
    assert res["ndof_after_adapt"] > res["ndof_before_adapt"], "nothing was refined"
    # The tangent going in is a proper one; this is the baseline the carry would have to preserve.
    assert res["invariant_before_adapt"] == pytest.approx(0.0, abs=1e-12)

    pre, post = res["locus_pre"], res["locus_post"]
    sign = -1.0 if reverse else 1.0
    # The pre-adapt steps always run forward -- ds is only flipped AFTER the adaptation -- so they
    # are checked unsigned, and the signed check starts at the junction: pre's last point is where
    # the post-adapt direction is taken from.
    pre_bs = [p[0] for p in pre]
    for earlier, later in zip(pre_bs, pre_bs[1:]):
        assert later > earlier, "the locus did not advance before the adaptation: %s" % pre_bs
    post_bs = [pre_bs[-1]] + [p[0] for p in post]
    for earlier, later in zip(post_bs, post_bs[1:]):
        assert sign * (later - earlier) > 0, \
            "the continuation went the wrong way after the adaptation (reverse=%s): b went %s" % (
                reverse, post_bs)
    # And it is still the fold locus: lam_c(b) rises with b on this problem, so lam must move with b.
    post_lams = [pre[-1][1]] + [p[1] for p in post]
    for earlier, later in zip(post_lams, post_lams[1:]):
        assert sign * (later - earlier) > 0, \
            "the continuation left the locus (reverse=%s): lam_c went %s" % (reverse, post_lams)

    # ds must not have been restarted or collapsed by the adaptation: oomph halves it on a rejected
    # step, so a shrinking sequence here is the signal that the step after the adapt went badly.
    assert all(abs(b) > abs(a) for a, b in zip(res["ds_post"], res["ds_post"][1:])), \
        "ds stopped growing after the adaptation: " + str(res["ds_post"])
    # Whatever the tangent's provenance, the one in force at the end must satisfy the constraint.
    assert res["invariant_end"] == pytest.approx(0.0, abs=1e-12)


# Serial vs --distribute on the carried-tangent path. This is _CROSS_RTOL and not something looser
# BECAUSE the tracker block is recomputed rather than carried: compute_arclength_tangent solves the
# assembled augmented system, which is partition-independent.
#
# It did not start that way. While the tangent was carried whole -- base block interpolated, tracker
# block zero -- this comparison needed 1e-3: the INTERPOLATED tangent inherits oomph-lib's
# distributed Z2 recovery, which neglects the flux contributions of patches assembled only from
# vertex nodes owned by another process, so |dU/ds| differed by 4.1e-7 serial vs --distribute and the
# arclength constraint amplified that to ~2.4e-5 in where the next step landed. Recomputing the
# tangent removed it: measured 4e-12 at 2 and 3 ranks. Kept as a comment because a future change that
# goes back to carrying the tracker block will have to loosen this again, and should know why.


@pytest.mark.parametrize("nproc,distribute", [(2, True), (3, True)],
                         ids=["distributed_n2", "distributed_n3"])
def test_locus_workaround_matches_serial_distributed(tmp_path, nproc, distribute):
    """The same path under --distribute: the locus is physics, so it cannot depend on the partition.

    Two separate claims, with deliberately different tolerances:

      * the ranks of ONE run agree EXACTLY. Everything reported is global, so a disagreement here
        would mean the restored tangent was assembled differently depending on who owns which row --
        which is what the zero-padding could plausibly get wrong, since it assumes the tracker
        unknowns are the last global indices. Measured bit-identical at 2 and 3 ranks.

      * the run agrees with serial on the LOCUS, to _CROSS_RTOL -- an equality in practice, because
        the tracker block is recomputed from the assembled system rather than interpolated. See the
        comment above this test for what it took to get there.
    """
    per_rank = _run(nproc, tmp_path, distribute, "locus_then_adapt")
    ref = _serial(tmp_path, "locus_then_adapt")

    for r in per_rank:
        assert r["mode_at_end"] == ref["mode_at_end"]
        assert r["ndof_after_adapt"] == ref["ndof_after_adapt"]
        # The carry must have happened: a tangent of length 0 here means the restore silently did
        # nothing on this partition, which is exactly the regression this path is about.
        assert r["len_tangent_after_adapt"] == r["ndof_after_adapt"], \
            "the carried tangent was not restored at %d ranks: length %d, ndof %d" % (
                nproc, r["len_tangent_after_adapt"], r["ndof_after_adapt"])
        assert r["invariant_end"] == pytest.approx(0.0, abs=1e-12)
        for (b, lam), (bref, lamref) in zip(r["locus_post"], ref["locus_post"]):
            assert b == pytest.approx(bref, rel=_CROSS_RTOL)
            assert lam == pytest.approx(lamref, rel=_CROSS_RTOL)

    # Rank against rank, exactly.
    first = per_rank[0]
    for r in per_rank[1:]:
        for (b, lam), (b0, lam0) in zip(r["locus_post"], first["locus_post"]):
            assert b == pytest.approx(b0, rel=_RANK_RTOL)
            assert lam == pytest.approx(lam0, rel=_RANK_RTOL)


@pytest.mark.parametrize("inner_product", ["ndof", "l2"])
def test_arclength_metric_survives_a_tracked_adaptation(tmp_path, inner_product):
    """theta^2 across a tracked adaptation, with a configured inner product.

    The rest of this suite runs at theta^2 = 1 (set_arc_length_parameter(scale_arc_length=False)),
    where capturing and restoring theta^2 is a no-op -- which is exactly why a restore that put the
    tangent back WITHOUT theta^2 looked exact here and was 3.7e-5 off the arclength constraint under
    an l2 inner product. Hence this case.

    Two things have to hold, and they pull in opposite directions:

      * the theta^2 restored with the tangent is the OLD mesh's, because that is the value the
        carried tangent is normalised against -- so the pair is coherent the moment it goes back and
        the invariant is satisfied immediately.

      * theta^2 must then CHANGE, because it is ndof-dependent for "ndof" and mass-matrix-dependent
        for "l2", and the adaptation changed both. _retune_arclength_theta() at the top of the next
        arclength step does that and renormalises the tangent with it.

    Serial only: _retune_arclength_theta refuses a distributed problem with a configured inner
    product outright, because the norms would be taken over each rank's local dofs.
    """
    res = _serial(tmp_path, "locus_then_adapt", inner_product=inner_product)

    assert res["mode_at_end"] == "fold"
    assert res["ndof_after_adapt"] > res["ndof_before_adapt"]
    assert res["len_tangent_after_adapt"] == res["ndof_after_adapt"], "the tangent was not restored"

    # theta^2 is genuinely not 1, or this case is testing nothing.
    assert res["theta_sqr_before_adapt"] != pytest.approx(1.0), \
        "theta^2 is 1, so this case cannot see a theta^2 that was reset to 1"

    # Restored coherently with the tangent. This is the assertion that fails on a tangent-only
    # restore: measured 3.7e-5 under l2.
    assert res["theta_sqr_after_adapt"] == pytest.approx(res["theta_sqr_before_adapt"], rel=1e-12)
    assert res["invariant_after_adapt"] == pytest.approx(0.0, abs=1e-12), \
        "the restored tangent does not satisfy the arclength constraint in the metric now in force"

    # And then re-derived for the new mesh by the next step, not left at the old value.
    assert res["theta_sqr_end"] != pytest.approx(res["theta_sqr_after_adapt"], rel=1e-6), \
        "theta^2 was not re-derived after the mesh changed: still %.12g" % res["theta_sqr_end"]
    assert res["invariant_end"] == pytest.approx(0.0, abs=1e-12)

    if inner_product == "ndof":
        # The mechanism, exactly: this metric IS 1/ndof, so the refinement has to halve it.
        assert res["theta_sqr_before_adapt"] == pytest.approx(1.0/res["ndof_before_adapt"], rel=1e-12)
        assert res["theta_sqr_end"] == pytest.approx(1.0/res["ndof_after_adapt"], rel=1e-12)
    else:
        # The l2 metric is a mean square over the refined mesh; it must move in the same direction
        # and by a comparable amount, without being pinned to an exact formula.
        ratio = res["theta_sqr_end"]/res["theta_sqr_before_adapt"]
        assert 0.1 < ratio < 1.0, "the l2 metric moved implausibly across the adaptation: %.4g" % ratio
