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

# Problem.find_bifurcation_via_eigenvalues: that it converges, that it works from either side of the
# axis, and the optional eigenvector tracking.
#
# The search did not converge. It kept no bracket -- only the last sign and a ds it rescaled -- so
# after the first sign change it ran a damped recursion with a fixed point of its own rather than a
# bracketed root-find. Measured before the fix, on the Hopf of the ramped Brusselator: it settled at
# B = 2.39063 with Re = +1.38e-3 against a root at 2.3875, stopped moving, and span until the
# collapsing arclength step killed the continuation with an OomphException.
#
# Three bugs compounded there, which is why the fix is a rewrite of the bracketed phase rather than
# a tweak:
#
#   1. no bracket was kept at all, so nothing forced the iteration towards the root;
#   2. ds is an ARCLENGTH, not a parameter increment -- the parameter moves by (dparameter/ds)*ds --
#      so halving ds does not halve the parameter step, and ds's sign does not decide the direction
#      along the branch (Continuation_direction does);
#   3. ds was re-read from arclength_continuation's RETURN value, which is the step oomph suggests
#      for next time, not the one just taken.
#
# What replaced it is false position on the PARAMETER with Illinois damping, converting to ds via
# dparameter/ds. The damping is load-bearing: plain regula falsi retains the same end repeatedly on
# a convex bracket and creeps towards the root from one side, which is the shape of the original
# failure.
#
# The unstable start was a separate, deliberate refusal ("Starting already with an unstable
# solution"). Nothing in the search needs it: a sign CHANGE is as detectable from above the axis as
# from below.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "bifurcation_scan_worker.py")

# The Hopf of the ramped Brusselator, from an independent bisection that does not use this code
# path at all (tests/adapt_while_tracking_worker.py's _run_hopf_locus brackets it by hand and then
# lets the Hopf TRACKER converge onto it): B_c = 2.38750181, omega = 1.0374478.
_B_C = 2.3875018
_OMEGA = 1.0374478


def _reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support"
    except Exception:
        return "petsc4py not available"
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP = _reason()
pytestmark = [pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)), pytest.mark.slow]

_CACHE = {}


def _scan(tmpdir, *, start_B=1.5, initstep=0.08, epsilon=1e-7, neigen=6, track=False,
          stay_stable=False, continuation_data_in_states=False):
    key = (start_B, initstep, epsilon, neigen, track, stay_stable, continuation_data_in_states)
    if key in _CACHE:
        return _CACHE[key]
    outdir = os.path.join(str(tmpdir), "scan_%s" % abs(hash(key)))
    cmd = [sys.executable, _WORKER, "--outdir", outdir,
           "--start-B", repr(start_B), "--initstep", repr(initstep),
           "--epsilon", repr(epsilon), "--neigen", str(neigen)]
    if track:
        cmd.append("--track")
    if stay_stable:
        cmd.append("--stay-stable")
    if continuation_data_in_states:
        cmd.append("--continuation-data-in-states")
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=1800)
    got = None
    for line in proc.stdout.splitlines():
        marker = "PYOOMPH_SCAN_RESULT "
        if marker in line:
            got = json.loads(line[line.index(marker) + len(marker):])
    if got is None:
        raise AssertionError("no result (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s"
                             % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert "traceback" not in got, got.get("traceback")
    _CACHE[key] = got
    return got


# ---------------------------------------------------------------------------------------------
# The overlap measure itself. Pure arithmetic, so no fixture and no solver.
# ---------------------------------------------------------------------------------------------

def test_eigenvector_overlap_is_scale_and_phase_invariant():
    """An eigenvector is defined up to a complex factor, so the measure must ignore one.

    Three distinct ambiguities, and they need distinct treatment:

      * SCALE. |<a,b>|/(|a||b|) handles it: rescaling multiplies numerator and denominator alike.
      * PHASE. Only the MODULUS of the Hermitian product is invariant; the real part alone would
        report two copies of one mode differing by a phase of pi as perfectly anti-correlated.
      * CONJUGATION. conj(v) is NOT a rescaling of v, so it needs its own branch -- a solver may
        return either member of a complex pair, which is the same reading as
        mpi_augmented_systems.md section 11's "compare |omega|".
    """
    from pyoomph import Problem
    ov = Problem.eigenvector_overlap
    rng = numpy.random.default_rng(0)
    v = rng.normal(size=64) + 1j*rng.normal(size=64)

    assert ov(v, v) == pytest.approx(1.0, abs=1e-14)
    for factor in (1e7, 1e-7, -1.0, numpy.exp(1.3j), 3.7*numpy.exp(-0.4j)):
        assert ov(v, factor*v) == pytest.approx(1.0, abs=1e-12), "not invariant under %r" % factor
    assert ov(v, numpy.conjugate(v)) == pytest.approx(1.0, abs=1e-12)
    assert ov(v, 3.7*numpy.exp(-0.4j)*numpy.conjugate(v)) == pytest.approx(1.0, abs=1e-12)

    # A real eigenvector is its own conjugate, so the two branches must agree rather than one of
    # them winning by accident.
    r = rng.normal(size=32)
    assert ov(r, numpy.conjugate(r)) == pytest.approx(1.0, abs=1e-14)

    # Unrelated modes read low. Random complex vectors in n dimensions overlap at ~1/sqrt(n), so
    # this is a real statement only because n is large enough: at n=64 the expectation is ~0.12,
    # which is why the default tolerance of 0.5 is not a coin toss.
    w = rng.normal(size=64) + 1j*rng.normal(size=64)
    assert ov(v, w) < 0.4

    # Orthogonal is exactly zero; the degenerate inputs report "tells us nothing" rather than raise.
    assert ov(numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 1.0, 0.0])) == 0.0
    assert ov(v, 0*v) == 0.0
    assert ov(v, v[:8]) == 0.0


# ---------------------------------------------------------------------------------------------
# The search.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("epsilon", [1e-3, 1e-7, 1e-9])
def test_the_search_converges_to_the_root(tmp_path, epsilon):
    """It must reach the root, at whatever tolerance is asked for.

    This is the regression. Before the fix every one of these tolerances stalled at Re = +1.38e-3
    and then raised OomphException from the collapsing arclength step; only epsilon >= ~1e-2
    "worked", and then by stopping wherever the march happened to be rather than at the root.
    """
    res = _scan(tmp_path, epsilon=epsilon)
    assert res["outcome"] == "converged", res.get("error", res["outcome"])
    assert abs(res["real_part"]) < epsilon, \
        "returned Re = %.3g for epsilon = %.3g" % (res["real_part"], epsilon)
    # And at the right place, against a value obtained without this code path.
    assert res["B"] == pytest.approx(_B_C, rel=1e-4)
    assert res["omega"] == pytest.approx(_OMEGA, rel=1e-4)


def test_tightening_the_tolerance_costs_steps_but_still_converges(tmp_path):
    """Monotone in the obvious direction, and bounded.

    A bracketed root-find on a near-linear function should take a handful of extra steps per extra
    digit, not fail and not spin. The old code's step count was unbounded by construction, because
    it was not converging at all.
    """
    loose = _scan(tmp_path, epsilon=1e-3)
    tight = _scan(tmp_path, epsilon=1e-7)
    assert loose["steps"] <= tight["steps"]
    assert tight["steps"] < 60, "took %d steps, which is not a bracketed search" % tight["steps"]
    # Both land on the same root; the tighter one just resolves it better.
    assert loose["B"] == pytest.approx(tight["B"], rel=1e-3)


def test_searching_from_an_unstable_start(tmp_path):
    """Marching DOWN from an unstable solution must find the same bifurcation.

    This was refused outright ("Starting already with an unstable solution"). The search needs a
    sign change, not a stable start, so a user who has walked past a bifurcation can come back to it
    instead of re-approaching from the other side.
    """
    res = _scan(tmp_path, start_B=2.6, initstep=-0.08)
    assert res["start_real_part"] > 0.0, "the start was supposed to be unstable"
    assert res["outcome"] == "converged", res.get("error", res["outcome"])
    assert res["B"] == pytest.approx(_B_C, rel=1e-4)

    # The same root as from the stable side, to the tolerance both were asked for -- the two
    # approach it from opposite directions, so agreement here is a statement about the root and not
    # about the path.
    from_below = _scan(tmp_path, start_B=1.5, initstep=0.08)
    assert res["B"] == pytest.approx(from_below["B"], rel=1e-6)
    assert res["omega"] == pytest.approx(from_below["omega"], rel=1e-6)


def test_eigenvector_tracking_does_not_change_a_healthy_search(tmp_path):
    """track_eigenvector must be inert when the index was right all along.

    It is off by default and exists to stop a lost mode being silently swapped for another. On a
    branch where the followed index stays the correct mode -- measured: overlap 1.000 at every step
    of this scan -- turning it on may not move the answer, or it is not a diagnostic but a change of
    algorithm.
    """
    off = _scan(tmp_path, track=False)
    on = _scan(tmp_path, track=True)
    assert on["outcome"] == off["outcome"] == "converged"
    assert on["steps"] == off["steps"]
    assert on["B"] == pytest.approx(off["B"], rel=1e-12)
    assert on["real_part"] == pytest.approx(off["real_part"], abs=1e-14)


# ---------------------------------------------------------------------------------------------
# stay_stable_file: never leave the stable side.
# ---------------------------------------------------------------------------------------------

def test_stay_stable_file_never_leaves_the_stable_side(tmp_path):
    """A step that lands unstable is discarded and retried, so the caller never solves past onset.

    No in-tree caller passes stay_stable_file, but external scripts do, and it had no coverage at
    all -- so this pins the behaviour rather than describing it.

    What it must do: save the state at every stable point, and when a step crosses the axis, reload
    that state and retry with a secant prediction instead of accepting the unstable point. The
    search therefore approaches the bifurcation from below and every yielded point is stable.
    """
    res = _scan(tmp_path, stay_stable=True)
    assert res["outcome"] == "converged", res.get("error", res["outcome"])
    assert res["reloads"] > 0, "the retry path was never taken, so this test proves nothing"
    assert res["stay_stable_dump_exists"] is True, "no state was saved"

    # Every point handed to the caller is on the stable side -- that is the whole contract. The
    # unstable probes are discarded before the yield.
    for b, re_part in res["yielded"]:
        assert re_part < res["epsilon"], \
            "yielded an unstable point: B=%.9g Re=%+.3g" % (b, re_part)

    # And it still lands on the bifurcation, approaching it from below.
    assert res["B"] == pytest.approx(_B_C, rel=1e-4)
    assert abs(res["real_part"]) < res["epsilon"]


@pytest.mark.parametrize("continuation_data_in_states", [False, True],
                         ids=["no_continuation_data", "with_continuation_data"])
def test_stay_stable_secant_relies_on_the_arclength_reset(tmp_path, continuation_data_in_states):
    """The retry's secant is a PARAMETER delta handed over as an ARCLENGTH, and that is correct here.

    Only while dparameter/ds is 1 at the moment of the handover, which is what the
    reset_arc_length_parameters() immediately before it guarantees: the parameter then moves by
    exactly ds (measured dB/ds = 1 to every digit).

    BOTH parametrisations are needed, and the reason is the useful part. With
    continuation_data_in_states False -- the default -- load_state restores no tangent, so
    dparameter/ds is already 1 when the reload returns and the reset is redundant: deleting it
    changes nothing and a test running only this case passes anyway (checked, by deleting it). With
    the setting True, load_state restores the real tangent -- 0.7071 on this problem -- and the reset
    is the only thing that puts it back to 1. So the first case pins the arithmetic and the second
    pins the reset.

    What a tidy-up would cost: converting the delta with the post-reload dparameter/ds instead of
    relying on the reset gives ds = dp/0.7071, and since the parameter then moves by ds, that
    overshoots the target by 41%.
    """
    res = _scan(tmp_path, stay_stable=True,
                continuation_data_in_states=continuation_data_in_states)
    assert res["reloads"] > 0
    assert len(res["dparam_ds_at_handover"]) == res["reloads"]

    # The invariant, in both configurations.
    for value in res["dparam_ds_at_handover"]:
        assert value == pytest.approx(1.0, abs=1e-12), \
            ("dparameter/ds is %.12g at the secant handover, not 1: the parameter delta is being "
             "scaled by it, so the retry no longer aims where the secant predicted" % value)
    # And its consequence, measured rather than assumed: ds IS the parameter increment.
    for value in res["dB_over_ds"]:
        assert value == pytest.approx(1.0, rel=1e-9), \
            "the secant step moved the parameter by %.9g x ds, not 1 x ds" % value

    if continuation_data_in_states:
        # This is the case that makes the reset load-bearing: without it the handover would see
        # whatever load_state restored.
        for value in res["dparam_ds_after_reload"]:
            assert value != pytest.approx(1.0, abs=1e-6), \
                ("load_state did not restore a tangent even with continuation_data_in_states=True, "
                 "so this parametrisation is no longer testing the reset")
    else:
        for value in res["dparam_ds_after_reload"]:
            assert value == pytest.approx(1.0, abs=1e-12)

    # Either way it still lands on the bifurcation.
    assert res["B"] == pytest.approx(_B_C, rel=1e-4)
