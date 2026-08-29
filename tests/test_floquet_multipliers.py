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

# Floquet multipliers of a periodic orbit, both formulations.
#
# The default "condensed" method exploits the block bidiagonal structure of the orbit Jacobian: each
# element of the time discretization is condensed to one nbase x nbase transfer matrix, and their
# product is the monodromy matrix. The older "eigenproblem" method instead hands the whole orbit
# Jacobian to a general eigensolver, paired with a mass matrix that is the identity on the
# wrap-around block alone -- a pencil whose mass matrix has rank nbase in an nT*nbase-dimensional
# space, so all but nbase of its eigenvalues are infinite and have to be thresholded away.
#
# WHAT EACH ASSERTION IS FOR:
#
#   - test_stuart_landau_multipliers: the two multipliers are known exactly (1 and exp(-4*pi)), so
#     this pins the value, not just self-consistency. The trivial 1 is the free accuracy check the
#     condensation gives on the whole orbit solve, and it lands at machine precision here.
#   - test_multiplier_converges_with_NT: the deviation from exp(-4*pi) must be discretization error
#     and nothing else, so refining the time mesh has to remove it -- superconvergently, since
#     order-3 Gauss collocation is O(h^6) at the nodes. A wrong block structure would produce a
#     wrong number that refinement does not fix.
#   - test_count_is_nbase_in_every_mode: the point of the whole exercise. The eigenproblem method
#     returns a variable number of multipliers (docs/source/tutorial/temporal/orbit/langford_floquet.py
#     used to carry a workaround saying so, and that it differed between serial and mpirun); the
#     condensation returns exactly nbase, in every discretization that has a wrap-around block.
#   - test_agrees_with_eigenproblem_method: the two formulations are of the same thing, so every
#     multiplier the old one manages to resolve must be among the new one's.
#   - test_eigenproblem_method_loses_the_small_multiplier: the flip side, asserted so that the
#     reason the default changed does not quietly stop being true.
#   - test_dae_*: a singular mass matrix, which is what rules out a shooting formulation. Gauss
#     collocation is not stiffly accurate, so the algebraic direction is NOT a zero multiplier: it
#     is exactly (-1)**(number of time intervals), which for an odd count sits on -1 where a
#     period-doubling bifurcation would. Asserted in both parities because it is a trap.
#   - test_matrix_free_agrees_with_dense: the large-nbase route, forced on a tiny problem.
#   - test_periodic_schur_*: the opt-in method="periodic_schur", which never forms the product and
#     therefore keeps the multipliers many orders below the dominant one. Measured against the SAME
#     transfer matrices multiplied in 120-digit arithmetic, so the comparison is of the two products
#     and not of two discretizations. It has to agree with the default method where the default is
#     accurate (which is nearly everywhere) and beat it where it is not.
#   - test_non_floquet_modes_are_refused: central/BDF2 keep no explicit end-of-period degree of
#     freedom, so neither formulation has a wrap-around block to read.
#
# Only ONE Problem is constructed per process (a second one segfaults in the JIT loader, see
# tests/test_multiple_problems.py), so every case runs the shared worker in its own subprocess.

import json
import os
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "floquet_worker.py")

# The radial multiplier of the Stuart-Landau limit cycle: r'=r(1-r^2) linearizes to -2 at r=1, over
# a period of 2*pi.
_EXACT_RADIAL = float(numpy.exp(-4 * numpy.pi))


def _no_eigensolver_reason():
    """Why the legacy method="eigenproblem" path cannot run here, or None.

    That path is the only one in this module that solves a generalised eigenproblem
    (Problem.get_floquet_multipliers -> get_eigen_solver().solve); the default condensation method
    needs no eigensolver at all, which is why the rest of the module is unaffected. Without
    petsc4py/slepc4py pyoomph falls back to the scipy solver, and scipy's ARPACK cannot build an
    Arnoldi factorization for this pencil - "ARPACK error -9999", raised out of the solve rather than
    returning a poor answer. Seen on every pyoomph wheel, which deliberately ships without PETSc
    (PYOOMPH_USE_MPI=OFF), in the test-wheel run of 29th August 2026.
    """
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "no slepc4py: the eigenproblem method falls back to scipy, whose ARPACK cannot factor this pencil"
    return None


_NO_EIGENSOLVER = _no_eigensolver_reason()


def _run(tmp_path, timeout=900, **kw):
    """Run one worker case in its own process and return its result dict."""
    os.makedirs(str(tmp_path), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmp_path)]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v is not False and v is not None:
            cmd += [flag, str(v)]
    out = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=timeout)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_FLOQUET_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_FLOQUET_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    return res


def _run_capturing_output(tmp_path, timeout=900, **kw):
    """Same as _run, but hands back what the worker PRINTED as well as what it returned.

    Needed because the thing under test here is a warning, and a warning that nobody prints is worth
    nothing. Deliberately does not go through _run: that one asserts on the result and would hide a
    case where the warning fires but the numbers are wrong.
    """
    os.makedirs(str(tmp_path), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmp_path)]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v is not False and v is not None:
            cmd += [flag, str(v)]
    out = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=timeout)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_FLOQUET_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_FLOQUET_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    return res, out.stdout


def _multipliers(res):
    return numpy.array(res["mult_re"]) + 1j * numpy.array(res["mult_im"])


def _trivial_error(F):
    """How far the multiplier closest to 1 -- the one time-shift invariance guarantees -- really is."""
    return float(numpy.min(numpy.abs(F - 1.0)))


def test_stuart_landau_multipliers(tmp_path):
    res = _run(tmp_path)
    F = _multipliers(res)
    assert res["nbase"] == 2 and len(F) == 2, res
    assert _trivial_error(F) < 1e-8, F
    radial = F[numpy.argmin(numpy.abs(F))]
    assert abs(radial.imag) < 1e-12, F
    assert abs(radial.real - _EXACT_RADIAL) / _EXACT_RADIAL < 1e-3, (radial, _EXACT_RADIAL)
    # The eigenfunction is returned over the whole orbit, with a trailing zero for the period
    # unknown -- the same layout the eigenproblem formulation produces.
    assert res["eigvec_shape"] == [2, res["nT"] * res["nbase"] + 1], res


def test_eigenfunction_closes_the_orbit(tmp_path):
    """v(s=1) = lambda*v(s=0) by construction, which is what pins the eigenfunction reconstruction.

    The reconstruction pushes every eigenvector through the chain in one batch rather than one column
    at a time -- asking for all multipliers of an nbase=1282 orbit spent 185 s on it column-by-column
    against 13 s batched. Only the batching changed, not the arithmetic, and this is the invariant
    that says so.
    """
    for case, nbase in (("sl", 2), ("dae", 3)):
        res = _run(tmp_path / case, case=case, NT=48)
        assert res["nbase"] == nbase, res
        assert res["eigvec_shape"] == [nbase, res["nT"] * nbase + 1], res
        assert res["closure_residual"] < 1e-9, (case, res["closure_residual"])


def test_multiplier_converges_with_NT(tmp_path):
    err = []
    for NT in (24, 48, 96):
        F = _multipliers(_run(tmp_path / ("NT%d" % NT), NT=NT))
        radial = F[numpy.argmin(numpy.abs(F))].real
        err.append(abs(radial - _EXACT_RADIAL) / _EXACT_RADIAL)
    # Order-3 Gauss collocation is O(h^6) at the nodes, i.e. a factor 64 per halving of h. Asserted
    # at 20 to leave room for the period itself converging alongside.
    assert err[1] < err[0] / 20 and err[2] < err[1] / 20, err


@pytest.mark.parametrize("mode,order", [("floquet", 1), ("collocation", 1),
                                        ("collocation", 2), ("collocation", 3)])
def test_count_is_nbase_in_every_mode(tmp_path, mode, order):
    res = _run(tmp_path, mode=mode, order=order, NT=48)
    F = _multipliers(res)
    assert len(F) == res["nbase"], res
    assert _trivial_error(F) < 1e-7, F


@pytest.mark.skipif(_NO_EIGENSOLVER is not None, reason=str(_NO_EIGENSOLVER))
def test_agrees_with_eigenproblem_method(tmp_path):
    cond = _multipliers(_run(tmp_path / "cond", case="dae", NT=48))
    old = _multipliers(_run(tmp_path / "old", case="dae", NT=48, method="eigenproblem", n=3))
    assert len(cond) == 3, cond
    assert len(old) > 0, old
    for z in old:
        assert numpy.min(numpy.abs(cond - z)) < 1e-6, (z, cond)


@pytest.mark.skipif(_NO_EIGENSOLVER is not None, reason=str(_NO_EIGENSOLVER))
def test_eigenproblem_method_loses_the_small_multiplier(tmp_path):
    """Why the default changed: the old filtering cannot tell exp(-4*pi) from an infinite eigenvalue."""
    cond = _multipliers(_run(tmp_path / "cond", NT=48))
    old = _multipliers(_run(tmp_path / "old", NT=48, method="eigenproblem", n=2))
    assert numpy.min(numpy.abs(cond)) < 1e-4, cond          # the condensation keeps it
    assert numpy.min(numpy.abs(old)) > 1e-2, old            # the eigenproblem one does not
    assert len(old) < len(cond), (old, cond)


# order 2 is deliberately only present at +1: its interval count is a multiple of 2, so
# (-1)**(intervals) can never be -1 there. Odd orders are the ones that can produce the spurious -1.
@pytest.mark.parametrize("order,NT,expected", [(3, 45, -1.0), (3, 48, +1.0),
                                               (1, 47, -1.0), (1, 48, +1.0),
                                               (2, 46, +1.0)])
def test_dae_algebraic_multiplier_sign(tmp_path, order, NT, expected):
    """A singular mass matrix goes through, but Gauss collocation puts its multiplier on +-1, not 0.

    The perturbation of an algebraic direction is the degree-``order`` polynomial vanishing at the
    ``order`` Gauss points of the element; those points are symmetric about the midpoint, so its
    value at the end of an element is (-1)**order times its value at the start, and over the orbit
    that accumulates to (-1)**(number of intervals) exactly.
    """
    res = _run(tmp_path, case="dae", NT=NT, order=order)
    F = _multipliers(res)
    assert res["nbase"] == 3 and len(F) == 3, res
    assert res["n_intervals"] == NT, res
    assert numpy.all(numpy.isfinite(F)), F
    assert numpy.min(numpy.abs(F - expected)) < 1e-9, (F, expected)
    # The genuine radial multiplier is still there, and still converging to the right value (which
    # the low collocation orders only manage loosely at these NT).
    radial = F[numpy.argmin(numpy.abs(F))]
    tol = {1: 0.2, 2: 1e-2, 3: 1e-3}[order]
    assert abs(radial.real - _EXACT_RADIAL) / _EXACT_RADIAL < tol, F


# --------------------------------------------------------------------------------------------
# The collocation artefact has to be SAID, not just be true. test_dae_algebraic_multiplier_sign
# above pins where the algebraic direction lands; these pin that a user is told, because the number
# itself is indistinguishable from the physics it imitates -- an odd interval count puts it on -1,
# which is exactly where a period doubling lives.
# --------------------------------------------------------------------------------------------

def test_an_odd_interval_dae_warns_that_the_minus_one_may_not_be_a_period_doubling(tmp_path):
    res, printed = _run_capturing_output(tmp_path, case="dae", NT=45, order=3)
    F = _multipliers(res)
    assert numpy.min(numpy.abs(F + 1.0)) < 1e-9, F      # the artefact really is there
    assert "sit on -1" in printed and "ODD" in printed, printed[-3000:]
    assert "not a period doubling" in printed, printed[-3000:]


def test_an_even_interval_dae_warns_that_the_plus_one_is_doubled(tmp_path):
    # Harmless to a stability verdict, but it makes the multiplicity of the trivial multiplier look
    # like a bifurcation of the orbit, which the pre-existing "multiple unity" warning used to say.
    res, printed = _run_capturing_output(tmp_path, case="dae", NT=48, order=3)
    F = _multipliers(res)
    assert numpy.sum(numpy.abs(F - 1.0) < 1e-9) == 2, F
    assert "sit on +1" in printed, printed[-3000:]


@pytest.mark.parametrize("NT", [45, 48])
def test_a_plain_ode_is_not_warned_about(tmp_path, NT):
    """The control, at both parities: no algebraic direction, so nothing to warn about.

    Without it the warnings above could be firing on every orbit, which would train the reader to
    ignore them -- the failure mode of a warning is not being wrong, it is being noise.
    """
    _res, printed = _run_capturing_output(tmp_path, case="sl", NT=NT, order=3)
    assert "sit on -1" not in printed, printed[-3000:]
    assert "sit on +1" not in printed, printed[-3000:]


def test_matrix_free_agrees_with_dense(tmp_path):
    dense = _multipliers(_run(tmp_path / "dense", case="dae", NT=48))
    free = _multipliers(_run(tmp_path / "free", case="dae", NT=48, n=1, dense_threshold=1))
    assert len(free) == 1, free
    assert abs(free[0] - dense[numpy.argmax(numpy.abs(dense))]) < 1e-9, (free, dense)


@pytest.mark.parametrize("mode", ["central", "BDF2"])
def test_non_floquet_modes_are_refused(tmp_path, mode):
    """central/BDF2 have no end-of-period dof and, unlike bspline, are not worth sampling.

    They are finite-difference stencils with no accuracy to speak of and nobody continues an orbit
    with them, so the refusal is left as a refusal rather than becoming a list of exceptions.
    """
    res = _run(tmp_path, mode=mode, expect_refusal=True)
    assert res["refused"] is not None, res
    assert "has no Floquet multipliers" in res["refused"], res
    assert "cannot be sampled" in res["refused"], res


@pytest.mark.parametrize("NT", [24, 48, 96])
def test_a_bspline_orbit_reports_the_collocation_multipliers(tmp_path, NT):
    """A B-spline orbit has no multipliers of its own, and answers with a sampling's.

    Its basis is periodic BY CONSTRUCTION, so the orbit Jacobian is block-circulant-banded (half
    bandwidth = the spline order, wrapping at the seam) rather than the block-bidiagonal chain the
    condensation slices: there is no end-of-period degree of freedom and no wrap-around row
    `v_{nT-1} - v_0 = 0` to key off. The orbit is a curve all the same, so it is sampled onto a
    collocation discretization for the computation.

    What has to hold is that this costs nothing in accuracy: the answer must be the one a
    natively-collocation orbit of the same resolution gives, and it must converge with NT the same
    way. Judged against exp(-4*pi), which is exact for this oscillator.
    """
    bspl = _multipliers(_run(tmp_path / "bspl", mode="bspline", order=3, NT=NT))
    coll = _multipliers(_run(tmp_path / "coll", mode="collocation", order=3, NT=NT))
    assert len(bspl) == 2 and len(coll) == 2, (bspl, coll)
    assert _trivial_error(bspl) < 1e-8, bspl

    radial_b = bspl[numpy.argmin(numpy.abs(bspl))].real
    radial_c = coll[numpy.argmin(numpy.abs(coll))].real
    assert radial_b == pytest.approx(radial_c, rel=1e-9), \
        "the sampled route must agree with a native collocation orbit"
    # ... and both are the same distance from the exact value, i.e. the sampling adds no error of
    # its own - only the collocation discretization's, which is what converges with NT.
    err_b = abs(radial_b - _EXACT_RADIAL) / _EXACT_RADIAL
    err_c = abs(radial_c - _EXACT_RADIAL) / _EXACT_RADIAL
    assert err_b == pytest.approx(err_c, rel=1e-3), (err_b, err_c)
    assert err_b < {24: 1e-2, 48: 1e-3, 96: 1e-5}[NT], err_b


def test_the_bspline_orbit_survives_its_multipliers(tmp_path):
    """The sampling is temporary: the orbit that was installed must come back, bit for bit.

    This is the whole risk of computing them this way. The blocks are read out of the augmented dof
    vector either side of the call (PeriodicOrbit._blocks) rather than re-sampled, because a resample
    interpolates and would hide exactly the error this is looking for.
    """
    res = _run(tmp_path, mode="bspline", order=3, NT=48)
    assert res["sampled"] is True, res
    assert res["mode_after"] == "bspline", res
    assert res["nT_after"] == res["nT"], res
    assert res["floquet_mode_after"] is False, res
    assert res["restore_shape_ok"], res
    assert res["restore_max_diff"] == 0.0, res["restore_max_diff"]
    assert res["restore_T_diff"] == 0.0, res["restore_T_diff"]


def test_the_bspline_orbit_survives_a_failed_resolve(tmp_path):
    """And it comes back even when the sampled orbit's re-solve diverges.

    The restore is in a `finally` for this reason: a re-solve that fails must cost the attempt and
    not the orbit the user is continuing.
    """
    res = _run(tmp_path, mode="bspline", order=3, NT=48, fail_resolve=True)
    assert res["resolve_failed_with"] is not None, "the substituted solve must have made this fail"
    assert res["mode_after"] == "bspline" and res["floquet_mode_after"] is False, res
    assert res["restore_max_diff"] == 0.0 and res["restore_T_diff"] == 0.0, res


def test_a_collocation_orbit_is_not_sampled(tmp_path):
    """The control: nothing about the ordinary path changed, and no sampling is taken for it."""
    res = _run(tmp_path, mode="collocation", order=3, NT=48)
    assert res["sampled"] is False, res
    assert res["mode_after"] == "collocation" and res["floquet_mode_after"] is True, res
    assert res["restore_max_diff"] == 0.0, res


def _sorted(F):
    return numpy.array(sorted(F, key=lambda z: (abs(z), z.real)))


def _relerr(got, ref):
    got, ref = _sorted(got), _sorted(ref)
    assert len(got) == len(ref), (got, ref)
    return numpy.abs(got - ref) / numpy.abs(ref)


@pytest.mark.parametrize("case,order,NT", [("sl", 3, 48), ("dae", 3, 45), ("dae", 1, 47)])
def test_periodic_schur_agrees_with_default(tmp_path, case, order, NT):
    """Where the plain product is accurate -- which is nearly everywhere -- the two must agree.

    The two dae cases are the ones subspace iteration cannot separate at all (the algebraic +-1 has
    the same modulus as the trivial multiplier), so they also exercise the block fallback.
    """
    a = _multipliers(_run(tmp_path / "a", case=case, order=order, NT=NT))
    b = _multipliers(_run(tmp_path / "b", case=case, order=order, NT=NT, method="periodic_schur"))
    assert len(a) == len(b), (a, b)
    assert numpy.max(_relerr(b, a)) < 1e-8, (a, b)


def test_periodic_schur_beats_the_product_at_the_bottom(tmp_path):
    """The reason the method exists, pinned against a 120-digit product of the same matrices.

    The stiff chain's spectrum spans 25 orders of magnitude and the strong coupling makes the
    monodromy non-normal, which is what stops the plain product from resolving its bottom. Every
    multiplier above ~1e-20 is at machine precision either way; only the two smallest separate them.
    """
    pytest.importorskip("mpmath")
    prod = _run(tmp_path / "prod", case="stiff", reference=True)
    schur = _run(tmp_path / "schur", case="stiff", reference=True, method="periodic_schur")
    ref = numpy.array(prod["ref_re"]) + 1j * numpy.array(prod["ref_im"])
    assert numpy.allclose(ref, numpy.array(schur["ref_re"]) + 1j * numpy.array(schur["ref_im"])), \
        "the two runs must share their transfer matrices for the comparison to mean anything"

    e_prod, e_schur = _relerr(_multipliers(prod), ref), _relerr(_multipliers(schur), ref)
    # Both are exact from the fourth-smallest multiplier (~1e-11) upwards.
    assert numpy.max(e_prod[2:]) < 1e-9 and numpy.max(e_schur[2:]) < 1e-9, (e_prod, e_schur)
    # The two smallest (~7e-23 and ~4e-26) are where forming the product costs everything.
    assert numpy.max(e_prod[:2]) > 1e-6, e_prod
    assert numpy.max(e_schur[:2]) < 1e-9, e_schur



# The matrix-free route seeks the multipliers near sigma rather than by largest magnitude. Plain
# Arnoldi has the right TARGET but converges badly when the wanted multipliers are clustered, which is
# the normal case for a PDE orbit -- on the 1D Brusselator, eight of them cost 277 s that way against
# 18 s shift-inverted, to 2.6e-13. The shift is available exactly, and without forming the monodromy:
# take the orbit Jacobian, keep its element rows, and replace the wrap-around row v_last - v_0 = 0 with
# v_last - sigma*v_0 = b. The element rows still force v_k+1 = C_k v_k, so the last row reads
# (Mono - sigma*I) v_0 = b, and one solve of a matrix the size of the orbit system is the shift-invert
# apply -- factorized once, at the cost the orbit's own Newton step already pays each iteration.
@pytest.mark.parametrize("shift_invert", [1, 0], ids=["shift_invert", "plain_arnoldi"])
def test_matrix_free_agrees_with_dense_on_a_clustered_spectrum(tmp_path, shift_invert):
    """The Stuart-Landau reaction-diffusion orbit, whose multipliers sit on top of each other at 0.95."""
    dense = _run(tmp_path / "dense", case="pde", what="orbit", sigma=-1.0, N=20, n=4)
    free = _run(tmp_path / "free", case="pde", what="orbit", sigma=-1.0, N=20, n=4,
                dense_threshold=1, shift_invert=shift_invert)
    # Folded onto the positive-imaginary representative before sorting: the two routes can list
    # opposite halves of a conjugate pair, which says nothing about either being wrong.
    def canon(res):
        F = _multipliers(res)
        F = numpy.where(F.imag >= 0, F, numpy.conjugate(F))
        return numpy.array(sorted(F, key=lambda z: (abs(z), z.real, z.imag)))
    a, b = canon(dense), canon(free)
    assert len(a) == len(b) == 4, (a, b)
    assert numpy.max(numpy.abs(a - b)) < 1e-8, (a, b)
