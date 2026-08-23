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


def test_agrees_with_eigenproblem_method(tmp_path):
    cond = _multipliers(_run(tmp_path / "cond", case="dae", NT=48))
    old = _multipliers(_run(tmp_path / "old", case="dae", NT=48, method="eigenproblem", n=3))
    assert len(cond) == 3, cond
    assert len(old) > 0, old
    for z in old:
        assert numpy.min(numpy.abs(cond - z)) < 1e-6, (z, cond)


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


def test_matrix_free_agrees_with_dense(tmp_path):
    dense = _multipliers(_run(tmp_path / "dense", case="dae", NT=48))
    free = _multipliers(_run(tmp_path / "free", case="dae", NT=48, n=1, dense_threshold=1))
    assert len(free) == 1, free
    assert abs(free[0] - dense[numpy.argmax(numpy.abs(dense))]) < 1e-9, (free, dense)


@pytest.mark.parametrize("mode", ["central", "BDF2"])
def test_non_floquet_modes_are_refused(tmp_path, mode):
    res = _run(tmp_path, mode=mode, expect_refusal=True)
    assert res["refused"] is not None and "Floquet mode not active" in res["refused"], res


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

