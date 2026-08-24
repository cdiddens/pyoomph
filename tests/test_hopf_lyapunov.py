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

# The first Lyapunov coefficient, and switching from a Hopf point onto the emerging orbit.
#
# get_hopf_lyapunov_coefficient() had no numerical test of any kind: the tutorials that reach it only
# check that they do not crash, so nothing pinned the coefficient, the amplitude, or the criticality.
# It is Kuznetsov's real-form algorithm generalised to a mass matrix, and that generalisation -- M in
# the h20 operator and in <p,Mq>=1, nowhere else -- is the pyoomph-specific part that most needed
# pinning.
#
# The system is the Hopf normal form in Cartesian coordinates (see tests/hopf_lyapunov_worker.py for
# the equations). Two properties make it the right anchor:
#
#   - The nonlinearity is CUBIC, so the quadratic form B vanishes at the origin and the coefficient
#     comes entirely from the C term -- the one that has to be finite-differenced out of the analytic
#     Hessian, and the one with no other coverage.
#   - For m1 == m2 the polar reduction r*rdot = x*xdot + y*ydot is exact whatever that mass is, giving
#     rdot = (mu*r + sigma*r^3)/m. So the limit cycle radius is exactly sqrt(-mu/sigma) and the period
#     exactly 2*pi*m, independently of the mass -- an end-to-end reference that needs no knowledge of
#     the normalisation convention.
#
# WHAT EACH ASSERTION IS FOR:
#
#   - test_coefficient_on_the_normal_form: ga == 2*sigma, which is Re<p,C(q,q,qb)>/(2*omega0) worked
#     out by hand for this system. It pins the C term, the 1/(2*omega0), and the sign convention that
#     starts_supercritically() reads.
#   - test_coefficient_is_invariant_under_a_common_mass: m1 == m2 == 2 halves omega0 and leaves ga
#     alone. That is the mass-matrix generalisation doing exactly what it should, and it is the only
#     test of it anywhere.
#   - test_coefficient_with_an_anisotropic_mass: m1 != m2, where M genuinely changes the eigenvectors.
#     omega0 = 1/sqrt(m1*m2) and ga = 1.5*sqrt(3)*sigma are characterisation values -- they are here to
#     catch a change, not because they are derived.
#   - test_orbit_matches_the_exact_limit_cycle: the end-to-end one. Radius against sqrt(-mu/sigma),
#     period against 2*pi*m, and the parameter step against eps^2. This is what actually says ga, al
#     and the guess construction agree with each other; any one of them being wrong moves the radius.
#   - test_orbit_radius_is_mesh_independent: ga itself is NOT mesh-independent -- it scales with the
#     normalisation of q, so spreading the same uniform mode over more nodes shrinks it (measured:
#     exactly ga_ode/41 on a 41-node mesh) while al grows by the same factor. The orbit radius is the
#     invariant combination, and this checks that the two really do cancel.
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
_WORKER = os.path.join(_HERE, "hopf_lyapunov_worker.py")


def _skip_reason():
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support"
    except Exception:
        return "petsc4py not available (PYTHONPATH must carry a complex PETSc build)"
    try:
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "slepc4py not available"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))


def _run(tmp_path, timeout=1800, **kw):
    os.makedirs(str(tmp_path), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmp_path)]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    out = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=timeout)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_HOPF_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_HOPF_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    return res


@pytest.mark.parametrize("sigma", [-1.0, 1.0])
def test_coefficient_on_the_normal_form(tmp_path, sigma):
    """ga = 2*sigma exactly, and the criticality that follows from its sign."""
    res = _run(tmp_path, what="coeff", sigma=sigma)
    assert abs(res["mu_hopf"]) < 1e-9, res            # the tracker really is at the Hopf point
    assert abs(res["omega"] - 1.0) < 1e-9, res
    assert abs(res["ga"] - 2.0 * sigma) < 1e-8, res
    assert res["dlam"] == (-1 if sigma < 0 else 1), res
    assert abs(res["al"] - 1.0 / numpy.sqrt(2.0)) < 1e-8, res


@pytest.mark.parametrize("sigma", [-1.0, 1.0])
def test_coefficient_is_invariant_under_a_common_mass(tmp_path, sigma):
    """m1 == m2 == m rescales time: omega0 -> 1/m, and the coefficient is untouched."""
    res = _run(tmp_path, what="coeff", sigma=sigma, m1=2.0, m2=2.0)
    assert abs(res["omega"] - 0.5) < 1e-9, res
    assert abs(res["ga"] - 2.0 * sigma) < 1e-8, res
    assert res["dlam"] == (-1 if sigma < 0 else 1), res


def test_coefficient_with_an_anisotropic_mass(tmp_path):
    """m1 != m2, where M actually changes the eigenvectors. Characterisation values."""
    res = _run(tmp_path, what="coeff", sigma=-1.0, m1=1.0, m2=3.0)
    assert abs(res["omega"] - 1.0 / numpy.sqrt(3.0)) < 1e-9, res
    assert abs(res["ga"] + 1.5 * numpy.sqrt(3.0)) < 1e-7, res
    assert abs(res["al"] - 2.0 / 3.0) < 1e-8, res


# m=2 is here as well as m=1 because the period is the one quantity the common mass does change.
@pytest.mark.parametrize("sigma,m,eps", [(-1.0, 1.0, 0.1), (-1.0, 1.0, 0.05),
                                         (-1.0, 2.0, 0.1), (1.0, 1.0, 0.1)])
def test_orbit_matches_the_exact_limit_cycle(tmp_path, sigma, m, eps):
    res = _run(tmp_path, what="orbit", sigma=sigma, m1=m, m2=m, eps=eps)
    # switch_to_hopf_orbit steps the parameter by eps^2 away from the Hopf point, to the side the
    # orbit lives on: mu>0 for the supercritical case, mu<0 for the subcritical one.
    assert abs(abs(res["mu_orbit"]) - eps ** 2) < 1e-9 * eps ** 2 + 1e-12, res
    assert (res["mu_orbit"] > 0) == (sigma < 0), res
    assert res["supercritical"] == (sigma < 0), res
    assert abs(res["T"] - res["T_exact"]) < 1e-6 * res["T_exact"], res
    # The residual error is the orbit's own time discretization, not the coefficient: it comes out at
    # 1.4e-3 and is independent of eps, of m and of the mesh.
    rel = abs(res["radius_mean"] - res["radius_exact"]) / res["radius_exact"]
    assert rel < 5e-3, (rel, res)
    assert res["radius_rel_spread"] < 1e-2, res


@pytest.mark.parametrize("N", [20, 40])
def test_orbit_radius_is_mesh_independent(tmp_path, N):
    """Spreading the same uniform mode over more nodes shrinks ga and grows al; the radius is fixed."""
    res = _run(tmp_path, case="pde", what="orbit", sigma=-1.0, eps=0.1, N=N)
    rel = abs(res["radius_mean"] - res["radius_exact"]) / res["radius_exact"]
    assert rel < 5e-3, (rel, res)
    assert res["supercritical"] is True, res
    assert abs(res["T"] - res["T_exact"]) < 1e-6 * res["T_exact"], res


# The normal form's nonlinearity is cubic, so its quadratic form B vanishes identically and the
# h11/h20 solves and the sig/d0 terms are never reached. The Brusselator has a genuine quadratic term
# and is the case that exercises them: measured |r| = 0.30 and |s| = 1.1 at A=1.5, against |r| = |s| = 0
# for the normal form. Its Hopf sits at B = 1+A^2 with omega0 = A, both exact.
_BRUSSELATOR_GA = {1.0: -0.5, 1.5: -17.0 / 66.0, 2.0: -1.0 / 6.0}


@pytest.mark.parametrize("A", sorted(_BRUSSELATOR_GA))
def test_brusselator_hopf_and_coefficient(tmp_path, A):
    res = _run(tmp_path, case="brusselator", A_bruss=A, what="coeff")
    assert abs(res["mu_hopf"] - (1.0 + A * A)) < 1e-9, res     # the Hopf really is at B = 1+A^2
    assert abs(res["omega"] - A) < 1e-9, res
    # Characterisation values: exact rationals, reproduced to 1e-10. They are pinned to catch a
    # change, not because they are derived -- and note ga scales with the normalisation of q, so they
    # are specific to the unit-norm convention used here.
    assert abs(res["ga"] - _BRUSSELATOR_GA[A]) < 1e-8, res
    assert res["dlam"] == -1, res                              # the Brusselator Hopf is supercritical


@pytest.mark.parametrize("eps", [0.1, 0.05])
def test_amplitude_prediction_converges(tmp_path, eps):
    """The guess amplitude 2*eps*al against the amplitude the Newton solve actually lands on.

    This is what isolates `al`, and through it c1: a wrong coefficient still converges to the true
    limit cycle, so only the guess is wrong. Works on any system, including one with no closed-form
    cycle, and the discrepancy has to vanish with eps -- measured 4.3% at eps=0.1 and 2.1% at 0.05.
    """
    res = _run(tmp_path, case="brusselator", A_bruss=1.0, what="orbit", eps=eps)
    assert res["supercritical"] is True, res
    assert res["radius_guess"] > 0 and res["radius_mean"] > 0, res
    assert res["guess_vs_solved"] < 0.6 * eps, res
