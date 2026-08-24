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

# Branch switching under MPI -- BOTH regimes, because it used to work in neither.
#
# What blocked it was not --distribute. NormalFormCalculator took every quantity it needed from the
# Python custom multi-assembly, and that throws from
# Problem::sparse_assemble_row_or_column_compressed_base_problem for ANY nproc > 1, replicated or
# not; the Python-side _require_non_distributed guard only ever caught the distributed half, so a
# plain mpirun failed five frames deep in C++ instead. It now takes each quantity from an accessor
# that is MPI-safe in both regimes -- the same move deflation and get_hopf_lyapunov_coefficient
# already made. See dev_docs/branch_switching.md.
#
# WHAT EACH ASSERTION IS FOR:
#
#   - the coefficients a, b1, b2, b3 and b2_rel against serial. These are the thing that was
#     unavailable under MPI. NOT against a formula: b2 and b3 scale with the normalisation of zeta,
#     so this is the same discretisation solved a different way and what is left over is the
#     assembly's summation order.
#   - the landed parameter and the mesh integral usqr, after the switch AND after four continuation
#     steps. End to end: b1, b3, the bordered solve and the acceptance test all have to agree for the
#     landing to come out at all. The INTEGRAL rather than the dofs, because distribute() renumbers.
#   - the landing against the ANALYTIC branch. That is the half which says the answer is right and
#     not merely reproducible.
#   - the classified type identical across runs, and b2_rel to 1e-9. The classification must not be
#     partition-dependent, and b2_rel is a ratio of two reductions -- a partial one on some rank moves
#     it while leaving everything else looking fine.
#   - direction=+1 and -1 reaching OPPOSITE arms at every rank count. This is the one assertion that
#     catches ranks disagreeing about the acceptance test's dot product.
#   - distributed is True, nproc is right, the global ndof agrees, evect_len == ndof. A silent
#     fallback to replicated would otherwise pass everything else, and evect_len catches a rank that
#     returned its own row block instead of gathering.
#   - that the distributed run COMPLETES inside an explicit timeout. Branch switching runs a ladder
#     of Newton solves in a loop with a collective in it; a rank that took `continue` while the others
#     carried on would hang rather than fail, and a hang is only visible as a timeout.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_branch_switch_worker.py")
_N = 8
_NPROC = 4
_ASPECT = 1.05


def _skip_reason():
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
            return "PETSc has no MUMPS support"
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "petsc4py/slepc4py not available (PYTHONPATH must carry a complex PETSc build)"
    return None


_SKIP = _skip_reason()
pytestmark = [pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)), pytest.mark.slow]


def _run(tmpdir, nproc, distribute, kind, phase="full", N=_N, timeout=1800):
    os.makedirs(str(tmpdir), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmpdir), "--kind", kind,
           "--phase", phase, "--N", str(N)]
    if nproc > 1:
        cmd = ["mpirun", "-n", str(nproc)] + cmd
    if distribute:
        cmd.append("--distribute")
    env = dict(os.environ)
    # Own TMPDIR for the nested mpirun; see tests/test_mpi_bifurcation_tracking.py for why.
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    out = subprocess.run(cmd, cwd=str(tmpdir), capture_output=True, text=True, timeout=timeout, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_BRANCH_SWITCH_RESULT ")]
    assert lines, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_BRANCH_SWITCH_RESULT "):])
    assert res.get("ok"), res.get("traceback", res.get("error"))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    return res


@pytest.fixture(scope="module")
def switch_runs(tmp_path_factory):
    """serial / replicated / distributed, for both PDE problems."""
    base = tmp_path_factory.mktemp("mpi_branch_switch")
    runs = {}
    for kind in ("transcritical", "pitchfork"):
        runs[kind] = (_run(base / (kind + "_serial"), 1, False, kind),
                      _run(base / (kind + "_replicated"), _NPROC, False, kind),
                      _run(base / (kind + "_distributed"), _NPROC, True, kind))
    return runs


@pytest.fixture(scope="module")
def odd_rank_run(tmp_path_factory):
    """np=3 --distribute on the transcritical: an odd count catches a layout assumed even."""
    base = tmp_path_factory.mktemp("mpi_branch_switch_np3")
    return (_run(base / "serial", 1, False, "transcritical", phase="coefficients"),
            _run(base / "np3", 3, True, "transcritical", phase="coefficients"))


@pytest.fixture(scope="module")
def bratu_runs(tmp_path_factory):
    """The Bratu fold -- the only case here with a NONZERO dR/dparameter.

    On the trivial branch u = 0 of the two branch-point problems, dR/dlam = -int(u v) vanishes
    identically, so R01 is exactly zero and the bordered solve is handed a zero right-hand side.
    Bratu is what actually exercises get_parameter_derivative and the bordered solve under MPI.
    """
    base = tmp_path_factory.mktemp("mpi_branch_switch_bratu")
    return (_run(base / "serial", 1, False, "bratu", phase="coefficients"),
            _run(base / "distributed", _NPROC, True, "bratu", phase="coefficients"))


_MODES = {1: "replicated", 2: "distributed"}


def _phi_integrals():
    """<phi,phi>, <phi^3>, <phi^4> for the first Dirichlet mode of the 1 x ASPECT rectangle."""
    nx = 600
    x = (numpy.arange(nx) + 0.5) / nx
    y = (numpy.arange(nx) + 0.5) / nx * _ASPECT
    P = numpy.sin(numpy.pi * x)[:, None] * numpy.sin(numpy.pi * y / _ASPECT)[None, :]
    dA = (1.0 / nx) * (_ASPECT / nx)
    return float(numpy.sum(P * P) * dA), float(numpy.sum(P ** 3) * dA), float(numpy.sum(P ** 4) * dA)


def test_the_distributed_run_really_was_distributed(switch_runs):
    for kind, (serial, rep, dist) in switch_runs.items():
        assert serial["distributed"] is False and rep["distributed"] is False, (kind, serial, rep)
        assert dist["distributed"] is True and dist["nproc"] == _NPROC, (kind, dist)
        # ndof is the GLOBAL count, so it must agree whichever way the mesh was cut up; evect_len
        # catches a rank that handed back its own row block instead of the gathered eigenvector.
        assert serial["ndof"] == rep["ndof"] == dist["ndof"], (kind, switch_runs[kind])
        for r in (serial, rep, dist):
            assert r["evect_len"] == r["ndof"], (kind, r)


@pytest.mark.parametrize("kind", ["transcritical", "pitchfork"])
@pytest.mark.parametrize("which", [1, 2], ids=["replicated", "distributed"])
def test_coefficients_match_serial(switch_runs, kind, which):
    serial, other = switch_runs[kind][0], switch_runs[kind][which]
    assert abs(other["lam_bif"] - serial["lam_bif"]) < 1e-9 * abs(serial["lam_bif"]), (serial, other)
    for key in ("b1", "b3"):
        assert abs(other[key] - serial[key]) < 1e-9 * abs(serial[key]), (key, serial, other)
    # b2 is exactly zero on the pitchfork, so an absolute floor is needed there.
    assert abs(other["b2"] - serial["b2"]) < 1e-9 * max(abs(serial["b2"]), 1e-12), (serial, other)


@pytest.mark.parametrize("kind", ["transcritical", "pitchfork"])
def test_classification_is_not_partition_dependent(switch_runs, kind):
    serial, rep, dist = switch_runs[kind]
    assert serial["type"] == rep["type"] == dist["type"] == kind, switch_runs[kind]
    for other in (rep, dist):
        assert abs(other["b2_rel"] - serial["b2_rel"]) < 1e-9 * max(abs(serial["b2_rel"]), 1e-12), \
            (serial, other)
    if kind == "pitchfork":
        # The odd nonlinearity makes the Hessian contraction identically zero at every quadrature
        # point, which is what the exact-zero fast path keys off. A change that made it merely small
        # would send this through the cosine instead, where 0/0 has no answer.
        for r in switch_runs[kind]:
            assert r["norm_b2v"] == 0.0, r


def test_b2_rel_is_mesh_independent_where_the_old_ratio_was_not(tmp_path_factory):
    """The reason the discriminant was changed, as an executable claim.

    The old test was 100*|b2/2| < |b3/6|, i.e. a fixed threshold on |b2/b3| -- and with zeta
    normalised to unit EUCLIDEAN length that ratio grows like sqrt(ndof), so the pitchfork/
    transcritical verdict moved with the mesh. b2_rel is the cosine between the quadratic term and
    the left null vector and does not.
    """
    base = tmp_path_factory.mktemp("mpi_branch_switch_mesh")
    coarse = _run(base / "n8", 1, False, "transcritical", phase="coefficients", N=8)
    fine = _run(base / "n16", 1, False, "transcritical", phase="coefficients", N=16)
    old_ratio = abs(coarse["b2"] / coarse["b3"]), abs(fine["b2"] / fine["b3"])
    growth = old_ratio[1] / old_ratio[0]
    ndof_ratio = fine["ndof"] / coarse["ndof"]
    assert 1.8 < growth < 2.6, ("the old ratio must move with the mesh", old_ratio, ndof_ratio)
    assert abs(fine["b2_rel"] - coarse["b2_rel"]) < 0.05 * coarse["b2_rel"], \
        ("b2_rel must not", coarse["b2_rel"], fine["b2_rel"])


@pytest.mark.parametrize("which", [1, 2], ids=["replicated", "distributed"])
def test_the_landing_matches_serial(switch_runs, which):
    for kind in ("transcritical", "pitchfork"):
        serial, other = switch_runs[kind][0], switch_runs[kind][which]
        assert len(serial["landed"]) == len(other["landed"]), (kind, serial, other)
        for a, b in zip(serial["landed"], other["landed"]):
            assert (a is None) == (b is None), (kind, a, b)
            if a is None:
                continue
            for sa, sb in zip(a["steps"], b["steps"]):
                assert abs(sb["lam"] - sa["lam"]) < 1e-8 * abs(sa["lam"]), (kind, sa, sb)
                assert abs(sb["usqr"] - sa["usqr"]) < 1e-8 * abs(sa["usqr"]), (kind, sa, sb)


@pytest.mark.parametrize("which", [0, 1, 2], ids=["serial", "replicated", "distributed"])
def test_the_landing_is_on_the_analytic_branch(switch_runs, which):
    """Leading order, so a few percent -- its job is to say WHICH branch, not to measure it."""
    pp, p3, p4 = _phi_integrals()
    for kind in ("transcritical", "pitchfork"):
        r = switch_runs[kind][which]
        step = r["landed"][0]["steps"][0]
        dlam = step["lam"] - r["lam_bif"]
        assert dlam > 0, (kind, r["lam_bif"], step)
        A = dlam * pp / p3 if kind == "transcritical" else numpy.sqrt(dlam * pp / p4)
        assert abs(abs(step["uphi"]) - A * pp) < 0.05 * A * pp, (kind, dlam, step["uphi"], A * pp)


@pytest.mark.parametrize("which", [0, 1, 2], ids=["serial", "replicated", "distributed"])
def test_the_two_pitchfork_arms_are_reached(switch_runs, which):
    """direction=+1 and -1 must reach OPPOSITE arms, which the acceptance test now guarantees.

    It used to compare abs(dot(moved,du)) against the threshold, so a landing on the arm opposite
    the one asked for was accepted and reported as that direction's result.
    """
    r = switch_runs["pitchfork"][which]
    plus, minus = r["landed"]
    assert plus is not None and minus is not None, r
    up, um = plus["steps"][0]["uphi"], minus["steps"][0]["uphi"]
    assert up * um < 0, ("the two directions must reach the two arms", up, um)
    assert abs(abs(up) - abs(um)) < 1e-8 * abs(up), ("the arms are mirror images", up, um)


def test_odd_rank_count_distributed(odd_rank_run):
    serial, np3 = odd_rank_run
    assert np3["distributed"] is True and np3["nproc"] == 3, np3
    assert np3["ndof"] == serial["ndof"] and np3["evect_len"] == np3["ndof"], (serial, np3)
    for key in ("b1", "b2", "b3", "b2_rel"):
        assert abs(np3[key] - serial[key]) < 1e-9 * max(abs(serial[key]), 1e-12), (key, serial, np3)


def test_nonzero_parameter_derivative_distributed(bratu_runs):
    """Bratu: the only case whose dR/dparameter is not identically zero.

    So it is the only one that actually exercises get_parameter_derivative, the bordered solve with a
    nonzero right-hand side, and dJdp_dot under MPI. It also keeps the fold end of the
    fold/branch-point measure honest -- a_rel near 1 here against 0 for the two branch points.
    """
    serial, dist = bratu_runs
    assert dist["distributed"] is True, dist
    assert serial["type"] == dist["type"] == "fold", bratu_runs
    assert serial["a_rel"] > 0.5 and abs(dist["a_rel"] - serial["a_rel"]) < 1e-9, bratu_runs
    for key in ("a", "b1", "b2", "b3"):
        assert abs(dist[key] - serial[key]) < 1e-9 * abs(serial[key]), (key, serial, dist)
    # The bordered solve really solved something, on both.
    for r in bratu_runs:
        assert r["psi01_residual"] < 1e-8, r
        assert r["psi01_orth"] < 1e-12, r
        # zeta and zeta_star annihilate the SAME L they border. Under MPI the matrix and the null
        # vectors come from separate calls, so this is what says they still belong together.
        assert r["L_zeta"] < 1e-9 and r["LT_zeta_star"] < 1e-9, r


def test_a_hopf_refuses_and_says_where_to_go_instead(tmp_path):
    """A Hopf sheds a periodic ORBIT, not a second steady branch.

    It used to die with a bare TypeError from inside a lambda: a Hopf's normal form DOES carry both
    predictor keys, so the "no branch predictor" guard could not catch it, and its
    perturbation_predictor takes (dp, omega*t) and returns an absolute state. Serial, and no MPI
    involved -- it lives here because this is where the other switch_branch refusals are.
    """
    import numpy as _np
    from pyoomph import Problem, ODEEquations, InitialCondition
    from pyoomph.expressions import var_and_test, partial_t, testfunction

    class NF(ODEEquations):
        def __init__(self, mu):
            super().__init__()
            self.mu = mu

        def define_fields(self):
            self.define_ode_variable("x", "y")

        def define_residuals(self):
            x, y = var_and_test("x")[0], var_and_test("y")[0]
            r2 = x ** 2 + y ** 2
            self.add_residual((partial_t(x) - (self.mu * x - y - x * r2)) * testfunction(x))
            self.add_residual((partial_t(y) - (x + self.mu * y - y * r2)) * testfunction(y))

    class P(Problem):
        def define_problem(self):
            self += (NF(self.get_global_parameter("mu")) + InitialCondition(x=0.0, y=0.0)) @ "nf"

    with P() as p:
        p.set_output_directory(str(tmp_path / "hopf"))
        p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.get_global_parameter("mu").value = -0.1
        p.solve()
        p.solve_eigenproblem(2)
        p.activate_bifurcation_tracking("mu", "hopf")
        p.solve()
        assert abs(float(p.get_global_parameter("mu").value)) < 1e-7, "the Hopf is at mu = 0"
        nf = p.classify_bifurcation("mu")
        assert nf.get("type") == "hopf", nf.get("type")
        with pytest.raises(RuntimeError, match="switch_to_hopf_orbit"):
            p.switch_branch("mu", normal_form=nf, quiet=True)

