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

# LyapunovExponentCalculator under MPI, in both regimes. There was no coverage of this class at all
# before -- only the Lorenz tutorial, a 3-dof ODE that cannot be distributed -- and it held two
# defects that only MPI reaches:
#
#   1. an unseeded numpy.random.rand for the initial perturbation basis, so the ranks drew different
#      vectors, took different Gram-Schmidt steps and would deadlock at the next collective. That is
#      B6 in dev_docs/mpi_augmented_systems.md, the same defect the deflation drivers had.
#   2. csr_matrix(..., shape=(n, n)) for the assembled J and M, where the assembly returns this
#      rank's (nrow_local, n) block with global column indices -- the B2 shape of mistake.
#
# It is now one code path for all three regimes: J and M are DistMatrix blocks on the base dof
# layout, every norm and projection in the Gram-Schmidt sweep is an allreduce, and the k
# perturbations are solved through one factorisation via LinearAlgebraBackend.solve_many.
#
# The problem is LINEAR on purpose (u_t = D u_xx + a u, Dirichlet). A linear system's Lyapunov
# exponents are the real parts of its Jacobian eigenvalues, a - D (m pi / L)^2, so there is an
# analytic answer and the test does not have to assume a chaotic trajectory survives a dof
# renumbering. k=2 is what reaches the Gram-Schmidt sweep, which is pure reductions.
#
# What is compared across regimes is the exponents and the Gram matrix of the final basis -- both
# numbering-independent. The basis itself is not: --distribute renumbers, so the seeded draw (made
# globally and sliced by global row) genuinely starts the regimes from different physical vectors.
# That is why the run has a prerelaxation time; see the comment in the worker.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_lyapunov_worker.py")


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
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:
        return "petsc4py not available"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# The regimes agree to 9-10 digits, measured; only the summation order of the reductions differs.
_ATOL = 1e-7


def _run(nproc, tmpdir, distribute, timeout=1800):
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, "--outdir", str(tmpdir)]
    if distribute:
        cmd += ["--distribute"]
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: the failure mode the unseeded RNG produced was a HANG, not a crash --
        # the ranks disagreed about the basis and sat in the next allreduce for ever.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%d distribute=%s)."
            "\n--- stdout tail ---\n%s" % (timeout, nproc, distribute, (e.stdout or "")[-3000:]))
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


def _serial_reference(tmpdir):
    sys.path.insert(0, _HERE)
    try:
        import mpi_lyapunov_worker
        return mpi_lyapunov_worker.run(outdir=os.path.join(str(tmpdir), "serial"))
    finally:
        sys.path.remove(_HERE)


@pytest.fixture(scope="module")
def serial_ref(tmp_path_factory):
    return _serial_reference(tmp_path_factory.mktemp("lyap_serial"))


def test_serial_exponents_match_the_analytic_spectrum(serial_ref):
    """The reference itself has to be right, or every comparison below is vacuous.

    Compared against the TIME-DISCRETE spectrum, not the continuous one. The perturbations are
    advanced by implicit Euler, so what is measured is log(1/(1 - lambda dt))/dt; at dt=0.01 that is
    1% below lambda_1 and 15% below lambda_2, so a comparison against the continuous value needs a
    20% tolerance and would pass on an exponent that was simply wrong. Against the discrete
    prediction the residual is the spatial discretisation alone, ~1.5% on both modes, so 5% here is
    a real assertion.
    """
    got, disc, cont = serial_ref["exponents"], serial_ref["discrete"], serial_ref["analytic"]
    assert len(got) == len(disc) == 2
    for i, (g, d) in enumerate(zip(got, disc)):
        assert abs(g - d) <= 0.05 * abs(d), \
            "exponent %d is %r; implicit Euler at dt=%r predicts %r (continuous %r)" % (
                i, g, serial_ref["dt"], d, cont[i])
    assert got[0] > got[1], "the exponents came back out of order: %r" % (got,)


def test_serial_basis_is_orthonormal(serial_ref):
    """The Gram-Schmidt sweep is the only place k>1 adds anything, and it is all reductions."""
    gram = numpy.array(serial_ref["gram"])
    assert numpy.allclose(gram, numpy.eye(gram.shape[0]), atol=1e-10), gram


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_exponents_match_serial(tmp_path, serial_ref, nproc, distribute):
    """The exponents, against the serial run, in both regimes.

    Under --distribute J and M are real row blocks and every norm and inner product in the sweep is
    an allreduce. A reduction done over the wrong rows does not crash and does not hang -- it
    returns a plausible exponent -- so this is the assertion that catches it.
    """
    per_rank = _run(nproc, tmp_path, distribute)
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert r["distributed"] is distribute
        assert r["ndof_global"] == serial_ref["ndof_global"], \
            "rank %d has %d global dofs, serial has %d" % (
                r["rank"], r["ndof_global"], serial_ref["ndof_global"])
    if distribute:
        # Not a formality: a non-distributed layout would make every assertion here pass while
        # testing the replicated path twice. The blocks must also tile the global system.
        assert sum(r["nrow_local"] for r in per_rank) == serial_ref["ndof_global"], \
            "the row blocks do not tile: %r" % ([r["nrow_local"] for r in per_rank],)
        assert all(r["nrow_local"] < r["ndof_global"] for r in per_rank), \
            "a rank owns every row, so this is not really distributed: %r" % (
                [(r["rank"], r["nrow_local"]) for r in per_rank],)
    else:
        assert all(r["nrow_local"] == r["ndof_global"] for r in per_rank)
    for r in per_rank:
        for i, (g, s) in enumerate(zip(r["exponents"], serial_ref["exponents"])):
            assert abs(g - s) <= _ATOL * max(1.0, abs(s)), \
                "rank %d exponent %d is %r, serial %r" % (r["rank"], i, g, s)
    # Every rank must report the same numbers: they are the result of reductions, so a rank left out
    # of one of them shows up here rather than in the comparison against serial.
    for r in per_rank[1:]:
        assert r["exponents"] == per_rank[0]["exponents"], \
            "rank %d disagrees with rank 0: %r vs %r" % (
                r["rank"], r["exponents"], per_rank[0]["exponents"])


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_basis_stays_orthonormal_under_mpi(tmp_path, nproc, distribute):
    """The Gram matrix of the final basis, which is numbering-independent where the basis is not."""
    per_rank = _run(nproc, tmp_path, distribute)
    for r in per_rank:
        gram = numpy.array(r["gram"])
        assert numpy.allclose(gram, numpy.eye(gram.shape[0]), atol=1e-10), \
            "rank %d: the perturbation basis is not orthonormal:\n%r" % (r["rank"], gram)
