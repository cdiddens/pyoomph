#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# Which solvers "import pyoomph" settles on when nobody says. Each case here is a fresh interpreter
# with some backends made unimportable, which is the only honest way to test it: what both cascades
# really depend on is whether an import succeeds. (The linear one also runs once, inline, at import
# time, so it cannot be re-evaluated in-process with a different answer at all.)
#
# LINEAR, non-macOS:
#     pardiso -> petsc_mumps -> mumps -> superlu
# pardiso and petsc_mumps stay in front so that no existing installation changes solver. "mumps" (the
# standalone MUMPS from the separate pyoomph_mumps package) sits ahead of superlu, and on macOS ahead
# of accelerate too, because it is a real sparse direct solver and they are the last resort.
#
# EIGEN:
#     slepc_mumps -> spectra(+Pardiso) -> mumps -> spectra(+SuperLU) -> pardiso -> accelerate -> scipy
# Spectra is compiled into pyoomph, so it is available on every build and nothing placed after it
# could ever be reached. What actually varies is the factorisation of J - sigma*M behind the same
# Arnoldi iteration - MKL Pardiso, MUMPS, or scipy's SuperLU - which is why the MUMPS backend belongs
# BETWEEN the two Spectra outcomes rather than after them.
#
# Getting either wrong is not loud: every one of these backends returns an answer, so a mis-ordered
# cascade shows up as a machine that is quietly several times slower than it should be.

import os
import subprocess
import sys

import pytest

_PROBE = r"""
import sys, warnings
BLOCK = set(a for a in sys.argv[1].split(",") if a)

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name in BLOCK or name.split(".")[0] in BLOCK:
            raise ImportError("blocked for the test: " + name)
        return None

sys.meta_path.insert(0, _Blocker())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import pyoomph
    from pyoomph.solvers.generic import get_default_linear_solver
    print("SOLVER:" + str(get_default_linear_solver()))
    for x in w:
        if "falling back" in str(x.message):
            print("WARN:" + " ".join(str(x.message).split()))
"""


def _autoselect(blocked):
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "always"
    out = subprocess.run([sys.executable, "-c", _PROBE, ",".join(blocked)],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    solver = warn = None
    for line in out.stdout.splitlines():
        if line.startswith("SOLVER:"):
            solver = line[len("SOLVER:"):]
        elif line.startswith("WARN:"):
            warn = line[len("WARN:"):]
    assert solver is not None, out.stdout + out.stderr
    return solver, warn


# The names that have to be made unimportable to knock each backend out. pardiso is reached through
# pyoomph's own module (there is no separate distribution to block), the other two are the packages.
_PARDISO = "pyoomph.solvers.pardiso"
_PETSC = "petsc4py"
_MUMPS = "pyoomph_mumps"

# What "the cascade fell all the way through" looks like. NOT just superlu: on macOS
# _select_default_linear_solver tries Apple's Accelerate first, so a mac with it ends there instead.
# Every "nothing better was installed" guard below has to accept both, or it reads a perfectly normal
# macOS cascade as a solver that should not have won - which is how two of these tests failed the
# 0.2.2 release gate on both macOS legs while linux passed.
_LAST_RESORT = ("superlu", "accelerate")


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS has its own cascade, with accelerate in it")
def test_the_standalone_mumps_is_preferred_over_superlu():
    """The rung this test file exists for: mumps before the last-resort superlu."""
    solver, _ = _autoselect([_PARDISO, _PETSC])
    if solver in _LAST_RESORT:
        pytest.skip("pyoomph_mumps is not installed here, so there is nothing to prefer")
    assert solver == "mumps"


def test_the_cascade_ends_in_a_last_resort_solver_and_says_why():
    """With pardiso, PETSc and the standalone MUMPS all gone, the cascade must still land somewhere
    and must explain itself.

    WHICH solver it lands on is platform-dependent - the same reason the test above carries a darwin
    skipif, which this one was missing:
    _select_default_linear_solver tries accelerate before superlu, so a macOS runner with Apple's
    Accelerate framework ends in "accelerate" and everything else ends in "superlu". Asserting
    "superlu" unconditionally is what made the 0.2.2 release gate's macOS legs fail with
    `assert 'accelerate' == 'superlu'` while linux passed. Both are legitimate last resorts and both
    warn through the same _warn_suboptimal_solver, so the interesting half of this test - that the
    warning names why each better candidate was skipped - is asserted on every platform rather than
    skipped on macOS.
    """
    solver, warn = _autoselect([_PARDISO, _PETSC, _MUMPS])
    assert solver in _LAST_RESORT, solver
    if sys.platform != "darwin":
        assert solver == "superlu", solver
    assert warn is not None, "falling back to " + solver + " must warn"
    # Both skipped candidates have to name their reason: "not installed" and "installed but built
    # against the other MPI setting" read identically otherwise, and only one of them is a rebuild.
    assert "PETSc/MUMPS was not used because" in warn, warn
    assert "standalone MUMPS" in warn, warn


def test_pardiso_still_wins_where_it_is_available():
    """Nothing added below it may take an existing installation's solver away."""
    solver, _ = _autoselect([])
    if solver != "pardiso":
        pytest.skip("pardiso is not available here")
    assert solver == "pardiso"


def test_petsc_mumps_stays_ahead_of_the_standalone_mumps():
    solver, _ = _autoselect([_PARDISO])
    if solver in _LAST_RESORT:
        pytest.skip("neither PETSc/MUMPS nor pyoomph_mumps is installed here")
    assert solver in ("petsc_mumps", "mumps")
    if solver == "mumps":
        pytest.skip("PETSc/MUMPS is not available here, so there is no ordering to check")


# =============================================================================================
# Eigensolver
# =============================================================================================

_EIGEN_PROBE = r"""
import sys
BLOCK = set(a for a in sys.argv[1].split(",") if a)

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name in BLOCK or name.split(".")[0] in BLOCK:
            raise ImportError("blocked for the test: " + name)
        return None

sys.meta_path.insert(0, _Blocker())
import pyoomph
print("EIGEN:" + str(pyoomph._autodetect_eigen_solver()))
"""


def _autoselect_eigen(blocked):
    out = subprocess.run([sys.executable, "-c", _EIGEN_PROBE, ",".join(blocked)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    for line in out.stdout.splitlines():
        if line.startswith("EIGEN:"):
            return line[len("EIGEN:"):]
    raise AssertionError(out.stdout + out.stderr)


def test_the_mumps_eigensolver_is_taken_when_slepc_and_pardiso_are_missing():
    """The rung this was added for: MUMPS rather than Spectra-on-SuperLU."""
    got = _autoselect_eigen([_PETSC, _PARDISO])
    if got == "spectra":
        pytest.skip("pyoomph_mumps is not installed here, so Spectra falls back to SuperLU as before")
    assert got == "mumps"


def test_spectra_still_wins_while_pardiso_is_there():
    """MKL Pardiso ahead of MUMPS: nothing may take an existing installation's backend away."""
    got = _autoselect_eigen([_PETSC])
    if got != "spectra":
        pytest.skip("MKL Pardiso is not available here")
    assert got == "spectra"


def test_slepc_stays_first():
    got = _autoselect_eigen([])
    if got != "slepc_mumps":
        pytest.skip("PETSc/SLEPc with MUMPS is not available here")
    assert got == "slepc_mumps"


def test_without_mumps_the_eigen_cascade_is_unchanged():
    """Blocking pyoomph_mumps must give exactly what the cascade gave before it existed."""
    assert _autoselect_eigen([_PETSC, _PARDISO, _MUMPS]) == "spectra"
