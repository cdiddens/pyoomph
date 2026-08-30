# ========================================================================
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

"""When a job promises a complex PETSc/SLEPc, check it really arrived.

The counterpart of tests/test_spectra_present.py, and it exists for the same reason: everything that
needs SLEPc asks whether petsc4py imports and SKIPS when it does not, so a job whose PETSc did not
reach the test interpreter - the wrong cpXY artifact, a PYTHONPATH that cibuildwheel did not pass
through, a tarball unpacked somewhere other than the prefix it was configured with - goes green
while quietly testing the spectra fallback instead. That is precisely the failure the macOS arm64
wheel jobs (.github/workflows/wheels.yml, test_wheel_suite.yml) install the prebuilt PETSc to avoid,
so they set PYOOMPH_EXPECT_COMPLEX_SLEPC=1 and this turns the skip into a red job.

Nothing else sets it: a from-source build, and every other platform's wheel, is entitled to have no
PETSc at all and run on spectra over Pardiso.
"""

import os

import pytest


def _expected():
    if not os.environ.get("PYOOMPH_EXPECT_COMPLEX_SLEPC", ""):
        pytest.skip("PYOOMPH_EXPECT_COMPLEX_SLEPC is not set; this installation may legitimately "
                    "have no PETSc/SLEPc")


def test_petsc4py_and_slepc4py_import_and_are_complex():
    _expected()
    import numpy
    from petsc4py import PETSc  # type:ignore
    import slepc4py  # type:ignore  # noqa: F401
    # A REAL build imports just as happily and then truncates every complex target to its real part,
    # which is a wrong answer rather than an error - so the scalar type is the assertion, not the
    # import. See CLAUDE.md on why the complex arch is the one the eigen tests need.
    assert PETSc.ScalarType is numpy.complex128, \
        "the PETSc on PYTHONPATH is a real build; PYTHONPATH must point at the complex arch"


def test_petsc_has_mumps():
    _expected()
    from petsc4py import PETSc  # type:ignore
    # Without MUMPS the autodetection below picks spectra instead, and the augmented (bifurcation
    # tracking) systems - whose shifted matrices have empty diagonal entries - have no factorisation
    # that can handle them.
    assert PETSc.Sys.hasExternalPackage("mumps"), "this PETSc was built without MUMPS"


def test_the_autodetected_eigensolver_is_slepc():
    _expected()
    from pyoomph.solvers.generic import get_default_eigen_solver
    assert get_default_eigen_solver() == "slepc_mumps", \
        "PETSc is importable but pyoomph did not select it; see pyoomph/__init__.py"
