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

"""The wheels must actually have Spectra - a cheap check that runs on every wheel.

Same argument as tests/test_openmp_present.py. tests/test_spectra_eigensolver.py skips wholesale when
``has_spectra`` is false, so a wheel whose Spectra/Eigen download failed at configure time passes it
green - and on Windows that wheel would have no eigensolver capable of targeting an eigenvalue at
all, since there is no PETSc/SLEPc there. Only the wheel builds set PYOOMPH_EXPECT_SPECTRA; a
from-source build is entitled to be configured with -DPYOOMPH_HAS_SPECTRA=OFF.
"""

import os

import pytest

from pyoomph import _pyoomph_core


def test_spectra_is_compiled_in_when_the_build_promised_it():
    if not os.environ.get("PYOOMPH_EXPECT_SPECTRA", ""):
        pytest.skip("PYOOMPH_EXPECT_SPECTRA is not set; this build may legitimately have no Spectra")
    assert _pyoomph_core.has_spectra, \
        "this build was configured with PYOOMPH_HAS_SPECTRA=ON but the extension reports no Spectra"


def test_the_spectra_backend_registers_when_it_is_compiled_in():
    if not getattr(_pyoomph_core, "has_spectra", False):
        pytest.skip("this pyoomph build was compiled without Spectra")
    # Compiled in but unimportable would mean the extension and pyoomph/solvers/spectra.py disagree,
    # which the eigensolver autodetection silently papers over by falling back to the next backend.
    from pyoomph.solvers.spectra import SpectraEigenSolver
    from pyoomph.solvers.generic import GenericEigenSolver
    assert GenericEigenSolver._registered_solvers["spectra"] is SpectraEigenSolver
