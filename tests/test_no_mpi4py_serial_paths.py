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

# The serial paths through pyoomph/generic/mpi.py, with mpi4py removed from the import system.
#
# Every wheel on PyPI is built with PYOOMPH_USE_MPI=OFF and installs no mpi4py, so a `from mpi4py
# import MPI` on a line a single process reaches is an immediate ModuleNotFoundError for every user
# who pip-installed pyoomph. Neither this machine nor the nightly can notice: both have an
# MPI-enabled build, hence mpi4py, so the import succeeds and the serial path looks fine.
#
# It has gone wrong once. mpi_allgather_vector() imported mpi4py ABOVE its own `nproc<=1` early
# return, which took out Temporal_ODEs/lorenz_lyapunov.py and
# Temporal_ODEs/bifurcation_branch_switching.py on three of the four platforms of the tutorial CI
# run of 12th October 2026 - the Lyapunov spectrum and classify_bifurcation() both reach it
# serially, through the replicating default of
# GenericLinearSystemSolver.solve_python_built_distributed().
#
# So mpi4py is blocked here AFTER pyoomph has been imported. Blocking it before would send mpi.py
# down its own no-MPI branch, which already works by construction and is not what broke.

import builtins
import sys

import numpy
import pytest
from scipy.sparse import csr_matrix

from pyoomph.generic import mpi as _mpi


@pytest.fixture
def without_mpi4py(monkeypatch):
    """mpi4py unimportable, and mpi.py answering the way a build without MPI answers.

    get_mpi_nproc() returns 0 there, not 1 - see the `no_mpi_file.exists()` branch of mpi.py - so
    that is what is installed here. A guard written as `== 1` would pass this test and still fail
    for the users it is about.
    """
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ModuleNotFoundError("No module named 'mpi4py'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_mpi, "get_mpi_nproc", lambda comm=None: 0)
    monkeypatch.setattr(_mpi, "get_mpi_world_comm", lambda: None)
    monkeypatch.setattr(builtins, "__import__", blocking_import)
    for name in [k for k in sys.modules if k == "mpi4py" or k.startswith("mpi4py.")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ModuleNotFoundError):
        __import__("mpi4py")
    yield


def test_allgather_vector_is_the_vector(without_mpi4py):
    b = numpy.array([1.0, 2.0, 3.0])
    got = _mpi.mpi_allgather_vector(3, 0, 3, b, context="serial, no mpi4py")
    assert numpy.allclose(got, b)
    # float64 even if the caller handed in something else: the MPI branch returns that, and a
    # caller that then hands the result to a solver must not see a different dtype serially.
    assert got.dtype == numpy.float64
    assert _mpi.mpi_allgather_vector(2, 0, 2, [4, 5]).dtype == numpy.float64


def test_allgather_square_csr_is_the_matrix(without_mpi4py):
    mat = csr_matrix(numpy.eye(3))
    assert _mpi.mpi_allgather_square_csr(3, 0, 3, mat) is mat
