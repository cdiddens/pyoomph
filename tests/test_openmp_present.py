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

"""The wheels must actually have OpenMP - a cheap check that runs on every wheel.

tests/test_openmp_assembly.py proves the threaded element loop is bit-identical to the serial one,
but every one of its tests is skipped when ``has_openmp`` is false, so a wheel that lost OpenMP
passes it green. That is precisely the regression the release build cares about, and it needs an
assertion rather than a skip.

Two separate failures are possible and both are checked here:

* OpenMP was not COMPILED IN. On macOS this is the normal state of affairs - AppleClang ships no
  OpenMP - and .github/workflows/wheels.yml builds a static libomp for it (citools/build_static_
  libomp.sh); on all platforms it then passes PYOOMPH_USE_OPENMP=ON, which makes a missing OpenMP a
  configure error. Only the wheel builds set PYOOMPH_EXPECT_OPENMP, because a from-source build
  defaults to AUTO and is entitled to have no OpenMP at all.
* OpenMP was compiled in but the threaded loop never RUNS. Nothing above catches that: the serial
  fallback gives the right answer, so it is invisible without asking how many parallel assemblies
  were done. This is the one that the macOS static-libomp arrangement could plausibly break.

Deliberately tiny (a 6x6 Poisson, no solve): this file is in the reduced test set the cp310/cp311
wheels run (pyproject.toml), which exists to stay well under a minute.
"""

import os

import pytest

from pyoomph import *
from pyoomph.equations.poisson import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph import _pyoomph_core


class _TinyPoisson(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(N=6)
        eqs = PoissonEquation(name="u", source=1)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        self += eqs @ "domain"


def test_openmp_is_compiled_in_when_the_build_promised_it():
    if not os.environ.get("PYOOMPH_EXPECT_OPENMP", ""):
        pytest.skip("PYOOMPH_EXPECT_OPENMP is not set; this build may legitimately have no OpenMP")
    assert _pyoomph_core.has_openmp, \
        "this build was configured with PYOOMPH_USE_OPENMP=ON but the extension reports no OpenMP"


def test_a_threaded_backend_is_compiled_in_when_the_build_promised_it():
    # The backend-agnostic promise, for platforms that thread through something other than OpenMP:
    # macOS uses GCD/libdispatch (has_openmp is false there, has_gcd true), so the wheel jobs set
    # PYOOMPH_EXPECT_THREADED rather than PYOOMPH_EXPECT_OPENMP on macOS. Either way a wheel that lost
    # its threaded loop must fail rather than silently make --omp N a no-op.
    if not os.environ.get("PYOOMPH_EXPECT_THREADED", ""):
        pytest.skip("PYOOMPH_EXPECT_THREADED is not set; this build may legitimately have no threaded loop")
    assert getattr(_pyoomph_core, "has_threaded_assembly", _pyoomph_core.has_openmp), \
        "this build was configured for a threaded element loop (OpenMP or GCD) but the extension reports none"


@pytest.mark.skipif(not getattr(_pyoomph_core, "has_threaded_assembly", _pyoomph_core.has_openmp),
                    reason="this build has no threaded element loop (neither OpenMP nor GCD)")
def test_the_threaded_element_loop_actually_runs(tmp_path):
    with _TinyPoisson() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p.assemble_jacobian(with_residual=True)   # serial, to get past any first-call setup
        before = p._get_parallel_assemblies_done()
        p._set_num_assembly_threads(2)
        p.assemble_jacobian(with_residual=True)
        ran = p._get_parallel_assemblies_done() - before
        p._set_num_assembly_threads(1)
    assert ran > 0, \
        "OpenMP is compiled in, but the threaded element loop declined to run - the runtime " \
        "OpenMP library is missing or unusable in this wheel"
