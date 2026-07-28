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

# Distributed (MPI) 3D adaptivity tests. Reuses the harness of test_mpi_adaptivity.py verbatim -- the
# worker takes a --cases module, and both box_cases (2D) and box_cases_3d expose the same
# solve_case()/case_id() interface -- so this module only has to choose the case matrix.
#
# The matrix is restricted to the configurations that work SERIALLY: it is not informative to run a
# distributed test of something that is already broken in serial. Concretely, the multi-space cases
# (coupled C2+C1, Taylor-Hood Stokes, ALE) are run under non-uniform refinement only on bricks and tets,
# and on every layout only at uniform refinement -- see the header of test_adaptive_3d_campaign.py and
# dev_docs/mixed_adapt_validation.md §9 for why. The single-space cases (Poisson, Neumann) run on all 11
# layouts at the full two-level non-uniform state.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_cases_3d
from box_mesh_3d import ALL_LAYOUTS
from test_mpi_adaptivity import _check, _SKIP_REASON  # the harness, shared with the 2D module

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))

_MULTISPACE_OK = ["hex", "tet"]  # families whose C1 hang slot survives non-uniform refinement

# DISTRIBUTED-ONLY defect: on the pure-tet layout at the two-level non-uniform state, get_elemental_errors()
# throws an OomphException on SOME ranks but not others during the initial adaption
# (Problem._adapt_with_interfacial_errors -> Mesh.get_elemental_errors). Because the throw is asymmetric,
# the ranks that did not throw block forever in the next collective, so the symptom is a deadlock rather
# than a clean error -- that is what the harness's bounded subprocess timeout turns into a failure. It is
# equation-dependent (C1/C2 Poisson are fine on the same mesh; Neumann, Taylor-Hood and ALE are not) and it
# does NOT occur serially, nor at uniform refinement, nor on the layouts that merely CONTAIN tets
# (hex_tet, all_four, tet_wedge all pass). Pinned by test_distributed_3d_pure_tet_nonuniform_xfail below.
_MPI_NONUNIFORM_BROKEN = {"tet"}


@pytest.mark.parametrize("eq", ["poisson1", "poisson2", "neumann"])
def test_distributed_3d_single_space_all_families(eq, tmp_path):
    # Single-space problems on EVERY family combination at the two-level non-uniform state: the case that
    # exercises distributed 2:1 hanging across brick/tet/wedge/pyramid interfaces, including the
    # brick-to-tet transition cells.
    kinds = [k for k in ALL_LAYOUTS if not (eq == "neumann" and k in _MPI_NONUNIFORM_BROKEN)]
    _check([(k, eq, (1, 2)) for k in kinds], 2, tmp_path, mod=box_cases_3d)


@pytest.mark.parametrize("eq", ["mixed12", "constrain12", "unconstrain12", "stokes_th", "ale"])
def test_distributed_3d_multispace_uniform_all_families(eq, tmp_path):
    # Two coexisting continuous spaces (a C1 field on C2 geometry) on every family combination, at uniform
    # refinement -- where all families work serially. Distributing this exercises the C1 field's own hang
    # slot across partition boundaries.
    cases = [(kind, eq, (1, 1)) for kind in ALL_LAYOUTS]
    _check(cases, 2, tmp_path, mod=box_cases_3d)


@pytest.mark.parametrize("eq", ["mixed12", "constrain12", "unconstrain12", "stokes_th", "ale"])
def test_distributed_3d_multispace_nonuniform(eq, tmp_path):
    # The same, at the two-level NON-uniform state, on the families where that works serially.
    kinds = [k for k in _MULTISPACE_OK if k not in _MPI_NONUNIFORM_BROKEN]
    _check([(k, eq, (1, 2)) for k in kinds], 2, tmp_path, mod=box_cases_3d)


def test_distributed_3d_four_ranks(tmp_path):
    # More partitions than families: with 4 ranks the partition boundaries no longer align with the
    # family boundaries of the layout, so a rank routinely owns a slice of two different element families.
    cases = [(kind, eq, (1, 2)) for kind in ["all_four", "hex_tet"]
             for eq in ["poisson2", "neumann"]]
    cases += [(kind, eq, (1, 1)) for kind in ["all_four", "hex_pyr_wedge"]
              for eq in ["stokes_th", "ale", "unconstrain12"]]
    _check(cases, 4, tmp_path, mod=box_cases_3d)


@pytest.mark.xfail(strict=True, reason="asymmetric get_elemental_errors() throw on a distributed pure-tet "
                                       "mesh under non-uniform refinement, which then deadlocks the other "
                                       "ranks (see _MPI_NONUNIFORM_BROKEN above)")
def test_distributed_3d_pure_tet_nonuniform_xfail(tmp_path):
    # Pins the one distributed 3D configuration that is known broken, so it stays visible and the suite
    # tells us when it is fixed. Kept to a single case and a short timeout: the failure mode is a hang, so
    # this must not be allowed to stall a suite run.
    _check([("tet", "neumann", (1, 2))], 2, tmp_path, mod=box_cases_3d, timeout=240)
