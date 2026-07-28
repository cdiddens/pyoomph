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
# Everything the serial campaign covers now passes, so the matrix here is restricted only for COST (see
# _REPRESENTATIVE below) and for the one remaining distributed-only defect (_MPI_NONUNIFORM_BROKEN).

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_cases_3d
from box_mesh_3d import ALL_LAYOUTS
from test_mpi_adaptivity import _check, _SKIP_REASON  # the harness, shared with the 2D module

pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

# The SERIAL campaign (test_adaptive_3d_campaign.py) already sweeps all 11 layouts exhaustively. The job of
# the distributed campaign is narrower -- to show that PARTITIONING does not break what serial already
# proved -- so most of its matrices use a representative subset instead of all 11: every pure family, plus
# "all_four", which puts bricks, tets, wedges, pyramids AND the brick-to-tet transition cells in one mesh
# and therefore carries every legal interface kind at once. Running the remaining six mixtures adds
# permutations of interfaces already covered here, at roughly six minutes.
# "neumann" is the exception and keeps all 11: boundary-facet propagation under refinement is the most
# shape-dependent thing in the campaign, so it is worth the full sweep.
_REPRESENTATIVE = ["hex", "tet", "wedge", "pyr", "all_four"]

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
    kinds = ALL_LAYOUTS if eq == "neumann" else _REPRESENTATIVE
    kinds = [k for k in kinds if not (eq == "neumann" and k in _MPI_NONUNIFORM_BROKEN)]
    _check([(k, eq, (1, 2)) for k in kinds], 2, tmp_path, mod=box_cases_3d)


@pytest.mark.parametrize("eq", ["mixed12", "constrain12", "unconstrain12", "stokes_th", "ale"])
def test_distributed_3d_multispace_uniform_all_families(eq, tmp_path):
    # Two coexisting continuous spaces (a C1 field on C2 geometry) at uniform refinement. Distributing this
    # exercises the C1 field's own hang slot across partition boundaries.
    _check([(k, eq, (1, 1)) for k in _REPRESENTATIVE], 2, tmp_path, mod=box_cases_3d)


@pytest.mark.parametrize("eq", ["mixed12", "constrain12", "unconstrain12", "stokes_th", "ale"])
def test_distributed_3d_multispace_nonuniform(eq, tmp_path):
    # The same at the two-level NON-uniform state -- distributed 2:1 hanging with two coexisting continuous
    # spaces, which is the combination that needed the wedge/pyramid per-value interpolation hooks.
    kinds = [k for k in _REPRESENTATIVE if k not in _MPI_NONUNIFORM_BROKEN]
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
