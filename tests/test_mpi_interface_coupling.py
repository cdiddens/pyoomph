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

# Distributed (MPI) half of the coupled-interface adaptivity campaign. Same harness as the 2D/3D
# adaptivity modules -- each test launches tests/mpi_worker.py under `mpirun -n N ... --distribute` and
# compares the per-rank results against a serial reference computed in-process from the SAME definitions
# (tests/two_domain_cases.py).
#
# Distribution makes the coupled-interface problem harder in a way that is worth spelling out, because
# it is not the same difficulty as in the serial case. Two coupled domains share no nodes, so the
# partitioner sees two DISCONNECTED components and is free to cut them differently. A rank therefore
# routinely holds one side of an interface facet pair and not the other. That has two consequences:
#
#   * the conformity decision has to be made from GLOBAL facet data, not from what a rank can see, or
#     ranks refine different elements and the halo layer drifts out of step with its owner (this is the
#     same class of defect as dev_docs/mixed_adapt_validation.md section 9.8);
#   * conformity alone is then still not sufficient for connect_interface_elements_by_kdtree, which is
#     rank-local: the opposite element must also be PRESENT on the rank that needs it. That is a
#     halo-coverage property, provided by the set_must_be_kept_as_halo marking in
#     Problem.actions_before_distribute, and Problem.check_interface_conformity() counts and reports the
#     two failure modes separately because the matcher's own error message cannot tell them apart.
#
# There is also an ordering constraint that shows up only here: oomph's Problem::distribute() refuses to
# run on a mesh that is "no longer uniformly refined". Coupled domains asked for different initial
# uniform levels are therefore levelled DOWN to what they share before distribution, and the remainder is
# applied afterwards (Problem._defer_uneven_initial_refinement). test_uneven_initial_levels_survive_
# distribution is what holds that in place.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import two_domain_cases
from test_mpi_adaptivity import _check, _SKIP_REASON  # the harness, shared with the 2D/3D modules

pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow, pytest.mark.campaign]

_KINDS = two_domain_cases.MESH_KINDS


def _run(cases, nproc, tmp_path, **kw):
    return _check(cases, nproc, tmp_path, mod=two_domain_cases, **kw)


@pytest.mark.parametrize("eq", ["connect1", "connect2", "connect12"])
def test_connected_fields_distributed(eq, tmp_path):
    # The asymmetric criterion is stated for "lower" only, so under distribution the ranks that hold
    # "upper" have to learn about it through the global facet exchange -- there is no local evidence for
    # it anywhere in their partition.
    _run([(k, eq, (1, 2, "level")) for k in _KINDS], 2, tmp_path)


@pytest.mark.parametrize("crit", ["size", "callback", "estimator"])
def test_refinement_criteria_distributed(crit, tmp_path):
    # One test per criterion, since what decides whether a criterion is dangerous is WHERE it is stated
    # rather than what it computes. "estimator" is the realistic one: no explicit level anywhere, just a
    # Z2 estimate on a field that is only sharp in the lower domain.
    _run([(k, "connect1", (1, 2, crit)) for k in _KINDS], 2, tmp_path)


def test_moving_mesh_distributed(tmp_path):
    # ConnectMeshAtInterface: the interface geometry is an unknown, so the facet positions the conformity
    # machinery keys on are solution-dependent -- and each rank has to arrive at the same keys.
    _run([(k, "ale", (1, 2, "level")) for k in _KINDS], 2, tmp_path)


def test_multilevel_jump_distributed(tmp_path):
    # A THREE-level jump across the interface, from an unrefined base: the repair has to iterate, and
    # every iteration is a collective that all ranks must enter together -- including ranks with nothing
    # of their own left to refine.
    _run([(k, "connect1", (0, 3, "level")) for k in _KINDS], 2, tmp_path)


def test_four_ranks(tmp_path):
    # More ranks than domains, so at least one rank is guaranteed to hold part of only one side of the
    # interface. That is the configuration in which a rank-local conformity decision would be wrong.
    _run([("quad", "connect1", (1, 2, "level")), ("mixed", "connect1", (1, 2, "callback")),
          ("tri_crossed", "connect2", (0, 3, "level"))], 4, tmp_path)


def test_uneven_initial_levels_survive_distribution(tmp_path):
    # The two domains ask for different INITIAL uniform levels, which is what forces the level-down /
    # apply-the-rest-afterwards split around distribute(). Before that split existed this died inside
    # Problem::distribute() with "at least one of your meshes is no longer uniformly refined" -- so this
    # case is the one that would notice the split being removed or reordered.
    _run([("quad", "connect1", (1, 2, "level")), ("quad", "connect1", (0, 3, "level"))], 2, tmp_path)


def test_four_domains_at_a_cross_point_distributed(tmp_path):
    # Four domains meeting at a cross point, so the coupling graph is a CYCLE and the partitioner sees
    # FOUR disconnected components to cut independently. A rank can therefore hold a piece of one domain
    # and nothing of the domain its refinement demand has to reach -- the global facet exchange is the
    # only thing that can carry it.
    _run([(k, "connect1", (0, 0, "level")) for k in two_domain_cases.FOUR_DOMAIN_KINDS], 2, tmp_path)


def test_conformity_check_stays_clean_under_throw(tmp_path):
    # Run the whole thing with the conformity/halo cross-check in THROWING mode, so a divergence between
    # the processes fails the job at the adapt that created it rather than surfacing later as a wrong
    # ndof, an inf residual or a deadlock.
    cases = [(k, "connect1", (1, 2, "callback")) for k in _KINDS]
    cases += [("quad", "ale", (1, 2, "level")), ("mixed", "connect12", (0, 3, "level"))]
    _run(cases, 2, tmp_path, extra_env={"PYOOMPH_CHECK_HALO_CONSISTENCY": "2"})
