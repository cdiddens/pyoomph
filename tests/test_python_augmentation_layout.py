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

# The dof layout of a PYTHON augmentation (DofAugmentations, as built by the
# AugmentedAssemblyHandler family of pyoomph/generic/bifurcation_tools.py).
#
# These are serial tests of a distributed mechanism, which is the point: DofAugmentations now goes
# through the same AugmentedDofDistributionHelper as the C++ trackers of src/bifurcation.cpp, and the
# replicated branch of that helper is required to reproduce the previous behaviour exactly. What the
# tests below pin down is that it does -- the block lengths, the split() indexing and the teardown --
# so that the distributed branch can be switched on later without having to rediscover what serial
# used to do.
#
# The base-vs-augmented layout test covers a real defect rather than a hypothetical one.
# Problem::BaseDofDistributionScope used to find its helper only by dynamic_cast-ing the installed
# ASSEMBLY HANDLER, and a Python augmentation leaves the DEFAULT handler installed -- so while a
# FoldTracker was active, get_base_dof_distribution_info() reported the AUGMENTED row layout. Every
# consumer of the base layout (the eigensolver's row split, and the distributed backend to come) would
# have been told 2N+1 where the answer is N.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.bifurcation_tools import FoldTracker, HopfTracker


class Bratu(Equations):
    """laplace(u) + lam*exp(u) = 0. Stationary, so there is no mass matrix: the eigenvector guess is
    supplied explicitly and no eigensolve is needed, which keeps these tests about the layout."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) - weak(self.lam * exp(u), v))


class BratuProblem(Problem):
    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += LineMesh(N=self.N, size=1, name="domain")
        self.lam = self.define_global_parameter(lam=1.0)
        eqs = Bratu(self.lam)
        for b in ["left", "right"]:
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"


def _solved(tmp_path, N=8):
    p = BratuProblem(N)
    p.set_output_directory(str(tmp_path))
    return p


def test_base_layout_is_visible_while_a_python_augmentation_is_installed(tmp_path):
    with _solved(tmp_path) as p:
        p.initialise()
        p.solve()
        base_ndof = p.ndof()
        p.set_custom_assembler(FoldTracker(p, "lam", eigenvector=numpy.ones(base_ndof)))

        aug_n = p._get_dof_distribution_info()[0]
        bas_n = p._get_base_dof_distribution_info()[0]
        assert aug_n == 2 * base_ndof + 1, "a fold augments to 2N+1 (N=%d), got %d" % (base_ndof, aug_n)
        assert bas_n == base_ndof, (
            "the BASE layout reports %d while a Python augmentation is installed; it must report the "
            "unaugmented %d" % (bas_n, base_ndof))
        assert p._get_n_unaugmented_dofs() == base_ndof
        p.set_custom_assembler(None)


def test_split_blocks_have_their_registered_lengths(tmp_path):
    """split() reads the vector blocks and scalars, not a flat walk over Dof_pt.

    Index 0 is the base block and index i>0 the (i-1)-th registered entry, which is the convention
    every caller in bifurcation_tools.py relies on (`split(startindex=1)` = "everything I added").
    """
    with _solved(tmp_path) as p:
        p.initialise()
        p.solve()
        n = p.ndof()
        tracker = FoldTracker(p, "lam", eigenvector=numpy.ones(n))
        p.set_custom_assembler(tracker)
        aug = tracker.get_augmented_dofs()

        # a single named block
        V, = aug.split(startindex=1, endindex=2)
        assert len(V) == n

        # "everything I added": a fold registers the eigenvector plus the parameter
        blocks = aug.split(startindex=1)
        assert [len(b) for b in blocks] == [n, 1]

        # index 0 is the base block
        base, = aug.split(startindex=0, endindex=1)
        assert len(base) == n

        # the parameter block reads the live global parameter
        assert blocks[1][0] == pytest.approx(p.lam.value)
        p.set_custom_assembler(None)


def test_hopf_registers_two_vectors_and_two_scalars(tmp_path):
    """A layout with more than one vector block and a scalar on either side of them.

    Fold alone would not catch an off-by-one in the block bookkeeping: it has exactly one of each.
    """
    with _solved(tmp_path) as p:
        p.initialise()
        p.solve()
        n = p.ndof()
        guess = numpy.ones(n) + 1j * numpy.ones(n)
        tracker = HopfTracker(p, "lam", eigenvector=guess, omega=1.0)
        p.set_custom_assembler(tracker)
        # [u | Phi | Psi | param | Omega]
        assert p._get_dof_distribution_info()[0] == 3 * n + 2
        assert p._get_base_dof_distribution_info()[0] == n
        assert [len(b) for b in tracker.get_augmented_dofs().split(startindex=1)] == [n, n, 1, 1]
        p.set_custom_assembler(None)


def test_teardown_restores_the_base_dof_vector(tmp_path):
    """Installing and removing an augmentation repeatedly must be a no-op on the layout.

    reset_augmented_dof_vector_to_nonaugmented() now asks the helper to put the distribution back
    rather than rebuilding it non-distributed unconditionally -- which was wrong whenever the base
    distribution was itself distributed.
    """
    with _solved(tmp_path) as p:
        p.initialise()
        p.solve()
        n = p.ndof()
        for _ in range(3):
            p.set_custom_assembler(FoldTracker(p, "lam", eigenvector=numpy.ones(n)))
            assert p.ndof() == 2 * n + 1
            p.set_custom_assembler(None)
            assert p.ndof() == n, "the dof vector was not restored to its unaugmented length"
            assert p._get_n_unaugmented_dofs() == 0


def test_a_wrongly_sized_guess_is_refused(tmp_path):
    """An augmented vector block must have one entry per base dof.

    Previously a short vector simply pushed fewer dof pointers than the layout assumed, which showed
    up later as a Newton update applied to the wrong unknowns.
    """
    with _solved(tmp_path) as p:
        p.initialise()
        p.solve()
        n = p.ndof()
        with pytest.raises(RuntimeError, match="one entry per base dof"):
            p.set_custom_assembler(FoldTracker(p, "lam", eigenvector=numpy.ones(n - 2)))
