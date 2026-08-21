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

# One Equations instance may be added to several domains of the equation tree and has to stay
# fully functional on each of them.
#
# Nothing per-domain is stored on an equation object: BaseEquations._master() resolves the domain
# that is being compiled or traversed *right now*, through the bound code generator, falling back
# to a scan of the domains holding the instance when the instance itself is not bound (which is
# what happens when another domain reaches in, e.g. via InterfaceEquations.get_parent_equations()).
#
# Before that, a per-object _final_element pointer named a single domain, so sharing an instance
# was only safe as long as no domain merged it with an explicitly given sibling -- one merge and
# the sibling domains resolved their master to a code generator that was not being compiled.

import pytest

from pyoomph import Problem, DirichletBC, Equations, InterfaceEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import Scaling
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _ScaleProbe(Equations):
    """Records the scaling it resolves, keyed by the domain it resolved it on."""

    def __init__(self):
        super().__init__()
        self.at_codegen = {}
        self.at_hook = {}

    def define_residuals(self):
        self.at_codegen[self.get_current_code_generator().get_full_name()] = float(self.get_scaling("u"))

    def after_newton_solve(self):
        self.at_hook[self.get_current_code_generator().get_full_name()] = float(self.get_scaling("u"))


class _TwoDomains(Problem):
    def __init__(self, probe, scales):
        super().__init__()
        self.probe, self.scales = probe, scales

    def define_problem(self):
        for dom, sc in self.scales.items():
            self += RectangularQuadMesh(N=3, size=[1, 1], name=dom)
            self += PoissonEquation(name="u", source=1) @ dom
            self += Scaling(u=sc) @ dom
            self += self.probe @ dom
            for b in ("left", "right", "top", "bottom"):
                self += DirichletBC(u=0) @ (dom + "/" + b)


def test_a_shared_instance_resolves_the_domain_it_runs_on(tmp_path):
    scales = {"A": 2.0, "B": 7.0}
    probe = _ScaleProbe()
    with _TwoDomains(probe, scales) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
    # Both while the element code is generated and while the tree is walked afterwards.
    assert probe.at_codegen == scales
    assert probe.at_hook == scales


def test_a_shared_instance_survives_a_domain_merging_it_with_a_sibling(tmp_path):
    # The case that used to break: "B" holds the shared probe *and* an extra equation, so the
    # probe is no longer the sole equation of that domain.
    scales = {"A": 2.0, "B": 7.0}
    probe = _ScaleProbe()

    class P(_TwoDomains):
        def define_problem(self):
            super().define_problem()
            self += Scaling(spatial=1) @ "B"

    with P(probe, scales) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
    assert probe.at_codegen == scales
    assert probe.at_hook == scales


def test_the_shared_instance_is_not_duplicated_in_the_tree(tmp_path):
    scales = {"A": 2.0, "B": 7.0}
    probe = _ScaleProbe()
    with _TwoDomains(probe, scales) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        for dom in scales:
            node = p._equation_system.get_by_path(dom)
            assert sum(1 for e in node._equations if e is probe) == 1


# Sharing works because a placed node keeps its identity, and EquationTree.__add__ hands the children
# of both operands to the new node it returns. That makes "+" destructive: adding to a node that is
# already sitting in a tree drops the addition into a new tree and leaves the placed children pointing
# at that new root. It surfaced as "Mesh is None" in pin_redundant_lagrange_multipliers, arbitrarily
# far from the line that caused it, so the placement is now checked up front.

def test_adding_to_a_placed_tree_is_refused():
    eqs = PoissonEquation(name="u") + DirichletBC(u=0) @ "left"
    placed = eqs @ "domain"
    with pytest.raises(RuntimeError, match="already been placed"):
        eqs += DirichletBC(u=1) @ "right"
    # The refusal has to leave the placed tree intact, not half-merged.
    node = placed.get_child("domain")
    assert node.get_child("left")._parent is node


def test_adding_a_placed_tree_to_something_else_is_refused():
    inner = PoissonEquation(name="u") @ "domain"
    other = DirichletBC(u=0) @ "other"
    with pytest.raises(RuntimeError, match="already been placed"):
        _ = other + inner.get_child("domain")


def test_assembling_before_placing_still_works():
    eqs = PoissonEquation(name="u") + DirichletBC(u=0) @ "left" + DirichletBC(u=1) @ "right"
    placed = eqs @ "domain"
    assert sorted(placed.get_child("domain")._children) == ["left", "right"]


def test_placing_the_same_tree_on_two_domains_still_works():
    common = PoissonEquation(name="u") + DirichletBC(u=0) @ "left"
    a, b = common @ "domA", common @ "domB"
    assert sorted(a._children) == ["domA"] and sorted(b._children) == ["domB"]
    assert a.get_child("domA").get_child("left") is not b.get_child("domB").get_child("left")
