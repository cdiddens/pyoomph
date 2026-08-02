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

"""Problem.desired_ndof: adapt towards a target problem size instead of a fixed error tolerance.

The controller replaces min/max_permitted_error before every adaptation with an order statistic of
the error distribution, so that roughly the right number of elements is selected. It runs before the
override stages, because those encode their verdict relative to those very thresholds -- see
dev_docs/spatial_error_estimators.md sections 4 and 5, and test_mandatory_refinement_survives_a_tight_budget
below, which is what pins that ordering.
"""

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class PeakedProblem(Problem):
    """A sharply peaked source in the middle of the unit square: a lot of refinement is warranted in
    one small region and none anywhere else, so a budget has somewhere obvious to spend itself."""

    def __init__(self, refine_top_to=None):
        super().__init__()
        self.refine_top_to = refine_top_to

    def define_problem(self):
        x = var("coordinate")
        eqs = PoissonEquation(source=exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.002))
        eqs += DirichletBC(u=0) @ "bottom"
        # Absolute errors: the controller itself is ordinal and does not care, but an absolute
        # measure is what makes a target meaningful across several meshes, so exercise that path.
        eqs += SpatialErrorEstimator(u=1, normalize_relative=0)
        if self.refine_top_to is not None:
            eqs += RefineToLevel(self.refine_top_to) @ "top"
        self += RectangularQuadMesh(N=8)
        self += eqs @ "domain"


def _adapt_until_settled(problem, max_steps=20):
    """Drive the adaptation the way the calling loops in Problem do, and return the ndof history."""
    history = []
    for _ in range(max_steps):
        nref, nunref = problem._adapt()
        problem.solve()
        history.append(problem.ndof())
        if nref == 0 and nunref == 0:
            return history
    raise AssertionError("desired_ndof controller did not settle in "+str(max_steps)+" steps: "+str(history))


@pytest.mark.parametrize("target", [3000, 20000])
def test_grows_to_the_target_and_terminates(target):
    """Both that it arrives, and that it stops -- a controller that oscillated about the target would
    never report nref==nunref==0 and every adaptation loop in Problem would run to its step limit."""
    with PeakedProblem() as problem:
        problem.desired_ndof = target
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve()
        history = _adapt_until_settled(problem)
        assert abs(history[-1]-target) <= problem.desired_ndof_tolerance*target
        # Monotone approach, not an overshoot that got corrected: the damping exists for that.
        assert history[-1] <= target*(1.0+problem.desired_ndof_tolerance)


def test_budget_is_spent_where_the_error_is():
    """A target on its own says nothing about *where*; the point is that the ranking decides."""
    with PeakedProblem() as problem:
        problem.desired_ndof = 8000
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve()
        _adapt_until_settled(problem)
        mesh = problem.get_mesh("domain")
        level = numpy.array([e.refinement_level() for e in mesh.elements()])
        cx = numpy.array([numpy.mean([e.node_pt(i).x(0) for i in range(e.nnode())]) for e in mesh.elements()])
        cy = numpy.array([numpy.mean([e.node_pt(i).x(1) for i in range(e.nnode())]) for e in mesh.elements()])
        r = numpy.hypot(cx-0.5, cy-0.5)
    assert level[r < 0.15].mean() > level[r > 0.4].mean()+1.5


def test_shrinks_back_when_the_target_is_lowered():
    """The direction that cannot be modelled well: oomph merges a father only if all its sons agree,
    so the number of elements below the threshold is an upper bound on what actually merges. The
    controller escalates its request when a step achieves nothing, which is what keeps this from
    creeping down one element at a time."""
    with PeakedProblem() as problem:
        problem.desired_ndof = 20000
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve()
        _adapt_until_settled(problem)
        grown = problem.ndof()
        assert grown > 15000

        problem.desired_ndof = 4000
        history = _adapt_until_settled(problem)
    assert history[-1] < 0.5*grown
    assert abs(history[-1]-4000) <= problem.desired_ndof_tolerance*4000


def test_target_below_the_initial_mesh_is_left_alone():
    """A target the mesh cannot reach is a statement about the mesh, not an error. Nothing can be
    merged at the coarsest level, so the controller must say so and stop rather than divide by an
    empty candidate list."""
    with PeakedProblem() as problem:
        problem.desired_ndof = 10
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve()
        coarse = problem.ndof()
        history = _adapt_until_settled(problem)
    assert history[-1] == coarse


def test_mandatory_refinement_survives_a_tight_budget():
    """The reason the controller runs before the override stages.

    RefineToLevel expresses itself as an error of 100*max_permitted_error. If the controller moved
    the thresholds afterwards, that sentinel would no longer sit above the refine threshold and the
    directive would be silently ignored -- or, worse for the sibling sentinel, every element an
    interface asked to protect would be unrefined instead. Deciding the thresholds first is what
    makes the directive win over the budget, which is the intended precedence.
    """
    with PeakedProblem(refine_top_to=3) as problem:
        problem.desired_ndof = 1200   # far too small to afford the directive
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve(spatial_adapt=12)
        mesh = problem.get_mesh("domain")
        ytop = max(n.x(1) for n in mesh.nodes())
        top_levels = [e.refinement_level() for e in mesh.elements()
                      if max(e.node_pt(i).x(1) for i in range(e.nnode())) > ytop-1e-9]
    assert top_levels and min(top_levels) >= 3


def test_thresholds_are_restored_when_desired_ndof_is_unset():
    """While it is set, min/max_permitted_error are outputs. Setting it back to None has to give the
    user their own values back, not leave the last controller state behind."""
    with PeakedProblem() as problem:
        problem.min_permitted_error = 1e-4
        problem.max_permitted_error = 1e-3
        problem.desired_ndof = 5000
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve(spatial_adapt=6)
        mesh = problem.get_mesh("domain")
        assert mesh.max_permitted_error != 1e-3   # the controller really did take them over

        problem.desired_ndof = None
        problem._adapt()
        assert mesh.max_permitted_error == 1e-3
        assert mesh.min_permitted_error == 1e-4


def test_works_through_the_ordinary_solve_entry_point():
    """Every adaptation path in Problem funnels through _adapt, so setting the attribute is all a
    user should have to do."""
    with PeakedProblem() as problem:
        problem.desired_ndof = 8000
        problem.max_refinement_level = 8
        problem.initial_adaption_steps = 0
        problem.quiet(True)
        problem.solve(spatial_adapt=12)
        ndof = problem.ndof()
    assert abs(ndof-8000) <= 0.2*8000
