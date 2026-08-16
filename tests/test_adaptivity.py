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

# Import pyoomph
from pyoomph import *
from pyoomph.expressions import *
# Also import the predefined harmonic oscillator equation
from pyoomph.meshes.simplemeshes import CircularMesh
from pyoomph.equations.poisson import *


class PoissonProblem(Problem):	
	def define_problem(self):
		self+=CircularMesh(radius=1,segments=["NE"])
		eqs=PoissonEquation(source=1)+DirichletBC(u=0)@"circumference"
		anasol=0.25*(1-dot(var("coordinate"),var("coordinate")))
		eqs+=IntegralObservables(error=(var("u")-anasol)**2)
		self+=eqs@"domain"
		

def test_without_adapt():
	with PoissonProblem() as problem:
		
		problem.solve()
		err=float(problem.get_mesh("domain").evaluate_observable("error"))
		assert err<1e-7
  
def test_with_adapt():
	with PoissonProblem() as problem:
		problem+=RefineToLevel(2)@"domain"
		problem+=RefineToLevel(4)@"domain/circumference"
		problem+=MeshFileOutput()@"domain"
		problem.solve()
		problem.output()
		err=float(problem.get_mesh("domain").evaluate_observable("error"))
		assert err<1e-10
  


# ---------------------------------------------------------------------------------------------
# SpatialErrorEstimator normalisation and compound-flux grouping: what the elemental errors mean,
# and how several independently added criteria combine.
# See dev_docs/spatial_error_estimators.md.
# ---------------------------------------------------------------------------------------------

import numpy
import pytest
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

_UNSET = object()


class _ScaledPoissonProblem(Problem):
	"""A peaked source of adjustable amplitude. Poisson is linear, so the amplitude scales the
	solution, its gradient, the flux jumps and the flux norm all by the same factor."""

	def __init__(self, normalize_relative=_UNSET, amplitude=1.0):
		super().__init__()
		self.normalize_relative = normalize_relative
		self.amplitude = amplitude

	def define_problem(self):
		x = var("coordinate")
		source = self.amplitude*exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.01)
		eqs = PoissonEquation(source=source)+DirichletBC(u=0)@"bottom"
		if self.normalize_relative is _UNSET:
			eqs += SpatialErrorEstimator(u=1)
		else:
			eqs += SpatialErrorEstimator(u=1, normalize_relative=self.normalize_relative)
		self += RectangularQuadMesh(N=8)
		self += eqs@"domain"


def _elemental_errors(normalize_relative=_UNSET, amplitude=1.0):
	with _ScaledPoissonProblem(normalize_relative, amplitude) as problem:
		problem.initial_adaption_steps = 0
		problem.max_refinement_level = 0
		problem.solve()
		return numpy.array(problem.get_mesh("domain").get_elemental_errors())


def test_normalize_relative_defaults_to_one():
	"""Not passing it must leave the historical, fully relative behaviour untouched."""
	# Not exactly equal: the threaded direct solver is not bit-reproducible run to run, and two runs
	# with identical arguments differ by the same ~1e-16 as these two do.
	assert numpy.allclose(_elemental_errors(), _elemental_errors(1.0), rtol=1e-12, atol=0.0)


def test_relative_error_is_blind_to_the_solution_scale():
	"""The defining property of the relative measure: it divides the mesh's own norm back out."""
	base = _elemental_errors(1.0, amplitude=1.0)
	scaled = _elemental_errors(1.0, amplitude=10.0)
	# Not exact only because of the +1e-9 offset on the denominator, which is an absolute constant
	# sitting in an otherwise scale-free expression (dev_docs/spatial_error_estimators.md 7.3).
	assert numpy.allclose(scaled, base, rtol=1e-6, atol=0.0)


def test_absolute_error_follows_the_solution_scale():
	"""And the defining property of the absolute one: it does not."""
	base = _elemental_errors(0.0, amplitude=1.0)
	scaled = _elemental_errors(0.0, amplitude=10.0)
	assert numpy.allclose(scaled, 10.0*base, rtol=1e-9, atol=0.0)


def test_intermediate_normalize_relative_is_the_geometric_blend():
	"""err/norm**p is exactly (err/norm)**p * err**(1-p), so a half-way value must be the geometric
	mean of the fully relative and the fully absolute error."""
	relative = _elemental_errors(1.0)
	absolute = _elemental_errors(0.0)
	half = _elemental_errors(0.5)
	assert numpy.allclose(half, numpy.sqrt(relative*absolute), rtol=1e-9, atol=0.0)


def test_absolute_error_shrinks_when_the_mesh_is_refined():
	"""What the relative measure cannot tell you: whether the mesh is good enough yet."""
	coarse = _elemental_errors(0.0)

	class Refined(_ScaledPoissonProblem):
		def define_problem(self):
			super().define_problem()
			self += RefineToLevel(2)@"domain"

	with Refined(0.0) as problem:
		problem.initial_adaption_steps = 2
		problem.max_refinement_level = 2
		problem.solve()
		fine = numpy.array(problem.get_mesh("domain").get_elemental_errors())
	assert fine.max() < 0.1*coarse.max()


def test_normalize_relative_out_of_range_is_rejected():
	with pytest.raises(ValueError, match="normalize_relative"):
		SpatialErrorEstimator(u=1, normalize_relative=1.5)


# --- compound-flux grouping -------------------------------------------------------------------


class _TwoFieldProblem(Problem):
	"""Two independent Poisson fields peaked in different corners of one domain, so that each has
	its own natural refinement region."""

	def __init__(self, criteria, weight=1.0, inner_factor=1.0):
		super().__init__()
		self.criteria, self.weight, self.inner_factor = criteria, weight, inner_factor

	def define_problem(self):
		x = var("coordinate")
		eqs = PoissonEquation(source=exp(-((x[0]-0.3)**2+(x[1]-0.3)**2)/0.01))+DirichletBC(u=0)@"bottom"
		eqs += PoissonEquation(name="w", source=exp(-((x[0]-0.8)**2+(x[1]-0.8)**2)/0.01))+DirichletBC(w=0)@"bottom"
		if "u" in self.criteria:
			eqs += SpatialErrorEstimator(u=1, group="u")
		if "w" in self.criteria:
			eqs += SpatialErrorEstimator(w=self.inner_factor, group="w", weight=self.weight)
		if self.criteria == "joint":
			eqs += SpatialErrorEstimator(u=1, w=1)
		self += RectangularQuadMesh(N=8)
		self += eqs@"domain"


def _two_field_errors(criteria, weight=1.0, inner_factor=1.0):
	with _TwoFieldProblem(criteria, weight, inner_factor) as problem:
		problem.initial_adaption_steps = 0
		problem.max_refinement_level = 0
		problem.solve()
		return numpy.array(problem.get_mesh("domain").get_elemental_errors())


def test_groups_are_combined_by_the_maximum():
	"""The composition rule that lets independent parts of a model contribute criteria without
	knowing about each other: two groups together are exactly the elementwise max of each alone."""
	u_only = _two_field_errors("u")
	w_only = _two_field_errors("w")
	both = _two_field_errors("uw")
	assert numpy.allclose(both, numpy.maximum(u_only, w_only), rtol=1e-10, atol=0.0)


def test_adding_a_criterion_can_only_raise_the_error():
	"""Monotonicity, which is what makes the max safe: a criterion can never mask another one, so
	adding one can only ever cause more refinement and less unrefinement."""
	u_only = _two_field_errors("u")
	both = _two_field_errors("uw")
	# The slack is set by the threaded direct solver, not by the estimator: re-running the *same*
	# problem moves the errors by ~1e-13 relative, so anything tighter than that tests the solver.
	assert numpy.all(both >= u_only*(1.0 - 1e-9))


def test_separate_groups_are_normalised_separately():
	"""Two groups get two norms; one joint group gets one. They must not agree."""
	grouped = _two_field_errors("uw")
	joint = _two_field_errors("joint")
	assert not numpy.allclose(grouped, joint, rtol=1e-6, atol=0.0)


def test_factor_inside_a_group_cancels_but_weight_does_not():
	"""A common factor on the expressions divides straight back out of that group's own norm; the
	weight is applied after the normalisation and therefore survives. This is the distinction that
	makes 'weight the droplet more than the gas' expressible at all."""
	base = _two_field_errors("uw")
	inner = _two_field_errors("uw", inner_factor=100.0)
	weighted = _two_field_errors("uw", weight=100.0)
	assert numpy.allclose(inner, base, rtol=1e-6, atol=0.0)
	assert not numpy.allclose(weighted, base, rtol=1e-3, atol=0.0)
	# Promoting the w criterion moves the worst element from u's peak to w's.
	assert numpy.argmax(weighted) != numpy.argmax(base)


def test_conflicting_settings_within_one_group_are_rejected():
	"""The norm is a property of the group, so all of a group's terms must agree about it. Different
	groups may of course disagree -- that is the point of having them."""

	class Conflicting(Problem):
		def define_problem(self):
			eqs = PoissonEquation(source=1)+DirichletBC(u=0)@"bottom"
			eqs += SpatialErrorEstimator(u=1, group="g", normalize_relative=1.0)
			eqs += SpatialErrorEstimator("mesh", group="g", normalize_relative=0.0)
			self += RectangularQuadMesh(N=4)
			self += eqs@"domain"

	with pytest.raises(RuntimeError, match="Conflicting normalization"):
		with Conflicting() as problem:
			problem.initialise()


# ---------------------------------------------------------------------------------------------
# Refinement criteria must survive a mesh replacement.
#
# RefineToLevel/RefineMaxElementSize state their criterion on the mesh object, from
# after_compilation. Remeshing builds new meshes but reuses the compiled code, so after_compilation
# is not called again: without Problem._reregister_refinement_directives() every criterion, bulk and
# interface alike, was gone from the first force_remesh() on, and the adaption inside the remesh had
# nothing left to act on - the mesh came back at its base level and stayed there.
# ---------------------------------------------------------------------------------------------

from pyoomph.equations.additional import RefineMaxElementSize


class _RemeshableQuarterCircle(GmshTemplate):
	def define_geometry(self):
		self.default_resolution = 0.3
		p00 = self.point(0, 0)
		if not self.is_remeshing():
			p10, p01 = self.point(1, 0), self.point(0, 1)
			self.circle_arc(p10, p01, center=p00, name="interface")
		else:
			# Rebuild the arc from where the nodes of the mesh being replaced actually are
			coords = self.get_boundary_coordinates("domain/interface", sort_along_axis="x+")
			pts = [self.point(x, y) for x, y in coords[0]]
			self.spline(pts, name="interface")
			p10, p01 = pts[-1], pts[0]
		self.create_lines(p10, "substrate", p00, "axis", p01)
		self.plane_surface("substrate", "axis", "interface", name="domain")


class _RemeshRefineProblem(Problem):
	def __init__(self, extra_eqs=None):
		super().__init__()
		self.extra_eqs = extra_eqs

	def define_problem(self):
		self += _RemeshableQuarterCircle()
		eqs = PoissonEquation(source=1)+DirichletBC(u=0)@"interface"
		if self.extra_eqs is not None:
			eqs += self.extra_eqs
		self += eqs@"domain"


def _levels(problem):
	"""(all bulk refinement levels, levels of the elements touching the interface)"""
	mesh = problem.get_mesh("domain")
	bulk = {e.refinement_level() for e in mesh.elements()}
	at_interface = {e.get_bulk_element().refinement_level() for e in mesh.get_mesh("interface").elements()}
	return bulk, at_interface


def test_refine_to_level_survives_remeshing():
	"""Both a bulk and an interface RefineToLevel must still act on the mesh a remesh builds."""
	pytest.importorskip("gmsh", reason="remeshing needs gmsh")
	eqs = RefineToLevel(1)+RefineToLevel(3)@"interface"
	with _RemeshRefineProblem(eqs) as problem:
		problem.max_refinement_level = 5
		problem.solve()
		bulk_before, interface_before = _levels(problem)
		assert min(bulk_before) >= 1 and max(interface_before) == 3, "wrong before remeshing already"

		problem.force_remesh()
		bulk_after, interface_after = _levels(problem)
		assert min(bulk_after) >= 1, \
			"the bulk RefineToLevel(1) did not act on the remeshed mesh: levels %s" % sorted(bulk_after)
		assert max(interface_after) == 3, \
			"the interface RefineToLevel(3) did not act on the remeshed mesh: levels %s" % sorted(interface_after)


def test_refine_max_element_size_survives_remeshing():
	"""Same for the size-based criterion, which uses the same registration mechanism."""
	pytest.importorskip("gmsh", reason="remeshing needs gmsh")
	# An element SIZE, i.e. the area in 2d: the template's resolution of 0.3 gives ~0.04 per element.
	with _RemeshRefineProblem(RefineMaxElementSize(0.02)) as problem:
		problem.max_refinement_level = 5
		# Explicitly, unlike RefineToLevel: only that one raises _initial_uniform_refinement_level and
		# so gets an adaption out of initialise() by itself.
		problem.solve(spatial_adapt=3)
		bulk_before, _ = _levels(problem)
		# Meeting the limit takes refinement - element counts alone would not tell the two cases
		# apart, since the remeshed template reproduces the fine boundary it is handed.
		assert min(bulk_before) >= 1, "wrong before remeshing already: levels %s" % sorted(bulk_before)
		problem.force_remesh()
		bulk_after, _ = _levels(problem)
		assert min(bulk_after) >= 1, \
			"RefineMaxElementSize did not act after remeshing: levels %s" % sorted(bulk_after)


# ---------------------------------------------------------------------------------------------
# An adaptation that decides to refine and unrefine nothing is skipped - all of it except the node
# reordering, which is the one part everything else depends on.
#
# Deciding nothing is the normal end state, not an edge case: oomph only leaves its own adaption
# loop once an adapt() has reported 0/0, so with spatial_adapt>0 the last adaptation of every solve
# is a no-op by construction, and a mesh sitting at max_refinement_level with errors still above the
# refinement tolerance never reports anything else. Skipping it is worth real time - otherwise every
# interface mesh is torn down and rebuilt and assign_eqn_numbers() invalidates the Jacobian sparsity
# pattern unconditionally, so the frozen sparsity is thrown away and rebuilt for a numbering that
# did not change.
#
# The first attempt skipped the reordering with it and had to be reverted: an executed adaptation
# puts the nodes into the order the elements walk them, and since the no-op adaptation is universal,
# that was what made every run agree on the order. Runs then disagreed depending on the route to the
# mesh, and the suite compared permuted states. The reordering is now done on its own - it is
# idempotent, so it costs one renumbering on the first adaptation and nothing afterwards.
#
# Hence two claims below, and they are not the same one: the numbering and the sparsity must survive
# a no-op adaptation, and the node order must be the one every other route produces.
# test_state_file_restart.py is the guard for the second: a restart is bit-identical to its writer
# only while both orders agree.
# ---------------------------------------------------------------------------------------------


class _SaturatingPoissonProblem(Problem):
	"""Peaked source on a mesh capped at max_refinement_level: once the peak is resolved as far as
	the cap allows, the errors there stay above the refine tolerance and every further adaptation
	decides to do nothing."""

	def define_problem(self):
		x = var("coordinate")
		xpeak = self.get_global_parameter("xpeak")
		xpeak.value = 0.5
		eqs = PoissonEquation(source=100*exp(-100*((x[0]-xpeak)**2+(x[1]-0.5)**2)))
		eqs += DirichletBC(u=0)@["left", "right", "top", "bottom"]
		eqs += SpatialErrorEstimator(u=1)
		self += RectangularQuadMesh(N=8)
		self += eqs@"domain"


def _node_order(problem):
	mesh = problem.get_mesh("domain")
	return [(mesh.node_pt(i).x(0), mesh.node_pt(i).x(1)) for i in range(mesh.nnode())]


def test_a_noop_adaptation_leaves_the_numbering_and_the_frozen_sparsity_alone():
	with _SaturatingPoissonProblem() as problem:
		problem.max_refinement_level = 2
		problem.keep_structural_zeros = True   # otherwise jacobian_structure_id is 0 and says nothing
		problem.solve(spatial_adapt=2)
		ndof, structure_id = problem.ndof(), problem.jacobian_structure_id
		nnz = problem._get_frozen_sparsity_nnz()
		rebuilds = problem._get_frozen_sparsity_rebuild_count()
		assert nnz > 0, "the frozen sparsity path did not engage at all, so this proves nothing"
		# Every one of these solves ends in an adaptation that decides nothing. The reordering the
		# first adaptation did is idempotent, so none of these has anything left to renumber for.
		for _ in range(3):
			problem.solve(spatial_adapt=2)
		assert problem.ndof() == ndof
		assert problem.jacobian_structure_id == structure_id, \
			"the equations were renumbered although nothing was refined or unrefined"
		assert problem._get_frozen_sparsity_nnz() == nnz
		assert problem._get_frozen_sparsity_rebuild_count() == rebuilds, \
			"the frozen sparsity pattern was rebuilt for an unchanged numbering"


def test_a_noop_adaptation_leaves_the_nodes_in_the_canonical_order():
	"""The skipped adaptation still has to leave the node order an executed one would produce.

	Repeating it is then a fixed point. Skipping the reordering as well is what made a run that never
	refined disagree with load_state() and with a distributed run about where each dof sits."""
	with _SaturatingPoissonProblem() as problem:
		problem.max_refinement_level = 2
		problem.solve(spatial_adapt=2)
		order = _node_order(problem)
		mesh = problem.get_mesh("domain")
		assert not mesh._reorder_nodes_if_needed(), \
			"the nodes were not in the canonical order after an adaptation that refined nothing"
		for _ in range(3):
			problem.solve(spatial_adapt=2)
		assert _node_order(problem) == order, \
			"a no-op adaptation moved the nodes, so two runs can end up with permuted states"


def test_an_adaptation_that_refines_nothing_reports_no_refinement():
	"""nrefined()/nunrefined() must report this adaptation, not the last one that changed something."""
	with _SaturatingPoissonProblem() as problem:
		problem.max_refinement_level = 2
		problem.solve(spatial_adapt=2)
		mesh = problem.get_mesh("domain")
		assert problem._adapt() == (0, 0)
		assert (mesh.nrefined(), mesh.nunrefined()) == (0, 0)


def test_an_adaptation_that_does_something_still_renumbers():
	"""The gate must not swallow a real adaptation: moving the peak has to re-adapt and renumber."""
	with _SaturatingPoissonProblem() as problem:
		problem.max_refinement_level = 2
		problem.keep_structural_zeros = True
		problem.solve(spatial_adapt=2)
		structure_id = problem.jacobian_structure_id
		problem.get_global_parameter("xpeak").value = 0.2
		problem.solve(spatial_adapt=2)
		mesh = problem.get_mesh("domain")
		assert problem.jacobian_structure_id != structure_id, \
			"the mesh followed the moved peak, so the equations must have been renumbered"
		# The refinement now sits around the new peak position, not the old one.
		fine = [e for e in mesh.elements() if e.refinement_level() == 2]
		assert fine and max(numpy.mean([e.node_pt(i).x(0) for i in range(e.nnode())]) for e in fine) < 0.5
