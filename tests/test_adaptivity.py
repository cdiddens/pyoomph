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
