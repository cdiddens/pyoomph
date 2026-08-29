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

# The integration rules themselves, tested without any physics on top.
#
# Nothing else in the suite looks at them directly. That is how Gauss<2,3>::Knot -- the 3x3
# Gauss-Legendre rule for 2D quadrilateral elements -- carried a typo for as long as it did: five of
# its nine entries read 0.774596662941483 where the outer knot is 0.774596669241483, two digits
# transposed. Only the POSITIVE knot was affected, so the rule stayed a valid quadrature of the right
# total weight but stopped being symmetric, and an asymmetric rule is invisible to any test whose
# reference value is itself computed on the same mesh. What it did instead was leave a fixed 6e-9
# defect in every 2D quadrilateral C2 assembly -- a defect of the RULE, so it did not shrink under
# refinement: a field lying exactly in the finite element space still produced a nonzero residual,
# and the answer was wrong by ~1e-9 however fine the mesh.
#
# The two tests below are the two ways to see it:
#
#   * integrate an odd monomial over a symmetric domain. Exactly zero for a symmetric rule; with the
#     transposed knot the three-point rule gives (5/9)*(b^5 - a^5) ~ 5*a^4*(b-a)*(5/9) ~ -1e-8 per
#     direction, so the test fails by six orders of magnitude, not by a hair.
#   * solve a Poisson problem whose exact solution lies in the finite element space. The discrete
#     residual of that solution is identically zero for an exact rule, so the computed answer must be
#     the interpolant to machine precision, on any mesh.
#
# Both are run across element geometries, because the rules are per-geometry: 1D lines, 2D quads, 2D
# triangles and 3D bricks each have their own table, and only the 2D quad one was wrong.

import itertools

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh, CuboidBrickMesh

_run_counter = itertools.count()


def _fresh(problem):
    problem.set_output_directory("run%d" % next(_run_counter))
    problem.set_c_compiler("system")
    return problem


def _centred_mesh(geometry):
    """A mesh spanning [-1,1]^dim, so that odd monomials integrate to zero by symmetry."""
    if geometry == "line":
        return LineMesh(N=2, size=2.0, minimum=-1.0), 1
    if geometry == "quad":
        return RectangularQuadMesh(N=2, size=2.0, lower_left="centered"), 2
    if geometry == "tri":
        return RectangularQuadMesh(N=2, size=2.0, lower_left="centered",
                                   split_in_tris="crossed"), 2
    if geometry == "brick":
        return CuboidBrickMesh(N=2, size=2.0, lower_left=[-1.0, -1.0, -1.0]), 3
    raise ValueError(geometry)


class _Monomials(Problem):
    """Carries a field only so that the elements exist; the observables are what is under test."""

    def __init__(self, geometry, space):
        super().__init__()
        self.geometry, self.space = geometry, space

    def define_problem(self):
        mesh, dim = _centred_mesh(self.geometry)
        self.add_mesh(mesh)
        x = var("coordinate_x")
        y = var("coordinate_y") if dim > 1 else 1
        eqs = PoissonEquation(name="u", space=self.space)
        eqs += DirichletBC(u=0) @ "left"
        obs = {"volume": 1, "odd_x": x ** 5, "even_x": x ** 4}
        if dim > 1:
            obs.update({"odd_y": y ** 5, "odd_mixed": x ** 3 * y ** 2, "even_mixed": x ** 2 * y ** 2})
        eqs += IntegralObservables(**obs)
        self.add_equations(eqs @ "domain")
        self.dim = dim


@pytest.mark.parametrize("geometry", ["line", "quad", "tri", "brick"])
def test_quadrature_is_symmetric(geometry):
    """Odd monomials over a symmetric domain must integrate to exactly zero.

    This is the direct probe of the knot table: it asks only that the positive knots be the exact
    negatives of the negative ones, which is a property of the numbers in the table and not of the
    mesh, so it cannot be satisfied by refinement.
    """
    prob = _Monomials(geometry, "C2")
    with _fresh(prob) as p:
        p.initialise()
        o = {k: float(v) for k, v in p.get_mesh("domain").evaluate_all_observables().items()}
    volume = 2.0 ** prob.dim
    assert o["volume"] == pytest.approx(volume, rel=1e-13), "the mesh is not what the test assumes"
    for name, value in o.items():
        if name.startswith("odd_"):
            assert abs(value) < 1e-13 * volume, "%s = %g, so the rule is not symmetric" % (name, value)


@pytest.mark.parametrize("geometry", ["line", "quad", "tri", "brick"])
def test_quadrature_is_exact_for_the_degree_it_claims(geometry):
    """The even monomials pin the magnitudes, which symmetry alone would not.

    A rule can be symmetric and still be scaled wrongly, and ``x^4`` (and ``x^2*y^2``) are inside the
    degree that every rule here integrates exactly, so these are equalities and not estimates.
    """
    prob = _Monomials(geometry, "C2")
    with _fresh(prob) as p:
        p.initialise()
        o = {k: float(v) for k, v in p.get_mesh("domain").evaluate_all_observables().items()}
    dim = prob.dim
    # int_{-1}^{1} x^4 dx = 2/5, int_{-1}^{1} x^2 dx = 2/3, int_{-1}^{1} dx = 2.
    assert o["even_x"] == pytest.approx(0.4 * 2.0 ** (dim - 1), rel=1e-13)
    if dim > 1:
        assert o["even_mixed"] == pytest.approx((2 / 3) ** 2 * 2.0 ** (dim - 2), rel=1e-13)


class _LinearPoisson(Problem):
    """Poisson with a linear exact solution -- which lies in every space here, at every order."""

    def __init__(self, geometry, space):
        super().__init__()
        self.geometry, self.space = geometry, space

    def exact(self):
        return -1.3 * var("coordinate_x") + 0.7

    def define_problem(self):
        mesh, dim = _centred_mesh(self.geometry)
        self.add_mesh(mesh)
        eqs = PoissonEquation(name="u", space=self.space, coefficient=2.0)
        for b in ["left", "right"]:
            eqs += DirichletBC(u=self.exact()) @ b
        self.add_equations(eqs @ "domain")
        self.dim = dim


@pytest.mark.parametrize("geometry", ["line", "quad", "tri", "brick"])
@pytest.mark.parametrize("space", ["C1", "C2"])
def test_a_solution_inside_the_space_is_reproduced_exactly(geometry, space):
    """The symptom the knot typo actually produced, and the reason it went unnoticed for so long.

    ``u = -1.3x + 0.7`` is in the space, so the discrete residual of its interpolant vanishes
    identically and the solver must return it to machine precision. It used to come back off by
    8.5e-10 at the mid-side nodes of a 2D quadrilateral C2 mesh -- and only there: the vertex nodes
    were exact, C1 was exact, triangles were exact, and the deviation was the same at N=2 and at
    N=16, which is what identifies a quadrature defect rather than a discretisation error.
    """
    prob = _LinearPoisson(geometry, space)
    with _fresh(prob) as p:
        p.solve()
        mesh = p.get_mesh("domain")
        iu = mesh.get_nodal_field_indices()["u"]
        dev = max(abs(n.value(iu) - (-1.3 * n.x(0) + 0.7)) for n in mesh.nodes())
    assert dev < 1e-12, "off by %g at a node, so the assembly is not exact" % dev
