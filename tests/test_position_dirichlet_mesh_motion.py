#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

"""A strongly prescribed nodal position must deform the mesh, once.

The Lagrangian coordinates are the *as-built* reference configuration. ``Problem.initialise()``
applies the Dirichlet conditions before it maps the nodes on the macro elements, so the mesh-wide
``xi := x`` that the moving-mesh equations used to run from
``after_mapping_on_macro_elements`` baked a prescribed boundary position into that reference. Two
symptoms followed, both silent:

* the smoothers write their residual in ``displ = x - X``, so with ``X`` redefined to the current
  ``x`` the undeformed interior solved the mesh-motion equation exactly with the boundary sitting
  where it had been told. The boundary moved and nothing else did, which crushes the first element
  layer and folds it on refinement;
* a value referring to ``var("lagrangian_*")`` was then evaluated against the displaced reference on
  the next pass, i.e. realised as ``f(f(Y))``: ``mesh_y = Y + h`` gave ``2h``.

A mesh carrying macro elements never showed either, because the macro map re-derives ``x`` from the
geometry just before the freeze. See dev_docs/position_dirichlet_mesh_motion.md.

The oracles, in order of sharpness:

1. the reference configuration itself - every node's Lagrangian coordinate must equal the one the
   same mesh has when nothing is prescribed. This is the defect with no solve in between;
2. ``EnforcedDirichlet``, the same condition through a Lagrange multiplier, which never writes a
   nodal position and was therefore always correct. It satisfies the condition in an integral rather
   than a pointwise sense, so it agrees only up to a discretisation error that has to *converge*;
3. for the linear Laplace smoother the interior is the discrete harmonic extension of the boundary
   data, hence strictly monotone and strictly inside the two prescribed boundaries.
"""

import math

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh, PseudoElasticMesh
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.meshes.simplemeshes import CircularMesh, RectangularQuadMesh

#: Bump amplitude. Large enough that the crushed first layer of the pre-fix behaviour inverts under
#: refinement, small enough that the smoothers converge in a couple of Newton steps.
AMP = 0.07

_SPELLINGS = ["separate", "one_object", "vector", "y_only"]


class BumpMesh(Problem):
    """Mesh motion only, no physics: a unit square whose bottom is pushed up by ``A sin(pi X)``.

    ``case`` selects how that bump is imposed; the top is held at its undeformed position and the
    sides keep their x, so the displacement has to decay from the bump to zero at the top - which is
    what the mesh-motion equation is for.
    """

    def __init__(self, case, mesh_eq="laplace", N=(6, 4)):
        super().__init__()
        self.case, self.mesh_eq, self.N = case, mesh_eq, N

    def define_problem(self):
        self += RectangularQuadMesh(N=list(self.N), size=[1.0, 1.0])
        base = LaplaceSmoothedMesh() if self.mesh_eq == "laplace" else PseudoElasticMesh()
        eqs = base + ElementSpace("C2")
        X, Y = var("lagrangian_x"), var("lagrangian_y")
        h = AMP * sin(pi * X)

        eqs += DirichletBC(mesh=var("lagrangian")) @ "top"
        for b in ("left", "right"):
            eqs += DirichletBC(mesh_x=X) @ b

        if self.case == "undeformed":        # the reference run: nothing is pushed anywhere
            eqs += DirichletBC(mesh_x=X, mesh_y=Y) @ "bottom"
        elif self.case == "separate":        # mesh_x and mesh_y in two DirichletBC objects
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += DirichletBC(mesh_y=h) @ "bottom"
        elif self.case == "one_object":      # both components in one DirichletBC
            eqs += DirichletBC(mesh_x=X, mesh_y=h) @ "bottom"
        elif self.case == "vector":          # the whole position vector at once
            eqs += DirichletBC(mesh=vector(X, h)) @ "bottom"
        elif self.case == "y_only":          # mesh_x left free
            eqs += DirichletBC(mesh_y=h) @ "bottom"
        elif self.case == "enforced":        # the oracle: the same condition, weakly
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += EnforcedDirichlet(mesh_y=h) @ "bottom"
        elif self.case == "lagr_plus_h":     # a value referring to the component it prescribes
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += DirichletBC(mesh_y=Y + h) @ "bottom"
        elif self.case == "two_lagr_plus_h":
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += DirichletBC(mesh_y=2 * Y + h) @ "bottom"
        elif self.case == "top_lagr_plus_h":  # the same, on a boundary where lagrangian_y == 1
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += DirichletBC(mesh=vector(X, Y + h)) @ "top"
        elif self.case == "top_literal":     # ... and the literal spelling of the same thing
            eqs += DirichletBC(mesh_x=X) @ "bottom"
            eqs += DirichletBC(mesh=vector(X, 1 + h)) @ "top"
        else:
            raise ValueError(self.case)

        self += eqs @ "domain"


def _column(problem, x=0.5):
    """The y coordinates of the node column at this x, bottom first."""
    return numpy.array(sorted(n.x(1) for n in problem.get_mesh("domain").nodes() if abs(n.x(0) - x) < 1e-9))


def _lagrangian_coordinates(problem):
    """Every node's Lagrangian coordinate, keyed by it, so two runs can be compared node by node."""
    return {(round(n.x_lagr(0), 12), round(n.x_lagr(1), 12)) for n in problem.get_mesh("domain").nodes()}


def _measure(case, tmp_path, mesh_eq="laplace", N=(6, 4), nsolve=1, measure=None):
    """Build the problem, initialise (``nsolve=0``) or solve it, and return what ``measure`` reads."""
    with BumpMesh(case, mesh_eq, N) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        if nsolve:
            for _ in range(nsolve):
                problem.solve()
        else:
            problem.initialise()
        return (measure or _column)(problem)


# --------------------------------------------------------------------------------------------
# 1. The reference configuration, with no solve in between
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", _SPELLINGS)
def test_prescribing_a_position_does_not_redefine_the_reference(case, tmp_path):
    """``initialise()`` must leave the Lagrangian coordinates at the as-built mesh."""
    def probe(problem):
        # Non-vacuity: the boundary really is where it was told to be, so there WAS something to bake.
        moved = max(n.x(1) for n in problem.get_mesh("domain").nodes() if abs(n.x_lagr(1)) < 1e-9)
        return _lagrangian_coordinates(problem), moved

    reference, _ = _measure("undeformed", tmp_path / "undeformed", nsolve=0, measure=probe)
    prescribed, moved = _measure(case, tmp_path / case, nsolve=0, measure=probe)
    assert abs(moved - AMP) < 1e-12
    assert prescribed == reference


# --------------------------------------------------------------------------------------------
# 2. The interior follows - the discrete harmonic extension of the boundary data
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_eq", ["laplace", "pseudoelastic"])
@pytest.mark.parametrize("case", _SPELLINGS)
def test_interior_follows_the_prescribed_boundary(case, mesh_eq, tmp_path):
    col = _measure(case, tmp_path, mesh_eq)
    undeformed = numpy.array([0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.0])
    assert len(col) == len(undeformed)
    assert abs(col[0] - AMP) < 1e-12                       # the boundary went where it was told
    assert abs(col[-1] - 1.0) < 1e-12                      # the pinned top stayed
    assert numpy.all(numpy.diff(col) > 0)                  # no crushed or folded layer
    assert numpy.all(col[1:-1] > AMP) and numpy.all(col[1:-1] < 1.0)
    # Every interior node moved: this is the assertion that failed before the fix, where the
    # interior sat at exactly its undeformed position.
    assert numpy.min(numpy.abs(col[1:-1] - undeformed[1:-1])) > 1e-3


@pytest.mark.parametrize("N", [(6, 4), (12, 8)])
def test_no_node_crosses_the_prescribed_boundary(N, tmp_path):
    """The pre-fix mesh put 1 and 2 nodes of this column *below* the prescribed wall at these two
    resolutions, and more the finer the mesh - the crushed first layer inverting."""
    col = _measure("y_only", tmp_path, "laplace", N)
    assert numpy.sum(col < AMP - 1e-9) == 0


# --------------------------------------------------------------------------------------------
# 3. Against EnforcedDirichlet, the formulation that was always right
# --------------------------------------------------------------------------------------------

def _gap_to_enforced(tmp_path, N):
    strong = _measure("one_object", tmp_path / ("strong%d" % N[0]), "laplace", N)
    weak = _measure("enforced", tmp_path / ("weak%d" % N[0]), "laplace", N)
    return float(numpy.max(numpy.abs(strong - weak)))


def test_strong_and_enforced_position_conditions_converge_to_each_other(tmp_path):
    coarse = _gap_to_enforced(tmp_path, (6, 4))
    fine = _gap_to_enforced(tmp_path, (12, 8))
    # EnforcedDirichlet satisfies the condition in an integral sense, so the two differ by a
    # discretisation error - the point is that it falls, and fast, rather than that it is small.
    assert coarse < 5e-4
    assert fine < 1e-4
    assert fine < coarse / 4


# --------------------------------------------------------------------------------------------
# 4. A value referring to lagrangian_* is applied once, not composed with itself
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["y_only", "lagr_plus_h", "two_lagr_plus_h"])
def test_a_lagrangian_referencing_value_is_applied_once(case, tmp_path):
    """On the bottom ``lagrangian_y == 0``, so ``h``, ``Y + h`` and ``2 Y + h`` all ask for ``h``.
    Composed once they give ``2h`` and ``3h``, which is what came out before the fix."""
    assert abs(_measure(case, tmp_path)[0] - AMP) < 1e-12


def test_a_lagrangian_referencing_value_is_stable_over_repeated_solves(tmp_path):
    """Each solve re-applies the conditions, so a composing one would walk - except that it had a
    fixed point at ``2h``, which is why the wrong value never looked unconverged."""
    def probe(problem):
        before = _column(problem)[0]
        problem.reapply_boundary_conditions()
        return before, _column(problem)[0]

    before, after = _measure("lagr_plus_h", tmp_path, nsolve=4, measure=probe)
    assert abs(before - AMP) < 1e-12
    assert abs(after - AMP) < 1e-12


def test_a_lagrangian_offset_agrees_with_its_literal_spelling(tmp_path):
    """On the top ``lagrangian_y == 1``, so ``Y + h`` must give ``1 + h`` and not ``1 + 2h``."""
    referenced = _measure("top_lagr_plus_h", tmp_path / "lagr")
    literal = _measure("top_literal", tmp_path / "literal")
    assert abs(referenced[-1] - (1.0 + AMP)) < 1e-12
    assert numpy.max(numpy.abs(referenced - literal)) < 1e-12


# --------------------------------------------------------------------------------------------
# 5. The macro-element case the freeze exists for must keep working
# --------------------------------------------------------------------------------------------

_R = 1.0


class _CurvedDisk(Problem):
    """A quarter disc of quads. Each rim facet carries a circular arc, so the macro map - and not
    the straight interpolant between the corners - is what places the rim nodes."""

    def __init__(self, squeeze=False):
        super().__init__()
        self.squeeze = squeeze

    def define_problem(self):
        self += CircularMesh(radius=_R, segments=["NE"])
        eqs = LaplaceSmoothedMesh() + ElementSpace("C2")
        # Each straight side slides along its own axis: pinning both components would hold those
        # nodes outside the squeezed rim and tangle the mesh rather than test it.
        eqs += DirichletBC(mesh_x=0) @ "center_to_north"
        eqs += DirichletBC(mesh_y=0) @ "center_to_east"
        if self.squeeze:
            # Pull the rim inwards along its own radius, keeping it a circle of radius 0.9.
            X = var("lagrangian")
            eqs += DirichletBC(mesh=0.9 * X) @ "circumference"
        else:
            eqs += DirichletBC(mesh=var("lagrangian")) @ "circumference"
        self += eqs @ "domain"


def _rim_radii(problem, lagrangian):
    mesh = problem.get_mesh("domain")
    bidx = mesh.get_boundary_index("circumference")
    out = []
    for n in mesh.nodes():
        if n.is_on_boundary(bidx):
            c = (n.x_lagr(0), n.x_lagr(1)) if lagrangian else (n.x(0), n.x(1))
            out.append(math.hypot(*c))
    return numpy.array(out)


def test_the_macro_map_still_defines_the_reference_on_a_curved_mesh(tmp_path):
    """The freeze exists because the mesh template sets ``xi := x`` *before* the macro map snaps the
    rim onto the arc. Restricting it to the macro-mapped nodes must not lose that: both the Eulerian
    and the Lagrangian rim have to be the exact circle, not the polynomial interpolant."""
    with _CurvedDisk() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        assert numpy.max(numpy.abs(_rim_radii(problem, False) - _R)) < 1e-10
        assert numpy.max(numpy.abs(_rim_radii(problem, True) - _R)) < 1e-10


def test_a_prescribed_position_on_a_curved_mesh_deforms_the_interior(tmp_path):
    with _CurvedDisk(squeeze=True) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.solve()
        # The rim is where it was told, and its reference is still the undeformed circle.
        assert numpy.max(numpy.abs(_rim_radii(problem, False) - 0.9 * _R)) < 1e-9
        assert numpy.max(numpy.abs(_rim_radii(problem, True) - _R)) < 1e-10
        # ... and the interior followed it inwards rather than staying put.
        mesh = problem.get_mesh("domain")
        rim = mesh.get_boundary_index("circumference")
        moved = [math.hypot(n.x(0), n.x(1)) - math.hypot(n.x_lagr(0), n.x_lagr(1))
                 for n in mesh.nodes()
                 if not n.is_on_boundary(rim) and math.hypot(n.x_lagr(0), n.x_lagr(1)) > 0.5 * _R]
        assert len(moved) > 0
        assert max(moved) < -1e-3


# --------------------------------------------------------------------------------------------
# 6. The re-bake paths warn, and only when the re-bake would change what the condition reads
# --------------------------------------------------------------------------------------------

class _RemeshedBump(Problem):
    """A remeshable version of the bump problem. ``reads_moving`` picks whether the prescribed
    value refers to the component that actually moves."""

    def __init__(self, reads_moving):
        super().__init__()
        self.reads_moving = reads_moving

    def define_problem(self):
        mesh = RectangularQuadMesh(N=6)
        mesh.remesher = Remesher2d(mesh)
        self += mesh
        eqs = LaplaceSmoothedMesh() + ElementSpace("C2")
        X, Y = var("lagrangian_x"), var("lagrangian_y")
        h = AMP * sin(pi * X)
        eqs += DirichletBC(mesh=var("lagrangian")) @ "top"
        eqs += DirichletBC(mesh_x=X) @ "left"
        eqs += DirichletBC(mesh_x=X) @ "right"
        if self.reads_moving:
            # mesh_y reads lagrangian_y, and the bottom's y is exactly what moves.
            eqs += DirichletBC(mesh_x=X, mesh_y=Y + h) @ "bottom"
        else:
            # mesh_y reads lagrangian_x, which nothing displaces - re-baking cannot change it.
            eqs += DirichletBC(mesh_x=X, mesh_y=h) @ "bottom"
        self += eqs @ "domain"


def _remesh_warnings(reads_moving, tmp_path, capsys):
    from pyoomph.meshes import interpolator
    interpolator._warned_lagrangian_rebake.clear()
    with _RemeshedBump(reads_moving) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.solve()
        problem.force_remesh()
        problem.force_remesh()
    return capsys.readouterr().out.count("re-bake the Lagrangian")


def test_a_remesh_warns_once_about_a_lagrangian_dependent_position_condition(tmp_path, capsys):
    """Remeshing redefines the reference configuration on purpose, so this is a warning rather than
    a refusal - but it must not be silent, since the condition then composes exactly as it did
    before the fix."""
    assert _remesh_warnings(True, tmp_path, capsys) == 1


def test_a_remesh_is_silent_when_the_referenced_component_has_not_moved(tmp_path, capsys):
    """A value reading a component the mesh has not been displaced in means the same thing after the
    re-bake. The shipped remeshing tutorial is this case, and must stay warning-free."""
    assert _remesh_warnings(False, tmp_path, capsys) == 0
