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

# Curved boundaries via MacroElements -- see dev_docs/macro_elements_generalisation.md.
#
# One acceptance criterion runs through all of this: a node that lies on a curved boundary must
# satisfy that boundary's implicit equation to machine precision, no matter which element shape it
# belongs to and no matter which code path created it. For a circle of radius R that is simply
# |r - R| ~ 1e-16, which makes the check independent of any reference solution.
#
# Written at stage S0 with most of it failing, as strict xfails carrying the measured value; S1 made
# them pass and the markers came off. The pre-S1 numbers are kept in the comments, because a test that
# says what it used to return is a much better guard than one that only says "< 1e-14" -- 5.4e-4 and
# 7.6e-2 are what silent regressions in this area look like.
#
# Two families run in a child process (see _worker_radius_error). Not for speed -- a case costs under
# a second -- but because before S1 they could take the interpreter down rather than raise: a curved
# triangular mesh threw "MACRO ELEM" mid-refinement and left a half-built tree whose teardown then
# aborted, so even catching the RuntimeError did not make the process reusable. That throw is gone,
# so the isolation is now redundant rather than wrong; it is kept because it costs almost nothing and
# is the right shape for any future case that crashes instead of failing.

import gc
import math
import os
import subprocess
import sys

import numpy
import pytest

from pyoomph import *
from pyoomph import _pyoomph_core as _pyoomph
from pyoomph.equations.generic import ConnectFieldsAtInterface
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.meshes.simplemeshes import CircularMesh, SphericalOctantMesh


_R = 1.0
_EXACT = 1e-14


def _max_radius_error(mesh, boundary_name, radius=_R, ndim=2):
    # Largest deviation from the circle/sphere over every node sitting on the named boundary.
    bidx = mesh.get_boundary_index(boundary_name)
    worst = 0.0
    for node in mesh.nodes():
        if node.is_on_boundary(bidx):
            r = math.sqrt(sum(node.x(i) ** 2 for i in range(ndim)))
            worst = max(worst, abs(r - radius))
    return worst


# --------------------------------------------------------------------------------------------
# Geometries
# --------------------------------------------------------------------------------------------

class _QuadDisk(Problem):
    # A quarter disc of quads; CircularMesh attaches a CurvedEntityCircleArc per rim facet.
    def __init__(self, space="C2"):
        super().__init__()
        self._space = space

    def define_problem(self):
        self += CircularMesh(radius=_R, segments=["NE"])
        eqs = PoissonEquation(source=1, space=self._space) + DirichletBC(u=0) @ "circumference"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _TriDiskTemplate(MeshTemplate):
    # A disc of N triangles fanning out of the centre, each rim edge carrying its own circular arc.
    # CircularMesh is quad-only, so the triangular counterpart is built by hand.
    def __init__(self, nseg=8):
        super().__init__()
        self._nseg = nseg
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        centre = self.add_node_unique(0, 0)
        rim = [self.add_node_unique(_R * math.cos(2 * math.pi * i / self._nseg),
                                    _R * math.sin(2 * math.pi * i / self._nseg))
               for i in range(self._nseg)]
        for i in range(self._nseg):
            j = (i + 1) % self._nseg
            domain.add_tri_2d_C1(centre, rim[i], rim[j])
            arc = _pyoomph.CurvedEntityCircleArc([0, 0, 0],
                                                 self.get_node_position(rim[i]),
                                                 self.get_node_position(rim[j]))
            self._entities.append(arc)
            self.add_facet_to_boundary("circumference", [rim[i], rim[j]], [rim[i], rim[j]], arc)


class _TriDisk(Problem):
    def __init__(self, space="C2"):
        super().__init__()
        self._space = space

    def define_problem(self):
        self += _TriDiskTemplate()
        eqs = PoissonEquation(source=1, space=self._space) + DirichletBC(u=0) @ "circumference"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _SphereOctant(Problem):
    # A spherical octant of bricks; SphericalOctantMesh attaches a CurvedEntitySpherePart to each of
    # the three shell faces. One curved facet per element, so no shared-edge correction is involved.
    def __init__(self, space="C1"):
        super().__init__()
        self._space = space

    def define_problem(self):
        self += SphericalOctantMesh(radius=_R)
        eqs = PoissonEquation(source=1, space=self._space) + DirichletBC(u=0) @ "shell"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _unit(v):
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v]


class _SphericalTetTemplate(MeshTemplate):
    # A single tetrahedron with all four vertices on the unit sphere, of which the first `ncurved`
    # faces are declared to lie on it. Any two faces of a tet share an edge, so ncurved >= 2 is exactly
    # the configuration the blend needs its inclusion-exclusion term for: without it each of the two
    # faces would add that shared edge's deviation and the edge would overshoot the sphere.
    _VERTS = [_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1]), _unit([1, 1, 1])]
    _FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    def __init__(self, ncurved):
        super().__init__()
        self._ncurved = ncurved
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        n = [self.add_node_unique(*p) for p in self._VERTS]
        domain.add_tetra_3d_C1(n[0], n[1], n[2], n[3])
        sphere = _pyoomph.CurvedEntitySpherePart([0, 0, 0], [_R, 0, 0])
        self._entities.append(sphere)
        for face in self._FACES[:self._ncurved]:
            nodes = [n[i] for i in face]
            self.add_facet_to_boundary("shell", nodes, nodes, sphere)


class _SphericalTet(Problem):
    def __init__(self, ncurved):
        super().__init__()
        self._ncurved = ncurved

    def define_problem(self):
        self += _SphericalTetTemplate(self._ncurved)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "shell"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _TetBallTemplate(MeshTemplate):
    # An octant of a ball as three tetrahedra fanning out of the centre, each with exactly one face on
    # the sphere. Unlike _SphericalTetTemplate this is a mesh someone might actually write, and every
    # node marked as a shell node genuinely lies on the shell.
    def __init__(self):
        super().__init__()
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        centre = self.add_node_unique(0, 0, 0)
        axis = [self.add_node_unique(_R, 0, 0), self.add_node_unique(0, _R, 0), self.add_node_unique(0, 0, _R)]
        diag = self.add_node_unique(*[c * _R for c in _unit([1, 1, 1])])
        sphere = _pyoomph.CurvedEntitySpherePart([0, 0, 0], [_R, 0, 0])
        self._entities.append(sphere)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            domain.add_tetra_3d_C1(centre, axis[a], axis[b], diag)
            face = [axis[a], axis[b], diag]
            self.add_facet_to_boundary("shell", face, face, sphere)


class _TetBall(Problem):
    def __init__(self, space="C1"):
        super().__init__()
        self._space = space

    def define_problem(self):
        self += _TetBallTemplate()
        eqs = PoissonEquation(source=1, space=self._space) + DirichletBC(u=0) @ "shell"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _WedgeShellTemplate(MeshTemplate):
    # A single wedge whose quadrilateral facet 4 (s0+s1=1, nodes 1,2,4,5) lies on the sphere.
    def __init__(self):
        super().__init__()
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        n0 = self.add_node_unique(0, 0, 0)
        n1 = self.add_node_unique(*[c * _R for c in _unit([1, 0, 0.35])])
        n2 = self.add_node_unique(*[c * _R for c in _unit([0, 1, 0.35])])
        n3 = self.add_node_unique(0, 0, 0.6 * _R)
        n4 = self.add_node_unique(*[c * _R for c in _unit([1, 0, 1.4])])
        n5 = self.add_node_unique(*[c * _R for c in _unit([0, 1, 1.4])])
        domain.add_wedge_3d_C1(n0, n1, n2, n3, n4, n5)
        sphere = _pyoomph.CurvedEntitySpherePart([0, 0, 0], [_R, 0, 0])
        self._entities.append(sphere)
        face = [n1, n2, n4, n5]
        self.add_facet_to_boundary("shell", face, face, sphere)


class _PyramidShellTemplate(MeshTemplate):
    # A single pyramid whose quadrilateral base (facet 4, s2=0) lies on the sphere. Refining it
    # produces six pyramids and four tetrahedra, so this also exercises a tet son of a pyramid
    # father -- the mixed-forest case, where son and father do not even have the same shape.
    def __init__(self):
        super().__init__()
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        quad = [self.add_node_unique(*[c * _R for c in _unit(v)])
                for v in ([1, -0.4, -0.4], [1, 0.4, -0.4], [1, 0.4, 0.4], [1, -0.4, 0.4])]
        apex = self.add_node_unique(0.2 * _R, 0, 0)
        domain.add_pyramid_3d_C1(quad[0], quad[1], quad[2], quad[3], apex)
        sphere = _pyoomph.CurvedEntitySpherePart([0, 0, 0], [_R, 0, 0])
        self._entities.append(sphere)
        self.add_facet_to_boundary("shell", quad, quad, sphere)


class _MixedShell(Problem):
    def __init__(self, kind):
        super().__init__()
        self._kind = kind

    def define_problem(self):
        self += _WedgeShellTemplate() if self._kind == "wedge" else _PyramidShellTemplate()
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "shell"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _CircleByNormal(_pyoomph.MeshTemplateCurvedEntity):
    # A user-defined entity charted REDUNDANTLY: a 1-dimensional manifold described by a 2-component
    # unit normal. That is the case a plain weighted sum of parametric coordinates gets wrong -- the
    # average of two unit normals is not a unit normal -- so it is what blend() exists for.
    # `renormalise=False` leaves the default sum in place, to show the difference is real.
    def __init__(self, centre, radius, renormalise=True):
        super().__init__(2)
        self._c = numpy.array(centre, dtype=float)
        self._r = radius
        self._renormalise = renormalise

    def get_intrinsic_dimension(self):
        return 1

    def pos_to_parametric(self, t, pos, param):
        d = numpy.array([pos[0], pos[1]]) - self._c
        param[:] = d / numpy.linalg.norm(d)

    def parametric_to_pos(self, t, param, pos):
        n = numpy.array(param[:2])
        pos[0] = self._c[0] + self._r * n[0]
        pos[1] = self._c[1] + self._r * n[1]

    def blend(self, weights, params, result):
        result[:] = weights @ params
        if self._renormalise:
            result /= numpy.linalg.norm(result)


class _NormalDiskTemplate(MeshTemplate):
    # The triangular disc again, but with its rim on a Python-defined entity rather than a built-in.
    def __init__(self, renormalise=True, nseg=8):
        super().__init__()
        self._renormalise, self._nseg = renormalise, nseg
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        centre = self.add_node_unique(0, 0)
        rim = [self.add_node_unique(_R * math.cos(2 * math.pi * i / self._nseg),
                                    _R * math.sin(2 * math.pi * i / self._nseg))
               for i in range(self._nseg)]
        entity = _CircleByNormal([0, 0], _R, self._renormalise)
        self._entities.append(entity)
        for i in range(self._nseg):
            j = (i + 1) % self._nseg
            domain.add_tri_2d_C1(centre, rim[i], rim[j])
            self.add_facet_to_boundary("circumference", [rim[i], rim[j]], [rim[i], rim[j]], entity)


class _NormalDisk(Problem):
    def __init__(self, renormalise=True):
        super().__init__()
        self._renormalise = renormalise

    def define_problem(self):
        self += _NormalDiskTemplate(self._renormalise)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "circumference"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _CurvedInterfaceTemplate(MeshTemplate):
    # Two concentric annular domains sharing a CIRCULAR interface, plus curved inner and outer walls.
    # The point is the shared interface: it belongs to both domains, and both must place its refined
    # nodes in the same places or the opposite-element matcher has nothing to pair up.
    _R = (0.4, 0.7, 1.0)

    def __init__(self, curved=True, nseg=16):
        super().__init__()
        self._curved, self._nseg = curved, nseg
        self._entities = []

    def define_geometry(self):
        inner = self.new_domain("inner")
        outer = self.new_domain("outer")
        n = self._nseg
        cache = {}

        def node(i, r):
            key = (i % n, r)
            if key not in cache:
                t = 2 * math.pi * (i % n) / n
                cache[key] = self.add_node_unique(r * math.cos(t), r * math.sin(t))
            return cache[key]

        def arc(a, b):
            if not self._curved:
                return None
            entity = self.create_curved_entity("circle_arc", a, b, center=[0, 0, 0])
            self._entities.append(entity)
            return entity

        r0, r1, r2 = self._R
        for i in range(n):
            a0, b0 = node(i, r0), node(i + 1, r0)
            a1, b1 = node(i, r1), node(i + 1, r1)
            a2, b2 = node(i, r2), node(i + 1, r2)
            inner.add_quad_2d_C1(a0, b0, a1, b1)
            outer.add_quad_2d_C1(a1, b1, a2, b2)
            self.add_facet_to_boundary("inner_wall", [a0, b0], [a0, b0], arc(a0, b0))
            self.add_facet_to_boundary("interface", [a1, b1], [a1, b1], arc(a1, b1))
            self.add_facet_to_boundary("outer_wall", [a2, b2], [a2, b2], arc(a2, b2))


class _CurvedInterfaceProblem(Problem):
    def __init__(self, curved=True):
        super().__init__()
        self._curved = curved

    def define_problem(self):
        self += _CurvedInterfaceTemplate(self._curved)
        self += PoissonEquation(name="u", source=0, space="C1") @ "inner"
        self += PoissonEquation(name="u", source=0, space="C1") @ "outer"
        self += DirichletBC(u=0) @ "inner/inner_wall"
        self += DirichletBC(u=1) @ "outer/outer_wall"
        self += ConnectFieldsAtInterface("u") @ "inner/interface"
        # Deliberately asymmetric: only "inner" is told to refine, so the interface-refinement coupler
        # has to carry the requirement across to "outer".
        self += (SpatialErrorEstimator(u=1) + RefineToLevel(2)) @ "inner"
        self += SpatialErrorEstimator(u=1) @ "outer"


class _GmshBallTemplate(GmshTemplate):
    # An octant of a ball meshed by gmsh with tetrahedra, its shell mapped onto the sphere. Unlike the
    # hand-built meshes above this is unstructured, which is what makes it the interesting case: most
    # elements touching the sphere own a face on it, but a substantial minority touch it along an edge
    # only, and those need the curved-edge registry to agree with their neighbours.
    def __init__(self, map_to_sphere=True):
        super().__init__()
        self._map = map_to_sphere

    def define_geometry(self):
        self.default_resolution = 0.35
        self.mesh_mode = "tetras"
        o = self.point(0, 0, 0)
        px, py, pz = self.point(_R, 0, 0), self.point(0, _R, 0), self.point(0, 0, _R)
        axy = self.circle_arc(px, py, center=o)
        ayz = self.circle_arc(py, pz, center=o)
        azx = self.circle_arc(pz, px, center=o)
        lx, ly, lz = self.line(o, px), self.line(o, py), self.line(o, pz)
        shell = self.ruled_surface(axy, ayz, azx, name="shell", map_to_sphere=self._map)[0]
        sxy = self.plane_surface(lx, axy, ly, name="plane_z0")[0]
        syz = self.plane_surface(ly, ayz, lz, name="plane_x0")[0]
        szx = self.plane_surface(lz, azx, lx, name="plane_y0")[0]
        self.volume(shell, sxy, syz, szx, name="domain")


class _GmshBall(Problem):
    def __init__(self, map_to_sphere=True):
        super().__init__()
        self._map = map_to_sphere

    def define_problem(self):
        self += _GmshBallTemplate(self._map)
        eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ "shell"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


_SEAM_CENTRE = 160.0  # the reported droplet sat this far up the axis, measured in its own radii
_SEAM_R = 2.0


def _circumcircle(p0, p1, p2):
    # Centre of the circle through three points, fitted the way GmshTemplate.circle_arc(through_point=)
    # does it: solve a*(x^2+y^2) + b*x + c*y + 1 = 0 and read off (-b/2a, -c/2a).
    m = numpy.array([[x * x + y * y, x, y] for x, y in (p0, p1, p2)])
    a, b, c = numpy.linalg.solve(m, -numpy.ones(3))
    return [-b / (2 * a), -c / (2 * a), 0.0]


class _SeamRingTemplate(MeshTemplate):
    # A half annulus whose outer rim is described by SEVERAL circle arcs, each fitted through its own
    # three points, with the whole thing sitting far from the origin. Both halves matter and both come
    # straight from the reported case: a droplet interface declared by three circle_arc(through_point=)
    # calls, 80 radii up the axis of an axisymmetric domain.
    #
    # Three fits of one circle do not produce one circle. They produce three circles whose centres and
    # radii differ in the last bits, so at each seam node the two arcs meeting there disagree about
    # where that node is -- by ~1e-16 relative, which the distance to the origin turns into ~1e-12
    # absolute. The elements on either side of the seam must not inherit that disagreement.
    def __init__(self, nseg=6, narcs=3):
        super().__init__()
        self._nseg, self._narcs = nseg, narcs
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        ang = [-0.5 * math.pi + math.pi * i / self._nseg for i in range(self._nseg + 1)]
        pos = lambda r, t: (r * math.cos(t), _SEAM_CENTRE + r * math.sin(t))
        inner = [self.add_node_unique(*pos(0.5 * _SEAM_R, t)) for t in ang]
        outer = [self.add_node_unique(*pos(_SEAM_R, t)) for t in ang]
        for i in range(self._nseg):
            # Same orientation as _ArcSectorTemplate: rim on the north edge, positive Jacobian.
            domain.add_quad_2d_C1(inner[i + 1], inner[i], outer[i + 1], outer[i])
        per = self._nseg // self._narcs
        for a in range(self._narcs):
            lo, hi = a * per, (a + 1) * per
            arc = _pyoomph.CurvedEntityCircleArc(
                _circumcircle(pos(_SEAM_R, ang[lo]), pos(_SEAM_R, ang[(lo + hi) // 2]), pos(_SEAM_R, ang[hi])),
                self.get_node_position(outer[lo]), self.get_node_position(outer[hi]))
            self._entities.append(arc)
            for i in range(lo, hi):
                self.add_facet_to_boundary("rim", [outer[i], outer[i + 1]], [outer[i], outer[i + 1]], arc)


class _SeamRing(Problem):
    def __init__(self):
        super().__init__()

    def define_problem(self):
        self += _SeamRingTemplate()
        eqs = PoissonEquation(source=1, coefficient=1, space="C2") + DirichletBC(u=0) @ "rim"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _ArcSectorTemplate(MeshTemplate):
    # One quad annulus sector spanning [a0, a1] degrees with its outer edge on a circular arc, used
    # to sweep the arc across the atan2 branch cut at +-pi. keep_entity=False deliberately drops the
    # only Python reference to the entity (see the lifetime regression below).
    # `order` picks which of the two possible node orderings the rim facet is declared in. It is a
    # free choice of the mesh author and describes the same geometry either way -- but it decides
    # which endpoint apply_periodicity() decides to mirror, and hence whether the mesh builds.
    def __init__(self, a0_deg, a1_deg, keep_entity=True, order="rev"):
        super().__init__()
        self._a0, self._a1 = math.radians(a0_deg), math.radians(a1_deg)
        self._keep = keep_entity
        self._order = order
        self._entities = []

    def define_geometry(self):
        domain = self.new_domain("domain")
        n = [self.add_node_unique(r * math.cos(t), r * math.sin(t))
             for r in (0.5 * _R, _R) for t in (self._a0, self._a1)]
        # oomph QElement<2,2> ordering is SW, SE, NW, NE; this orientation puts the outer arc on the
        # north edge with a positive Jacobian.
        domain.add_quad_2d_C1(n[1], n[0], n[3], n[2])
        rim = [n[3], n[2]] if self._order == "rev" else [n[2], n[3]]
        arc = _pyoomph.CurvedEntityCircleArc([0, 0, 0],
                                             self.get_node_position(rim[0]),
                                             self.get_node_position(rim[1]))
        if self._keep:
            self._entities.append(arc)
        self.add_facet_to_boundary("arc", rim, rim, arc)


class _ArcSector(Problem):
    def __init__(self, a0_deg, a1_deg, keep_entity=True, order="rev"):
        super().__init__()
        self._a0, self._a1, self._keep, self._order = a0_deg, a1_deg, keep_entity, order

    def define_problem(self):
        self += _ArcSectorTemplate(self._a0, self._a1, self._keep, self._order)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "arc"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _MovingGmshDiskTemplate(GmshTemplate):
    # A gmsh triangular disc, rim on circle arcs. Gmsh-based on purpose: only a template that
    # records a .msh file in the state can come back from load_state as a *different* template
    # (MeshTemplate.define_state_file), which is what makes the meshes be rebuilt there.
    def define_geometry(self):
        self.default_resolution = 0.4
        self.mesh_mode = "tris"
        self.create_circle_lines((0, 0), _R, line_name="circumference")
        self.plane_surface("circumference", name="domain")


class _MovingGmshDisk(Problem):
    # ALE: the nodal positions are dofs, so the macro elements must not survive setup.
    def define_problem(self):
        from pyoomph.equations.ALE import LaplaceSmoothedMesh
        self += _MovingGmshDiskTemplate()
        eqs = PoissonEquation(source=1, space="C2") + DirichletBC(u=0) @ "circumference"
        eqs += LaplaceSmoothedMesh()
        self += eqs @ "domain"


# --------------------------------------------------------------------------------------------
# Child-process driver
# --------------------------------------------------------------------------------------------

def _worker_radius_error(tmp_path, *args):
    # Run one case in a fresh interpreter and return the max |r - R| it reported. Anything other
    # than a clean run with a RESULT line is an assertion failure carrying the child's diagnostics,
    # so a throw and an abort are both reported rather than silently swallowed.
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), str(tmp_path), *map(str, args)],
                          capture_output=True, text=True, timeout=600,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return float(line.split()[1])
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
    raise AssertionError(f"worker {args} did not report a result (exit {proc.returncode}):\n{tail}")


def _worker_lines(tmp_path, *args):
    # Like _worker_radius_error, but for cases reporting several named quantities. Returns them as a
    # dict of the child's "KEY value" lines.
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), str(tmp_path), *map(str, args)],
                          capture_output=True, text=True, timeout=600,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isupper() and parts[0].isalpha():
            out[parts[0]] = parts[1]
    if not out:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
        raise AssertionError(f"worker {args} reported nothing (exit {proc.returncode}):\n{tail}")
    return out


def _worker_main(argv):
    outdir, kind = argv[0], argv[1]
    if kind == "tri":
        space, nref = argv[2], int(argv[3])
        problem, mesh_name, boundary = _TriDisk(space=space), "domain", "circumference"
    elif kind == "arc":
        a0, nref, order = float(argv[2]), int(argv[3]), argv[4]
        problem, mesh_name, boundary = _ArcSector(a0, a0 + 30.0, order=order), "domain", "arc"
    elif kind == "sphere":
        space, nref = argv[2], int(argv[3])
        problem = _SphereOctant(space=space)
        problem.set_output_directory(outdir)
        problem.max_refinement_level = 4
        problem.initialise()
        for _ in range(nref):
            problem.refine_uniformly()
        print("RESULT", _max_radius_error(problem.get_mesh("domain"), "shell", ndim=3))
        return
    elif kind == "overmark":
        # Compare the nodes MARKED as being on a boundary against the nodes that actually lie on one of
        # its facets. The interface mesh for a boundary is generated from those facets, so its node set
        # is the truth; positions are the key, since the two meshes hand out distinct wrappers for the
        # same node.
        which, nref = argv[2], int(argv[3])
        boundary = "circumference" if which == "tri" else "shell"
        if which == "tetball":
            template = _TetBallTemplate()
        elif which == "gmshball":
            template = _GmshBallTemplate()
        elif which == "tri":
            template = _TriDiskTemplate()
        else:
            template = _SphericalTetTemplate(4 if which == "singletet" else 2)

        class _P(Problem):
            def define_problem(self):
                self += template
                eqs = PoissonEquation(source=1, space="C1") + DirichletBC(u=0) @ boundary
                eqs += SpatialErrorEstimator(u=1)
                self += eqs @ "domain"

        with _P() as problem:
            problem.set_output_directory(outdir)
            problem.max_refinement_level = nref + 1
            problem.initialise()
            for _ in range(nref):
                problem.refine_uniformly()
            mesh = problem.get_mesh("domain")
            bidx = mesh.get_boundary_index(boundary)

            def key(nd):
                return tuple(round(nd.x(i), 12) for i in range(nd.ndim()))
            marked = set(key(nd) for nd in mesh.nodes() if nd.is_on_boundary(bidx))
            on_facets = set(key(nd) for nd in problem.get_mesh("domain/" + boundary).nodes())
            print("MARKED", len(marked))
            print("SPURIOUS", len(marked - on_facets))
            # The other direction. The interface mesh is built through build_face_element(), which is
            # the routine that knows about oddities like the C2TB tet's face bubble, so this is an
            # INDEPENDENT oracle for the face-node enumeration the repair relies on -- a self-consistent
            # check against the same tables could not catch a wrong one.
            print("UNMARKED", len(on_facets - marked))
            # And the mesh's own view of the same question, straight off the face tags.
            sp, mi = mesh.check_boundary_node_membership()
            print("SELFSPURIOUS", sp)
            print("SELFMISSING", mi)
        return
    elif kind == "coupled":
        curved = argv[2] == "1"
        with _CurvedInterfaceProblem(curved) as problem:
            problem.set_output_directory(outdir)
            problem.max_refinement_level = 3
            problem.initialise()
            problem.solve()
            r1 = _CurvedInterfaceTemplate._R[1]
            worst = 0.0
            sets = []
            for dom in ("inner", "outer"):
                mesh = problem.get_mesh(dom)
                bidx = mesh.get_boundary_index("interface")
                pts = []
                for nd in mesh.nodes():
                    if nd.is_on_boundary(bidx):
                        worst = max(worst, abs(math.hypot(nd.x(0), nd.x(1)) - r1))
                        pts.append((round(nd.x(0), 12), round(nd.x(1), 12)))
                sets.append(sorted(pts))
            print("RESULT", worst)
            print("IDENTICAL", 1 if sets[0] == sets[1] and sets[0] else 0)
            print("NIFACE", len(sets[0]))
        return
    elif kind == "statereload":
        # save and load run in separate interpreters under separate output directories, because the
        # .msh path recorded in the state is relative to the state file and it is precisely a
        # *changed* path that makes load_state rebuild the meshes -- the same thing a remesh does.
        phase = argv[2]
        statefile = os.path.join(outdir, "state.dump")
        with _MovingGmshDisk() as problem:
            problem.set_output_directory(os.path.join(outdir, phase))
            problem.initialise()
            if phase == "save":
                problem.save_state(statefile)
                print("SAVED 1")
                return
            problem.load_state(statefile)
            mesh = problem.get_mesh("domain")
            print("NMACRO", sum(1 for e in mesh.elements() if e.get_macro_element() is not None))
            # Move the free surface, as an ALE solve would: scale the whole disc up by 20%.
            scale = 1.2
            for i in range(mesh.nnode()):
                nd = mesh.node_pt(i)
                for t in range(nd.ntstorage()):
                    nd.set_x_at_t(t, 0, nd.x_at_t(t, 0) * scale)
                    nd.set_x_at_t(t, 1, nd.x_at_t(t, 1) * scale)
            problem.refine_uniformly()
            bidx = mesh.get_boundary_index("circumference")
            moved, frozen = 0.0, 1.0e30
            for i in range(mesh.nnode()):
                nd = mesh.node_pt(i)
                if not nd.is_on_boundary(bidx):
                    continue
                r = math.hypot(nd.x(0), nd.x(1))
                moved = max(moved, abs(r - scale * _R))
                frozen = min(frozen, abs(r - _R))
            print("RESULT", moved)
            print("FROZEN", frozen)
        return
    elif kind == "seam":
        problem = _SeamRing()
        problem.set_output_directory(outdir)
        problem.check_mesh_integrity = False  # measured below instead, so a failure is a number
        problem.initialise()
        mesh = problem.get_mesh("domain")
        # 1. Does the macro map reproduce the element's own vertex positions? Everything else follows
        #    from this: two elements meeting at a node hold the same Node, so if each returns exactly
        #    what the node says, they cannot disagree.
        worst = 0.0
        for e in mesh.elements():
            if e.get_macro_element() is None:
                continue
            for (s0, s1), ln in (((-1, -1), 0), ((1, -1), 2), ((-1, 1), 6), ((1, 1), 8)):
                x = e.get_macro_element_position_at_s([float(s0), float(s1)])
                nd = e.node_pt(ln)
                worst = max(worst, abs(x[0] - nd.x(0)), abs(x[1] - nd.x(1)))
        print("VERTEXDEV", worst)
        # 2. And the end-to-end consequence: oomph compares neighbouring elements' get_x() at the
        #    shared corners against an absolute 1e-14 and takes the run down if they differ.
        try:
            mesh.check_integrity()
            print("INTEGRITY 1")
        except RuntimeError:
            print("INTEGRITY 0")
        return
    elif kind == "normaldisk":
        renormalise, nref = argv[2] == "1", int(argv[3])
        problem = _NormalDisk(renormalise=renormalise)
        problem.set_output_directory(outdir)
        problem.max_refinement_level = 3
        problem.initialise()
        for _ in range(nref):
            problem.refine_uniformly()
        print("RESULT", _max_radius_error(problem.get_mesh("domain"), "circumference"))
        return
    elif kind == "gmshball":
        nref, mapped = int(argv[2]), argv[3] == "1"
        problem = _GmshBall(map_to_sphere=mapped)
        problem.set_output_directory(outdir)
        problem.max_refinement_level = 3
        problem.initialise()
        for _ in range(nref):
            problem.refine_uniformly()
        mesh = problem.get_mesh("domain")
        bidx = mesh.get_boundary_index("shell")
        nmacro = sum(1 for e in mesh.elements() if e.get_macro_element() is not None)
        nface = 0
        for e in mesh.elements():
            if sum(1 for i in range(e.nnode()) if e.node_pt(i).is_on_boundary(bidx)) >= 3:
                nface += 1
        print("RESULT", _max_radius_error(mesh, "shell", ndim=3))
        print("NMACRO", nmacro)
        print("NFACE", nface)
        return
    elif kind in ("wedge", "pyramid"):
        nref = int(argv[3])
        problem = _MixedShell(kind)
        problem.set_output_directory(outdir)
        problem.max_refinement_level = 4
        problem.initialise()
        for _ in range(nref):
            problem.refine_uniformly()
        print("RESULT", _max_radius_error(problem.get_mesh("domain"), "shell", ndim=3))
        return
    elif kind == "tetball":
        space, nref = argv[2], int(argv[3])
        problem = _TetBall(space=space)
        problem.set_output_directory(outdir)
        problem.max_refinement_level = 4
        problem.initialise()
        for _ in range(nref):
            problem.refine_uniformly()
        print("RESULT", _max_radius_error(problem.get_mesh("domain"), "shell", ndim=3))
        return
    elif kind == "gridtet":
        _worker_grid_tet(outdir, int(argv[2]))
        return
    elif kind == "hang":
        _worker_hanging(outdir, argv[2])
        return
    else:
        raise SystemExit(f"unknown worker case {kind!r}")
    # No "with": the teardown of a mesh left half-refined by the MACRO ELEM throw aborts, which
    # would replace the real error message with a signal.
    problem.set_output_directory(outdir)
    problem.max_refinement_level = 4
    problem.initialise()
    for _ in range(nref):
        problem.refine_uniformly()
    print("RESULT", _max_radius_error(problem.get_mesh(mesh_name), boundary))


def _worker_grid_tet(outdir, ncurved):
    # Sample the macro map densely over each of the tet's four faces and report the worst deviation
    # from the sphere, separately for the faces declared curved and the ones not. This asks only
    # whether the blend is right, independent of which nodes the mesh happens to mark as boundary
    # nodes -- see test_spherical_tet_blend_is_exact_on_every_curved_face for why that matters.
    problem = _SphericalTet(ncurved)
    problem.set_output_directory(outdir)
    problem.max_refinement_level = 0
    problem.initialise()
    element = problem.get_mesh("domain").element_pt(0)
    # lambda = (s0, s1, s2, 1-s0-s1-s2), so face k is the set where lambda_k == 0.
    curved_sets = [set(f) for f in _SphericalTetTemplate._FACES[:ncurved]]
    worst_curved, worst_flat, npts = 0.0, 0.0, 21
    for k in range(4):
        verts = sorted({0, 1, 2, 3} - {k})
        worst = 0.0
        for i in range(npts):
            for j in range(npts - i):
                bary = [i / (npts - 1.0), j / (npts - 1.0), 0.0]
                bary[2] = 1.0 - bary[0] - bary[1]
                lam = [0.0] * 4
                for slot, v in enumerate(verts):
                    lam[v] = bary[slot]
                r = element.get_macro_element_position_at_s(lam[:3])
                worst = max(worst, abs(math.sqrt(sum(c * c for c in r)) - _R))
        if set(verts) in curved_sets:
            worst_curved = max(worst_curved, worst)
        else:
            worst_flat = max(worst_flat, worst)
    print("CURVED", worst_curved)
    print("FLAT", worst_flat)


def _worker_hanging(outdir, shape):
    # Refine the rim region harder than the interior, so the mesh ends up genuinely non-conforming,
    # and report how many nodes hang in total, how many of those are on the curved boundary, and how
    # far the boundary nodes are off the circle.
    problem = _QuadDisk() if shape == "quad" else _TriDisk()
    problem.set_output_directory(outdir)
    problem.max_refinement_level = 4
    problem += RefineToLevel(1) @ "domain"
    problem += RefineToLevel(3) @ "domain/circumference"
    problem.initialise()
    mesh = problem.get_mesh("domain")
    bidx = mesh.get_boundary_index("circumference")

    nhang = sum(1 for n in mesh.nodes() if n.is_hanging())
    nhang_boundary, max_rim_err = 0, 0.0
    for node in mesh.nodes():
        if not node.is_on_boundary(bidx):
            continue
        if node.is_hanging():
            nhang_boundary += 1
        max_rim_err = max(max_rim_err, abs(math.hypot(node.x(0), node.x(1)) - _R))
    print("NHANG", nhang)
    print("NHANGBOUNDARY", nhang_boundary)
    print("MAXRIMERR", max_rim_err)


if __name__ == "__main__":
    _worker_main(sys.argv[1:])


# --------------------------------------------------------------------------------------------
# Behaviour that already holds, and must keep holding
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["C1", "C2"])
def test_curved_quad_template_mesh_is_exact(space):
    # map_nodes_on_macro_element() runs when the template mesh is generated, so the unrefined mesh
    # sits exactly on the circle. This is the one part of the macro-element machinery that works for
    # every quad today.
    with _QuadDisk(space=space) as problem:
        problem.max_refinement_level = 0
        problem.initialise()
        assert _max_radius_error(problem.get_mesh("domain"), "circumference") < _EXACT


def test_map_nodes_on_macro_elements_is_idempotent():
    # Until S1, refinement placed new nodes by FE interpolation and this global pass was what repaired
    # them -- so this test used to assert the mesh had drifted before calling it. Now nodes are placed
    # correctly when they are created and the pass has nothing left to do. Idempotence is the stronger
    # statement of the two: it says the two routes onto the geometry (at creation, and by re-snapping
    # afterwards) agree, which is what makes the pass safe to keep.
    with _QuadDisk() as problem:
        problem.max_refinement_level = 4
        problem.initialise()
        problem.refine_uniformly()
        mesh = problem.get_mesh("domain")
        before = [(n.x(0), n.x(1)) for n in mesh.nodes()]
        assert _max_radius_error(mesh, "circumference") < _EXACT
        problem.map_nodes_on_macro_elements()
        after = [(n.x(0), n.x(1)) for n in mesh.nodes()]
        assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(before, after)) < _EXACT


def test_curved_entity_survives_dropped_python_reference():
    # MeshTemplateFacet stores the curved entity as a bare borrowed pointer, so before the
    # nb::keep_alive<1,5> on add_facet_to_boundary this segfaulted during mesh generation whenever
    # the caller did not happen to keep the Python object alive. Here define_geometry() drops it on
    # purpose.
    with _ArcSector(10, 40, keep_entity=False) as problem:
        problem.max_refinement_level = 0
        problem.initialise()
        gc.collect()
        problem.map_nodes_on_macro_elements()
        assert _max_radius_error(problem.get_mesh("domain"), "arc") < _EXACT


# --------------------------------------------------------------------------------------------
# T1 -- every element shape places refined nodes on the curve
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["C1", "C2"])
def test_curved_quad_uniform_refinement_is_exact(space):
    # Before S1: 7.6e-2 (C1) / 5.4e-4 (C2), because RefineableSolidQElement<2>::build overwrote the
    # macro-element position with the FE one and nothing put it back.
    with _QuadDisk(space=space) as problem:
        problem.max_refinement_level = 4
        problem.initialise()
        for _ in range(2):
            problem.refine_uniformly()
        assert _max_radius_error(problem.get_mesh("domain"), "circumference") < _EXACT


@pytest.mark.parametrize("space", ["C1", "C2"])
def test_curved_tri_uniform_refinement_is_exact(space, tmp_path):
    # Before S1 this threw "MACRO ELEM" outright (refineable_telements.cpp:743).
    assert _worker_radius_error(tmp_path, "tri", space, 1) < _EXACT


@pytest.mark.parametrize("space", ["C1", "C2"])
def test_curved_tri_template_mesh_is_exact(space, tmp_path):
    # Before S1 the C2 case gave 7.6e-2 = 1 - cos(22.5 deg): map_nodes_on_macro_element() returned
    # early for T-elements, so the mid-edge node each rim facet gains from convert_for_C2_space stayed
    # at the chord midpoint. The triangular gap was not merely "cannot refine" -- an *unrefined*
    # curved triangular mesh was already wrong, i.e. the macro element did nothing at all.
    assert _worker_radius_error(tmp_path, "tri", space, 0) < _EXACT


# --------------------------------------------------------------------------------------------
# T6 -- the runtime adapt() path, which never calls the global re-snap
# --------------------------------------------------------------------------------------------

def test_curved_quad_runtime_adapt_is_exact():
    # Before S1: 2.2e-06. map_nodes_on_macro_elements() is only called from the initial-adaption and
    # remeshing paths, so error-estimator driven adaptation during a solve never got the repair.
    # Refine after initialisation without invoking the repair pass -- i.e. what error-estimator
    # driven adaptation does during a time loop.
    with _QuadDisk() as problem:
        problem.max_refinement_level = 2
        problem += RefineToLevel(2) @ "domain"
        problem.initialise()
        assert _max_radius_error(problem.get_mesh("domain"), "circumference") < _EXACT
        problem.max_refinement_level = 4
        problem.refine_uniformly()
        assert _max_radius_error(problem.get_mesh("domain"), "circumference") < _EXACT


# --------------------------------------------------------------------------------------------
# T2 -- the arc must build at every orientation, seam or no seam
# --------------------------------------------------------------------------------------------

# CurvedEntityCircleArc parametrises by atan2, so its chart is cut along the negative x axis and an
# arc straddling that cut arrives with endpoints near +pi and -pi. apply_periodicity() now unwraps
# every node of the facet onto the branch nearest the first one's, which is correct for any number
# of facet nodes and for any orientation.
#
# It did not used to be. Measured on 2026-07-29, before that fix, a 30 deg arc across the cut failed
# in *both* available ways, and which one it hit depended on the order in which the facet's two
# nodes were declared -- a free choice of the mesh author describing identical geometry:
#
#   order         a0 = 155..165          a0 = 170..180
#   start-to-end  throws                 exact
#   end-to-start  builds, 1.9e-2..3.4e-2 throws
#
# The lower-left cell is why this is parametrised over both orders rather than over angles alone.
# There the old heuristic replaced the angle p by -p, a reflection across the x axis rather than an
# unwrap by 2*pi, so the entity reported the mirror-image point for that corner, the Coons blend's
# corners stopped agreeing, and the mesh came out silently wrong -- no error, just a boundary node
# up to 3.4e-2 off a unit circle. Both orders are kept permanently so that a future rewrite of the
# parametrisation (S2) cannot reintroduce an orientation- or order-dependent seam unnoticed.
_ARC_CASES = [
    (order, a0) for order in ("fwd", "rev")
    for a0 in (0, 60, 120, 150, 155, 160, 165, 170, 175, 180, 190, 240, 300)
]


@pytest.mark.parametrize("order,a0", [pytest.param(o, a, id=f"{o}-{a}") for o, a in _ARC_CASES])
def test_curved_arc_is_exact_at_every_orientation(order, a0, tmp_path):
    assert _worker_radius_error(tmp_path, "arc", a0, 0, order) < _EXACT


# --------------------------------------------------------------------------------------------
# T7 -- hanging nodes on a curved boundary
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["quad", "tri"])
def test_non_uniform_refinement_keeps_curved_boundary_exact(shape, tmp_path):
    # map_nodes_on_macro_element() skips hanging nodes, because a hanging node's position is dictated
    # by its masters and snapping it onto the curve would put it somewhere its own constraint does not.
    # In 2d that guard turns out never to fire, and the reason is worth recording rather than leaving
    # to be rediscovered: a node interior to a *boundary* edge belongs to exactly one element, since a
    # boundary facet has no neighbour across it, so nothing coarser can constrain it. Boundary nodes in
    # 2d therefore cannot hang at all. (In 3d two boundary faces do share an edge, so a node on that
    # shared edge can hang and the guard becomes load-bearing -- that case belongs to S3.)
    #
    # What this does test, and what actually matters here, is that a strongly non-conforming mesh --
    # 24 hanging nodes for the quad disc, 64 for the triangular one, measured 2026-07-29 -- leaves the
    # curved boundary exact anyway.
    out = _worker_lines(tmp_path, "hang", shape)
    assert int(out["NHANG"]) > 0, "refinement was uniform after all, so this proves nothing"
    assert int(out["NHANGBOUNDARY"]) == 0
    assert float(out["MAXRIMERR"]) < _EXACT


# --------------------------------------------------------------------------------------------
# T4 -- 3d
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nref", [0, 1, 2])
@pytest.mark.parametrize("space", ["C1", "C2"])
def test_curved_brick_sphere_is_exact(space, nref, tmp_path):
    # The first 3d curved boundary pyoomph can actually build. Before S3 the shell entity of
    # SphericalOctantMesh sat behind "if False: # TODO: This does not work yet", and the sphere entity
    # it would have used was parametrised by (theta, phi) -- a chart with a branch cut and a genuine
    # degeneracy at the pole. It is now the outward unit normal, which has neither.
    assert _worker_radius_error(tmp_path, "sphere", space, nref) < _EXACT


# --------------------------------------------------------------------------------------------
# T13 -- the 3d shared-edge correction
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ncurved", [1, 2, 3, 4])
def test_spherical_tet_blend_is_exact_on_every_curved_face(ncurved, tmp_path):
    # Any two faces of a tetrahedron share an edge, so ncurved >= 2 exercises the inclusion-exclusion
    # term: each face's deviation includes that of the shared edge, and without the correction the
    # edge would receive it twice. Before the correction existed this configuration was refused
    # outright rather than answered wrongly.
    #
    # This samples the macro map itself rather than looking at node positions, deliberately. A tet
    # with only some of its faces on a boundary is the pathological case for oomph's rule that a new
    # node inherits the boundaries shared by its parents: the midpoint of an edge joining two shell
    # nodes gets marked as a shell node even when the edge itself runs through the interior. That
    # happens with no macro element in sight -- measured with straight-sided elements, 1 of 35 nodes
    # at two refinements -- so it is not this machinery's doing, and measuring node positions here
    # would charge it for someone else's bookkeeping.
    out = _worker_lines(tmp_path, "gridtet", ncurved)
    assert float(out["CURVED"]) < _EXACT
    # The faces that were not declared curved must NOT be pulled onto the sphere: they are ruled
    # surfaces carrying whatever curved edges they inherit, which is the geometrically right answer
    # for a partially curved element and the thing the old flat-face treatment got wrong.
    if ncurved < 4:
        assert float(out["FLAT"]) > 1e-3


@pytest.mark.parametrize("space", ["C1", "C2"])
@pytest.mark.parametrize("nref", [0, 1, 2])
def test_curved_tet_ball_is_exact(space, nref, tmp_path):
    # The realistic 3d simplex case: an octant of a ball as three tets, each with exactly one face on
    # the sphere, so the boundary bookkeeping above is unambiguous.
    assert _worker_radius_error(tmp_path, "tetball", space, nref) < _EXACT


# --------------------------------------------------------------------------------------------
# T5 -- wedges, pyramids, and the mixed forest
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nref", [0, 1, 2])
@pytest.mark.parametrize("kind", ["wedge", "pyramid"])
def test_curved_wedge_and_pyramid_are_exact(kind, nref, tmp_path):
    # The pyramid is the interesting one twice over. Its C1 shape functions are rational with a
    # removable singularity at the apex, where the whole quadrilateral base collapses to a point --
    # the shipped shape function divides by 1-s2 unguarded, so the macro version takes the limit
    # explicitly. And refining it yields six pyramids and four tetrahedra, so a son inherits a macro
    # element from a father of a *different shape*: 10 elements after one refinement, 92 after two.
    # That works only because the son's region is carried as vertex coordinates rather than as
    # oomph's axis-aligned box, which cannot express it.
    assert _worker_radius_error(tmp_path, kind, "C1", nref) < _EXACT


# --------------------------------------------------------------------------------------------
# gmsh in 3d, and the curved-edge registry it needs
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nref", [0, 1, 2])
def test_gmsh_ball_shell_is_exact(nref, tmp_path):
    # Curved boundaries reachable from gmsh in 3d, which they were not before: _curved_entities2d was
    # declared with a "TODO: Set those" and never populated, so a gmsh 3d mesh had no curved geometry
    # at all regardless of how it was built.
    out = _worker_lines(tmp_path, "gmshball", nref, 1)
    assert float(out["RESULT"]) < _EXACT


def test_gmsh_ball_needs_the_curved_edge_registry(tmp_path):
    # The reason an unstructured mesh needs §3.2.2's registry, stated as a measurement. On the coarse
    # gmsh ball 45 of 126 elements own a face on the sphere -- but 70 carry a macro element, because
    # another 25 touch the sphere along an edge only. Those 25 have no curved facet of their own, and
    # before the registry they placed that edge's new nodes on the chord while the element on the
    # other side placed them on the sphere; whichever built the node first won, and the shell came out
    # 1.1e-2 off rather than exact.
    out = _worker_lines(tmp_path, "gmshball", 0, 1)
    nmacro, nface = int(out["NMACRO"]), int(out["NFACE"])
    assert nface == 45
    assert nmacro == 70, "expected the 45 face-touching elements plus 25 edge-only ones"
    assert float(out["RESULT"]) < _EXACT


def test_gmsh_map_to_sphere_is_opt_in(tmp_path):
    # Gmsh's built-in kernel does not produce an exact sphere from a ruled surface even when the
    # bounding curves are great-circle arcs, so pyoomph must not assume one. Without the opt-in the
    # shell stays polyhedral, as it always has.
    out = _worker_lines(tmp_path, "gmshball", 1, 0)
    assert float(out["RESULT"]) > 1e-3


# --------------------------------------------------------------------------------------------
# The entity interface: a user-defined chart that needs its own blending rule
# --------------------------------------------------------------------------------------------

def test_user_entity_can_own_its_blending_rule(tmp_path):
    # A parametric coordinate is an opaque vector whose meaning belongs to the entity, so how two of
    # them combine has to belong to the entity too. The default is a weighted sum, which is right for
    # a flat chart (an angle, an arclength, a spline parameter) and wrong for a redundant one: the
    # average of two unit normals is not a unit normal.
    #
    # Both halves are asserted, because the interesting claim is not "it works" but "the hook is
    # load-bearing": the same geometry with the default sum is off by 7.6e-2, exactly the chord
    # sagitta of a 45 degree facet, i.e. no better than no curved treatment at all.
    assert _worker_radius_error(tmp_path, "normaldisk", 1, 1) < _EXACT
    assert _worker_radius_error(tmp_path, "normaldisk", 0, 1) > 1e-3


def test_intrinsic_and_parametric_dimensions_are_reported_separately():
    # A sphere patch is a 2-manifold charted by 3 numbers, and a curve can be charted by 2. The
    # redundancy is the point (no 2-component chart of a sphere avoids a degeneracy at the poles), so
    # it is reported rather than left looking like an inconsistency.
    sphere = _pyoomph.CurvedEntitySpherePart([0, 0, 0], [_R, 0, 0])
    assert sphere.get_parametric_dimension() == 3
    assert sphere.get_intrinsic_dimension() == 2
    arc = _pyoomph.CurvedEntityCircleArc([0, 0, 0], [_R, 0, 0], [0, _R, 0])
    assert arc.get_parametric_dimension() == 1
    assert arc.get_intrinsic_dimension() == 1
    user = _CircleByNormal([0, 0], _R)
    assert user.get_parametric_dimension() == 2
    assert user.get_intrinsic_dimension() == 1


# --------------------------------------------------------------------------------------------
# Coupled interfaces (dev_docs 19.4)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("curved", [True, False])
def test_curved_shared_interface_agrees_from_both_sides(curved, tmp_path):
    # A curved boundary shared by two coupled domains. The two sides refine independently -- only
    # "inner" carries the refinement requirement, and InterfaceRefinementCoupler carries it across --
    # so each places the interface's new nodes through its own macro elements. They must land in the
    # same places, or connect_interface_elements_by_kdtree has nothing to pair up.
    #
    # dev_docs 19.4 listed this as an argument rather than a measurement: both sides attach the same
    # entity to the same facets, so agreement "should" follow. This is the measurement.
    out = _worker_lines(tmp_path, "coupled", 1 if curved else 0)
    assert int(out["NIFACE"]) > 0
    assert int(out["IDENTICAL"]) == 1, "the two domains disagree about where the interface nodes are"
    if curved:
        assert float(out["RESULT"]) < _EXACT
    else:
        assert float(out["RESULT"]) > 1e-3      # straight-sided, as a control


# --------------------------------------------------------------------------------------------
# Boundary-node membership (dev_docs 15.2, characterised in 23, repaired per
# dev_docs/boundary_node_membership_repair.md)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["tri", "tetball", "gmshball"])
@pytest.mark.parametrize("nref", [1, 2])
def test_boundary_node_membership_matches_the_facets(which, nref, tmp_path):
    # A node is on a boundary iff it belongs to one of that boundary's facets. The refinement rules
    # approximate this, by giving a new node the boundaries SHARED BY ALL its generating nodes
    # (refineable_telements.cpp:458-467 for 2d tris, and the tet/wedge/pyramid/brick equivalents) --
    # and two nodes can share a boundary label without the edge between them lying on that boundary.
    #
    # On every mesh anyone would actually write, the approximation is exact anyway, because an element
    # meets a given boundary in a single face and so no interior edge has both ends on it. That is why
    # the repair is a no-op here, and this pins BOTH halves of that: no spurious marks, and -- the more
    # important direction now that memberships are actively removed -- nothing missing either.
    out = _worker_lines(tmp_path, "overmark", which, nref)
    assert int(out["MARKED"]) > 0
    assert int(out["SPURIOUS"]) == 0
    assert int(out["UNMARKED"]) == 0, "the repair dropped a node the interface mesh does own"
    assert (int(out["SELFSPURIOUS"]), int(out["SELFMISSING"])) == (0, 0)


@pytest.mark.parametrize("which,nref,was", [("halftet", 1, 1), ("halftet", 3, 84), ("singletet", 3, 35)])
def test_boundary_node_membership_is_repaired_when_a_boundary_wraps_an_element(which, nref, was, tmp_path):
    # The configuration the approximation breaks on: a tetrahedron with several of its faces on one
    # boundary. Its other faces then have all their vertices on that boundary, so every edge of them is
    # mislabelled, and the error compounds with refinement -- 51% of the marked nodes at three
    # refinements for the two-face case.
    #
    # `was` is what these cases measured before the post-adapt repair landed, when this was a strict
    # xfail; it is quoted in the failure message because "84 spurious" is a much better description of a
    # regression here than "not 0". The counts after the repair are 165-84=81 and 165-35=130 marked.
    out = _worker_lines(tmp_path, "overmark", which, nref)
    assert int(out["MARKED"]) > 0
    assert int(out["SPURIOUS"]) == 0, "%s at nref=%d: %s spurious (was %d before the repair)" % (
        which, nref, out["SPURIOUS"], was)
    assert int(out["UNMARKED"]) == 0, "the repair overshot and dropped a genuine membership"
    assert (int(out["SELFSPURIOUS"]), int(out["SELFMISSING"])) == (0, 0)


# --------------------------------------------------------------------------------------------
# T15 -- macro elements must not survive a state load onto a moving mesh
# --------------------------------------------------------------------------------------------

def test_state_load_drops_macro_elements_on_a_moving_mesh(tmp_path):
    # A macro element freezes the geometry of the template it was built from. That is exactly right
    # while the nodes are fixed, and exactly wrong once they are dofs: refinement takes every new
    # node's position from father->get_x(), which goes through the macro map whenever one is
    # attached, so on an ALE mesh a new node gets placed on the *template's* boundary rather than on
    # the boundary the solve has moved the mesh to. That is why initialise() and force_remesh() both
    # end with remove_macro_elements().
    #
    # load_state is the third place that hands a problem a freshly built mesh -- when the state
    # carries a different mesh template, which is what a state written after a remesh does -- and it
    # used to be the one that forgot. The symptom is undramatic enough to be missed: the mesh does
    # not throw or invert, the new boundary nodes are simply somewhere else, and on a free surface
    # they land off the interface entirely (found on a remeshed axisymmetric free-surface run, where
    # a refined interface node sat several elements away from the segment it was inserted into).
    #
    # Here the disc is blown up by 20% after the load, so the two answers are 0.2 apart and no
    # tolerance argument is needed: 4.2e-5 (the FE chord error of this mesh) if the new rim nodes
    # follow the mesh, 2.0e-1 if they snap back onto the template circle -- which is what they did,
    # exactly, with FROZEN measuring 0.0 because the macro map put them on r = 1 to machine
    # precision.
    _worker_lines(tmp_path, "statereload", "save")
    out = _worker_lines(tmp_path, "statereload", "load")
    assert int(out["NMACRO"]) == 0
    assert float(out["RESULT"]) < 1e-3, "new rim nodes do not follow the moved mesh"
    assert float(out["FROZEN"]) > 0.1, "a rim node sits on the template circle, i.e. on the macro map"


# --------------------------------------------------------------------------------------------
# T14 -- the macro map must reproduce the element's own vertex positions exactly
# --------------------------------------------------------------------------------------------

def test_arcs_meeting_at_a_node_do_not_move_it(tmp_path):
    # The blend is x = sum_v lambda_v X_v + sum_F w_F d_F, and it only reproduces X_v at a vertex if
    # d_F vanishes there exactly. Measuring d_F against the vertex node positions X_k did not achieve
    # that: X_k and the entity's own image C(p_k) of the same vertex agree only to round-off, so the
    # residue survived -- and it is entity-dependent, because an element evaluates a shared corner
    # through whichever curved entity it owns. Two arcs meeting at a node then place that node in two
    # places at once.
    #
    # Round-off is not too small to matter here. It is relative to the coordinate, and a feature far
    # from the origin (see _SEAM_CENTRE) has coordinates much larger than itself, while oomph's
    # neighbour check is an absolute 1e-14. On the reported case -- an axisymmetric droplet 80 radii up
    # the axis, its interface declared by three circle_arc(through_point=) calls -- that came to
    # 7.7e-13 and initialise() died in QuadTreeForest::check_all_neighbours before the first solve.
    #
    # Measuring d_F against C(p_k) instead makes the deviation vanish at the vertices to the last bit.
    # Before that fix this geometry gave VERTEXDEV 2.5e-13 and INTEGRITY 0; the assertion is on exactly
    # zero rather than on a tolerance because that is the property being claimed -- a tolerance here
    # would just be a second, smaller number to regress past.
    out = _worker_lines(tmp_path, "seam")
    assert float(out["VERTEXDEV"]) == 0.0, "the macro map does not reproduce its own vertex positions"
    assert int(out["INTEGRITY"]) == 1
