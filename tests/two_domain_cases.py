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

# Shared problem definitions for the COUPLED-INTERFACE adaptivity campaign. Same three-way sharing as
# box_cases.py: the serial tests (test_adaptive_interface_coupling.py), the MPI harness
# (test_mpi_interface_coupling.py) and the worker it launches under mpirun all import this one module, so
# the serial and distributed runs solve a bit-identical problem and differ only in how it is partitioned.
# It exposes the same solve_case()/case_id() interface as box_cases, so mpi_worker.py needs no changes.
#
# What is under test here is NOT the refinement engine (box_cases covers that) but the fact that two
# domains sharing an interface are adapted INDIVIDUALLY by oomph-lib. Each case therefore drives
# refinement ASYMMETRICALLY -- the criterion is stated for the "lower" domain only -- and the "upper"
# domain has no reason of its own to follow. Without Problem.enforce_interface_conformity() the
# opposite-element matcher (InterfaceMesh::connect_interface_elements_by_kdtree, which pairs interface
# elements by exact vertex-position sets) then has nothing to pair up and the run dies with
# "Cannot locate opposite element". See dev_docs/interface_refinement_coupling.md.
#
# The unit square, split at y=0.5 into "lower" and "upper" with the shared boundary named "interface":
#
#   kinds  quad / tri_left / tri_crossed  -- both domains the same family
#          mixed                          -- quads below, triangles above: the two sides of the interface
#                                            belong to DIFFERENT element families, so a facet subdivided
#                                            by a quad split has to be matched by a triangle split
#          curved / curved_tri            -- the interface is a circular ARC carrying a curved entity, so
#                                            a C2 side's midside nodes sit off the chord
#          box_hex / box_tet              -- 3d: the unit cube split at z=0.5, where a facet is a quad or
#                                            a triangle rather than a line
#
# and the ELEMENT SPACES of the two domains are an axis of their own ("connect:C2/C1TB"), because nothing
# in the coupling machinery requires them to agree and nobody had ever swept the matrix. The "move" family
# is the configuration in which they genuinely can disagree: the interface is prescribed as a curve on both
# sides, so a C2 domain's midside node sits ON it while a C1 domain has only the chord. See
# dev_docs/interface_refinement_coupling.md section 14 for what that sweep found.
#
# The measured quantities are the same partition-independent set as box_cases (gathered residual, global
# ndof, MPI-reduced integral observables) plus "nonconforming", which must be 0: the direct statement of
# the invariant, checked rather than inferred from the absence of a crash.

import math

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.ALE import LaplaceSmoothedMesh, ConnectMeshAtInterface
from pyoomph.equations.generic import ConnectFieldsAtInterface, RefineToLevel, SpatialErrorEstimator, ElementSpace
from pyoomph.equations.additional import RefineMaxElementSize, RefineAccordingToElement
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

MESH_KINDS = ["quad", "tri_left", "tri_crossed", "mixed"]
# Kinds with a CURVED interface, kept out of MESH_KINDS so the existing matrix is unchanged.
CURVED_KINDS = ["curved", "curved_tri"]
# The 3d two-domain kinds: a facet is a quad / a triangle rather than a line.
BOX3D_KINDS = ["box_hex", "box_tet"]
EQUATIONS = ["connect1", "connect2", "connect12", "ale"]

# The spaces a family can actually carry. The bubble spaces exist only on simplices, so a quad domain
# has two choices and a triangular one four. "mixed" is quads BELOW and triangles ABOVE.
SPACES_OF_FAMILY = {"quad": ["C1", "C2"], "tri": ["C1", "C1TB", "C2", "C2TB"]}


def families_of_kind(kind):
    """(lower family, upper family) for a mesh kind -- which decides which spaces each side may take."""
    if kind == "mixed":
        return ("quad", "tri")
    if kind in ("quad", "curved", "box_hex"):
        return ("quad", "quad")
    return ("tri", "tri")


def space_pairs_of_kind(kind):
    lo, up = families_of_kind(kind)
    return [(a, b) for a in SPACES_OF_FAMILY[lo] for b in SPACES_OF_FAMILY[up]]


# Legacy equation selectors, kept verbatim so the existing suite keeps measuring what it measured.
_LEGACY_EQ = {"connect1": ("connect", ("C1", "C1")),
              "connect2": ("connect", ("C2", "C2")),
              "connect12": ("connect", ("C2", "C1")),
              "ale": ("ale", ("C2", "C2"))}


def parse_eq(eq):
    """(family, (lower space, upper space)) for an equation selector.

    The three legacy "connectN" names and "ale" keep their meaning. The general forms are
    "connect:<lower>/<upper>" -- Poisson tied across the interface, the case the space matrix sweeps --
    and "move:<lower>/<upper>" -- a Laplace-smoothed mesh in both domains with the interface POSITION
    coupled, i.e. the case where the interface geometry is itself an unknown and the two sides can only
    agree on a shape both of them can represent.
    """
    if eq in _LEGACY_EQ:
        return _LEGACY_EQ[eq]
    if ":" not in eq:
        raise ValueError("unknown equation: " + str(eq))
    fam, spaces = eq.split(":", 1)
    if fam not in ("connect", "move"):
        raise ValueError("unknown equation family: " + str(fam))
    lo, up = spaces.split("/", 1)
    return (fam, (lo, up))
# (level applied uniformly to BOTH domains, level the asymmetric criterion drives in "lower", criterion)
LEVELS = [(0, 0, "level"), (1, 2, "level"), (1, 2, "size"), (1, 2, "callback"),
          (1, 2, "interface"), (1, 2, "estimator"), (0, 3, "level")]

_LOWER_BND = ["left", "right", "bottom"]
_UPPER_BND = ["left", "right", "top"]
_LOWER_BND_3D = ["left", "right", "front", "back", "bottom"]
_UPPER_BND_3D = ["left", "right", "front", "back", "top"]


class MixedTwoDomainMesh(MeshTemplate):
    """The unit square as quads for y<0.5 and triangles for y>0.5, sharing the boundary "interface".

    The point of this kind is that the two sides of the coupled interface are DIFFERENT element families.
    They still meet in matching line facets, so conformity is well-defined -- but a quad refining its
    bottom edge and a triangle refining its top edge have to end up with the same two sub-segments, which
    is exactly the property the opposite-element matcher relies on and nothing else in the suite checks.
    """

    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        half = N // 2
        lower = self.new_domain("lower")
        upper = self.new_domain("upper")
        idx = {}

        def node(i, j):
            if (i, j) not in idx:
                idx[(i, j)] = self.add_node_unique(1.0 * i / N, 1.0 * j / N)
            return idx[(i, j)]

        for i in range(N):
            for j in range(N):
                a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
                if j < half:
                    lower.add_quad_2d_C1(a, b, c, d)
                else:
                    upper.add_tri_2d_C1(a, b, d)
                    upper.add_tri_2d_C1(a, d, c)
        for j in range(half):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for j in range(half, N):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("bottom", [node(i, 0), node(i + 1, 0)])
            self.add_facet_to_boundary("top", [node(i, N), node(i + 1, N)])
            self.add_facet_to_boundary("interface", [node(i, half), node(i + 1, half)])


class CurvedTwoDomainMesh(MeshTemplate):
    """The unit square split by a circular ARC rather than by a straight line at y=0.5.

    Every other kind here has a FLAT interface, and on a flat interface a C1 domain and a C2 one place a
    refinement node in exactly the same spot: the C2 side reuses the father's midside node, which sits at
    the chord midpoint, and the C1 side creates one there. The two sides therefore agree by accident, and
    no amount of mixing spaces can show it.

    Here the interface facets carry a circle_arc curved entity, so the C2 side's midside nodes sit ON the
    arc, off the chord -- the two domains' interface GEOMETRY differs at level 0 even though their
    vertices coincide. What refinement then does with that is the question every position-keyed step of
    the coupling machinery depends on.
    """

    def __init__(self, N=4, sagitta=0.15, split_in_tris=False):
        super().__init__()
        self.N, self.sagitta, self.split_in_tris = N, sagitta, split_in_tris
        h = sagitta
        self.R = 0.5 * h + 1.0 / (8.0 * h)      # circle through (0,0.5) and (1,0.5) with sagitta h
        self.cy = 0.5 + h - self.R

    def interface_y(self, x):
        return self.cy + math.sqrt(max(self.R ** 2 - (x - 0.5) ** 2, 0.0))

    def define_geometry(self):
        N, half = self.N, self.N // 2
        lower, upper = self.new_domain("lower"), self.new_domain("upper")
        idx = {}

        def node(i, j):
            if (i, j) not in idx:
                x = 1.0 * i / N
                yc = self.interface_y(x)
                y = yc * j / half if j <= half else yc + (1.0 - yc) * (j - half) / (N - half)
                idx[(i, j)] = self.add_node_unique(x, y)
            return idx[(i, j)]

        for i in range(N):
            for j in range(N):
                a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
                dom = lower if j < half else upper
                if self.split_in_tris:
                    dom.add_tri_2d_C1(a, b, d)
                    dom.add_tri_2d_C1(a, d, c)
                else:
                    dom.add_quad_2d_C1(a, b, c, d)
        arc = self.create_curved_entity("circle_arc", node(0, half), node(N, half),
                                        center=[0.5, self.cy])
        for j in range(N):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("bottom", [node(i, 0), node(i + 1, 0)])
            self.add_facet_to_boundary("top", [node(i, N), node(i + 1, N)])
            self.add_facet_to_boundary("interface", [node(i, half), node(i + 1, half)],
                                       curved_entity=arc)


class SplitBoxMesh3D(MeshTemplate):
    """The unit cube split at z=0.5 into "lower" and "upper", sharing the boundary "interface".

    The 3d counterpart of the flat two-domain kinds. Everything about the coupling machinery that is
    stated on FACETS is dimension-dependent in one place only -- a facet here is a quadrilateral (hex) or
    a triangle (tet) instead of a line -- and nothing in the matrix exercised that until now. The cell
    filling is the same Kuhn split box_mesh_3d.py uses, so the two files agree about what a tet cell is.
    """

    def __init__(self, N=2, family="hex"):
        super().__init__()
        self.N, self.family = N, family

    def define_geometry(self):
        from box_mesh_3d import _KUHN, _CUBE_FACES_CYCLIC
        N, h = self.N, 1.0 / self.N
        half = N // 2
        lower, upper = self.new_domain("lower"), self.new_domain("upper")
        nodes, coords, faces = {}, {}, []

        def node(x, y, z):
            key = (round(x, 9), round(y, 9), round(z, 9))
            if key not in nodes:
                nodes[key] = self.add_node_unique(x, y, z)
                coords[nodes[key]] = key
            return nodes[key]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    x0, y0, z0 = i * h, j * h, k * h
                    c = [node(x0 + bx * h, y0 + by * h, z0 + bz * h)
                         for bz in (0, 1) for by in (0, 1) for bx in (0, 1)]
                    dom = lower if k < half else upper
                    if self.family == "hex":
                        dom.add_brick_3d_C1(*c)
                        faces.extend([tuple(c[q] for q in f) for f in _CUBE_FACES_CYCLIC])
                    else:
                        for t in _KUHN:
                            nn = [c[q] for q in t]
                            dom.add_tetra_3d_C1(*nn)
                            faces.extend([(nn[1], nn[2], nn[3]), (nn[0], nn[2], nn[3]),
                                          (nn[0], nn[1], nn[3]), (nn[0], nn[1], nn[2])])

        walls = {"left": (0, 0.0), "right": (0, 1.0), "front": (1, 0.0), "back": (1, 1.0),
                 "bottom": (2, 0.0), "top": (2, 1.0), "interface": (2, 0.5)}
        for f in faces:
            for name, (axis, val) in walls.items():
                if all(abs(coords[n][axis] - val) < 1e-9 for n in f):
                    self.add_facet_to_boundary(name, list(f))
                    break


def make_mesh(kind, N=4):
    if kind == "box_hex":
        return SplitBoxMesh3D(N=max(N // 2, 2), family="hex")
    if kind == "box_tet":
        return SplitBoxMesh3D(N=max(N // 2, 2), family="tet")
    if kind == "curved":
        return CurvedTwoDomainMesh(N=N)
    if kind == "curved_tri":
        return CurvedTwoDomainMesh(N=N, split_in_tris=True)
    if kind == "mixed":
        return MixedTwoDomainMesh(N=N)
    split = False if kind == "quad" else kind.split("_", 1)[1]
    return RectangularQuadMesh(name=lambda x, y: "lower" if y < 0.5 else "upper",
                               size=1, N=N, lower_left=[0, 0], split_in_tris=split,
                               boundary_names={"lower_upper": "interface"})


def centroid_x(e):
    return e.get_Eulerian_midpoint()[0]


def max_vertex_level_jump(problem, domain, boundary):
    """How far an element touching `boundary` at a VERTEX ONLY lags behind the interface at that vertex.

    Everything that enforces conformity is stated on FACETS, and an element can share a single vertex
    with a coupled interface while carrying no facet on it -- it is not a boundary element, it puts no
    key into either side's facet set, and so nothing in the conformity machinery can see it. When the
    OPPOSITE domain is what forces the refinement, its facet-carrying neighbours follow and it does
    not, and the level jump across that shared vertex grows without bound. In the tri kinds here every
    interface cell has exactly one such triangle (the second of the two the cell splits into), so this
    returns 0 for "quad" and the full driven level for the others unless the closure in
    enforce_interface_conformity() is doing its job. 2:1 balance means <= 1.

    Two-dimensional and rank-local, both on purpose. An element is taken to carry a facet when two of
    its vertex nodes lie on the boundary, which is exact for the straight interface these cases use but
    not in 3d (where two boundary vertices mean a shared EDGE, not a shared face). And it can only see
    the elements this process holds, so under MPI it may UNDER-report -- safe for an upper bound, and
    the reason the distributed harness does not compare it against the serial value.
    """
    m = problem.get_mesh(domain)
    bind = m.get_boundary_names().index(boundary)
    finest, corners = {}, []

    def key(n):
        return tuple(round(n.x(d) * 1e8) for d in range(n.ndim()))

    for e in m.elements():
        bn = e.boundary_vertex_nodes(bind)
        lvl = e.refinement_level()
        if len(bn) >= 2:
            for n in bn:
                k = key(n)
                finest[k] = max(finest.get(k, 0), lvl)
        elif len(bn) == 1:
            corners.append((key(bn[0]), lvl))
    return max([finest[k] - lvl for k, lvl in corners if k in finest] + [0])


def interface_facet_vertices(problem, interface):
    """The positions of the vertex nodes of the FACETS of `interface` (e.g. "lower/interface").

    Facet vertices, not "every vertex node flagged as being on the boundary": those are not the same set,
    and the difference is not academic. In 3d tets, refining an element that touches the interface only
    along an EDGE creates a node on that edge -- on the interface plane, and marked as being on the
    interface -- while carrying no interface face. See interface_stray_boundary_vertices.

    This is the set connect_interface_elements_by_kdtree actually looks up, and the set
    collect_interface_side keys the conformity test on.
    """
    m = problem.get_mesh(interface)
    # Keyed on the position, not on id(node): nanobind hands out a fresh Python wrapper per call and a
    # freed one's id is reused, so an id-keyed dict silently collapses to a handful of entries.
    seen = set()
    for e in m.elements():
        for i in range(e.nvertex_node()):
            n = e.vertex_node_pt(i)
            seen.add(tuple(n.x(d) for d in range(n.ndim())))
    return sorted(seen)


def interface_stray_boundary_vertices(problem, domain, boundary, interface):
    """Vertex nodes of `domain` marked as lying on `boundary` that are not a vertex of any facet there.

    Zero for every 2d kind. Non-zero in 3d tets, where a refinement of an element touching the interface
    along an edge puts a new node on that edge. Reported separately because it is a boundary-MEMBERSHIP
    question, not a conformity one: check_interface_conformity is stated on facets and is silent about it.
    """
    m = problem.get_mesh(domain)
    bind = m.get_boundary_names().index(boundary)
    marked = set()
    for e in m.elements():
        for n in e.boundary_vertex_nodes(bind):
            marked.add(tuple(n.x(d) for d in range(n.ndim())))
    return len(marked - set(interface_facet_vertices(problem, interface)))


def interface_vertex_gap(problem, sides, tol=1e-8):
    """How far the two sides' interface facet vertices are from actually coinciding.

    Everything that pairs the two sides up -- refinement_coupling's node_key, the KD-tree in
    connect_interface_elements_by_kdtree, the permutation search in analyze_opposite_orientation -- is a
    quantised EULERIAN position, so all of it rests on this one quantity being zero. Measuring it
    directly separates "the two sides refined different facets" (a topological mismatch, which
    check_interface_conformity already reports) from "they refined the same facet but put the new vertex
    in a different place" (a geometric divergence, which nothing today can see until the matcher throws).

    Returns (max distance from a vertex to the nearest one on the other side, number of vertices with no
    counterpart within tol). Rank-local under MPI, like max_vertex_level_jump, and for the same reason.
    """
    import numpy as np
    A = np.asarray(interface_facet_vertices(problem, sides[0]), dtype=float)
    B = np.asarray(interface_facet_vertices(problem, sides[1]), dtype=float)
    if not len(A) or not len(B):
        return (0.0, 0)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    nearest = np.concatenate([d.min(axis=1), d.min(axis=0)])
    return (float(nearest.max()), int((nearest > tol).sum()))


def interface_node_pairing(problem, interface):
    """Per interface element: how many of its nodes have no counterpart on the opposite side.

    A C2 element facing a C1 one legitimately has unmatched midside nodes -- opposite_node_pt() returns
    None for them by design (src/elements.hpp, "a lower-to-higher order mismatch"). What this reports is
    whether that count is the one the two spaces predict, or whether nodes are going unmatched for some
    other reason. Returns (nodes per element on this side, nodes per element on the other side, number
    of unmatched nodes, number of elements with no opposite element at all).
    """
    m = problem.get_mesh(interface)
    nself, nopp, unmatched, unpaired = set(), set(), 0, 0
    for e in m.elements():
        opp = e.get_opposite_interface_element()
        if opp is None:
            unpaired += 1
            continue
        nself.add(e.nnode())
        nopp.add(opp.nnode())
        # DO NOT ask a QUADRILATERAL face element for its opposite nodes when the two sides carry a
        # different number of them. Quad2dFaceOrientation::analyze (src/elements_interface.cpp) ends with
        #     node_index = node_index_map(orientation, nnode_1d);
        # using THIS element's nnode_1d, so a C2 face (9 nodes) against a C1 one (4) is left holding
        # indices up to 8 into a 4-node element. The 1d line elements handle the same mismatch explicitly
        # (elements_concrete.hpp, "opposite is C1: no midside node" -> index -1); the quad face does not.
        # The entries are latent -- the case itself runs and converges -- until something dereferences
        # them, and opposite_node_pt() then segfaults inside the nanobind caster.
        if e.nnode() in (4, 9) and e.nnode() != opp.nnode():
            continue
        for i in range(e.nnode()):
            if e.opposite_node_pt(i) is None:
                unmatched += 1
    return (sorted(nself), sorted(nopp), unmatched, unpaired)


class TwoDomainProblem(Problem):
    def __init__(self, kind="quad", eq="connect1", levels=(1, 2, "level"), N=4):
        super().__init__()
        self.kind, self.eq, self.levels, self.N = kind, eq, tuple(levels), N

    def define_problem(self):
        self += make_mesh(self.kind, self.N)
        x = var("coordinate_x")

        family, spaces = parse_eq(self.eq)
        # In 3d the box has four side walls and the driven direction is z, not y.
        is3d = self.kind in BOX3D_KINDS
        lower_bnd = _LOWER_BND_3D if is3d else _LOWER_BND
        upper_bnd = _UPPER_BND_3D if is3d else _UPPER_BND

        if family == "connect":
            # Poisson in both domains, u tied across the interface by Lagrange multipliers. u == y is the
            # exact solution of every variant, and it is representable in every discretisation here, so
            # the field is pinned down independently of the mesh -- see solve_case's "maxuerr".
            #
            # connect12 gives the two domains DIFFERENT spaces (C2 below, C1 above). That is the case
            # where the coupling space itself has to be negotiated
            # (get_interface_field_connection_space), and where a hanging node on one side of the
            # interface meets a differently-interpolated node on the other.
            self += PoissonEquation(name="u", source=0, space=spaces[0]) @ "lower"
            self += PoissonEquation(name="u", source=0, space=spaces[1]) @ "upper"
            self += DirichletBC(u=0) @ "lower/bottom"
            self += DirichletBC(u=1) @ "upper/top"
            self += ConnectFieldsAtInterface("u") @ "lower/interface"
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ "lower"
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ "upper"
        elif family == "move":
            # A moving mesh in both domains with the interface SHAPE prescribed, on BOTH sides, as a
            # curve. No ConnectMeshAtInterface: the point here is not to couple the positions but to
            # reproduce the geometry a free surface produces, which is what makes the two coordinate
            # spaces disagree:
            #
            #   * the lower domain, if C2, has its interface midside nodes pinned ON the curve, off the
            #     chord; a refinement promotes one of them to a VERTEX, still on the curve;
            #   * a C1 domain has no such node and creates its new vertex at the chord midpoint.
            #
            # The two sides' vertices then differ, and they differ precisely in the window in which the
            # opposite-element matcher runs: Problem.actions_after_adapt pairs the interfaces up BEFORE
            # it calls reapply_boundary_conditions, so the C1 side's new vertex has not yet been snapped
            # onto the curve.
            #
            # Note what does NOT produce this. Driving the mesh from an outer boundary leaves the
            # interface flat to machine precision (measured -- and so does the pre-existing "ale" case),
            # and prescribing the shape on one side while coupling the positions with
            # ConnectMeshAtInterface is over-constrained: the multiplier is then redundant on the pinned
            # side and the free side never follows at all, which fails for EQUAL spaces too and so
            # measures nothing about spaces.
            if is3d:
                raise ValueError("the 'move' family is 2d only")
            for dom, bnds, sp in (("lower", lower_bnd, spaces[0]), ("upper", upper_bnd, spaces[1])):
                eqs = ElementSpace(sp)
                eqs += PoissonEquation(name="u", source=0, space=sp)
                eqs += LaplaceSmoothedMesh()
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ bnds
                eqs += DirichletBC(mesh_x=True, mesh_y=0.5 + 0.1 * sin(2 * pi * x)) @ "interface"
                eqs += IntegralObservables(area=1, intu2=var("u") ** 2)
                self += eqs @ dom
            self += DirichletBC(u=0) @ "lower/bottom"
            self += DirichletBC(u=1) @ "upper/top"
            self += ConnectFieldsAtInterface("u") @ "lower/interface"
        elif family == "ale":
            # Stokes on a Laplace-smoothed mesh in both domains, with the mesh POSITIONS coupled across the
            # interface (ConnectMeshAtInterface) as well as the velocity. Two things this adds over the
            # Poisson cases: the interface geometry is itself an unknown, so the facet positions the
            # conformity machinery keys on are solution-dependent; and the position dofs carry their own
            # hanging-node structure.
            for dom, bnds in (("lower", lower_bnd), ("upper", upper_bnd)):
                eqs = StokesEquations(mode="TH", dynamic_viscosity=1, bulkforce=vector(0, -1))
                eqs += LaplaceSmoothedMesh()
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ bnds
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ bnds
                eqs += IntegralObservables(area=1, intu2=dot(var("velocity"), var("velocity")))
                self += eqs @ dom
            self += DirichletBC(pressure=0) @ "lower/bottom"
            self += ConnectFieldsAtInterface(["velocity_x", "velocity_y"]) @ "lower/interface"
            self += ConnectMeshAtInterface() @ "lower/interface"
        else:
            raise ValueError("unknown equation: " + str(self.eq))

        self._add_refinement_criterion()

    def _add_refinement_criterion(self):
        """The asymmetric part: state a refinement requirement for "lower" that "upper" cannot see.

        Which criterion is used matters, and not for the reason one might expect. What decides whether a
        criterion reaches the opposite domain is not the criterion itself but WHERE it is stated:

          * on the bulk ("level", "size", "callback", "estimator") -- reads only the lower domain's own
            elements. The upper domain is told nothing, so the two sides diverge. These are the cases
            that fail without enforce_interface_conformity(); verified by running the suite under
            PYOOMPH_DISABLE_INTERFACE_CONFORMITY=1.
          * on the interface ("interface") -- InterfaceMesh._override_bulk_errors_where_necessary already
            pushes the error onto BOTH adjacent bulk elements, so this one is symmetric by construction
            and passes with or without the fix. It is kept because that symmetry is a property that must
            not silently regress, not because it exercises the new code.
        """
        lo, hi = self.levels[0], self.levels[1]
        crit = self.levels[2] if len(self.levels) > 2 else "level"
        if lo:
            # Applied to BOTH domains: the conforming starting point the asymmetry is measured against.
            self += RefineToLevel(lo) @ "lower"
            self += RefineToLevel(lo) @ "upper"
        if not hi or hi == lo:
            return
        h = 1.0 / self.N
        if crit == "level":
            self += RefineToLevel(hi) @ "lower"
        elif crit == "size":
            # Cartesian element SIZE, i.e. an AREA in 2d -- and the base area depends on how the family
            # fills a cell of the h x h grid. Deriving it from h alone (as if everything were a quad)
            # made this criterion a silent no-op on triangles: their elements were already below the
            # threshold at level lo, so nothing was refined and the case tested nothing at all.
            base = {"quad": h * h, "tri_left": 0.5 * h * h, "tri_crossed": 0.25 * h * h,
                    "mixed": h * h, "curved": h * h, "curved_tri": 0.5 * h * h,
                    "box_hex": h ** 3, "box_tet": h ** 3 / 6}[self.kind]  # "mixed" is quads BELOW, and the criterion is on "lower"
            # Just under the area at level hi-1: those elements refine, level-hi ones do not.
            self += RefineMaxElementSize(0.99 * base / 4 ** (hi - 1)) @ "lower"
        elif crit == "callback":
            # Position-dependent, so the refinement level jumps ALONG the interface as well as across it:
            # the upper domain then has to follow a pattern, not just a uniform level.
            self += RefineAccordingToElement(lambda e: hi if centroid_x(e) < 0.5 else lo) @ "lower"
        elif crit == "interface":
            self += RefineToLevel(hi) @ "lower/interface"
        elif crit == "estimator":
            # The realistic case: no explicit level anywhere, just a Z2 error estimator on a field that is
            # only sharp in the lower domain. Nothing states a level, so the two domains genuinely
            # disagree about how fine the interface should be.
            self += PoissonEquation(name="w", source=1.0 / (0.01 + var("coordinate_x") ** 2),
                                    space=parse_eq(self.eq)[1][0]) @ "lower"
            self += DirichletBC(w=0) @ ["lower/bottom", "lower/left", "lower/right"]
            self += SpatialErrorEstimator(w=1) @ "lower"
        else:
            raise ValueError("unknown refinement criterion: " + str(crit))


def solve_case(kind, eq, levels, N=4, outdir=None):
    """Solve one case and return the partition-independent measurements.

    `kind` also accepts the four-domain layouts (see FOUR_DOMAIN_KINDS at the bottom of this file), for
    which `eq` and `levels` are ignored -- that keeps the MPI harness, which drives everything through
    solve_case/case_id, working for both topologies without needing to know about either.
    """
    import numpy as np
    if kind.split(":")[0] in FOUR_DOMAIN_KINDS:
        return solve_four_domain_case(kind, outdir=outdir)
    prob = TwoDomainProblem(kind=kind, eq=eq, levels=tuple(levels), N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.max_refinement_level = max(levels[0], levels[1]) + 1
        p.initialise()
        # Repairs done while setting the mesh up (the uneven part of the initial uniform refinement is
        # deliberately applied after distribute, so it genuinely has to be repaired). Only what happens
        # from here on is a statement about the ADAPT path.
        repairs_at_init = p._interface_conformity_repairs
        # The refinement is driven by explicit criteria, not by a converging error estimate, so a couple
        # of adapt steps is enough for all of them to have taken effect.
        # The gap is measured after EVERY adapt, not only at the end. A divergence introduced by one
        # refinement can be washed out by the next solve (ConnectMeshAtInterface re-equalises the
        # vertices), so a final-state-only reading would report success on exactly the runs where the
        # matcher is about to be handed two sides that do not line up.
        sides = ("lower/interface", "upper/interface")
        gap, gap_unmatched = interface_vertex_gap(p, sides)
        for _ in range(3):
            p.solve(spatial_adapt=1)
            g, gu = interface_vertex_gap(p, sides)
            gap, gap_unmatched = max(gap, g), max(gap_unmatched, gu)
        repairs_during_adapt = p._interface_conformity_repairs - repairs_at_init
        nself, nopp, unmatched_nodes, unpaired_elems = interface_node_pairing(p, "lower/interface")
        strays = sum(interface_stray_boundary_vertices(p, d, "interface", d + "/interface")
                     for d in ("lower", "upper"))

        # The Newton history of that last solve is useless as a Jacobian oracle: the field was already
        # converged from the previous pass, so it starts at machine zero and one step "reduces" nothing.
        # Wipe the coupled field on the FINAL mesh and solve it once more, from a residual of O(1).
        # Done nodally, so it stays correct per rank on a distributed mesh. Not done for "ale", whose
        # dofs include the nodal POSITIONS -- zeroing those collapses the mesh rather than resetting it.
        if parse_eq(eq)[0] == "connect":
            for dom in ("lower", "upper"):
                m = p.get_mesh(dom)
                iu = m.get_nodal_field_indices()["u"]
                for n in m.nodes():
                    if not n.is_pinned(iu):
                        n.set_value(iu, 0.0)
            p.solve()
        conv = p.get_last_residual_convergence()
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
            # The invariant itself, stated rather than inferred. Zero means both sides of the interface
            # carry identical boundary facets, which is precisely what the opposite-element matcher needs.
            # Collective under MPI, so every rank calls it and every rank gets the same number.
            "nonconforming": int(p.check_interface_conformity(throw_on_mismatch=False, when="end of case")),
            # How many elements the post-adapt repair had to refine AFTER the fact. Zero is the good
            # case: it means the two sides agreed before either acted, rather than one of them being
            # merged away and then refined back -- a round trip that is correct but re-interpolates the
            # sons from the merged father and loses the fine-scale solution.
            "repairs_during_adapt": int(repairs_during_adapt),
            # The 2:1 balance at the interface VERTICES, which facet conformity says nothing about.
            # See max_vertex_level_jump: not compared between serial and distributed runs, since it is
            # rank-local by construction.
            "vertex_jump": max(max_vertex_level_jump(p, dom, "interface") for dom in ("lower", "upper")),
            # The invariant the position-keyed matching actually rests on: the two sides' interface
            # VERTICES coincide. Rank-local under MPI, like vertex_jump.
            "max_vertex_gap": float(gap),
            "gap_unmatched_vertices": int(gap_unmatched),
            # What the differing element spaces cost at the pairing: nodes with no counterpart should be
            # exactly the ones the two spaces cannot both represent.
            "iface_nnode": [nself, nopp],
            "unmatched_nodes": int(unmatched_nodes),
            "unpaired_elems": int(unpaired_elems),
            # Vertex nodes marked as being on the interface that carry no facet there -- a boundary
            # MEMBERSHIP question the facet-stated conformity check cannot see. Zero in 2d.
            "stray_boundary_vertices": int(strays),
        }
        for dom in ("lower", "upper"):
            for name, val in p.get_mesh(dom).evaluate_all_observables().items():
                res["obs_" + dom + "_" + name] = float(val)
        if parse_eq(eq)[0] == "connect":
            # u == y exactly. A torn interface shows up here long before it shows up in an integral: a
            # mis-paired opposite element leaves the two sides connected to the WRONG neighbour, which
            # this catches even when the residual is happily converged.
            #
            # NB the tolerance. The Lagrange-multiplier formulation of ConnectFieldsAtInterface does not
            # reproduce a linear field to machine zero even on a single UNREFINED, unadapted mesh (~3e-10
            # on 81 dofs), so this bound is about the coupling, not about the refinement. A torn interface
            # misses it by orders of magnitude.
            worst = 0.0
            for dom in ("lower", "upper"):
                m = p.get_mesh(dom)
                # Look the index up by NAME. The "estimator" criterion adds a second field to the lower
                # domain, so index 0 is not necessarily u -- reading it positionally would silently
                # measure the wrong field (and did, until this was keyed on the name).
                iu = m.get_nodal_field_indices()["u"]
                for n in m.nodes():
                    worst = max(worst, abs(n.value(iu) - n.x(n.ndim() - 1)))
            res["maxuerr"] = float(worst)
        return res


def case_id(kind, eq, levels):
    if kind.split(":")[0] in FOUR_DOMAIN_KINDS:
        return kind.replace(":", "_").replace(",", "-")
    crit = levels[2] if len(levels) > 2 else "level"
    # eq may now be "connect:C2/C1TB"; the id is used as a directory name, so flatten the separators.
    return "%s-%s-%d%d-%s" % (eq.replace(":", "_").replace("/", "-"), kind, levels[0], levels[1], crit)


# --- Four domains meeting at a cross point ----------------------------------------------------------
#
#         A | B
#         --+--
#         C | D
#
# A different topology from everything above, and the two things it adds are worth stating.
#
# The coupling graph is a CYCLE (A-B-D-C-A), not a chain, so the reconciliation has to close a loop
# rather than propagate along one. D shares no interface with A at all -- they touch only at the cross
# point -- so a refinement demand raised in A can only reach D by travelling around the cycle.
#
# And the cross point itself is four DISTINCT nodes, one per domain, tied pairwise by four Lagrange
# multipliers (A=B, A=C, B=D, C=D). Only three of those four constraints are independent; the fourth
# follows. That is a genuine over-constraint at a single point, and it is
# ConnectFieldsAtInterface.pin_redundant_lagrange_multipliers that has to notice.
#
# Exact solution u = y everywhere, so a mis-paired interface shows up directly as a nodal error.

FOUR_DOMAIN_KINDS = ["four_corner", "four_diagonal", "four_away"]
_FOUR_DOMS = ["A", "B", "C", "D"]
# RectangularQuadMesh names an auto-generated internal interface after the two domains it separates.
_FOUR_IFACES = [("A", "A_B"), ("A", "A_C"), ("B", "B_D"), ("C", "C_D")]


def _four_domain_of(x, y):
    if y >= 0.5:
        return "A" if x < 0.5 else "B"
    return "C" if x < 0.5 else "D"


def parse_four_kind(kind):
    """("four_corner", {A: space, ...}) for a four-domain selector.

    "four_corner" keeps its meaning (C1 everywhere). The general form appends the four domains' spaces
    in A,B,C,D order: "four_corner:C1,C2,C2TB,C1". A junction is where the space matrix has to be
    stated per DOMAIN rather than per side, since four of them meet at one point.
    """
    if ":" not in kind:
        return kind, {d: "C1" for d in _FOUR_DOMS}
    base, spaces = kind.split(":", 1)
    sp = spaces.split(",")
    if len(sp) != 4:
        raise ValueError("a four-domain space selector needs four spaces (A,B,C,D): " + kind)
    return base, dict(zip(_FOUR_DOMS, sp))


class FourDomainProblem(Problem):
    def __init__(self, kind="four_corner", N=4):
        super().__init__()
        self.kind, self.spaces = parse_four_kind(kind)
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=[self.N, self.N], size=[1, 1], name=_four_domain_of)
        for d in _FOUR_DOMS:
            self += PoissonEquation(name="u", source=0, space=self.spaces[d]) @ d
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ d
        self += DirichletBC(u=0) @ "C/bottom"
        self += DirichletBC(u=0) @ "D/bottom"
        self += DirichletBC(u=1) @ "A/top"
        self += DirichletBC(u=1) @ "B/top"
        for dom, nm in _FOUR_IFACES:
            self += ConnectFieldsAtInterface("u") @ (dom + "/" + nm)

        if self.kind == "four_corner":
            # Refinement concentrated ON the cross point, in A only: the level jump sits exactly where
            # all four domains meet, and both of A's interfaces have to carry it into B and C, and then
            # around to D.
            def lev(e):
                x, y = e.get_Eulerian_midpoint()[0], e.get_Eulerian_midpoint()[1]
                d = max(abs(x - 0.5), abs(y - 0.5))
                return 3 if d < 0.2 else (1 if d < 0.35 else 0)
            self += RefineAccordingToElement(lev) @ "A"
        elif self.kind == "four_diagonal":
            # The DIAGONAL pair driven, to different levels: A and D are each pulled by two neighbours
            # that disagree with each other about how fine the interface should be.
            self += RefineAccordingToElement(lambda e: 3) @ "B"
            self += RefineAccordingToElement(lambda e: 2) @ "C"
        elif self.kind == "four_away":
            # Refinement in A but AWAY from every interface. Nothing may propagate: this is the case
            # that separates "the neighbours follow where they must" from "the neighbours follow
            # always", which the other two cannot distinguish.
            def lev(e):
                x, y = e.get_Eulerian_midpoint()[0], e.get_Eulerian_midpoint()[1]
                return 3 if (x < 0.2 and y > 0.8) else 0
            self += RefineAccordingToElement(lev) @ "A"
        else:
            raise ValueError("unknown four-domain kind: " + str(self.kind))


def solve_four_domain_case(kind, outdir=None):
    import numpy as np
    prob = FourDomainProblem(kind=kind)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.max_refinement_level = 3
        p.initialise()
        repairs_at_init = p._interface_conformity_repairs
        for _ in range(4):
            p.solve(spatial_adapt=1)
        # Wipe and re-solve on the final mesh, so the Newton history is a Jacobian oracle rather than
        # the tail of an already-converged solve (see solve_case).
        for d in _FOUR_DOMS:
            m = p.get_mesh(d)
            iu = m.get_nodal_field_indices()["u"]
            for n in m.nodes():
                if not n.is_pinned(iu):
                    n.set_value(iu, 0.0)
        p.solve()
        conv = p.get_last_residual_convergence()
        worst = 0.0
        levels = {}
        for d in _FOUR_DOMS:
            m = p.get_mesh(d)
            iu = m.get_nodal_field_indices()["u"]
            for n in m.nodes():
                worst = max(worst, abs(n.value(iu) - n.x(1)))
            levels[d] = sorted({e.refinement_level() for e in m.elements()})
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
            "nonconforming": int(p.check_interface_conformity(False, "four-domain")),
            "repairs_during_adapt": int(p._interface_conformity_repairs - repairs_at_init),
            "maxuerr": float(worst),
            "maxlevel": {d: max(levels[d]) for d in _FOUR_DOMS},
        }
        for d in _FOUR_DOMS:
            for name, val in p.get_mesh(d).evaluate_all_observables().items():
                res["obs_" + d + "_" + name] = float(val)
        return res
