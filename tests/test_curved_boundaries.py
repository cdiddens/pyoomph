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

import pytest

from pyoomph import *
from pyoomph import _pyoomph_core as _pyoomph
from pyoomph.equations.poisson import PoissonEquation
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
