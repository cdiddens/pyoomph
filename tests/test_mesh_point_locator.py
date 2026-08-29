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

#  Mesh point location and mesh-to-mesh transfer (dev_docs/mesh_point_locator.md).
#
#  Guards the campaign that replaced oomph::MeshAsGeomObject with pyoomph's own MeshPointLocator and
#  rebuilt the transfer paths on top of it. Nearly every defect that campaign found was SILENT - a
#  wrong element matched, a field quietly zeroed, a node placed at the antipode of where it belonged -
#  so the tests here are written to pin numbers down rather than to check that things merely run.
#
#  Three kinds of assertion recur, and it is worth knowing why each is used:
#
#   * A LINEAR field is reproduced exactly by an isoparametric FE space. Interpolating one and
#     comparing against its analytic value therefore measures pure transfer error, with no
#     discretisation error mixed in. This is what exposed the old path being ~1% wrong on curved
#     elements, and the locator's Lagrangian/Eulerian mix-up in the projection.
#   * SELF-LOCATION: every integration point of a mesh must be found in that same mesh. Anything less
#     than "0 unlocated" is a defect, whatever the values look like.
#   * The INTEGRAL of the field. An exact L2 projection conserves it exactly, because constants lie in
#     the FE space and Galerkin orthogonality then gives integral(u_new - u_old) = 0. That identity is
#     far sharper than the pointwise error and is what caught the projection solving in the wrong
#     coordinate space.

import math
import subprocess
import sys

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import IntegralObservables
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CircularMesh, CuboidBrickMesh
from pyoomph.meshes.interpolator import InternalInterpolator, ProjectionInternalInterpolator
from pyoomph.meshes.zeta import (AssignZetaCoordinatesByArclength,
                                 AssignZetaCoordinatesByEulerianCoordinate)
import pyoomph._pyoomph_core as _pyoomph


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _linear2(x, y):
    return 0.4 + 1.7 * x - 1.1 * y


class _Blob(GmshTemplate):
    """Quarter disc: two straight sides and one curved arc, so both element kinds occur."""

    def __init__(self, res=0.09):
        super().__init__()
        self.res = res

    def define_geometry(self):
        self.default_resolution = self.res
        self.mesh_mode = "tri"
        p00, p10, p01 = self.point(0, 0), self.point(1, 0), self.point(0, 1)
        self.line(p00, p10, name="bottom")
        self.circle_arc(p10, p01, center=p00, name="interface")
        self.line(p01, p00, name="axis")
        self.plane_surface("bottom", "interface", "axis", name="domain")


class _SurfField(InterfaceEquations):
    """An interface-ONLY dof, which travels through inter_field_map rather than the bulk path."""

    def __init__(self, name="c", space="C2"):
        super().__init__()
        self.fname, self.space = name, space

    def define_fields(self):
        self.define_scalar_field(self.fname, self.space)

    def define_residuals(self):
        u, v = var_and_test(self.fname)
        self.add_residual(weak(partial_t(u), v))


def _stamp_bulk(mesh, f, ndim=2):
    for n in mesh.nodes():
        n.set_value(0, f(*[n.x(i) for i in range(ndim)]))


def _bulk_error(mesh, f, ndim=2):
    return max(abs(n.value(0) - f(*[n.x(i) for i in range(ndim)])) for n in mesh.nodes())


# ----------------------------------------------------------------------------------------------
# 1. zeta validity guards (phase 0)
# ----------------------------------------------------------------------------------------------

class _CircleProb(Problem):
    def __init__(self, zeta_eqs):
        super().__init__()
        self.zeta_eqs = zeta_eqs

    def define_problem(self):
        self.add_mesh(CircularMesh(radius=1.0))
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "circumference"
        eqs += self.zeta_eqs @ "circumference"
        self.add_equations(eqs @ "domain")


class _RectProb(Problem):
    def __init__(self, zeta_eqs):
        super().__init__()
        self.zeta_eqs = zeta_eqs

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=6))
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
        eqs += self.zeta_eqs @ "top"
        self.add_equations(eqs @ "domain")


def test_eulerian_zeta_rejects_a_fold_back():
    # A circle parameterised by x folds back on itself: two elements claim the same zeta, so
    # locate_zeta can invert either. Detected as a TILING failure - the element intervals cover more
    # than they span - which is robust where "one element looks too wide" false-positives on a coarse
    # interface.
    with pytest.raises(RuntimeError, match="not invertible"):
        with _CircleProb(AssignZetaCoordinatesByEulerianCoordinate("x")) as p:
            p.quiet()
            p.solve()


def test_open_boundaries_still_accept_both_assigners():
    for eqs in (AssignZetaCoordinatesByArclength(sort_along_axis="x+"),
                AssignZetaCoordinatesByEulerianCoordinate("x")):
        with _RectProb(eqs) as p:
            p.quiet()
            p.solve()   # must not raise


def test_closed_loop_gets_a_periodic_zeta():
    # A circle has no single-valued zeta. It is parameterised by arclength and declared periodic
    # instead, so the seam element runs from z_last to z_first + period rather than backwards across
    # the whole range.
    with _CircleProb(AssignZetaCoordinatesByArclength(sort_along_axis="x+")) as p:
        p.quiet()
        p.solve()
        m = p.get_mesh("domain")
        assert m.get_boundary_zeta_period(m.get_boundary_index("circumference")) == pytest.approx(1.0)


def test_closed_loop_zeta_is_the_angle():
    # The seam is a CONTINUOUS geometric anchor (the loop's intersection with a ray from its
    # centroid), not a node. A node-quantised seam differs between two discretisations by up to one
    # element, which is exactly the failure a sign error in the ray/edge intersection produced.
    with _CircleProb(AssignZetaCoordinatesByArclength(sort_along_axis="x+")) as p:
        p.quiet()
        p.solve()
        m, im = p.get_mesh("domain"), p.get_mesh("domain/circumference")
        bind = m.get_boundary_index("circumference")
        for n in im.nodes():
            theta = math.atan2(n.x(1), n.x(0)) % (2 * math.pi)
            assert n.get_coordinates_on_boundary(bind)[0] * 2 * math.pi == pytest.approx(theta, abs=1e-9)


# ----------------------------------------------------------------------------------------------
# 2. element geometry classification (phase 1)
# ----------------------------------------------------------------------------------------------

def _self_locate(problem, domain="domain"):
    """Self-location: every integration point of a mesh must be found in that same mesh."""
    m = problem.get_mesh(domain)
    m.prepare_interpolation()
    m.prepare_zeta_interpolation(m)
    return m


class _PoissonOn(Problem):
    def __init__(self, mesh, bc):
        super().__init__()
        self.m, self.bc = mesh, bc

    def define_problem(self):
        self.add_mesh(self.m)
        self.add_equations((PoissonEquation(source=1) + DirichletBC(u=0) @ self.bc) @ "domain")


@pytest.mark.parametrize("mesh,bc", [
    (RectangularQuadMesh(N=6, split_in_tris="crossed"), "left"),   # triangles
    (RectangularQuadMesh(N=6), "left"),                            # quads
    (CuboidBrickMesh(N=3), "left"),                                # hexes
])
def test_self_location_finds_every_integration_point(mesh, bc, capfd):
    _pyoomph.Mesh.set_report_interpolation_timing(True)
    try:
        with _PoissonOn(mesh, bc) as p:
            p.quiet()
            p.solve()
            _self_locate(p)
        out = capfd.readouterr().out
    finally:
        _pyoomph.Mesh.set_report_interpolation_timing(False)
    assert "0 unlocated" in out, out
    # A straight-sided mesh must invert exactly, with no element left to Newton: that is the whole
    # point of the affine classification, and a regression there is invisible except as a slowdown.
    assert " newton" not in out, out


def test_curved_elements_are_not_classified_affine(capfd):
    # The converse of the test above: a genuinely curved element MUST be rejected by the straightness
    # check, or it would be inverted affinely and give a wrong answer rather than a slow one.
    _pyoomph.Mesh.set_report_interpolation_timing(True)
    try:
        with _PoissonOn(CircularMesh(radius=1.0), "circumference") as p:
            p.quiet()
            p.solve()
            _self_locate(p)
        out = capfd.readouterr().out
    finally:
        _pyoomph.Mesh.set_report_interpolation_timing(False)
    assert "0 unlocated" in out, out
    assert " newton" in out, out


# ----------------------------------------------------------------------------------------------
# 3. location accuracy (phase 1)
# ----------------------------------------------------------------------------------------------

def test_linear_field_is_reproduced_exactly_on_a_curved_mesh():
    # oomph's locate_zeta stops at a residual of 1e-7 (elements.cc:1654), and on curved elements that
    # left interpolated values ~1e-2 wrong at arbitrary interior points. The locator polishes to
    # machine precision, so a field the space represents exactly must come back exactly.
    pts = [[r * math.cos(a), r * math.sin(a)]
           for r in (0.15, 0.45, 0.7, 0.88) for a in (0.11 + 2 * math.pi * k / 12 for k in range(12))]
    with _PoissonOn(CircularMesh(radius=1.0), "circumference") as p:
        p.quiet()
        p.solve()
        m = p.get_mesh("domain")
        m.prepare_interpolation()
        _stamp_bulk(m, _linear2)
        got = [nd.value(0) for nd in m.add_interpolated_nodes_at(pts, False)]
    worst = max(abs(g - _linear2(*q)) for g, q in zip(got, pts))
    assert worst < 1e-12, worst


def test_projection_locates_points_beside_a_surface():
    # Codimension-1: a 2d surface in 3d has no chart, so location is "nearest point on the surface".
    # Points pushed off the surface must be found with an offset equal to how far they were pushed,
    # and points pushed far must be rejected rather than matched to something implausible.
    class Cube(Problem):
        def define_problem(self):
            self.add_mesh(CuboidBrickMesh(N=4))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
            eqs += DirichletBC(u=0) @ "top"
            self.add_equations(eqs @ "domain")

    with Cube() as p:
        p.quiet()
        p.solve()
        im = p.get_mesh("domain/top")
        im.prepare_interpolation()
        base = [[sum(e.node_pt(i).x(d) for i in range(e.nnode())) / e.nnode() for d in range(3)]
                for e in im.elements()]
        normal = (0.0, 1.0, 0.0)   # the cube's "top" is the y = const face
        for off, expect in ((0.0, 0.0), (0.01, 0.01), (-0.03, 0.03)):
            pts = [[b[d] + off * normal[d] for d in range(3)] for b in base]
            rows = im.locate_points(pts, True)
            assert all(r[0] > 0.5 for r in rows), off
            assert max(r[1] for r in rows) == pytest.approx(expect, abs=1e-9)
        far = [[b[d] + 0.5 * normal[d] for d in range(3)] for b in base]
        assert not any(r[0] > 0.5 for r in im.locate_points(far, True)), "the offset guard let a far point through"


# ----------------------------------------------------------------------------------------------
# 4. interface-owned data across ADAPTATION (phase 3b - still open)
# ----------------------------------------------------------------------------------------------

def test_interface_d0_survives_adaptation():
    class IfaceD0(InterfaceEquations):
        def define_fields(self):
            self.define_scalar_field("idata", "D0")

        def define_residuals(self):
            u, v = var_and_test("idata")
            self.add_residual(weak(u - 7, v))

    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
            eqs += IfaceD0() @ "top" + SpatialErrorEstimator(u=1)
            self.add_equations(eqs @ "domain")

    with Prob() as p:
        p.quiet()
        p.max_refinement_level = 2
        p.solve()
        p.refine_uniformly()
        im = p.get_mesh("domain/top")
        vals = [d.value(j)
                for e in im.elements()
                for d in (e.internal_data_pt(i) for i in range(e.ninternal_data()))
                for j in range(d.nvalue())]
    assert vals and all(v == pytest.approx(7.0) for v in vals), vals


def test_interface_dl_reproduces_a_linear_field_across_adaptation():
    """A DL field can represent x exactly, so a correct transfer leaves it exactly right.

    This is the part a constant cannot test: the snapshot is fitted back per element by least
    squares, and a fit that collapsed to a constant - or picked up points from the wrong element -
    would still pass the D0 test above while failing here.
    """
    class IfaceDL(InterfaceEquations):
        def define_fields(self):
            self.define_scalar_field("idata", "DL")

        def define_residuals(self):
            u, v = var_and_test("idata")
            self.add_residual(weak(u - var("coordinate_x"), v))

    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
            eqs += (IfaceDL() + IntegralObservables(err=(var("idata") - var("coordinate_x"))**2)) @ "top"
            eqs += SpatialErrorEstimator(u=1)
            self.add_equations(eqs @ "domain")

    with Prob() as p:
        p.quiet()
        p.max_refinement_level = 2
        p.solve()
        before = float(p.get_mesh("domain/top").evaluate_all_observables()["err"])
        p.refine_uniformly()
        after = float(p.get_mesh("domain/top").evaluate_all_observables()["err"])
    # Not machine zero any more, and deliberately so: setup_initial_conditions used to build the DL
    # coefficients as a midpoint value plus a finite difference along each local coordinate, which is
    # exact for a linear field on an affine quad and left this solve with nothing to do. It was also
    # wrong in general - wrong basis, and a Lagrangian IC evaluated at the origin - so it is a
    # least-squares fit onto the real DL basis now, exact only up to the normal-equation round-off it
    # seeds this projection solve with. 2.3e-18 here against 5e-3 for a fit that collapses to a
    # constant, i.e. fifteen orders of margin on what this guard is actually for.
    assert before < 1e-16, f"the field was not set up exactly to begin with: {before}"
    # ~1e-8 RMS, not machine zero: the fit is least-squares over points the locator projected onto
    # the new interface, so it inherits the projection's local-coordinate round-off. A fit that
    # collapsed to a constant, or drew points from the neighbouring element, lands near 5e-3 - four
    # orders above this and thirteen above what a correct transfer gives.
    assert after < 1e-16, f"DL transfer lost the linear field across adaptation: {after}"


def test_interface_d0_history_survives_adaptation():
    """The case that was silently wrong rather than merely reset.

    A D0 field its own residual determines algebraically recovers at the next solve even with no
    transfer at all, which is why the reset went unnoticed. One carrying a time derivative does not:
    its history levels are what the next timestep is computed from.
    """
    class IfaceRate(InterfaceEquations):
        def define_fields(self):
            self.define_scalar_field("idata", "D0")

        def define_residuals(self):
            u, v = var_and_test("idata")
            self.add_residual(weak(partial_t(u) - 1, v))

    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
            eqs += IfaceRate() @ "top" + SpatialErrorEstimator(u=1)
            self.add_equations(eqs @ "domain")

    def history(mesh):
        return [[d.value_at_t(t, 0)
                 for e in mesh.elements()
                 for d in (e.internal_data_pt(i) for i in range(e.ninternal_data()))]
                for t in range(3)]

    with Prob() as p:
        p.quiet()
        p.max_refinement_level = 2
        p.run(0.3, outstep=False, startstep=0.1, temporal_error=None, do_not_set_IC=False)
        before = history(p.get_mesh("domain/top"))
        p.refine_uniformly()
        after = history(p.get_mesh("domain/top"))

    # u = t, so the levels must differ by the timestep - otherwise "history survived" is vacuous,
    # since every level being zero would pass a same-before-and-after check just as well.
    # abs=1e-4 because run() stretches the last step slightly to land on the end time.
    assert before[0] and before[0][0] - before[1][0] == pytest.approx(0.1, abs=1e-4), before
    assert before[1][0] - before[2][0] == pytest.approx(0.1, abs=1e-9), before
    for t in range(3):
        assert all(v == pytest.approx(before[t][0], abs=1e-12) for v in after[t]), \
            f"time level {t}: {before[t][0]} -> {after[t]}"


# ----------------------------------------------------------------------------------------------
# 4b. evaluating fields at located points (LocationSet::evaluate, via Mesh.evaluate_at_points)
# ----------------------------------------------------------------------------------------------

def test_evaluate_at_points_reproduces_the_analytic_solution():
    """-u'' = 1 with u(0) = u(1) = 0 gives u = x(1-x)/2, which the C2 space represents exactly.

    So this is not an "about right" check: any error is the locator's or the evaluation's, not the
    discretisation's.
    """
    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=8))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
            self.add_equations(eqs @ "domain")

    probes = [[0.5, 0.5], [0.25, 0.75], [0.0, 0.0], [0.9375, 0.1]]
    with Prob() as p:
        p.quiet()
        p.solve()
        rows = p.get_mesh("domain").evaluate_at_points(probes, False, True, 0)

    assert len(rows) == len(probes)
    for probe, row in zip(probes, rows):
        assert row[0] == 1.0, f"{probe} was not located"
        # 1e-9 rather than machine zero: the quadratic solution lies in the C2 space, so what is
        # left is the Newton/linear-solver residual of the solve itself, not the evaluation.
        assert row[1] == pytest.approx(probe[0] * (1 - probe[0]) / 2, abs=1e-9), (probe, row)
        # position echoes the query back, which is the round trip locate -> interpolated_x
        assert row[2] == pytest.approx(probe[0], abs=1e-12)
        assert row[3] == pytest.approx(probe[1], abs=1e-12)


def test_evaluate_at_points_reports_a_point_outside_the_mesh():
    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            self.add_equations((PoissonEquation(source=1) + DirichletBC(u=0) @ "left") @ "domain")

    with Prob() as p:
        p.quiet()
        p.solve()
        rows = p.get_mesh("domain").evaluate_at_points([[0.5, 0.5], [5.0, 5.0]], False, False, 0)

    assert rows[0][0] == 1.0 and len(rows[0]) == 2
    # An unlocated point must be visibly unlocated, not a plausible-looking zero field value.
    assert rows[1] == [0.0]


def test_evaluate_at_points_carries_interface_d0_and_dl():
    """The discontinuous blocks follow the continuous one, in the DL-then-D0 order evaluate() fixes."""
    class Iface(InterfaceEquations):
        def define_fields(self):
            self.define_scalar_field("da", "DL")
            self.define_scalar_field("db", "D0")

        def define_residuals(self):
            a, va = var_and_test("da")
            b, vb = var_and_test("db")
            self.add_residual(weak(a - var("coordinate_x"), va) + weak(b - 7, vb))

    class Prob(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
            eqs += Iface() @ "top"
            self.add_equations(eqs @ "domain")

    with Prob() as p:
        p.quiet()
        p.solve()
        rows = p.get_mesh("domain/top").evaluate_at_points([[0.3, 1.0], [0.8, 1.0]], False, False, 0)

    for probe, row in zip([0.3, 0.8], rows):
        assert row[0] == 1.0, row
        # [found, u (continuous), da (DL), db (D0)]
        assert len(row) == 4, row
        assert row[2] == pytest.approx(probe, abs=1e-9), row
        assert row[3] == pytest.approx(7.0, abs=1e-12), row


# ----------------------------------------------------------------------------------------------
# 5. transfer across a remesh (phases 2, 3, 4b) - these build meshes with gmsh, so they are slow
# ----------------------------------------------------------------------------------------------

class _BlobProb(Problem):
    def __init__(self, extra=None, res=0.09, with_zeta=None, extra_on="interface"):
        super().__init__()
        self.extra, self.res, self.with_zeta = extra, res, with_zeta
        self.extra_on = extra_on

    def define_problem(self):
        m = _Blob(self.res)
        m.remesher = Remesher2d(m)
        self.add_mesh(m)
        eqs = PoissonEquation(source=1)
        eqs += IntegralObservables(u_int=var("u"))
        if self.extra is not None:
            eqs += self.extra @ self.extra_on
        if self.with_zeta is not None:
            eqs += self.with_zeta @ "interface"
        self.add_equations(eqs @ "domain")


@pytest.mark.slow
@pytest.mark.parametrize("interp", [InternalInterpolator, ProjectionInternalInterpolator])
def test_bulk_transfer_reproduces_a_linear_field(interp):
    with _BlobProb() as p:
        p.quiet()
        p.initialise()
        _stamp_bulk(p.get_mesh("domain"), _linear2)
        p.force_remesh(interpolator=interp)
        worst = _bulk_error(p.get_mesh("domain"), _linear2)
    assert worst < 1e-7, worst


@pytest.mark.slow
def test_projection_conserves_the_integral_better_than_interpolation():
    # The reason to prefer the projection at all. An exact L2 projection conserves exactly (constants
    # are in the space); nodal interpolation does not. Measured on a field the space CANNOT represent,
    # since on a representable one both are exact and the question does not arise.
    def bump(x, y):
        return math.exp(-30.0 * ((x - 0.45) ** 2 + (y - 0.3) ** 2))

    drift = {}
    for interp in (InternalInterpolator, ProjectionInternalInterpolator):
        with _BlobProb() as p:
            p.quiet()
            p.initialise()
            m = p.get_mesh("domain")
            _stamp_bulk(m, bump)
            before = float(m.evaluate_all_observables()["u_int"])
            p.force_remesh(interpolator=interp)
            after = float(p.get_mesh("domain").evaluate_all_observables()["u_int"])
        drift[interp] = abs(after - before) / abs(before)
    assert drift[ProjectionInternalInterpolator] < drift[InternalInterpolator], drift


@pytest.mark.slow
@pytest.mark.parametrize("space,tol", [("C2", 1e-8), ("C1", 1e-8)])
def test_interface_only_dofs_transfer_with_their_history(space, tol):
    # Interface dofs travel through inter_field_map, not the bulk field path, so they need checking
    # separately - and the fallback for an unlocated node used to skip them entirely, silently
    # zeroing the field on that node. Stamped on a STRAIGHT boundary: a C1 field cannot represent a
    # linear function of position along an arc (chord versus arc, ~h^2/8), which is representation
    # error and not the transfer's.
    def hist(v):
        return -0.5 * v + 0.25

    # "bottom" is straight; see the note above about C1 on an arc.
    with _BlobProb(extra=_SurfField("c", space), extra_on="bottom") as p:
        p.quiet()
        p.initialise()
        m, im = p.get_mesh("domain"), p.get_mesh("domain/bottom")
        ifid = m.has_interface_dof_id("c")
        if ifid is None:
            pytest.skip("no interface dof id for 'c'")
        for n in im.nodes():
            i = n.additional_value_index(ifid)
            if i >= 0:
                n.set_value(i, _linear2(n.x(0), n.x(1)))
                n.set_value_at_t(1, i, hist(_linear2(n.x(0), n.x(1))))
        p.force_remesh(interpolator=InternalInterpolator)
        m, im = p.get_mesh("domain"), p.get_mesh("domain/bottom")
        ifid = m.has_interface_dof_id("c")
        now, past = [], []
        for n in im.nodes():
            i = n.additional_value_index(ifid)
            if i >= 0:
                exact = _linear2(n.x(0), n.x(1))
                now.append(abs(n.value(i) - exact))
                past.append(abs(n.value_at_t(1, i) - hist(exact)))
    assert now and max(now) < tol, max(now) if now else None
    assert past and max(past) < tol, max(past) if past else None


@pytest.mark.slow
def test_remeshing_a_closed_boundary_keeps_it_ordered():
    # A closed boundary used to go to gmsh as ONE spline with its first point repeated. That spline
    # has a seam, and the element straddling it took its mid-side node from the average of the
    # endpoints' curve parameters - placing it HALFWAY AROUND THE LOOP, on the boundary and at the
    # right radius, so nothing downstream noticed while that element was grossly distorted.
    class Disc(GmshTemplate):
        def define_geometry(self):
            self.default_resolution = 0.25
            self.mesh_mode = "tri"
            self.create_circle_lines((0, 0), 1.0, line_name="rim")
            self.plane_surface("rim", name="domain")

    class Prob(Problem):
        def define_problem(self):
            m = Disc()
            m.remesher = Remesher2d(m)
            self.add_mesh(m)
            self.add_equations((PoissonEquation(source=1) + DirichletBC(u=0) @ "rim") @ "domain")

    with Prob() as p:
        p.quiet()
        p.solve()
        p.force_remesh()
        im = p.get_mesh("domain/rim")
        worst = 0.0
        for e in im.elements():
            th = [math.atan2(e.node_pt(i).x(1), e.node_pt(i).x(0)) for i in range(e.nnode())]
            for a, b in zip(th, th[1:]):
                worst = max(worst, abs(((b - a + math.pi) % (2 * math.pi)) - math.pi))
    # one element spans ~0.25 rad here; an antipodal node shows up as ~pi
    assert worst < 0.5, worst


_REFUSAL_SCRIPT = '''
import sys
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

space = sys.argv[1]

class Eqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("d", space)

    def define_residuals(self):
        u, ut = var_and_test("u")
        d, dt = var_and_test("d")
        self.add_residual(weak(u - var("coordinate_x"), ut))
        self.add_residual(weak(d - var("coordinate_y"), dt))

class P(Problem):
    def define_problem(self):
        m = RectangularQuadMesh(N=4)
        m.remesher = Remesher2d(m)
        self.add_mesh(m)
        self.add_equations(Eqs() @ "domain")

p = P()
p.quiet()
p.solve()
try:
    p.force_remesh()
except RuntimeError as e:
    print("REFUSED: " + str(e))
else:
    print("ACCEPTED")
'''


@pytest.mark.parametrize("space", ["DL", "D0", "D1", "D2"])
def test_remeshing_refuses_a_discontinuous_field(tmp_path, space):
    # The limit of the transfer, pinned so that raising it is deliberate. nodal_interpolate_from()
    # rejects any mesh carrying a discontinuous field; the DL/D0 transfer inside it and the
    # discontinuous pooling in share_interpolation_across_ranks() are therefore both dead code today.
    # That is a narrowing: until 41b438f2 the check was on the nodal DG spaces alone and DL/D0 fields
    # did survive a remesh. Whoever lifts it has to revisit both of those places, since each has to
    # address the internal data as [DG spaces][DL][D0] rather than from index 0.
    #
    # In a subprocess because the refusal leaves the Problem half-remeshed and tearing that down
    # aborts the interpreter - the test would take the rest of the file with it.
    script = tmp_path / "refusal.py"
    script.write_text(_REFUSAL_SCRIPT)
    proc = subprocess.run([sys.executable, str(script), space], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=300)
    assert "REFUSED: " in proc.stdout, \
        "a %s field was remeshed rather than refused -- if that is intended, the DL/D0 indexing in " \
        "nodal_interpolate_from and share_interpolation_across_ranks needs checking first:\n%s\n%s" % (
            space, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "Cannot interpolate DG fields" in proc.stdout, proc.stdout[-2000:]


# ----------------------------------------------------------------------------------------------
# 6. selecting the interpolator on the Problem
# ----------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_problem_setting_selects_the_interpolator():
    # Problem.mesh_interpolator has to reach the paths that do NOT take an argument - the remesh
    # handler used during continuation calls force_remesh() bare - which is the whole reason for
    # having it rather than passing interpolator= at each call site.
    used = []

    class _SpyNodal(InternalInterpolator):
        def __init__(self, old, new):
            used.append("nodal")
            super().__init__(old, new)

    class _SpyProjection(ProjectionInternalInterpolator):
        def __init__(self, old, new):
            used.append("projection")
            super().__init__(old, new)

    with _BlobProb() as p:
        p.quiet()
        p.mesh_interpolator = _SpyProjection
        p.initialise()
        _stamp_bulk(p.get_mesh("domain"), _linear2)
        p.force_remesh()                       # deliberately no interpolator= argument
        assert used == ["projection"], used
        # an explicit argument must still win over the setting
        used.clear()
        p.force_remesh(interpolator=_SpyNodal)
        assert used == ["nodal"], used


def test_default_interpolator_is_the_nodal_one():
    # Stated explicitly because it is the question every new simulation asks: the locator, the
    # projection-based boundary transfer and the zeta fixes are all in the DEFAULT path. Only the
    # L2 projection of the bulk fields is opt-in.
    class _Trivial(Problem):
        def define_problem(self):
            pass

    assert _Trivial().mesh_interpolator is InternalInterpolator


# ----------------------------------------------------------------------------------------------
# 7. diagnostics name the mesh and the nodes
# ----------------------------------------------------------------------------------------------

def test_full_domain_path():
    with _BlobProb(extra=_SurfField("c", "C2")) as p:
        p.quiet()
        p.initialise()
        assert p.get_mesh("domain").get_full_domain_path() == "domain"
        assert p.get_mesh("domain/interface").get_full_domain_path() == "domain/interface"


@pytest.mark.slow
def test_transfer_warnings_name_the_boundary_and_the_unset_nodes(capfd):
    # A diagnostic that says "boundary 2 of domain" and "3 nodes got nothing" is not actionable: the
    # index is an internal number, and the nodes that silently kept their initial values are almost
    # always in one identifiable place. Both are checked here because both were once missing.
    orig = InternalInterpolator.__init__

    def patched(self, old, new):
        orig(self, old, new)
        self.project_on_boundary_without_zeta = False        # force the legacy blend
        self.boundary_max_distances = {"interface": 1e-4}    # and make it fail to find anything

    InternalInterpolator.__init__ = patched
    try:
        with _BlobProb(extra=_SurfField("c", "C2"), res=0.12) as p:
            p.quiet()
            p.initialise()
            m = p.get_mesh("domain")
            for n in m.nodes():          # move it, so the remesh really does relocate the nodes
                n.set_x(0, n.x(0) * 0.93)
            p.force_remesh()
        out = capfd.readouterr().out
    finally:
        InternalInterpolator.__init__ = orig

    assert "domain/interface" in out, out          # the path, not a bare index
    assert "received NO value" in out, out
    assert "Nodes at: (" in out, out               # and where they are


@pytest.mark.slow
def test_boundary_pass_touches_only_its_own_boundary(capfd):
    # A boundary with no interface mesh of its own is handled by calling nodal_interpolate_from on
    # the BULK mesh with that boundary's index. That call used to ignore the index and walk every
    # node of the mesh - so, running after the interface passes, it re-did their nodes and
    # OVERWROTE the ones it could not locate with a nearest-node blend, undoing correct work. It also
    # reported sixteen failures for a boundary with two nodes on it.
    with _BlobProb() as p:
        p.quiet()
        p.initialise()
        _stamp_bulk(p.get_mesh("domain"), _linear2)
        capfd.readouterr()
        p.force_remesh()
        out = capfd.readouterr().out
        worst = _bulk_error(p.get_mesh("domain"), _linear2)

    assert "WARNING: interpolating" not in out, out
    assert worst < 1e-7, worst
    # and the two passes on the bulk mesh must be distinguishable from each other
    assert "(interior nodes only)" in out, out
    assert "(boundary nodes only)" in out, out


def test_interface_mesh_has_no_node_list_of_its_own():
    # Not a curiosity: nodal_interpolate_from's nearest-node fallback looped over from->nnode(), so on
    # an interface mesh it searched an EMPTY list, found nothing, and left the node with no value at
    # all rather than a poor one. The fallback now gathers the nodes from the elements instead.
    class _P(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            self.add_equations((PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
                                + _SurfField("c", "C2") @ "top") @ "domain")

    with _P() as p:
        p.quiet()
        p.solve()
        im = p.get_mesh("domain/top")
        assert im.nelement() > 0
        assert im.nnode() == 0, "if this ever gains its own node list, the fallback can be simplified"
        via_elements = {id(e.node_pt(i)) for e in im.elements() for i in range(e.nnode())}
        assert len(via_elements) > 0
