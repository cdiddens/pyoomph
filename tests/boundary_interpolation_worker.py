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

# Worker for tests/test_boundary_interpolation_fixes.py -- one Problem per process, since a second one
# segfaults in the JIT loader. Two cases, one per defect family; see the test module for what each of
# them is about.
#
# Every transferred field here is a QUADRATIC of the coordinates, which the C2 space represents
# exactly on straight-sided elements: the projection before the remesh is exact to round-off and a
# correct transfer stays exact, so the assertions can be about 1e-10 rather than about an
# interpolation error that would be of the same size as the defects.

import argparse
import sys
import traceback

from pyoomph import Problem, Equations, InterfaceEquations, ElementSpace, var
from pyoomph.expressions import var_and_test, weak
from pyoomph.equations.generic import ProjectExpression
from pyoomph.meshes.gmsh import GmshTemplate

X, Y = var("coordinate_x"), var("coordinate_y")
BULK_FIELD = 1 + 0.3 * X + 0.5 * Y + 0.7 * X ** 2 - 0.4 * X * Y + 0.9 * Y ** 2
#: Linear along the (straight, horizontal) interface, so that even the two-nearest-node blend that
#: the legacy boundary transfer uses reproduces it wherever the two matches bracket the new node.
SURFACTANT = 0.5 + 1.7 * X
CONTACT_LINE_VALUE = 3.25

#: Where the left domain's outer wall sits after remeshing. Moving it is what makes the codimension-2
#: corner MOVE, and only a corner that moved can show that the codim-2 pass overwrote the value the
#: per-boundary pass had interpolated there: for a corner that stays put, the nearest old node is the
#: old corner itself and the blend happens to be right.
XLEFT_REMESHED = 0.1


def bulk_exact(x, y):
    return 1 + 0.3 * x + 0.5 * y + 0.7 * x * x - 0.4 * x * y + 0.9 * y * y


class TwoBoxes(GmshTemplate):
    """Two squares sharing the boundary "middle", which carries no equations on either side.

    The remeshed geometry differs from the first one in two ways that both matter:

    * the points are created in a different order, which permutes the per-mesh boundary indices (they
      are handed out in the order the template's nodes are visited, see MeshFromTemplate2d). Old and
      new therefore disagree on them, which is the situation the bulk fallback in
      InternalInterpolator.interpolate got wrong;
    * the left wall moves outwards, so the corner where "bottom_l" meets "left_w" is at a different
      place afterwards.
    """

    def define_geometry(self):
        first = self.is_first_time()
        self.default_resolution = 0.25 if first else 0.2
        xl = 0.0 if first else XLEFT_REMESHED
        order = [(xl, 0), (1, 0), (2, 0), (xl, 1), (1, 1), (2, 1)] if first else \
                [(xl, 1), (1, 1), (2, 1), (xl, 0), (1, 0), (2, 0)]
        pts = {c: self.point(*c) for c in order}
        p00, p10, p20 = pts[(xl, 0)], pts[(1, 0)], pts[(2, 0)]
        p01, p11, p21 = pts[(xl, 1)], pts[(1, 1)], pts[(2, 1)]
        self.create_lines(p00, "bottom_l", p10, "bottom_r", p20, "right_w", p21,
                          "top_r", p11, "top_l", p01, "left_w", p00)
        self.line(p10, p11, name="middle")
        self.plane_surface("bottom_l", "middle", "top_l", "left_w", name="left")
        self.plane_surface("bottom_r", "right_w", "top_r", "middle", name="right")


class SingleBox(GmshTemplate):
    """The unit square, whose "top" boundary carries an interface-only field."""

    def define_geometry(self):
        self.default_resolution = 0.25 if self.is_first_time() else 0.2
        p00, p10 = self.point(0, 0), self.point(1, 0)
        p11, p01 = self.point(1, 1), self.point(0, 1)
        self.create_lines(p00, "bottom", p10, "right", p11, "top", p01, "left", p00)
        self.plane_surface("bottom", "right", "top", "left", name="dom")


class ForceLegacyBoundaryTransfer(Equations):
    """Send the boundaries of this domain down Mesh::nodal_interpolate_along_boundary.

    That is the path taken whenever no boundary coordinate is defined and the projection onto the old
    interface is switched off; it is also the only path that transfers a codimension-2 mesh at all.
    """

    def _before_mesh_to_mesh_interpolation(self, eqtree, interpolator):
        interpolator.try_to_use_zeta_on_boundary = False
        interpolator.project_on_boundary_without_zeta = False


class Surfactant(InterfaceEquations):
    """A field that exists on the interface only, i.e. an additional dof on the interface nodes."""

    def define_fields(self):
        self.define_scalar_field("Gamma", "C2")

    def define_residuals(self):
        g, gt = var_and_test("Gamma")
        self.add_residual(weak(g - SURFACTANT, gt))


class ContactLineValue(InterfaceEquations):
    """The same one level down: a dof of the codimension-2 mesh itself."""

    def define_fields(self):
        self.define_scalar_field("cl", "C2")

    def define_residuals(self):
        c, ct = var_and_test("cl")
        self.add_residual(weak(c - CONTACT_LINE_VALUE, ct))


def _report_boundary_indices(problem, domain):
    """How the old and the new mesh number the boundaries of `domain`, before the transfer runs.

    Reported so that the test can tell a genuine pass from a vacuous one: if the two happen to agree,
    the bulk fallback cannot be wrong about them and the case proves nothing.
    """
    from pyoomph.meshes import interpolator as interp_mod
    original = interp_mod.InternalInterpolator.interpolate

    def wrapped(self):
        if self.new.get_name() == domain:
            n_mismatch = 0
            for bn in self.new.get_boundary_names():
                if not bn:
                    continue
                if self.new.get_boundary_index(bn) != self.old.get_boundary_index(bn):
                    n_mismatch += 1
            print("PYOOMPH_INDICES domain=%s mismatched=%d" % (domain, n_mismatch), flush=True)
        return original(self)

    interp_mod.InternalInterpolator.interpolate = wrapped


def case_two_domain(outdir):
    problem = Problem()
    problem.set_output_directory(outdir)
    problem.quiet()
    problem += TwoBoxes()
    for dom in ("left", "right"):
        problem += (ElementSpace("C2") + ProjectExpression(u=BULK_FIELD)) @ dom
    # An interface mesh on ONE boundary of "left": that is what takes "bottom_l" out of the bulk
    # fallback loop and therefore leaves one destination boundary index unclaimed by it.
    problem += (Equations() + Equations() @ "left_w") @ "left/bottom_l"
    _report_boundary_indices(problem, "left")
    problem.initialise()
    problem.solve()
    for dom in ("left", "right"):
        mesh = problem.get_mesh(dom)
        worst = max(abs(n.value(0) - bulk_exact(n.x(0), n.x(1))) for n in mesh.nodes())
        print("PYOOMPH_PRE domain=%s worst=%.12g" % (dom, worst), flush=True)
    problem.force_remesh()
    for dom in ("left", "right"):
        mesh = problem.get_mesh(dom)
        per_boundary = {}
        worst_all = 0.0
        for n in mesh.nodes():
            err = abs(n.value(0) - bulk_exact(n.x(0), n.x(1)))
            worst_all = max(worst_all, err)
        for bn in mesh.get_boundary_names():
            if not bn:
                continue
            bi = mesh.get_boundary_index(bn)
            nodes = [mesh.boundary_node_pt(bi, i) for i in range(mesh.nboundary_node(bi))]
            if not nodes:
                continue
            per_boundary[bn] = max(abs(n.value(0) - bulk_exact(n.x(0), n.x(1))) for n in nodes)
        print("PYOOMPH_POST domain=%s worst=%.12g" % (dom, worst_all), flush=True)
        for bn, w in sorted(per_boundary.items()):
            print("PYOOMPH_BND domain=%s name=%s worst=%.12g" % (dom, bn, w), flush=True)
    # The codimension-2 corner itself: "bottom_l" meets "left_w" there, and it has moved.
    mesh = problem.get_mesh("left")
    for n in mesh.nodes():
        if abs(n.x(0) - XLEFT_REMESHED) < 1e-9 and abs(n.x(1)) < 1e-9:
            print("PYOOMPH_CORNER x=%.12g y=%.12g err=%.12g"
                  % (n.x(0), n.x(1), abs(n.value(0) - bulk_exact(n.x(0), n.x(1)))), flush=True)


def _report_interface_fields(problem, tag):
    imesh = problem.get_mesh("dom/top")
    gi = imesh.get_nodal_field_indices()["Gamma"]
    vals = [(n.x(0), n.value(gi)) for n in imesh.nodes()]
    worst = max(abs(g - (0.5 + 1.7 * x)) for x, g in vals)
    print("PYOOMPH_SURF tag=%s n=%d min=%.12g max=%.12g worst=%.12g"
          % (tag, len(vals), min(g for _, g in vals), max(g for _, g in vals), worst), flush=True)
    for side in ("left", "right"):
        cmesh = problem.get_mesh("dom/top/" + side)
        ci = cmesh.get_nodal_field_indices()["cl"]
        for n in cmesh.nodes():
            print("PYOOMPH_CL tag=%s side=%s value=%.12g" % (tag, side, n.value(ci)), flush=True)


def case_interface_field(outdir):
    problem = Problem()
    problem.set_output_directory(outdir)
    problem.quiet()
    problem += SingleBox()
    problem += (ElementSpace("C2") + ProjectExpression(u=BULK_FIELD)
                + ForceLegacyBoundaryTransfer()
                + (Surfactant() + ContactLineValue() @ "left"
                   + ContactLineValue() @ "right") @ "top") @ "dom"
    problem.initialise()
    problem.solve()
    _report_interface_fields(problem, "pre")
    problem.force_remesh()
    _report_interface_fields(problem, "post")


CASES = {"two_domain": case_two_domain, "interface_field": case_interface_field}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--outdir", required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    try:
        CASES[args.case](args.outdir)
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_RAISED %s: %s" % (type(e).__name__, " | ".join(str(e).splitlines())),
              flush=True)
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
