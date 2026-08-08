"""Check GmshTemplate.fix_2d_orientation over all four planar element types.

Companion to dev_docs/mesh_construction.md.

Each geometry is meshed three ways: with the curve loop wound counter-clockwise, with it
wound clockwise and the fix disabled (the control, which must produce an inside-out mesh),
and with it wound clockwise and the fix enabled.

Asserted for every case:
  * the C++ inversion detector accepts the mesh, i.e. every det(dx/ds) > 0;
  * the integrated area is exact, which a wrongly permuted element - a mid-side node moved
    onto another edge, say - would not reproduce;
  * the Poisson solution matches the counter-clockwise arm to solver tolerance;
  * a correctly wound mesh is left completely alone (no element flipped).

The second geometry is a disk, so the curved-boundary/macro-element path is covered too.

A final section checks that interface normals do not depend on the bulk winding, which is
what makes the fix a pure relabelling rather than a change of results.
"""
import sys

sys.argv = [sys.argv[0]]

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.generic import IntegralObservables
from pyoomph._pyoomph_core import set_detect_inverted_elements

# L-shape: unit square minus the top-right quarter. Non-convex, so the winding of the loop
# is not something Gmsh can silently normalise away.
_LPTS = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
_RADIUS = 0.7


class LMesh(GmshTemplate):
    def __init__(self, mode, reverse, fix):
        super().__init__()
        self.default_resolution = 0.2
        self.mesh_mode = "quads" if mode.startswith("quads") else "tris"
        self.order = 1 if mode.endswith("1") else 2
        self.reverse = reverse
        self.fix_2d_orientation = fix

    def define_geometry(self):
        pts = [self.point(x, y) for x, y in _LPTS]
        lines = [self.line(pts[i], pts[(i + 1) % len(pts)], name="bnd")
                 for i in range(len(pts))]
        self.plane_surface(*lines, name="domain", reversed_order=self.reverse)


class DiskMesh(GmshTemplate):
    """Curved boundary, i.e. the macro-element path with attached curved entities."""

    def __init__(self, mode, reverse, fix):
        super().__init__()
        self.default_resolution = 0.15
        self.mesh_mode = "quads" if mode.startswith("quads") else "tris"
        self.order = 1 if mode.endswith("1") else 2
        self.reverse = reverse
        self.fix_2d_orientation = fix

    def define_geometry(self):
        c = self.point(0, 0)
        r = _RADIUS
        rim = [self.point(r, 0), self.point(0, r), self.point(-r, 0), self.point(0, -r)]
        arcs = [self.circle_arc(rim[i], rim[(i + 1) % 4], center=c, name="bnd")
                for i in range(4)]
        self.plane_surface(*arcs, name="domain", reversed_order=self.reverse)


class Poisson(Equations):
    def __init__(self, space):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("u", self.space)

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) - weak(1, v))


class OrientProblem(Problem):
    def __init__(self, geom, mode, reverse, fix, with_normals=False):
        super().__init__()
        self.geom, self.mode, self.reverse, self.fix = geom, mode, reverse, fix
        self.with_normals = with_normals

    def define_problem(self):
        cls = LMesh if self.geom == "L" else DiskMesh
        self.mesh = cls(self.mode, self.reverse, self.fix)
        self.add_mesh(self.mesh)
        # Real unknowns, otherwise get_residuals() has nothing to assemble and the
        # inversion detector is never reached
        eqs = Poisson("C1" if self.mode.endswith("1") else "C2")
        eqs += IntegralObservables(area=1, uint=var("u"))
        eqs += DirichletBC(u=0) @ "bnd"
        if self.with_normals:
            eqs += IntegralObservables(ndotx=dot(var("normal"), var("coordinate"))) @ "bnd"
        self.add_equations(eqs @ "domain")


def check(geom, mode, reverse, fix, exact_area, reference=None):
    set_detect_inverted_elements(False)
    with OrientProblem(geom, mode, reverse, fix) as problem:
        problem.set_output_directory("out_orient_%s_%s_%s_%s" % (geom, mode, reverse, fix))
        problem.initialise()
        nflip = problem.mesh.num_flipped_2d_elements
        nelem = len(list(problem.get_mesh("domain").elements()))
        area = float(problem.get_mesh("domain").evaluate_observable("area"))

        set_detect_inverted_elements(True)
        try:
            problem.get_residuals()
            verdict = "ok"
        except Exception:
            verdict = "INVERTED"
        set_detect_inverted_elements(False)

        uint = float("nan")
        if verdict == "ok":
            problem.solve()
            uint = float(problem.get_mesh("domain").evaluate_observable("uint"))

    # A disk is only meshed to the accuracy of its boundary discretisation, so its area
    # cannot be compared as tightly as the straight-edged L
    tol = 1e-9 if geom == "L" else 2e-2
    ok_area = abs(area - exact_area) < tol
    ok_u = True if reference is None else abs(uint - reference) < 1e-8
    print("%-4s %-6s reverse=%-5s fix=%-5s  nelem=%3d  flipped=%3d  area=%.8f %-5s  "
          "int(u)=%.8f %-5s  detector=%s"
          % (geom, mode, reverse, fix, nelem, nflip, area, "OK" if ok_area else "WRONG",
             uint, "-" if reference is None else ("OK" if ok_u else "WRONG"), verdict))
    return verdict, nflip, ok_area, ok_u, uint


def check_normals(reverse, fix):
    """Interface normals come out of the bulk element, so they might follow its winding.

    The divergence theorem settles it without ambiguity: int_bnd n.x ds = 2*area, so a run
    that reports the negative of that had its normals reversed.
    """
    with OrientProblem("L", "tris2", reverse, fix, with_normals=True) as problem:
        problem.set_output_directory("out_normals_%s_%s" % (reverse, fix))
        problem.initialise()
        problem.solve()
        area = float(problem.get_mesh("domain").evaluate_observable("area"))
        ndotx = float(problem.get_mesh("domain/bnd").evaluate_observable("ndotx"))
        nflip = problem.mesh.num_flipped_2d_elements
    ok = abs(ndotx - 2 * area) < 1e-9
    print("NORMALS reverse=%-5s fix=%-5s  flipped=%3d  int n.x ds=%+.6f (expected %+.6f) %s"
          % (reverse, fix, nflip, ndotx, 2 * area, "OK" if ok else "WRONG"))
    return ok


if __name__ == "__main__":
    failures = []
    for geom, exact_area in (("L", 0.75), ("disk", numpy.pi * _RADIUS ** 2)):
        for mode in ("quads1", "quads2", "tris1", "tris2"):
            tag = "%s/%s" % (geom, mode)

            # Correctly wound loop: nothing must be touched, mesh must be valid
            verdict, nflip, ok_area, _, uref = check(geom, mode, False, True, exact_area)
            if verdict != "ok" or nflip != 0 or not ok_area:
                failures.append(tag + " ccw")

            # Reversed loop with the fix off: the control that the reversal really does
            # produce an inside-out mesh
            verdict_off = check(geom, mode, True, False, exact_area)[0]
            if verdict_off != "INVERTED":
                failures.append(tag + " control-not-inverted")

            # Reversed loop with the fix on: repaired, same solution as the ccw arm
            verdict, nflip, ok_area, ok_u, _ = check(geom, mode, True, True, exact_area, uref)
            if verdict != "ok" or not ok_area or not ok_u or nflip == 0:
                failures.append(tag + " cw")
            print()

    for reverse, fix in ((False, True), (True, False), (True, True)):
        if not check_normals(reverse, fix):
            failures.append("normals %s/%s" % (reverse, fix))

    print()
    print("FAILURES:", failures if failures else "none")
