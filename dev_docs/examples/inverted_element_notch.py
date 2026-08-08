"""Prescribed 2d deformation that folds a Laplace-smoothed mesh.

Companion to dev_docs/mesh_construction.md.

A unit square is meshed with quads and smoothed harmonically. The only thing driving the
problem is a prescribed boundary deformation: a Gaussian notch of growing depth is pushed
into the top edge. The *domain* stays perfectly meshable the whole time - it is only the
harmonic extension into that non-convex shape that folds, which is precisely the situation
remeshing is supposed to repair.

A diffusing scalar rides along; it is there so that (a) the problem has a non-geometric dof
to build a temporal error norm from and (b) a remesh has something to interpolate.

The mesh folds (min det(dx/ds) over the integration points goes negative) at t = 0.1565.
Past that point no reduction of dt helps: the deformation is prescribed as a function of t
alone, so the fold sits at a fixed time, not at a fixed step size.

First positional argument is a comma-separated set of independent flags:
  detect     set_detect_inverted_elements(True)
  adaptive   solve with a temporal error tolerance (so dt can be rejected/reduced)
  remesh     attach Remesher2d + the existing quality-based RemeshWhen
  tight      with "adaptive": use a much tighter temporal tolerance
Anything else (e.g. "plain") just runs with all of them off.
"""

import sys

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import RemeshWhen, RemeshingOptions
from pyoomph.meshes.remesher import Remesher2d
from pyoomph._pyoomph_core import set_detect_inverted_elements


class DiffusingScalar(Equations):
    """A scalar diffusing on the moving mesh - the only real (non-geometric) unknown."""

    def __init__(self, D=0.02):
        super().__init__()
        self.D = D

    def define_fields(self):
        self.define_scalar_field("c", "C2")

    def define_residuals(self):
        c, ctest = var_and_test("c")
        self.add_residual(weak(partial_t(c, ALE="auto"), ctest) + weak(self.D * grad(c), grad(ctest)))


class NotchProblem(Problem):
    def __init__(self, mode="plain"):
        super().__init__()
        self.mode = mode
        self.N = 10               # elements per direction
        self.notch_width = 0.12   # Gaussian half width of the notch
        self.notch_rate = 0.85    # notch depth per unit time
        self.with_remesher = "remesh" in mode.split(",")
        self.remesh_options = RemeshingOptions(max_expansion=3, min_expansion=0.2,
                                               min_quality_decrease=0.3)

    def define_problem(self):
        mesh = RectangularQuadMesh(N=self.N)
        if self.with_remesher:
            mesh.remesher = Remesher2d(mesh)
        self.add_mesh(mesh)

        eqs = LaplaceSmoothedMesh()
        eqs += DiffusingScalar()
        eqs += MeshFileOutput()

        xi = var("lagrangian")
        t = var("time")
        # Nodes on the sides and the bottom slide along their edge; the top edge is pushed
        # down by the notch. The notch depth grows linearly in time.
        eqs += DirichletBC(mesh_x=0, mesh_y=True) @ "left"
        eqs += DirichletBC(mesh_x=1, mesh_y=True) @ "right"
        eqs += DirichletBC(mesh_y=0, mesh_x=True) @ "bottom"
        notch = self.notch_rate * t * exp(-((xi[0] - 0.5) / self.notch_width) ** 2)
        eqs += DirichletBC(mesh_y=1 - notch, mesh_x=True) @ "top"

        eqs += InitialCondition(c=exp(-((xi[0] - 0.5) ** 2 + (xi[1] - 0.5) ** 2) / 0.05))

        if self.with_remesher:
            eqs += RemeshWhen(self.remesh_options)

        self.add_equations(eqs @ "domain")


# 3x3 Gauss points of the biquadratic quad - the same places the C++ core tests, so the
# number reported here is the one that decides whether an element counts as inverted.
_G = numpy.sqrt(3.0 / 5.0)
_GAUSS = [(a, b) for b in (-_G, 0.0, _G) for a in (-_G, 0.0, _G)]


def _dshape_quad9(s0, s1):
    """d(N_i)/d(s0), d(N_i)/d(s1) for the 9-node quad, nodes ordered s0 fastest."""
    def l(s):
        return numpy.array([0.5 * s * (s - 1), 1 - s * s, 0.5 * s * (s + 1)])

    def dl(s):
        return numpy.array([s - 0.5, -2 * s, s + 0.5])

    l0, l1, d0, d1 = l(s0), l(s1), dl(s0), dl(s1)
    return (numpy.outer(l1, d0).ravel(), numpy.outer(d1, l0).ravel())


def jacobian_stats(problem):
    """(smallest signed det(dx/ds), number of elements with any non-positive one)."""
    msh = problem.get_mesh("domain")
    worst, nbad, nskip = None, 0, 0
    for e in msh.elements():
        n = e.nnode()
        if n != 9:
            nskip += 1
            continue
        X = numpy.array([[e.node_pt(i).x(0), e.node_pt(i).x(1)] for i in range(n)])
        bad = False
        for s0, s1 in _GAUSS:
            g0, g1 = _dshape_quad9(s0, s1)
            J = numpy.array([g0 @ X, g1 @ X])  # rows: dx/ds0, dx/ds1
            d = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if worst is None or d < worst:
                worst = d
            if d <= 0:
                bad = True
        nbad += bad
    return worst, nbad, nskip


if __name__ == "__main__":
    mode = "plain"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        mode = sys.argv.pop(1)
    flags = set(mode.split(","))

    if "detect" in flags:
        set_detect_inverted_elements(True)

    with NotchProblem(mode) as problem:
        # Relative to the working directory, not to this file - run it from a scratch dir
        problem.set_output_directory("out_" + mode.replace(",", "_"))
        problem.initialise()

        def report(tag):
            worst, nbad, nskip = jacobian_stats(problem)
            print("### %-8s t=%.6f  min detJ=%.4e  inverted elems=%d  (skipped %d)"
                  % (tag, float(problem.get_current_time(dimensional=False)), worst, nbad, nskip))

        problem.output()
        report("start")
        t, dt = 0.0, 0.02
        while t < 1.0 - 1e-9:
            terr = (1e-7 if "tight" in flags else 1e-3) if "adaptive" in flags else None
            dt = float(problem.solve(timestep=dt, temporal_error=terr))
            dt = min(dt, 0.05)
            t = float(problem.get_current_time(dimensional=False))
            problem.output()
            report("step")
        print("### finished at t=%.6f" % t)
