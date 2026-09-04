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

# Does a REMESH carry the arclength continuation tangent across, or only half of it?
#
# force_remesh() parks the two continuation vectors in history slots 5 and 6, rebuilds the mesh, and
# reads them back. It used to read them back BEFORE interp.interpolate() had run, i.e. off a mesh that
# had never been told what was in those slots: the field values there were still exactly zero and the
# positions were the ones the mesh generator had just produced, not d(position)/ds.
#
# The result was a tangent whose FIELD half was identically zero and whose POSITION half was
# coordinates. The renormalisation afterwards then scaled that to unit length, so
#
#     (dparameter/ds)^2 + theta^2*|dU/ds|^2 = 1
#
# held to machine precision while the direction pointed somewhere else entirely - which is why the
# invariant alone cannot catch this and the two response magnitudes below are measured instead.
#
# A moving mesh is essential: on a fixed mesh the position half does not exist and the defect is
# invisible.

import argparse
import sys

import numpy

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, var_and_test, grad, weak, testfunction, vector
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import RemeshWhen, RemeshingOptions, IntegralObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.meshes.remesher import Remesher2d


class DrivenByP(Equations):
    """Makes BOTH halves of the tangent depend on the continuation parameter, through the residual.

    Through the residual is the point: a parameter that only enters a DirichletBC value is applied by
    setting a pinned value, so d(residual)/d(parameter) is zero and the whole continuation tangent
    comes out zero - measured, which is why the source term and the pseudo-solid body force below are
    both written against P rather than the lid position being moved.
    """

    def __init__(self, P):
        super().__init__()
        self.P = P

    def define_fields(self):
        self.define_scalar_field("c", "C2")

    def define_residuals(self):
        c, ctest = var_and_test("c")
        self.add_residual(weak(grad(c), grad(ctest)) - weak(1 + self.P, ctest))
        # A body force on the pseudo-solid, so the interior node POSITIONS move with P too.
        self.add_residual(-weak(self.P * vector(0, 1), testfunction("mesh")))


class SquashedBox(Problem):
    """A box with a Poisson field inside, both deformed and forced by the continuation parameter."""

    def define_problem(self):
        P = self.get_global_parameter("P")
        mesh = RectangularQuadMesh(N=8)
        mesh.remesher = Remesher2d(mesh)
        self.add_mesh(mesh)
        eqs = LaplaceSmoothedMesh()
        eqs += DrivenByP(P)
        for b, bc in (("left", dict(mesh_x=0)), ("right", dict(mesh_x=1)),
                      ("bottom", dict(mesh_y=0)), ("top", dict(mesh_y=1))):
            eqs += DirichletBC(**bc) @ b
            eqs += DirichletBC(c=0) @ b
        eqs += IntegralObservables(csqr=var("c") ** 2)
        eqs += RemeshWhen(RemeshingOptions())
        self.add_equations(eqs @ "domain")


def responses(problem):
    """(|d(dof)/ds|, max|dX/ds|, |d(int c^2)/ds|) read by perturbing the dofs ALONG the tangent.

    Deliberately measured through the actual mesh and nodal values rather than by slicing the dof
    vector: the dof ordering changes across a remesh, so a slice would compare two different things.
    """
    eps = 1e-6
    backup, _ = problem.get_current_dofs()
    backup = numpy.asarray(backup, dtype=float)
    ddof = numpy.asarray(problem.get_arclength_dof_derivative_vector(), dtype=float)
    if len(ddof) != len(backup) or not len(ddof):
        return 0.0, 0.0, 0.0

    def snap():
        # Coordinates read straight off the nodes, but the FIELD through a named observable: the
        # nodal value INDEX of "c" is not the same on both sides of a remesh, so reading value(0)
        # silently compares c before against a mesh-position component afterwards (measured: the two
        # responses then came out bit-identical, which is what gave it away).
        xs = numpy.array([(n.x(0), n.x(1)) for n in problem.get_mesh("domain").nodes()])
        return xs, float(problem.get_mesh("domain").evaluate_observable("csqr"))

    x0, c0 = snap()
    problem.set_current_dofs(backup + eps * ddof)
    x1, c1 = snap()
    problem.set_current_dofs(backup)
    return (float(numpy.linalg.norm(ddof)),
            float(numpy.abs(x1 - x0).max() / eps),
            abs(c1 - c0) / eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with SquashedBox() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.quiet()
        problem.set_arc_length_parameter(scale_arc_length=False)
        problem.get_global_parameter("P").value = 0.0
        problem.solve()

        ds = 0.05
        for _ in range(3):
            ds = problem.arclength_continuation("P", ds)

        n_before, x_before, c_before = responses(problem)
        print("BEFORE |ddof| {:.6g}  max|dX/ds| {:.6g}  |dcsqr/ds| {:.6g}".format(
            n_before, x_before, c_before))
        assert x_before > 0.0 and c_before > 0.0, \
            "the test is not measuring anything: the tangent has no position or no field part " \
            "before the remesh (" + str((x_before, c_before)) + ")"

        problem.force_remesh()
        n_after, x_after, c_after = responses(problem)

        print("AFTER  |ddof| {:.6g}  max|dX/ds| {:.6g}  |dcsqr/ds| {:.6g}".format(
            n_after, x_after, c_after))

        # The field half must survive at all. It came back as EXACTLY zero before the fix, so this is
        # the assertion that names the defect.
        assert c_after > 0.0, \
            "the remesh destroyed the field part of the continuation tangent: |d(int c^2)/ds| is " \
            + str(c_after) + " (it was " + str(c_before) + " before the remesh)"
        assert x_after > 0.0, \
            "the remesh destroyed the position part of the continuation tangent: max|dX/ds| is " \
            + str(x_after)

        # And the DIRECTION must be carried, not just both halves being nonzero. The ratio of the two
        # responses is invariant under the renormalisation force_remesh applies, so it compares the
        # directions and not the lengths. Measured 1.987 -> 1.987 on a two-phase moving mesh.
        r_before, r_after = x_before / c_before, x_after / c_after
        print("RATIO dX/dcsqr  {:.6g} -> {:.6g}".format(r_before, r_after))
        assert abs(r_after - r_before) < 0.05 * r_before, \
            "the remesh turned the continuation tangent: the position/field response ratio went " \
            + str(r_before) + " -> " + str(r_after)

        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
