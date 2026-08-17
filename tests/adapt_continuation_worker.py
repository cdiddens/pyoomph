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

# Does the arclength continuation tangent survive a mesh adaptation in a USABLE state?
#
# pyoomph carries it across by stashing the two continuation vectors in history slots 5 and 6 before the
# adapt and reading them back afterwards (Problem._adapt_with_interfacial_errors), so oomph's projection
# onto the new mesh interpolates them - much better than oomph's own path, which zero-fills
# Dof_derivative when the dof count changes.
#
# What it did not do was renormalise. The interpolation preserves the tangent's DIRECTION (measured:
# cos = 0.999999 against a freshly computed one) but not its length, since |dU/ds|^2 is a sum over dofs:
# refining 39 -> 79 dofs grew it by sqrt(2). The constraint
#
#     (dparameter/ds)^2 + theta^2*|dU/ds|^2 = 1
#
# is what gives ds its meaning as a step length, so the first step after an adapt came out 29% short.
#
# Bratu, u'' + lam*exp(u) = 0 with a time derivative added so a mass matrix exists, is used because
# refining it changes ndof without changing the physics.

import argparse
import sys

import numpy

from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, grad, exp, partial_t
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.meshes.simplemeshes import LineMesh


class Bratu(Equations):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_weak(partial_t(u), v)
        self.add_weak(grad(u), grad(v))
        self.add_weak(-self.lam*exp(u), v)


class Prob(Problem):
    def define_problem(self):
        self.add_mesh(LineMesh(N=20))
        eqs = Bratu(self.get_global_parameter("lam"))
        eqs += InitialCondition(u=0)
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def invariant_error(problem):
    dp = problem.get_arc_length_parameter_derivative()
    v = problem.get_arclength_dof_derivative_vector()
    theta = problem.get_arc_length_theta_sqr()
    return abs(dp*dp + theta*float(numpy.dot(v, v)) - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--inner-product", default="none", choices=["none", "l2", "ndof"])
    args = ap.parse_args()

    with Prob() as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.max_refinement_level = 4
        # The defaults leave this smooth solution alone, and an adapt that changes nothing proves
        # nothing at all.
        problem.max_permitted_error = 1e-7
        problem.min_permitted_error = 1e-9
        problem.quiet()
        if args.inner_product == "none":
            problem.set_arc_length_parameter(scale_arc_length=False)
        else:
            problem.set_arclength_inner_product(args.inner_product)

        problem.get_global_parameter("lam").value = 1.0
        problem.solve()

        ds = 0.05
        for _ in range(4):
            ds = problem.arclength_continuation("lam", ds)

        ndof_before = problem.ndof()
        err_before = invariant_error(problem)
        assert err_before < 1e-10, "the invariant was already broken before adapting: " + str(err_before)

        nref, _nunref = problem.adapt()
        ndof_after = problem.ndof()
        err_after = invariant_error(problem)
        print("ADAPT ndof {:d} -> {:d} (refined {:d})  invariant {:.3e} -> {:.3e}".format(
            ndof_before, ndof_after, nref, err_before, err_after))
        assert ndof_after != ndof_before, \
            "the mesh did not change, so nothing is being tested (raise max_refinement_level?)"
        assert err_after < 1e-10, \
            "the carried tangent is not normalised: the constraint is off by {:.3e}".format(err_after)

        # The direction has to be right too, not merely the length. Capture the carried tangent HERE:
        # history slot 5 is consumed by the read-back and then overwritten by the
        # assign_initial_values_impulsive() that ends the adapt, so reading it later gives zeros.
        carried = problem.get_arclength_dof_derivative_vector().copy()
        # One step lets oomph recompute the tangent from scratch on the new mesh; the carried one must
        # point the same way.
        problem.arclength_continuation("lam", ds)
        fresh = problem.get_arclength_dof_derivative_vector()
        if len(fresh) == len(carried):
            cos = float(numpy.dot(carried, fresh)/(numpy.linalg.norm(carried)*numpy.linalg.norm(fresh)))
            print("DIRECTION cos={:.6f}".format(cos))
            assert abs(cos) > 0.999, "the carried tangent points the wrong way: cos = " + str(cos)
        assert invariant_error(problem) < 1e-10, "the invariant broke on the step after the adapt"

        # And continuation has to keep working: lam must advance and the solve must converge.
        lam0 = problem.get_global_parameter("lam").value
        for _ in range(3):
            ds = problem.arclength_continuation("lam", ds)
        assert problem.get_global_parameter("lam").value > lam0, "the branch stopped advancing"
        print("CONTINUED to lam={:.6f} at ndof={:d}".format(
            problem.get_global_parameter("lam").value, problem.ndof()))

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
