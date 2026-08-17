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

"""Does set_arclength_inner_product() make the metric mesh-independent?

Bratu with a time derivative (same steady states, but a mass matrix exists):

    du/dt = laplace(u) + lam*exp(u),   u(0)=u(1)=0,   fold at lam = 3.513830719

Refining the mesh changes ndof without changing the physics, so anything that moves with ndof is the
metric and not the problem. Three things are checked per mesh:

  1. the constraint invariant  (dp/ds)^2 + theta^2*|dU/ds|^2 == 1  still holds after retuning,
  2. what theta^2 comes out as - "ndof" must give exactly 1/ndof, "l2" must land on the same 1/ndof
     scaling without being told to,
  3. the parameter increment per unit ds, which is what a user actually feels: mesh-INDEPENDENT is the
     whole point, against the 21x drift measured for the plain dof-sum metric.

One Problem per process, so --N comes from the caller.
"""
import argparse
import sys

import numpy

from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, grad, exp, partial_t
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
    def __init__(self, N):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(LineMesh(N=self.N))
        eqs = Bratu(self.get_global_parameter("lam"))
        eqs += InitialCondition(u=0)
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        self += eqs @ "domain"


def sweep(problem, kind, nsteps=6, ds=0.05):
    """Take a few steps with a FIXED ds and report the metric and the parameter stride."""
    problem.reset_arc_length_parameters()
    problem.set_arclength_inner_product(kind)
    if kind is None:
        problem.set_arc_length_parameter(scale_arc_length=False)
    problem.get_global_parameter("lam").value = 1.0
    problem.set_initial_condition()
    problem.solve()

    worst_invariant = 0.0
    strides = []
    prev = problem.get_global_parameter("lam").value
    for _ in range(nsteps):
        problem.arclength_continuation("lam", ds)
        lam = problem.get_global_parameter("lam").value
        strides.append(lam - prev)
        prev = lam
        # The constraint that gives ds its meaning as a step length.
        p = problem.get_arc_length_parameter_derivative()
        v = problem.get_arclength_dof_derivative_vector()
        theta = problem.get_arc_length_theta_sqr()
        worst_invariant = max(worst_invariant, abs(p*p + theta*float(numpy.dot(v, v)) - 1.0))
    return problem.get_arc_length_theta_sqr(), strides, worst_invariant


def fold_traverse(problem, kind, ds0=0.05, maxsteps=100, target=2.0):
    """Round the fold and come back to lam=target. The turn is detected from lam reversing."""
    problem.reset_arc_length_parameters()
    problem.set_arclength_inner_product(kind)
    if kind is None:
        problem.set_arc_length_parameter(scale_arc_length=False)
    problem.get_global_parameter("lam").value = 1.0
    problem.set_initial_condition()
    problem.solve()
    ds = ds0
    prev = problem.get_global_parameter("lam").value
    peak = prev
    rising, turned = True, False
    for i in range(maxsteps):
        try:
            ds = problem.arclength_continuation("lam", ds)
        except Exception:
            return None, peak
        lam = problem.get_global_parameter("lam").value
        peak = max(peak, lam)
        if rising and lam < prev:
            rising, turned = False, True
        prev = lam
        if turned and lam <= target:
            return i + 1, peak
    return None, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with Prob(args.N) as problem:
        problem.set_output_directory(args.outdir)
        problem.set_linear_solver("superlu")
        problem.quiet()
        problem.get_global_parameter("lam").value = 1.0
        problem.solve()
        ndof = problem.ndof()

        for kind, label in ((None, "plain dof sum"), ("ndof", "ndof"), ("l2", "l2 (mass matrix)")):
            theta, strides, inv = sweep(problem, kind)
            # The constraint is what gives ds its meaning as a step length; retuning theta^2 without
            # renormalising the tangent would break it silently.
            assert inv < 1e-10, "the arclength invariant broke: " + str(inv)
            if kind == "ndof":
                assert abs(theta*ndof - 1.0) < 1e-9, "theta^2 must be exactly 1/ndof, got " + str(theta*ndof)
            elif kind == "l2":
                # Not told about ndof at all, yet it has to land on the same scaling.
                assert abs(theta*ndof - 1.0) < 0.01, "the mass-matrix norm should scale as 1/ndof, got " + str(theta*ndof)
            print("IP {:6d} {:<18s} theta2 {:12.6g}  theta2*ndof {:9.5f}  "
                  "dlam/ds {:9.6f}  invariant_err {:.2e}".format(
                      ndof, label, theta, theta*ndof,
                      # the settled stride, ignoring the first step where there is no tangent yet
                      sum(strides[2:])/len(strides[2:])/0.05, inv), flush=True)

        for kind, label in ((None, "plain dof sum"), ("ndof", "ndof"), ("l2", "l2 (mass matrix)")):
            steps, peak = fold_traverse(problem, kind)
            print("FOLD {:6d} {:<18s} steps {:s}  peak_lam {:.4f}".format(
                ndof, label, str(steps) if steps is not None else "x", peak), flush=True)

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
