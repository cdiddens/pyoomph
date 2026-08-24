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

# Worker for tests/test_hopf_lyapunov.py, also usable under mpirun.
#
# The system is the Hopf normal form written in Cartesian coordinates, with a diagonal mass matrix:
#
#   m1*xdot = mu*x - w*y + sigma*x*(x^2+y^2)
#   m2*ydot = w*x + mu*y + sigma*y*(x^2+y^2)
#
# Why this one. The nonlinearity is CUBIC, so the second-order form B vanishes at the origin and the
# whole first Lyapunov coefficient comes from the C term -- the half that has to be finite-differenced
# out of the analytic Hessian, and the half nothing else pins. And for m1 == m2 the polar reduction is
# exact whatever that common mass is:
#
#   r*rdot = x*xdot + y*ydot  =>  m*rdot = mu*r + sigma*r^3
#
# so the Hopf sits at mu=0 with omega0 = w/m, and the limit cycle for mu*sigma < 0 has radius exactly
# sqrt(-mu/sigma) and period exactly 2*pi*m/w. That gives an end-to-end reference for the orbit that
# needs no knowledge of pyoomph's normalisation convention for the coefficient itself.
#
# The "pde" case applies the same kinetics pointwise on a 1D mesh and adds diffusion. Diffusion
# annihilates a spatially uniform field, so the uniform state is still an exact solution with the same
# normal form, while the mesh is now big enough to distribute. Every other spectral branch is diffusive
# and decays, so the Hopf pair the tracker finds is the uniform one.

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import Problem, ODEEquations, Equations, InitialCondition, LineMesh
from pyoomph.expressions import var, testfunction, partial_t, var_and_test, weak, grad
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc

_W = 1.0            # the linear rotation rate; omega0 is w/m
_PDE_DIFFUSIVITY = 0.05


class NormalFormODE(ODEEquations):
    def __init__(self, sigma, m1, m2):
        super().__init__()
        self.sigma, self.m1, self.m2 = sigma, m1, m2

    def define_fields(self):
        self.define_ode_variable("x", "y")

    def define_residuals(self):
        x, y = var(["x", "y"])
        mu = self.get_problem().mu
        r2 = x ** 2 + y ** 2
        self.add_residual((self.m1 * partial_t(x) - (mu * x - _W * y + self.sigma * x * r2)) * testfunction(x))
        self.add_residual((self.m2 * partial_t(y) - (_W * x + mu * y + self.sigma * y * r2)) * testfunction(y))


class NormalFormPDE(Equations):
    """The same kinetics pointwise, plus diffusion; the uniform state is still exact."""

    def __init__(self, sigma, m):
        super().__init__()
        self.sigma, self.m = sigma, m

    def define_fields(self):
        self.define_scalar_field("x", "C2")
        self.define_scalar_field("y", "C2")

    def define_residuals(self):
        x, xt = var_and_test("x")
        y, yt = var_and_test("y")
        mu = self.get_problem().mu
        r2 = x ** 2 + y ** 2
        self.add_residual(weak(self.m * partial_t(x) - (mu * x - _W * y + self.sigma * x * r2), xt)
                          + weak(_PDE_DIFFUSIVITY * grad(x), grad(xt)))
        self.add_residual(weak(self.m * partial_t(y) - (_W * x + mu * y + self.sigma * y * r2), yt)
                          + weak(_PDE_DIFFUSIVITY * grad(y), grad(yt)))


class BrusselatorODE(ODEEquations):
    """xdot = A - (B+1)x + x^2 y,  ydot = B x - x^2 y.

    Unlike the normal form this has a NON-ZERO quadratic term, so it is the case that actually
    exercises the h11/h20 solves and the sig/d0 terms; for the normal form they are identically zero.
    Its Hopf sits at B = 1 + A^2 on the fixed point (A, B/A) and is supercritical.
    """

    def __init__(self, A):
        super().__init__()
        self.A = A

    def define_fields(self):
        self.define_ode_variable("x", "y")

    def define_residuals(self):
        x, y = var(["x", "y"])
        B = self.get_problem().mu
        self.add_residual((partial_t(x) - (self.A - (B + 1) * x + x**2 * y)) * testfunction(x))
        self.add_residual((partial_t(y) - (B * x - x**2 * y)) * testfunction(y))


class NormalFormProblem(Problem):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # For the Brusselator this global parameter is B, whose Hopf is at 1+A^2; start just below.
        self.mu = self.define_global_parameter(
            mu=(1.0 + args.A_bruss**2 - 0.2) if args.case == "brusselator" else -0.1)

    def define_problem(self):
        if self.args.case == "brusselator":
            self += BrusselatorODE(self.args.A_bruss) @ "nf"
            self += InitialCondition(x=self.args.A_bruss,
                                     y=self.mu.value / self.args.A_bruss) @ "nf"
        elif self.args.case == "pde":
            self.add_mesh(LineMesh(N=self.args.N, size=1.0))
            self += (NormalFormPDE(self.args.sigma, self.args.m1)
                     + InitialCondition(x=0.0, y=0.0)) @ "domain"
        else:
            self += NormalFormODE(self.args.sigma, self.args.m1, self.args.m2) @ "nf"
            self += InitialCondition(x=0.0, y=0.0) @ "nf"


def to_the_hopf_point(problem):
    """Solve the trivial state, find the Hopf pair, and converge the tracker onto mu=0."""
    problem.solve()
    problem.solve_eigenproblem(4 if problem.args.case == "pde" else 2)
    problem.activate_bifurcation_tracking("mu", "hopf")
    problem.solve()


def run(args):
    problem = NormalFormProblem(args)
    with problem:
        problem.set_output_directory(os.path.join(args.outdir, "out"))
        problem.quiet()
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.set_eigensolver("slepc").use_mumps()
        to_the_hopf_point(problem)

        res = {"nproc": get_mpi_nproc(), "case": args.case, "sigma": args.sigma,
               "m1": args.m1, "m2": args.m2, "distributed": bool(problem.is_distributed()),
               "ndof": int(problem.ndof()), "mu_hopf": float(problem.mu.value)}

        if args.what == "orbit":
            # The dof description has to be taken BEFORE the orbit handler is installed: it is sized
            # by ndof(), which is the augmented count while tracking, and its mesh walk only fills the
            # base entries -- so asking during the sample loop classifies the time-block copies by
            # whatever name index -1 happens to hit.
            types, names = problem.get_dof_description()
            is_x = numpy.array([names[t].endswith("/x") for t in types])
            # ...and trimmed to the BASE dofs: the Hopf tracker is still installed here, so the
            # description covers [u | Phi | Psi | p | Omega]. The base block comes first.
            is_x = is_x[:problem.assembly_handler_pt().get_base_ndof()]
            with problem.switch_to_hopf_orbit(eps=args.eps, NT=args.NT, order=3,
                                              do_solve=False) as orbit:
                mu = float(problem.mu.value)
                centre = numpy.array(problem.get_current_dofs()[0][:2]) if args.case == "brusselator" \
                    else None

                def sample_radii():
                    out = []
                    for _ in orbit.iterate_over_samples(N=args.nsample):
                        d = orbit._current_base_dofs()
                        if args.case == "pde":
                            out.append(float(numpy.hypot(d[is_x].mean(), d[~is_x].mean())))
                        elif centre is not None:
                            out.append(float(numpy.linalg.norm(d[:2] - centre)))
                        else:
                            out.append(float(numpy.hypot(d[0], d[1])))
                    return numpy.array(out)

                # The guess amplitude is 2*eps*al*|Re(exp(i w t) q)|; if al is wrong the Newton solve
                # still lands on the true cycle, so guess-vs-solved isolates al.
                guess = sample_radii()
                problem.solve()
                res["radius_guess"] = float(guess.mean())
                radii = []
                for _ in orbit.iterate_over_samples(N=args.nsample):
                    # Not get_current_dofs()[0][:nbase]: under --distribute the augmented rows are
                    # interleaved per rank, which is what _current_base_dofs() is for.
                    d = orbit._current_base_dofs()
                    if args.case == "pde":
                        # Uniform state: every x dof carries the same value, likewise every y dof.
                        radii.append(float(numpy.hypot(d[is_x].mean(), d[~is_x].mean())))
                    else:
                        radii.append(float(numpy.hypot(d[0], d[1])) if centre is None
                                     else float(numpy.linalg.norm(d[:2] - centre)))
                radii = numpy.array(radii)
                res.update({"mu_orbit": mu, "eps": args.eps,
                            "T": float(orbit.get_T(dimensional=False)),
                            "T_exact": 2 * numpy.pi * args.m1 / _W,
                            "radius_mean": float(radii.mean()),
                            "radius_rel_spread": float(radii.ptp() / radii.mean()),
                            "radius_exact": (float(numpy.sqrt(-mu / args.sigma))
                                             if args.case != "brusselator" else float("nan")),
                            "guess_vs_solved": float(abs(res["radius_guess"] - radii.mean())
                                                     / radii.mean()),
                            "supercritical": bool(orbit.starts_supercritically())})
            return res

        # what == "coeff": the coefficient itself, from the state the tracker converged onto.
        from pyoomph.generic.bifurcation_tools import get_hopf_lyapunov_coefficient
        omega = problem.get_last_eigenvalues()[0].imag
        q = numpy.array(problem.assembly_handler_pt().get_nicely_rotated_eigenfunction())
        if omega < 0:
            omega, q = -omega, numpy.conj(q)
        param = problem._bifurcation_tracking_parameter_name
        problem.deactivate_bifurcation_tracking()
        problem.timestepper.make_steady()
        ga, dlam, al, qR, qI = get_hopf_lyapunov_coefficient(problem, param, omega=omega, q=q)
        res.update({"omega": float(omega), "ga": float(ga), "dlam": int(dlam), "al": float(al)})
        return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--case", default="ode", choices=["ode", "pde", "brusselator"])
    p.add_argument("--A-bruss", dest="A_bruss", type=float, default=1.0)
    p.add_argument("--what", default="coeff", choices=["coeff", "orbit"])
    p.add_argument("--sigma", type=float, default=-1.0)
    p.add_argument("--m1", type=float, default=1.0)
    p.add_argument("--m2", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--NT", type=int, default=48)
    p.add_argument("--nsample", type=int, default=32)
    p.add_argument("--N", type=int, default=20)
    # parse_known_args, not parse_args: pyoomph reads its own flags (--distribute) off sys.argv.
    args, _ = p.parse_known_args()
    try:
        res = run(args)
    except Exception as e:
        res = {"error": str(e), "traceback": traceback.format_exc()}
    if get_mpi_rank() == 0:
        print("PYOOMPH_HOPF_RESULT " + json.dumps(res), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
