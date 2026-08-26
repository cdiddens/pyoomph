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

# Worker for tests/test_floquet_multipliers.py, and usable under mpirun for the replicated case.
#
# Both systems here have an orbit that is known in closed form, so the guess handed to the orbit
# handler is already the answer and the Newton solve neither wanders nor needs a Hopf tracker (and
# therefore needs no complex PETSc): what is under test is the multiplier computation alone.
#
#   "sl"  - the Stuart-Landau oscillator in Cartesian form. Limit cycle x=cos(t), y=sin(t), T=2*pi,
#           and the linearization decouples into the phase direction (multiplier 1) and the radial
#           one, r'=r(1-r^2) linearizing to -2, hence exp(-2*T) = exp(-4*pi).
#   "dae" - the same, plus a purely algebraic w=x*y. The mass matrix then has an exactly zero row,
#           which is the case a shooting formulation could not take. Gauss collocation is not
#           stiffly accurate, so the algebraic direction comes out at (-1)**(number of intervals)
#           rather than at 0; see pyoomph/generic/floquet.py.
#   "stiff" - the same oscillator plus an upper-triangular chain z_k' = -a_k z_k + c z_{k+1}. Being
#           triangular, the chain contributes one multiplier per a_k whatever c is, so the spectrum
#           spans 25 orders of magnitude by construction; c makes the monodromy non-normal, which is
#           what stops the plain product of transfer matrices from resolving the bottom of it. With
#           --reference the SAME transfer matrices are multiplied again in 120-digit arithmetic, so
#           the comparison isolates the product's own roundoff from the time-discretization error.

#   "pde"  - the same Stuart-Landau kinetics applied pointwise on a 1D mesh, plus diffusion. The
#           spatially UNIFORM state is still exactly u=cos(t), v=sin(t) with T=2*pi, because
#           diffusion annihilates a uniform field -- so the exact orbit is available on a mesh with
#           as many degrees of freedom as wanted, and in particular on a DISTRIBUTED one. This is the
#           case tests/test_mpi_floquet.py drives under mpirun --distribute.

_STIFF_RATES = [2.0, 4.0, 8.0, 16.0, 32.0]
_PDE_DIFFUSIVITY = 0.05

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import Problem, ODEEquations, Equations, InitialCondition, LineMesh
from pyoomph.expressions import var, testfunction, partial_t, var_and_test, weak, grad
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class StuartLandau(ODEEquations):
    """r'=r(1-r^2), phi'=1 in Cartesian coordinates: a limit cycle at r=1 with period 2*pi."""

    def __init__(self, algebraic: bool = False, coupling: float | None = None):
        super().__init__()
        self.algebraic = algebraic
        self.coupling = coupling

    def define_fields(self):
        extra = ["w"] if self.algebraic else []
        if self.coupling is not None:
            extra += ["z%d" % k for k in range(len(_STIFF_RATES))]
        self.define_ode_variable("x", "y", *extra)

    def define_residuals(self):
        x, y = var(["x", "y"])
        r2 = x ** 2 + y ** 2
        self.add_residual((partial_t(x) - (x - y - x * r2)) * testfunction(x))
        self.add_residual((partial_t(y) - (x + y - y * r2)) * testfunction(y))
        if self.algebraic:
            w = var("w")
            self.add_residual((w - x * y) * testfunction(w))  # no partial_t: a zero mass-matrix row
        if self.coupling is not None:
            for k, a in enumerate(_STIFF_RATES):
                z = var("z%d" % k)
                rhs = -a * z
                if k + 1 < len(_STIFF_RATES):
                    rhs = rhs + self.coupling * var("z%d" % (k + 1))
                self.add_residual((partial_t(z) - rhs) * testfunction(z))


class StuartLandauProblem(Problem):
    def __init__(self, algebraic: bool = False, coupling: float | None = None):
        super().__init__()
        self.algebraic = algebraic
        self.coupling = coupling

    def define_problem(self):
        self += StuartLandau(algebraic=self.algebraic, coupling=self.coupling) @ "sl"
        ic = {"x": 1.0, "y": 0.0}
        if self.algebraic:
            ic["w"] = 0.0
        if self.coupling is not None:
            ic.update({"z%d" % k: 0.0 for k in range(len(_STIFF_RATES))})
        self += InitialCondition(**ic) @ "sl"


class StuartLandauPde(Equations):
    """Stuart-Landau kinetics pointwise plus diffusion; the uniform limit cycle is exact."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("v", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        v, vt = var_and_test("v")
        r2 = u ** 2 + v ** 2
        self.add_residual(weak(partial_t(u) - (u - v - u * r2), ut) + weak(_PDE_DIFFUSIVITY * grad(u), grad(ut)))
        self.add_residual(weak(partial_t(v) - (u + v - v * r2), vt) + weak(_PDE_DIFFUSIVITY * grad(v), grad(vt)))


class StuartLandauPdeProblem(Problem):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(LineMesh(N=self.N, size=1.0))
        self += (StuartLandauPde() + InitialCondition(u=1.0, v=0.0)) @ "domain"


def exact_orbit(t, algebraic, coupling):
    x, y = numpy.cos(t), numpy.sin(t)
    state = [x, y]
    if algebraic:
        state.append(x * y)
    if coupling is not None:
        state += [0.0] * len(_STIFF_RATES)
    return state


def high_precision_multipliers(elems):
    """Eigenvalues of the SAME transfer matrices, multiplied in 120-digit arithmetic.

    The reference the two double-precision routes are measured against: it shares their
    discretization exactly, so any difference is the product's own roundoff.
    """
    import mpmath
    mpmath.mp.dps = 120
    Cs = [el.transfer() for el in elems]
    n = Cs[0].shape[0]
    M = mpmath.eye(n)
    for Ci in Cs:
        M = mpmath.matrix([[mpmath.mpf(float(Ci[i, j])) for j in range(n)] for i in range(n)]) * M
    return [complex(z) for z in mpmath.eig(M, left=False, right=False)]


def run_pde(args):
    """The distributed case: an exact uniform orbit on a real mesh."""
    problem = StuartLandauPdeProblem(N=args.N)
    with problem:
        problem.set_output_directory(os.path.join(args.outdir, "out"))
        problem.quiet()
        problem.setup_for_stability_analysis(analytic_hessian=True)
        # No steady solve first: the only steady state of this system is the trivial one.
        problem.initialise()
        T = 2 * numpy.pi
        # get_dof_description() is documented to be merged across ranks and full length under
        # --distribute, and set_current_dofs takes a global vector, so every rank builds the same guess.
        types, names = problem.get_dof_description()
        is_u = numpy.array([names[t].endswith("/u") for t in types])
        states = [numpy.where(is_u, numpy.cos(t), numpy.sin(t))
                  for t in numpy.linspace(0, T, args.NT, endpoint=False)]
        problem.set_current_dofs(states[0])
        res = {"nproc": get_mpi_nproc(), "case": "pde", "N": args.N, "NT": args.NT,
               "distributed": bool(problem.is_distributed())}
        orbit = problem.activate_periodic_orbit_handler(T, states[1:], mode="collocation",
                                                        order=args.order)
        problem.solve()
        res["nbase"] = orbit._get_handler().get_base_ndof()
        res["nT"] = orbit.get_num_time_steps()
        res["ndof"] = int(problem.ndof())
        res["T"] = float(orbit.get_T(dimensional=False))
        F = orbit.get_floquet_multipliers(n=args.n if args.n > 0 else 6,
                                          dense_threshold=args.dense_threshold,
                                          shift_invert=bool(args.shift_invert))
        res["mult_re"] = [float(numpy.real(z)) for z in F]
        res["mult_im"] = [float(numpy.imag(z)) for z in F]
        # Drives set_dofs_to_interpolated_values() and the halo push that follows it, which is the
        # part of the handler that writes base dofs wholesale rather than through an element.
        orbit.update_phase_constraint()
        res["sample_absum"] = [float(numpy.sum(numpy.abs(problem.get_current_dofs()[0])))
                               for _ in orbit.iterate_over_samples(N=8)]
        problem.deactivate_bifurcation_tracking()
        return res


def run(args):
    if args.case == "pde":
        return run_pde(args)
    algebraic = args.case == "dae"
    coupling = args.coupling if args.case == "stiff" else None
    problem = StuartLandauProblem(algebraic=algebraic, coupling=coupling)
    with problem:
        problem.set_output_directory(os.path.join(args.outdir, "out"))
        problem.quiet()
        problem.setup_for_stability_analysis(analytic_hessian=True)
        T = 2 * numpy.pi
        states = [exact_orbit(t, algebraic, coupling) for t in numpy.linspace(0, T, args.NT, endpoint=False)]
        problem.initialise()
        problem.set_current_dofs(states[0])
        res = {"nproc": get_mpi_nproc(), "case": args.case, "mode": args.mode,
               "order": args.order, "NT": args.NT}
        with problem.activate_periodic_orbit_handler(T, states[1:], mode=args.mode,
                                                     order=args.order) as orbit:
            problem.solve()
            res["nbase"] = orbit._get_handler().get_base_ndof()
            res["nT"] = orbit.get_num_time_steps()
            res["n_intervals"] = res["nT"] - 1
            res["T"] = float(orbit.get_T(dimensional=False))
            res["orbit_ndof"] = problem.ndof()

            if args.expect_refusal:
                # central/BDF2 keep no explicit degree of freedom at the end of the period, so there
                # is no wrap-around block to read a multiplier off in either formulation.
                try:
                    orbit.get_floquet_multipliers()
                except RuntimeError as e:
                    res["refused"] = str(e)
                else:
                    res["refused"] = None
                return res

            kwargs = {"method": args.method, "quiet": True}
            if args.method == "eigenproblem":
                kwargs["n"] = args.n if args.n > 0 else res["nbase"]
                kwargs["shift"] = args.shift
            else:
                if args.n > 0:
                    kwargs["n"] = args.n
                kwargs["dense_threshold"] = args.dense_threshold
                kwargs["shift_invert"] = bool(args.shift_invert)
            # A B-spline orbit has no end-of-period degree of freedom, so its multipliers are taken
            # on a collocation SAMPLING of it and the B-spline orbit is put back afterwards. The
            # blocks either side of the call are what says the restore was exact rather than close.
            sampled = not orbit._get_handler().is_floquet_mode()
            res["sampled"] = bool(sampled)
            before, Tbefore = orbit._blocks()
            if args.fail_resolve:
                def _diverge(*a, **kw):
                    raise RuntimeError("pretend the re-solve of the sampling diverged")
                problem.solve = _diverge
                try:
                    orbit.get_floquet_multipliers(**kwargs)
                except RuntimeError as e:
                    res["resolve_failed_with"] = str(e)
                else:
                    res["resolve_failed_with"] = None
                del problem.solve
                after, Tafter = orbit._blocks()
                res["mode_after"] = str(orbit.mode)
                res["nT_after"] = int(orbit.get_num_time_steps())
                res["floquet_mode_after"] = bool(orbit._get_handler().is_floquet_mode())
                res["restore_shape_ok"] = bool(before.shape == after.shape)
                res["restore_max_diff"] = (float(numpy.max(numpy.abs(before - after)))
                                           if before.shape == after.shape else None)
                res["restore_T_diff"] = float(abs(Tafter - Tbefore))
                return res
            F = orbit.get_floquet_multipliers(**kwargs)
            after, Tafter = orbit._blocks()
            res["mode_after"] = str(orbit.mode)
            res["nT_after"] = int(orbit.get_num_time_steps())
            res["floquet_mode_after"] = bool(orbit._get_handler().is_floquet_mode())
            res["restore_shape_ok"] = bool(before.shape == after.shape)
            res["restore_max_diff"] = (float(numpy.max(numpy.abs(before - after)))
                                       if before.shape == after.shape else None)
            res["restore_T_diff"] = float(abs(Tafter - Tbefore))
            res["mult_re"] = [float(numpy.real(z)) for z in F]
            res["mult_im"] = [float(numpy.imag(z)) for z in F]
            ev = problem.get_last_eigenvectors()
            res["eigvec_shape"] = list(numpy.shape(ev))
            # The eigenfunction closes the orbit by construction: its last time block is the first
            # one pushed once round, i.e. lambda times it. Anything wrong in the reconstruction --
            # the batching above all -- shows up here and nowhere else.
            nb, nTb = res["nbase"], res["nT"]
            # Skipped when sampled: the eigenfunctions are laid out over the COLLOCATION sampling,
            # not over this orbit's own time blocks, so nT here is the wrong stride for them.
            if numpy.size(ev) and not sampled:
                first = ev[:, :nb]
                last = ev[:, (nTb - 1) * nb:nTb * nb]
                lam = numpy.array(res["mult_re"]) + 1j * numpy.array(res["mult_im"])
                scale = numpy.maximum(numpy.max(numpy.abs(first), axis=1), 1e-300)
                res["closure_residual"] = float(numpy.max(
                    numpy.abs(last - lam[:, None] * first).max(axis=1) / scale))
            if args.reference and not sampled:
                from pyoomph.generic.floquet import time_elements
                elems = time_elements(orbit._get_handler())
                J = problem.assemble_jacobian(with_residual=False)
                for el in elems:
                    el.factorize(J)
                ref = high_precision_multipliers(elems)
                res["ref_re"] = [float(numpy.real(z)) for z in ref]
                res["ref_im"] = [float(numpy.imag(z)) for z in ref]
        return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--case", default="sl", choices=["sl", "dae", "stiff", "pde"])
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--coupling", type=float, default=1e6)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--mode", default="collocation")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--NT", type=int, default=48)
    p.add_argument("--method", default="condensed")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--dense-threshold", dest="dense_threshold", type=int, default=2000)
    p.add_argument("--shift-invert", dest="shift_invert", type=int, default=1)
    p.add_argument("--expect-refusal", action="store_true")
    # Makes the re-solve of the collocation sampling diverge, to pin that the B-spline orbit comes
    # back anyway - the restore is in a finally precisely so a failed re-solve costs the attempt and
    # not the orbit.
    p.add_argument("--fail-resolve", action="store_true")
    # parse_known_args, not parse_args: pyoomph reads its own flags (--distribute above all) straight
    # off sys.argv, and argparse would reject them here first.
    args, _ = p.parse_known_args()
    try:
        res = run(args)
    except Exception as e:
        res = {"error": str(e), "traceback": traceback.format_exc()}
    if get_mpi_rank() == 0:
        print("PYOOMPH_FLOQUET_RESULT " + json.dumps(res), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
