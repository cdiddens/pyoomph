#!/usr/bin/env python3
"""Leading Lyapunov exponents of a LINEAR problem, in every MPI regime.

The problem is deliberately linear: u_t = D u_xx + a*u on (0,L) with u=0 at both ends. A linear
system's Lyapunov exponents ARE the real parts of its Jacobian eigenvalues, so the answer is known
in closed form -- lambda_m = a - D*(m*pi/L)^2 -- and the test does not have to trust a chaotic
trajectory to be reproducible across a dof renumbering. It also exercises the part that only k>1
reaches: the Gram-Schmidt sweep, which is pure reductions and is where a distributed run would
disagree first.

What is measured is the time-discrete exponent, which differs from lambda_m at O(dt). That is the
same in every regime, so the regimes are compared to round-off and the analytic value only loosely.
"""
import argparse
import json
import os
import sys

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.utils.lyapunov import LyapunovExponentCalculator

D_VAL = 1.0
A_VAL = 2.0
L_VAL = 1.0


def analytic_exponents(k):
    """The continuous spectrum: lambda_m = a - D (m pi / L)^2."""
    return [A_VAL - D_VAL * ((m * numpy.pi / L_VAL) ** 2) for m in range(1, k + 1)]


def discrete_exponents(k, dt):
    """What the calculator actually measures, which is not quite ``analytic_exponents``.

    The perturbations are advanced by implicit Euler (BDF1), so one step multiplies mode m by
    1/(1 - lambda_m dt) and the measured exponent is log of that over dt. At dt=0.01 that is 1% below
    lambda_1 and 15% below lambda_2 -- the steeper the mode, the worse, which is why comparing the
    second exponent against the continuous value needs a 20% tolerance and says almost nothing. What
    is left over against THIS prediction is the spatial discretisation (a C2 FEM eigenvalue of
    -D u_xx sits slightly above D (m pi / L)^2), measured at ~1.5% for both modes on N=80.
    """
    return [float(numpy.log(1.0 / (1.0 - lam * dt)) / dt) for lam in analytic_exponents(k)]


class LinearDecayEquations(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_weak(partial_t(u) - A_VAL * u, ut).add_weak(D_VAL * grad(u), grad(ut))


class LinearDecayProblem(Problem):
    def __init__(self, N=80, k=2, seed=0, prerelax=0.3):
        super().__init__()
        self.N, self.k, self.seed, self.prerelax = N, k, seed, prerelax

    def define_problem(self):
        self += LineMesh(N=self.N, size=L_VAL, name="domain")
        eqs = LinearDecayEquations()
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        eqs += InitialCondition(u=0)
        self += eqs @ "domain"
        # relative_to_output=False and an absolute name would collide between ranks; the calculator
        # only writes on rank 0, so the default output directory is fine.
        # prerelaxation_time is load-bearing, not tidiness. The seeded initial basis is drawn
        # globally and sliced by GLOBAL ROW, and --distribute renumbers the dofs, so the three
        # regimes genuinely start from different vectors in physical space. The exponents do not
        # care asymptotically -- any basis aligns with the leading subspace -- but Lambdas is a
        # running mean from Tstart2, so a transient averaged in from t=0 is never forgotten and
        # decays only like 1/T. Measured without it: serial, replicated and np=2 agreed to 10
        # digits while np=3 was 0.3% away, purely from the starting basis.
        self.lyap = LyapunovExponentCalculator(k=self.k, random_seed=self.seed,
                                               prerelaxation_time=self.prerelax,
                                               store_as_eigenvectors=False)
        self += self.lyap


def run(N=80, k=2, nsteps=100, dt=0.01, seed=0, prerelax=0.3, outdir=None):
    with LinearDecayProblem(N=N, k=k, seed=seed, prerelax=prerelax) as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.solve()                       # u == 0; the trajectory is the trivial one
        # Fixed step, no output: the exponents are accumulated per step, so a varying step would
        # make the regimes comparable only if they happened to choose the same ones.
        p.run(endtime=nsteps * dt, outstep=False, maxstep=dt, startstep=dt, temporal_error=None)
        t = p.get_current_time(dimensional=True, as_float=True)
        Tdiff = t - p.lyap._Tstart2
        exponents = [float(x) for x in (p.lyap.Lambdas / Tdiff)]
        # The perturbation basis itself, as one global number per vector that does not depend on the
        # dof numbering: a renumbering permutes the entries, so the vector cannot be compared
        # entry by entry across --distribute, but its norm and its inner products can.
        layout = p.lyap._layout
        from pyoomph.generic.distributed_la import DistVector
        grams = []
        for i in range(k):
            vi = DistVector(p.lyap.B[:, i], layout)
            grams.append([float(vi.dot(DistVector(p.lyap.B[:, j], layout))) for j in range(k)])
        return {
            "ndof": int(p.ndof()),
            "ndof_global": int(layout.n),
            "distributed": bool(p.is_distributed()),
            "nrow_local": int(layout.nrow_local),
            "exponents": exponents,
            "gram": grams,
            "analytic": analytic_exponents(k),
            "discrete": discrete_exponents(k, dt),
            "dt": float(dt),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--nsteps", type=int, default=100)
    ap.add_argument("--prerelax", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()
    payload = {"rank": int(get_mpi_rank()), "nproc": int(get_mpi_nproc())}
    try:
        payload.update(run(N=args.size, k=args.k, nsteps=args.nsteps, seed=args.seed,
                           prerelax=args.prerelax, outdir=args.outdir))
    except BaseException as e:
        import traceback
        payload["error"] = repr(e)
        payload["traceback"] = traceback.format_exc()
    # sys.__stdout__: pyoomph's MPI console mutes stdout on every rank but 0, so a plain print()
    # would report from rank 0 alone and the between-rank assertions would pass vacuously.
    sys.__stdout__.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
