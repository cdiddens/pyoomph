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

# Worker for tests/test_mpi_deflation.py -- launched under `mpirun ...`, with and without
# --distribute. Runs deflated solving and deflated continuation and prints one PYOOMPH_MPI_RESULT
# line per rank.
#
# Deflation does NOT go through the custom-assembler pipeline (which still refuses MPI): it scales
# the assembled residual by a scalar and rescales the Newton step, both on top of the ordinary
# assembly. See dev_docs/deflation.md. What that leaves to get wrong under MPI is exactly two
# things, and both are what this worker is for:
#
#   - the two reductions. ||u-w_i|| is summed over the DOF layout (a no-op replicated, an allreduce
#     over owned rows distributed), and grad(log M).dU over the SOLVER's row layout (an allreduce as
#     soon as nproc>1, even replicated, because oomph row-splits the linear algebra there). Getting
#     either wrong changes the deflation factor and hence which solutions are found.
#   - rank-independent control flow. The search is a sequence of random perturbations and Newton
#     solves whose success decides what happens next; a rank that draws a different perturbation or
#     disagrees about convergence takes a different branch and the job deadlocks in the next
#     collective rather than returning a wrong answer.
#
# Everything reported is numbering-independent -- solution counts and mesh integrals -- because
# distribute() renumbers the dofs, so a dof vector cannot be compared across the two.

import argparse
import json
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class PitchforkPDE(Equations):
    """u_t = laplace(u) + lam*u - u^3 with u=0 on the boundary.

    u=0 solves it for every lam; above the first eigenvalue of -laplace (2*pi^2 on the unit square)
    a symmetric pair +-u1 branches off. So for lam in between the first and second eigenvalue there
    are exactly THREE solutions, which is what deflation has to find -- and the pair is the part a
    wrong deflation factor loses, because it is only reachable by deflating u=0 away.
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(self.lam * u - u ** 3, v))


class PitchforkProblem(Problem):
    def __init__(self, N=8, lam0=25.0):
        super().__init__()
        self.N, self.lam0 = N, lam0

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        self.lam = self.define_global_parameter(lam=self.lam0)
        eqs = PitchforkPDE(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        # Partition-independent by construction: evaluate_integral_function skips halo elements and
        # MPI_Allreduce-sums, so these are the same number however the mesh was cut up.
        eqs += IntegralObservables(uint=var("u"), usqr=var("u") ** 2)
        self += eqs @ "domain"


def _observables(p):
    return {k: float(v) for k, v in p.get_mesh("domain").evaluate_all_observables().items()}


def _sorted_solutions(sols):
    """The found solution set in a comparable order.

    Sorted on the observables, not on the dofs: the dof numbering differs between a serial and a
    distributed run, and the ORDER in which deflation finds the branches depends on which random
    perturbation converged first, which is not something worth pinning down.
    """
    key = lambda s: (round(s["uint"], 9), round(s["usqr"], 9))
    return sorted(sols, key=key)


def deflated_solve_case(N=8, lam0=25.0, outdir=None):
    """All solutions at one parameter value: u=0 and the symmetric pair.

    The pair is the real assertion: +u1 and -u1 have the SAME |u| and are told apart only by the
    signed integral, and neither is reachable without the deflation factor doing its job.
    """
    prob = PitchforkProblem(N=N, lam0=lam0)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        p.initialise()
        sols = []
        for _dofs in p.iterate_over_multiple_solutions_by_deflation(
                deflation_alpha=0.1, deflation_p=2, perturbation_amplitude=0.5,
                # Three tries and this seed find all THREE solutions; with two, the random walk
                # happens to reach only one of the symmetric pair. Which is not a defect -- the
                # search is random -- but a run that found the pair is a far better certificate,
                # since +u1 and -u1 have the same |u| and are told apart only by the signed integral.
                # Eigenperturbation AND random tries. The eigenvector is a field, so perturbing
                # along it means the same thing however the mesh was partitioned; a random dof-index
                # vector does not, because --distribute renumbers and global index i is then a
                # different node. With both, every configuration below finds all three solutions;
                # with random tries alone, np=3 --distribute explores a different sequence and
                # reaches only the trivial one. See dev_docs/deflation.md on reproducibility.
                use_eigenperturbation=True, num_random_tries=3, random_seed=0):
            sols.append(_observables(p))
        return {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "nsolutions": len(sols),
            "solutions": _sorted_solutions(sols),
        }


def deflated_continuation_case(N=6, outdir=None):
    """Deflated continuation over lam, straight through the pitchfork.

    Starts below the bifurcation, where u=0 is the only solution, and ends above it, where there are
    three: the branch bookkeeping (which branches survive, which are new) is decided from Newton
    successes and dof distances, i.e. exactly the rank-dependent control flow that has to agree.
    """
    prob = PitchforkProblem(N=N, lam0=15.0)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        per_branch = {}
        for branch_index, lamvalue, _dofs in p.deflated_continuation(
                lam=numpy.linspace(15.0, 30.0, 7), perturbation_amplitude=0.5,
                num_random_tries=2, random_seed=1234):
            per_branch.setdefault(branch_index, []).append((float(lamvalue), _observables(p)))
        last = [{"lam": v[-1][0], **v[-1][1]} for v in per_branch.values()]
        return {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "nbranches": len(per_branch),
            "branch_lengths": sorted(len(v) for v in per_branch.values()),
            "final": _sorted_solutions(last),
        }


_CASES = {"solve": deflated_solve_case, "continuation": deflated_continuation_case}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--case", default="solve", choices=sorted(_CASES))
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "%s_N%d" % (args.case, args.size)}
    try:
        payload.update(_CASES[args.case](N=args.size, outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
