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

# Worker for tests/test_mpi_eigen_adapt.py -- launched under `mpirun ...`, with and without
# --distribute. Adapts the mesh to an EIGENFUNCTION (Problem.refine_eigenfunction) and prints one
# PYOOMPH_MPI_RESULT line per rank.
#
# refine_eigenfunction() used to be refused outright on a distributed problem: adapt() carries the
# eigenfunction across the adaptation in history levels 3 and 4, and the history dof accessors were
# unsupported there. They work now, and this is what holds the rest of the path.
#
# Three things are measured, for three different reasons:
#
#   eigfunc_before / eigfunc_carry   The eigenfunction integrals before the adaptation, and again
#       after the eigenvector has been round-tripped through the history levels the adaptation
#       carries it in. With ONE adaptation and no unrefinement these must agree to round-off,
#       because refining an element leaves the FE function it interpolates exactly unchanged. This
#       is the one oracle that does not care WHICH elements were refined, and it is the mechanism
#       the old refusal was about.
#
#   fingerprint / ndof / nelement    The refined mesh itself. Identical between serial and a
#       REPLICATED `mpirun` (every rank computes every patch). Between serial and --distribute it
#       need NOT be: oomph-lib's distributed Z2 recovery neglects the flux contributions of patches
#       that can only be assembled from vertex nodes on another process (the long comment in
#       LagrZ2ErrorEstimator::setup_patches, src/lagr_error_estimator.cpp), so elements near the
#       threshold can be decided differently. That is a property of the estimator, not of the
#       eigenfunction -- which is why --driver base exists: it drives the SAME estimator from the
#       base state alone, and shows the same effect.
#
#   the eigenvalue after the re-solve  Physics on whatever mesh came out, so it is compared with a
#       mesh-difference tolerance rather than to round-off.
#
# The dof numbering is NOT comparable between a serial and a distributed run -- distribute()
# renumbers so each rank owns a contiguous block -- so everything compared across the two is
# numbering-independent: integrals over the mesh, and element centroids.

import argparse
import hashlib
import json
import traceback

import numpy

from pyoomph import Problem, DirichletBC, Equations
from pyoomph.expressions import var, var_and_test, grad, weak, partial_t, exp, axisymmetric
from pyoomph.equations.generic import SpatialErrorEstimator, AxisymmetryBC, IntegralObservables
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc, get_mpi_sum, get_mpi_world_comm
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


def _observables():
    """Integrals of the eigenfunction, all partition-independent (evaluate_all_observables skips
    halo elements and MPI_Allreduce-sums).

    ``usqr`` alone is NOT a placement oracle: SLEPc B-orthonormalises against the mass matrix, which
    for this problem IS the L2 product, so int u^2 comes out at 1 whatever the eigenfunction looks
    like. The first moments are what say WHERE on the mesh it sits, and they are the numbers a
    misplaced (mis-scattered) eigenvector moves.
    """
    x = var("coordinate")
    return IntegralObservables(usqr=var("u") ** 2, usqr_x=var("u") ** 2 * x[0],
                               usqr_y=var("u") ** 2 * x[1])


class BumpDiffusion(Equations):
    """-div(D grad u) - f, with a first-order time derivative as the mass matrix.

    D has a localised bump and f a localised source, deliberately at DIFFERENT and asymmetric
    places: the base-state error field and the eigenfunction error field then peak in different
    regions, so a run that silently used only one of the two is visible in the refined mesh.
    """

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        x = var("coordinate")
        D = 1 + 50 * exp(-((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2) / 0.004)
        f = exp(-((x[0] - 0.72) ** 2 + (x[1] - 0.25) ** 2) / 0.004)
        self.add_residual(weak(partial_t(u), v) + weak(D * grad(u), grad(v)) - weak(f, v))


class _Base(Problem):
    def __init__(self, N=8):
        super().__init__()
        self.N = N
        # The eigen-driven adaptation is what is under test, so the mesh must arrive at it
        # unrefined: an initial adaption loop would already have refined the base-state features.
        self.initial_adaption_steps = 0
        self.max_refinement_level = 3
        self.max_permitted_error = 0.002
        self.min_permitted_error = 0.0002
        # Filled by _adapt below, so the carried eigenfunction can be measured before the re-solve.
        self.carried_eigenvector = None
        self.n_refined = 0
        self.n_unrefined = 0

    def _adapt(self):
        nref, nunref = super()._adapt()
        self.n_refined += nref
        self.n_unrefined += nunref
        if self._adapted_eigeninfo is not None:
            self.carried_eigenvector = numpy.array(self._adapted_eigeninfo[0]).copy()
        return nref, nunref


class CartesianProblem(_Base):
    """Plane diffusion, Dirichlet left/right, Neumann top/bottom -- a REAL eigenproblem."""

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        eqs = BumpDiffusion()
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        eqs += SpatialErrorEstimator(u=1, for_which="both")
        eqs += _observables()
        self += eqs @ "domain"


class AzimuthalProblem(_Base):
    """Axisymmetric, with AxisymmetryBC on the axis -- the COMPLEX eigenproblem and, at m=1, the
    forced-zero-dof matrix manipulator (a scalar field is unconstrained at r=0 for the base state
    but must vanish for |m|=1)."""

    def define_problem(self):
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = BumpDiffusion()
        eqs += DirichletBC(u=0) @ "right"
        eqs += DirichletBC(u=0) @ "top"
        eqs += DirichletBC(u=0) @ "bottom"
        eqs += AxisymmetryBC(verbose=False) @ "left"
        eqs += SpatialErrorEstimator(u=1, for_which="both")
        eqs += _observables()
        self += eqs @ "domain"


_PROBLEMS = {"cartesian": CartesianProblem, "azimuthal": AzimuthalProblem}


def _mesh_fingerprint(p, name="domain"):
    """A hash of the set of non-halo element centroids, agreed over all ranks.

    The centroids are used ONLY as a fingerprint of the refinement pattern -- nothing is matched by
    position. Quantised to 1e-7 of the unit box; every coordinate here is a dyadic rational, so the
    quantisation is exact rather than a tolerance.
    """
    mesh = p.get_mesh(name)
    keys = []
    nlocal = 0
    for e in mesh.elements():
        if e.is_halo():
            continue
        nlocal += 1
        nn = e.nnode()
        c = tuple(int(round(1e7 * sum(e.node_pt(k).x(d) for k in range(nn)) / nn)) for d in range(2))
        keys.append(c)
    # Only a DISTRIBUTED mesh is partitioned. Without --distribute every rank holds the whole mesh
    # and nothing is a halo, so gathering would count every element once per rank.
    comm = get_mpi_world_comm()
    if p.is_distributed() and get_mpi_nproc() > 1 and comm is not None:
        allkeys = [k for part in comm.allgather(keys) for k in part]
        nlocal = int(get_mpi_sum(nlocal))
    else:
        allkeys = keys
    allkeys.sort()
    return hashlib.sha1(repr(allkeys).encode()).hexdigest()[:16], nlocal


def _eigenfunction_integral(p, vector):
    """The eigenfunction observables of ``vector``, with the dofs restored afterwards.

    mode="abs" because an eigenvector is only fixed up to a complex phase, which serial and
    distributed runs do pick differently; |v| is phase-invariant, so what is compared is the shape.
    """
    backup = p._last_eigenvectors
    try:
        p._last_eigenvectors = numpy.array([vector])
        dofs, pinned = p.set_eigenfunction_as_dofs(0, mode="abs")
        try:
            obs = p.get_mesh("domain").evaluate_all_observables()
            return {k: float(v) for k, v in obs.items()}
        finally:
            p.set_all_values_at_current_time(dofs, pinned, False)
    finally:
        p._last_eigenvectors = backup


def _solve_eigen(p, azimuthal_m):
    if azimuthal_m is not None:
        p.solve_eigenproblem(1, azimuthal_m=azimuthal_m, quiet=True)
    else:
        p.solve_eigenproblem(1, quiet=True)


def solve_case(case="cartesian", N=8, numadapt=2, azimuthal_m=None, outdir=None, driver="eigen"):
    prob = _PROBLEMS[case](N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        if azimuthal_m is not None:
            p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        p.initialise()
        p.solve()
        _solve_eigen(p, azimuthal_m)

        res = {"distributed": bool(p.is_distributed()), "ndof_before": int(p.ndof())}
        fp, nel = _mesh_fingerprint(p)
        res["fingerprint_before"], res["nelement_before"] = fp, nel
        # The eigenfunction BEFORE any adaptation. Pure refinement leaves an FE function untouched,
        # so these integrals must come back unchanged from the history levels the adaptation carried
        # the eigenfunction through -- whichever elements were refined, and on whatever partition.
        res["eigfunc_before"] = _eigenfunction_integral(p, p.get_last_eigenvectors()[0])
        if driver == "base":
            # Control arm: the SAME estimator and the same number of adaptations, but driven by the
            # base state alone. Whatever this arm does distributed-vs-serial is pre-existing
            # behaviour of the estimator, not something the eigenfunction introduced. It carries
            # nothing through the history levels, so it reports no eigfunc_carry.
            for _ in range(numadapt):
                p.adapt()
                p.solve()
            _solve_eigen(p, azimuthal_m)
            ev = p.get_last_eigenvalues()[0]
            evec = p.get_last_eigenvectors()[0]
        else:
            ev, evec = p.refine_eigenfunction(numadapt=numadapt, use_startvector=True)
        fp, nel = _mesh_fingerprint(p)
        res.update({
            "ndof": int(p.ndof()), "nelement": nel, "fingerprint": fp,
            "n_refined": int(p.n_refined), "n_unrefined": int(p.n_unrefined),
            "eval_re": float(numpy.real(ev)), "eval_im": float(numpy.imag(ev)),
            "evect_len": int(len(evec)),
            "eigfunc": _eigenfunction_integral(p, evec),
        })
        if p.carried_eigenvector is not None:
            # The eigenvector as it came back out of history levels 3/4, i.e. the carry-across
            # itself, before the re-solve had a chance to repair it.
            res["eigfunc_carry"] = _eigenfunction_integral(p, p.carried_eigenvector)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cartesian", choices=sorted(_PROBLEMS))
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--numadapt", type=int, default=2)
    ap.add_argument("--azimuthal-m", type=int, default=-1)
    ap.add_argument("--driver", default="eigen", choices=["eigen", "base"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()
    azi = None if args.azimuthal_m < 0 else args.azimuthal_m
    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "%s_N%d_m%s_%s" % (args.case, args.size, str(azi), args.driver)}
    try:
        payload.update(solve_case(case=args.case, N=args.size, numadapt=args.numadapt,
                                  azimuthal_m=azi, outdir=args.outdir, driver=args.driver))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-3000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
