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

# Worker for tests/test_mpi_facet_fields.py -- launched under `mpirun ...`, with and without
# `--distribute`.
#
# Everything here lives or dies by one invariant: an interior facet is enumerated by exactly ONE of the
# two elements sharing it (the "near" side), on exactly one rank. Distributed, the two elements can sit
# on different ranks, and the near-side choice is made independently on each of them. If the two ever
# disagree, the facet flux is either counted twice or not at all -- and neither shows up as a failure:
# Newton converges, to the solution of a different PDE.
#
# The reported quantities are chosen so that cannot pass:
#
#   * `obs_meas` is the total measure of the skeleton, summed over non-halo facet elements and
#     MPI-reduced. A duplicated facet inflates it by that facet's length, a dropped one deflates it --
#     both by an amount far outside round-off, and both independently of the solution.
#   * `obs_uerr2` with a LINEAR manufactured solution. Interior-penalty DG is consistent, so the exact
#     linear solution lies in the "D1" space and is reproduced to machine precision -- but only if every
#     facet contributes exactly once: the flux term -avg(grad u).jump(v)n does NOT vanish at the exact
#     solution, so dropping or doubling a facet moves the answer.
#   * moments of u for the non-linear (sin) case, against a serial reference computed in-process from
#     this same module.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc, get_mpi_sum


class _FacetObservables(Equations):
    """Integral functions on the interior-facet skeleton, with no unknowns of their own.

    `meas` is the whole point: it certifies the ENUMERATION of the skeleton independently of anything
    the solution does, because it does not involve the solution at all."""

    def define_residuals(self):
        u = var("u")
        n = var("normal")
        self.add_integral_function("meas", 1 * self.get_dx())
        self.add_integral_function("jump2", jump(u) ** 2 * self.get_dx())
        self.add_integral_function("avgu", avg(u) * self.get_dx())
        # The ORIENTATION, which `meas` cannot see. The facet normal points out of the near-side
        # element, so a facet enumerated from the other side contributes with the opposite sign -- and
        # `jump()`, which is near minus far, flips with it. A rank that picks the other near side
        # therefore moves these by twice that facet's contribution.
        self.add_integral_function("nx", n[0] * self.get_dx())
        self.add_integral_function("ny", n[1] * self.get_dx())


class _FacetTrace(Equations):
    """An UNKNOWN on the skeleton: the L2 projection of the bulk trace onto a facet field.

    This is the half that needs the halo scheme. The facet's dofs live in the facet element's own
    internal Data, and a facet shared by two ranks exists on both - so without an owner they would be
    numbered twice and the single-valued trace would become two independent copies. A linear bulk field
    has a linear trace, which "DL" represents exactly, so the projection error is machine-zero and any
    mis-wiring shows up as a real number rather than as "small"."""

    def __init__(self, space="DL"):
        super().__init__()
        self.space = space

    def define_fields(self):
        self.define_scalar_field("p", self.space)

    def define_residuals(self):
        p, pt = var_and_test("p")
        u = avg(var("u"))
        self.add_residual(weak(p - u, pt))
        self.add_integral_function("perr2", (p - u) ** 2 * self.get_dx())
        self.add_integral_function("perr1", (p - u) * self.get_dx())
        self.add_integral_function("psum", p * self.get_dx())


class DGPoisson(Problem):
    """Interior-penalty DG Poisson on the unit square with weakly imposed Dirichlet data.

    `exact="linear"` puts the manufactured solution inside the discrete space, so the discrete answer is
    the exact one and every error below is a defect rather than a discretisation error."""

    def __init__(self, N=8, space="D1", tris=False, exact="linear", facet_space=None, adapt=0):
        super().__init__()
        self.N, self.space, self.tris, self.exact_kind = N, space, tris, exact
        # None: skeleton residuals only (the mode that already worked distributed). "DL"/"D0": a facet
        # UNKNOWN, which is what the halo scheme exists for.
        self.facet_space, self.adapt = facet_space, adapt

    def _exact(self):
        x, y = var("coordinate_x"), var("coordinate_y")
        if self.exact_kind == "linear":
            return 1 + 2 * x + 3 * y, 0
        return sin(pi * x) * sin(pi * y), 2 * pi ** 2 * sin(pi * x) * sin(pi * y)

    def define_problem(self):
        self += RectangularQuadMesh(name="domain", N=self.N,
                                    split_in_tris="left" if self.tris else False)
        exact, source = self._exact()
        eqs = PoissonEquation(space=self.space, source=source)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=exact) @ b
        feqs = _FacetObservables()
        if self.facet_space is not None:
            feqs += _FacetTrace(self.facet_space)
        eqs += feqs @ "_internal_facets_"
        if self.adapt:
            from pyoomph.equations.generic import SpatialErrorEstimator
            eqs += SpatialErrorEstimator(u=1)
        # Partition-independent bulk quantities. Moments rather than a single integral: a distributed
        # dof vector has no partition-independent ordering and cannot be compared entry by entry, and a
        # single integral is a weak statement about a field.
        x, y = var("coordinate_x"), var("coordinate_y")
        u = var("u")
        eqs += IntegralObservables(uerr2=(u - exact) ** 2, u1=u, u2=u ** 2,
                                   mx=x * u, my=y * u,
                                   gu=dot(grad(u), grad(u)))
        self += eqs @ "domain"
        # A nodal Dx facet space cannot be carried through an adaptation; "DL"/"D0" can.
        self.max_refinement_level = self.adapt
        self.initial_adaption_steps = 0


def _facet_element_count(p):
    """The number of interior facets the assembly actually sees, globally.

    Distributed, the mesh is partitioned and a facet belongs to the rank on whose share of the mesh its
    near-side element is non-halo, so the global count is the MPI sum over non-halo facet elements.
    Replicated, there are no halo flags at all -- every rank holds the whole skeleton and the element
    LOOP is what is split -- so the local count already is the global one and summing it would just
    report nproc times the answer."""
    mesh = p.get_mesh("domain/_internal_facets_")
    n = sum(1 for e in range(mesh.nelement()) if not mesh.element_pt(e).is_halo())
    return int(get_mpi_sum(n)) if p.is_distributed() else int(n)


def solve_case(N=8, space="D1", tris=False, exact="linear", outdir=None, facet_space=None,
               adapt=0, state=None):
    with DGPoisson(N=N, space=space, tris=tris, exact=exact, facet_space=facet_space,
                   adapt=adapt) as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        if state == "load":
            # The whole state comes back from the file, the mesh refinement included, so this rank's
            # share of the skeleton is whatever the loaded mesh implies and the halo scheme has to be
            # rebuilt for it. Nothing is adapted afterwards: the point is to reproduce the SAVED state,
            # which was reached by adapting, on a possibly different partition.
            p.load_state("state.dump", relative_to_output=True)
            adapt = 0
        p.solve()
        if adapt:
            # One error-driven adaptation, then solve again. Distributed, this is where the skeleton is
            # destroyed and rebuilt with a different set of facets on each rank, so the halo scheme (and
            # the facet values that go with it) has to be rebuilt with it. The threshold is tightened so
            # that something is actually refined: with the default the estimator reports "not enough
            # benefit" on this problem and the test would prove nothing.
            p.max_permitted_error = 1e-7
            p.min_permitted_error = 1e-9
            p.solve(spatial_adapt=1)
        if state == "save":
            p.save_state("state.dump", relative_to_output=True)
        res = {
            "ndof": int(p.ndof()),
            "newton_steps": len(list(p.get_last_residual_convergence())),
            "maxres": float(numpy.max(numpy.abs(numpy.asarray(p.get_residuals())))),
            "n_facet_elements": _facet_element_count(p),
        }
        # The expected interior-facet count, from the mesh topology rather than from the assembly. On a
        # distributed mesh this is a per-rank number, so it is only compared against itself serially;
        # what is cross-checked in MPI is n_facet_elements.
        summary = p.get_mesh("domain").facet_adjacency_summary()
        res["n_interior_facets_local"] = int(summary[2])
        for name, val in p.get_mesh("domain").evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        for name, val in p.get_mesh("domain/_internal_facets_").evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--space", default="D1")
    ap.add_argument("--tris", type=int, default=0)
    ap.add_argument("--exact", default="linear", choices=["linear", "sin"])
    ap.add_argument("--facet-space", default=None)
    ap.add_argument("--adapt", type=int, default=0)
    ap.add_argument("--state", default=None, choices=["save", "load"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        payload.update(solve_case(N=args.size, space=args.space, tris=bool(args.tris),
                                  exact=args.exact, outdir=args.outdir,
                                  facet_space=args.facet_space, adapt=args.adapt, state=args.state))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    # Straight to the real stdout: pyoomph's default MPI console mode MUTES every rank but 0, and a test
    # that only ever sees rank 0 cannot notice a rank disagreeing.
    sys.__stdout__.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
