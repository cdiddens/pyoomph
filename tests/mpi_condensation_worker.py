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

# Worker for tests/test_mpi_static_condensation.py -- launched under `mpirun ... --distribute`.
#
# Solves a Crouzeix-Raviart lid-driven cavity with static condensation on or off and prints one
# PYOOMPH_MPI_RESULT line per rank. It is a separate worker from mpi_structural_worker.py because the
# discretisation has to be CR on triangles (the condensation this feature was designed around needs an
# element-local bubble velocity and a discontinuous pressure), and because what is measured is different:
# integral observables of the RETAINED field would not notice a broken reconstruction of the ELIMINATED
# dofs at all, so the eliminated element-internal values are reported explicitly.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class _InterfacePressureCoupling(Equations):
    """A boundary field that reads the bulk DL pressure, so the interface element adopts the adjacent
    bulk elements' pressure Data and writes both the (lam, p) and the (p, lam) blocks.

    Distributed this is the interesting case twice over: the condensed dofs' OWN rows are written by an
    element that does not own them (requirement 2 of dev_docs/static_condensation.md section 2), and the
    interface element may sit on a different rank than the bulk element whose Data it adopts."""

    def define_fields(self):
        self.define_scalar_field("lam", "C2")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        p, ptest = var_and_test("pressure", domain="..")
        self.add_residual(weak(lam - 0.1 * p, lamtest) + weak(0.05 * lam, ptest))


class CRCavity(Problem):
    """Lid-driven cavity with CR elements on triangles: C2TB velocities whose cell-interior bubble node
    belongs to exactly one element, and a DL pressure (one constant plus the gradient modes per element).

    `condense` states the classical CR elimination in the equation tree; `condense_all_pressure` states
    the structurally singular one (the whole pressure, constant mode included), which exists to check
    that the refusal is collective rather than one-sided."""

    def __init__(self, N=12, Re=100, condense=True, interface=False, condense_all_pressure=False,
                 dg=False):
        super().__init__()
        self.N, self.Re, self.condense = N, Re, condense
        self.interface = interface
        self.condense_all_pressure = condense_all_pressure
        self.dg = dg

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N, split_in_tris="left"))
        ns = NavierStokesEquations(dynamic_viscosity=1, mass_density=self.Re, mode="CR")
        eqs = ns + ns.create_pressure_fixation(value=0)
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        if self.condense:
            if self.condense_all_pressure:
                # Structurally singular on purpose: the continuity equation contains no pressure, so
                # the constant mode has no equation determining it.
                eqs += StaticCondensation(pressure="all")
            else:
                eqs += StaticCondensation(velocity="bubble", pressure=[1, 2])
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        if self.interface:
            eqs += _InterfacePressureCoupling() @ "top"
        # Partition-independent observables: evaluate_integral_function skips halo elements and
        # MPI_Allreduce-sums, so these certify the FIELD and not one rank's slice of it.
        #
        # Several weighted MOMENTS rather than one integral, because a distributed dof vector has no
        # partition-independent ordering and cannot be compared entry by entry against a serial one.
        # A single integral of a field is a weak statement about it; a set of moments (and the gradient
        # energy, which is what the condensed bubble modes actually carry) pins it down.
        x, y = var("coordinate_x"), var("coordinate_y")
        eqs += IntegralObservables(ke=dot(var("velocity"), var("velocity")),
                                   vx=var("velocity_x"), vy=var("velocity_y"),
                                   p2=var("pressure") ** 2,
                                   mx=x * var("velocity_x"), my=y * var("velocity_y"),
                                   px=x * var("pressure"),
                                   gu=contract(grad(var("velocity")), grad(var("velocity"))))
        self.add_equations(eqs @ "domain")


def _internal_value_checksums(p):
    """Partition-independent scalars built from the ELEMENT-INTERNAL (DL pressure) values.

    The integral observables above are dominated by the retained dofs; a reconstruction that silently
    left the eliminated ones at their previous values would barely move them. The DL gradient modes are
    condensed, so these two sums are the direct evidence that the reconstruction ran and ran correctly.
    Summed over non-halo elements and MPI-reduced, so the answer does not depend on the partition --
    except on a REPLICATED run, where there are no halo flags at all and every rank already holds the
    whole mesh, so reducing would report nproc times the answer."""
    from pyoomph.generic.mpi import get_mpi_sum
    mesh = p.get_mesh("domain")
    s1 = 0.0
    s2 = 0.0
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        if el.is_halo():
            continue
        for k in range(el.ninternal_data()):
            d = el.internal_data_pt(k)
            for v in range(d.nvalue()):
                x = d.value(v)
                s1 += x
                s2 += x * x
    if not p.is_distributed():
        return float(s1), float(s2)
    return float(get_mpi_sum(s1)), float(get_mpi_sum(s2))


def solve_case(N, condense, outdir=None, transient=0, interface=False, Re=100):
    prob = CRCavity(N=N, Re=Re, condense=condense, interface=interface)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        steps = []
        if transient:
            for _ in range(transient):
                p.solve(timestep=0.05)
                steps.append(len(list(p.get_last_residual_convergence())))
        else:
            p.solve()
            steps.append(len(list(p.get_last_residual_convergence())))
            # A second solve from the converged state: the first can never reuse a plan, so only a
            # repeat exercises "pattern unchanged -> keep the plan".
            p.solve()
            steps.append(len(list(p.get_last_residual_convergence())))
        s1, s2 = _internal_value_checksums(p)
        stats = p._get_static_condensation_stats()
        res = {
            "ndof": int(p.ndof()),
            "newton_steps": steps,
            # get_residuals() is gathered to full length, so this is identical on every rank.
            "maxres": float(numpy.max(numpy.abs(numpy.asarray(p.get_residuals())))),
            "condensed": bool(p._last_jacobian_was_condensed()),
            "plan_rebuilds": int(stats["plan_rebuilds"]),
            "n_selected": int(stats["n_selected"]),
            "internal_sum": s1,
            "internal_sqsum": s2,
        }
        # The cross-rank half of the feature, per rank. All zero would mean a "distributed" plan that
        # did nothing a serial one would not -- i.e. an equivalence test proving nothing about MPI.
        for k in ("n_components_owned", "n_components_remote", "n_foreign_E",
                  "n_operator_sends", "n_operator_recvs"):
            res[k] = int(stats.get(k, 0))
        for name, val in p.get_mesh("domain").evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


class DGPoisson(Problem):
    """Interior-penalty DG Poisson: the facet terms couple every element to its neighbours, so the whole
    selected field is ONE connected component spanning the mesh -- and therefore spanning the ranks.

    That is the distributed-only refusal: no rank holds the block to be inverted in full. Serving it
    would be a distributed dense solve per component, so it is refused, collectively."""

    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_problem(self):
        from pyoomph.equations.poisson import PoissonEquation
        self.add_mesh(RectangularQuadMesh(N=self.N, split_in_tris="left"))
        eqs = PoissonEquation(space="D1", source=1)
        eqs += StaticCondensation("u")
        self.add_equations(eqs @ "domain")


def straddle_case(N, outdir=None):
    """A component split across ranks must be refused on EVERY rank, with the same message."""
    prob = DGPoisson(N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        # Raised beyond the whole-mesh component size, so it is the CROSS-RANK guard that speaks and
        # not the size guard that would otherwise fire first.
        p.static_condensation_max_component_size = 100000
        try:
            p.solve()
        except Exception as e:
            return {"refused": True, "message": str(e)[:400], "ndof": int(p.ndof())}
        return {"refused": False, "message": "", "ndof": int(p.ndof())}


class HDGPoisson(Problem):
    """Hybridised DG Poisson: the bulk unknowns of an element couple to that element and to the trace on
    its own facets, never to a neighbour's bulk unknowns.

    This is the selection a REPLICATED run can serve. Its condensed dofs are element-INTERNAL, so
    oomph-lib numbers each element's nine of them consecutively, and the row cut points can be moved off
    the blocks (Problem::condensation_row_cuts). A Crouzeix-Raviart selection cannot: it mixes the nodal
    bubble velocity with the internal pressure modes, and every nodal value is numbered before any
    internal one, so the two halves of a block sit hundreds of equations apart.

    Reuses the tutorial's equations rather than restating them, so this cannot drift away from the
    documented formulation."""

    def __init__(self, N=8, condense=True, space="D2", facet_space="D2"):
        super().__init__()
        self.N, self.condense = N, condense
        self.space, self.facet_space = space, facet_space

    def define_problem(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "docs", "source", "tutorial", "dg"))
        try:
            from hdg_poisson import HDGPoissonEquations, HDGDirichletBC
        finally:
            sys.path.pop(0)
        x, y = var("coordinate_x"), var("coordinate_y")
        exact = sin(pi * x) * sin(pi * y)
        source = 2 * pi ** 2 * exact
        tau = 10 * self.N
        self.max_refinement_level = 0
        self.initial_adaption_steps = 0
        self += RectangularQuadMesh(N=self.N)
        # The tutorial declares the trace and its facet residual inside HDGPoissonEquations itself
        # (at_internal_facets=True / add_interior_facet_residual), so this is the whole formulation.
        eqs = HDGPoissonEquations(source, tau, self.space, self.facet_space)
        eqs += HDGDirichletBC(exact, tau) @ ["left", "right", "top", "bottom"]
        eqs += IntegralObservables(err2=(var("u") - exact) ** 2, u1=var("u"), u2=var("u") ** 2,
                                   mx=x * var("u"), gu=dot(grad(var("u")), grad(var("u"))))
        if self.condense:
            eqs += StaticCondensation("u")
        self += eqs @ "domain"


def hdg_case(N, condense, outdir=None, space="D2", facet_space="D2"):
    prob = HDGPoisson(N=N, condense=condense, space=space, facet_space=facet_space)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.solve()
        steps = [len(list(p.get_last_residual_convergence()))]
        p.solve()
        steps.append(len(list(p.get_last_residual_convergence())))
        s1, s2 = _internal_value_checksums(p)
        stats = p._get_static_condensation_stats()
        res = {
            "ndof": int(p.ndof()),
            "newton_steps": steps,
            "maxres": float(numpy.max(numpy.abs(numpy.asarray(p.get_residuals())))),
            "condensed": bool(p._last_jacobian_was_condensed()),
            "plan_rebuilds": int(stats["plan_rebuilds"]),
            "n_selected": int(stats["n_selected"]),
            "internal_sum": s1,
            "internal_sqsum": s2,
            # Empty serially and on a distributed run; nproc+1 ascending rows on a replicated one, and
            # then the direct evidence that the cuts were moved off the element blocks.
            "row_cuts": [int(c) for c in p._condensation_row_cuts()],
        }
        for name, val in p.get_mesh("domain").evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


def refusal_case(N, outdir=None):
    """The structurally singular selection: every rank must throw the SAME refusal, and none may hang.

    This is the collective-vote test. The guard is decided from each rank's own owned block, so on a
    partition where one rank happens to hold no offending dof, a rank-local throw would leave it in the
    next collective for ever -- which is a hung job, not a failed one."""
    prob = CRCavity(N=N, Re=100, condense=True, condense_all_pressure=True)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        try:
            p.solve()
        except Exception as e:
            return {"refused": True, "message": str(e)[:400], "ndof": int(p.ndof())}
        return {"refused": False, "message": "", "ndof": int(p.ndof())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--condense", type=int, default=1)
    ap.add_argument("--transient", type=int, default=0)
    ap.add_argument("--interface", type=int, default=0)
    ap.add_argument("--space", default="D2")
    ap.add_argument("--facet-space", default="D2")
    ap.add_argument("--mode", default="solve", choices=["solve", "refuse", "straddle", "hdg"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        if args.mode == "refuse":
            payload.update(refusal_case(args.size, outdir=args.outdir))
        elif args.mode == "straddle":
            payload.update(straddle_case(args.size, outdir=args.outdir))
        elif args.mode == "hdg":
            payload.update(hdg_case(args.size, bool(args.condense), outdir=args.outdir,
                                    space=args.space, facet_space=args.facet_space))
        else:
            payload.update(solve_case(args.size, bool(args.condense), outdir=args.outdir,
                                      transient=args.transient, interface=bool(args.interface)))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    # Straight to the real stdout, not through print(): pyoomph's default MPI console mode
    # ("condensed") wraps sys.stdout and MUTES every rank but 0, so a plain print() here would report
    # from rank 0 alone -- and a test that only ever sees rank 0 cannot notice a rank disagreeing.
    sys.__stdout__.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
