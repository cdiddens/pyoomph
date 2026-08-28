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

"""Assembly / solve benchmark for the structural-assembly work (dev_docs/structural_assembly.md).

Not a pytest module: it is a measurement tool, run by hand when a change should move a number.

    python tests/benchmarks/bench_assembly.py                       # all cases, serial
    python tests/benchmarks/bench_assembly.py --case ns3d --size 8
    python tests/benchmarks/bench_assembly.py --newton --solver pardiso
    mpirun -n 4 python tests/benchmarks/bench_assembly.py --case ns2d --newton \\
           --solver petsc_mumps --distribute

Three things are measured, because they are what the work has to trade off against each other:

  * the ELEMENTAL half of an assembly (problem._benchmark_elemental_assembly), i.e. evaluating each
    element's residual/Jacobian and throwing it away. Subtracting it from a full assembly isolates the
    scatter + CSR compression, which is the part a precomputed sparsity pattern would remove.
  * the cost of keeping structural zeros -- more nonzeros to scatter and to factorise, against a
    pattern that is stable and hence reusable.
  * the end-to-end Newton solve, which is the only number that actually matters.

Under MPI every rank prints its own line; wall-clock is dominated by the slowest, so compare the max.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import NavierStokesFreeSurface
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.solid import DeformableSolidEquations, GeneralizedHookeanSolidConstitutiveLaw
from pyoomph.meshes.simplemeshes import CuboidBrickMesh, RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


# ale2d/solid3d are the moving-mesh (coordinates_as_dofs) arms. They exist because every other case
# here has a static mesh, and the position-dof Jacobian columns - which are the largest expressions the
# generator emits - therefore never appear at all. ale2d is the ordinary production shape (free surface
# on a smoothed mesh, first derivatives only); solid3d is the shape dev_docs/code_generation.md
# repeatedly names as the largest generated code in the project.
CASES = ["ns3d", "ns2d", "coupled", "poisson3d", "ale2d", "solid3d"]
DEFAULT_SIZE = {"ns3d": 8, "ns2d": 60, "coupled": 40, "poisson3d": 8, "ale2d": 24, "solid3d": 6}

# Cases measured at their initial condition rather than after a steady solve. solid3d starts from a
# 90-degrees-per-metre torsion, which is the deformed configuration we WANT to assemble at, and which a
# steady Newton solve cannot unwind (it hits the 10-iteration limit). Solving is only ever here to
# avoid measuring at a trivial state; for this case the initial state is the non-trivial one.
NO_STEADY_SOLVE = {"solid3d"}


class BenchProblem(Problem):
    def __init__(self, case: str, N: int):
        super().__init__()
        self.case = case
        self.N = N

    def define_problem(self):
        if self.case == "ns3d":
            self.add_mesh(CuboidBrickMesh(N=self.N))
            eqs = NavierStokesEquations(dynamic_viscosity=0.02, mass_density=1)
            for b in ["left", "right", "front", "back", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0, velocity_z=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left/front"
        elif self.case == "ale2d":
            self.add_mesh(RectangularQuadMesh(N=self.N))
            eqs = NavierStokesEquations(dynamic_viscosity=0.02, mass_density=1)
            eqs += LaplaceSmoothedMesh()
            eqs += NavierStokesFreeSurface(surface_tension=1) @ "top"
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_x=True, mesh_y=True) @ b
            eqs += DirichletBC(pressure=0) @ "bottom/left"
        elif self.case == "solid3d":
            # The torsioned hyperelastic beam of docs/.../ale/solid/solid_oscillations.py, kept
            # dimensionless here: the benchmark only needs the weak form's shape, and the unit
            # prescan of dev_docs/code_generation.md 2.2 would otherwise sit in the middle of it.
            self.add_mesh(CuboidBrickMesh(size=[1.0, 0.05, 0.05], N=[self.N * 4, self.N, self.N]))
            claw = GeneralizedHookeanSolidConstitutiveLaw(E=1.0, nu=0.38)
            eqs = DeformableSolidEquations(constitutive_law=claw, coordinate_space="C2", mass_density=1)
            X = var("lagrangian")
            theta = 1.5 * X[0]
            eqs += InitialCondition(mesh_y=X[1] * cos(theta) + X[2] * sin(theta),
                                    mesh_z=-X[1] * sin(theta) + X[2] * cos(theta))
            eqs += DirichletBC(mesh_x=0, mesh_y=True, mesh_z=True) @ "left"
        elif self.case == "poisson3d":
            self.add_mesh(CuboidBrickMesh(N=self.N))
            eqs = PoissonEquation(name="u", source=1, space="C2")
            eqs += DirichletBC(u=0) @ "left"
        else:
            self.add_mesh(RectangularQuadMesh(N=self.N))
            eqs = NavierStokesEquations(dynamic_viscosity=0.02, mass_density=1)
            if self.case == "coupled":
                eqs += AdvectionDiffusionEquations(fieldnames="c", diffusivity=0.01, space="C2")
                eqs += DirichletBC(c=1) @ "left"
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


def _time(fn, repeats):
    fn()  # warm up: first call pays for lazily allocated scratch buffers
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _say(msg):
    if get_mpi_nproc() > 1:
        print("[rank %d] %s" % (get_mpi_rank(), msg), flush=True)
    else:
        print(msg, flush=True)


def bench_assembly(case, N, repeats):
    """Assembly breakdown with the structural pattern off and on."""
    with BenchProblem(case, N) as p:
        p.set_c_compiler("system").optimize_for_max_speed()
        p.quiet()
        p.initialise()
        if case not in NO_STEADY_SOLVE:
            p.solve()  # measure at a realistic state, not at U=0

        t_el_res = p._benchmark_elemental_assembly(repeats, False, False)
        t_el_jac = p._benchmark_elemental_assembly(repeats, True, False)

        _say("%s N=%d ndof=%d" % (case, N, p.ndof()))
        _say("  elemental only : residual %.4f s   jacobian %.4f s" % (t_el_res, t_el_jac))
        base = None
        for flag in (False, True):
            p.keep_structural_zeros = flag
            nnz = p.assemble_jacobian(with_residual=False).nnz
            t_res = _time(p.get_residuals, repeats)
            t_asm = _time(p.assemble_jacobian, repeats)
            t_eig = _time(lambda: p.assemble_eigenproblem_matrices(0.0), repeats)
            if base is None:
                base = (t_asm, nnz)
            _say("  structural=%d   : residual %.4f s   res+jac %.4f s   res+jac+mass %.4f s"
                 "   nnz %d (%+.1f%%, time %+.1f%%)   scatter share %.0f%%"
                 % (flag, t_res, t_asm, t_eig, nnz,
                    100 * (nnz / base[1] - 1), 100 * (t_asm / base[0] - 1),
                    100 * max(0.0, 1 - t_el_jac / t_asm)))


def bench_newton(case, N, solver, repeats):
    """End-to-end Newton solve with the structural pattern (and hence solver symbolic reuse) off and on."""
    for flag in (False, True):
        with BenchProblem(case, N) as p:
            p.set_c_compiler("system").optimize_for_max_speed()
            p.quiet()
            p.set_linear_solver(solver)
            p.initialise()
            p.keep_structural_zeros = flag
            if case in NO_STEADY_SOLVE:
                p.assemble_jacobian()  # warm: JIT compile, without a solve this case cannot do
            else:
                p.solve()  # warm: JIT compile + first factorisation
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                p.solve()
                times.append(time.perf_counter() - t0)
            # Partition-independent oracles: get_residuals() is gathered to full length and the integral
            # observables are MPI_Allreduce-summed, so these must agree across ranks AND across flags.
            R = numpy.abs(numpy.asarray(p.get_residuals())).max()
            _say("%s N=%d ndof=%d solver=%s structural=%d : %.4f s/solve  |R|=%.3e  sid=%d"
                 % (case, N, p.ndof(), solver, flag, min(times), R, p.jacobian_structure_id))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=CASES + ["all"], default="all")
    ap.add_argument("--size", type=int, default=None, help="mesh resolution N (per case default)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--newton", action="store_true", help="benchmark full Newton solves instead")
    ap.add_argument("--solver", default="pardiso", help="linear solver for --newton")
    args, _ = ap.parse_known_args()  # --distribute is consumed by pyoomph itself

    cases = CASES if args.case == "all" else [args.case]
    for case in cases:
        N = args.size if args.size is not None else DEFAULT_SIZE[case]
        if args.newton:
            bench_newton(case, N, args.solver, max(args.repeats // 2, 1))
        else:
            bench_assembly(case, N, args.repeats)


if __name__ == "__main__":
    main()
