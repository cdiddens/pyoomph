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

"""Check that the macOS Accelerate backend reuses its symbolic factorization.

Run on an arm64 Mac (see .github/workflows/test_mac_accelerate.yml). Exits non-zero with a specific
message on the first thing that is wrong.

What is actually being checked, in order of how much it matters:

  1. The results are RIGHT. Accelerate with reuse must agree with Accelerate without reuse, and with
     an independent solver (scipy's SuperLU), on the converged solution. A reuse bug that returned stale factors would still converge to
     something, so agreement is checked against an independent solver, not just against itself.
  2. The reuse actually HAPPENED. num_symbolic_factorizations() must stop growing while
     num_numeric_refactorizations() climbs. Timings are not used as evidence: a path that silently
     fell back would look similar on a small problem, and that exact false negative already cost an
     afternoon once in this project (see dev_docs/structural_assembly.md, Phase 2).
  3. It is invalidated when it must be. Re-solving on an unchanged pattern must take no new symbolic
     factorization, and renumbering the equations must force one.
"""

import sys

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


FAILURES = []

# An INDEPENDENT solver to check the answer against, so a reuse bug that returned stale factors cannot
# hide by being self-consistent. "superlu" is scipy's SuperLU wrapper -- note the linear solver is
# registered under that name, not "scipy" (which is the eigen solver). MKL Pardiso is deliberately not
# used: it is unavailable on arm64 Macs, which is where this script runs.
_REFERENCE_SOLVER = "superlu"


def check(condition, message):
    print(("  ok   " if condition else "  FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


class _Bratu(Problem):
    """-laplace(u) + dt(u) = lam*exp(u). Nonlinear, so the Newton loop takes several steps and the
    Jacobian values genuinely change between them while the pattern does not -- which is the situation
    the reuse exists for."""

    def __init__(self, N=12):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        lam = self.get_global_parameter("lam")
        lam.value = 3.0

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(lam * exp(u), v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


def solve_with(solver_name, reuse):
    with _Bratu() as p:
        p.quiet()
        p.set_linear_solver(solver_name)
        p.initialise()
        la = p.get_la_solver()
        if hasattr(la, "reuse_symbolic_factorization"):
            la.reuse_symbolic_factorization = reuse
        p.solve()
        dofs = numpy.asarray(p.get_history_dofs(0)).copy()
        residual = float(numpy.abs(numpy.asarray(p.get_residuals())).max())
        counts = None
        if hasattr(la, "solver") and hasattr(la.solver, "num_symbolic_factorizations"):
            counts = (la.solver.num_symbolic_factorizations(), la.solver.num_numeric_refactorizations())
        return dofs, residual, counts


def main():
    print("pyoomph macOS Accelerate symbolic-factorization reuse check", flush=True)

    try:
        import pyoomph.solvers.accelerate  # noqa: F401
    except Exception as e:
        print("FATAL: the Accelerate solver could not be imported: %r" % (e,))
        return 2

    print("\n[1] correctness: accelerate(reuse) vs accelerate(no reuse) vs %s" % _REFERENCE_SOLVER, flush=True)
    ref_dofs, ref_res, _ = solve_with(_REFERENCE_SOLVER, False)
    off_dofs, off_res, off_counts = solve_with("accelerate", False)
    on_dofs, on_res, on_counts = solve_with("accelerate", True)

    check(ref_res < 1e-8, "%s reference converged (|R| = %.3e)" % (_REFERENCE_SOLVER, ref_res))
    check(off_res < 1e-8, "accelerate without reuse converged (|R| = %.3e)" % off_res)
    check(on_res < 1e-8, "accelerate with reuse converged (|R| = %.3e)" % on_res)
    d_self = float(numpy.abs(on_dofs - off_dofs).max()) if on_dofs.shape == off_dofs.shape else float("inf")
    d_ref = float(numpy.abs(on_dofs - ref_dofs).max()) if on_dofs.shape == ref_dofs.shape else float("inf")
    check(d_self < 1e-9, "reuse agrees with no-reuse (max|du| = %.3e)" % d_self)
    check(d_ref < 1e-7, "reuse agrees with the independent %s solve (max|du| = %.3e)" % (_REFERENCE_SOLVER, d_ref))

    print("\n[2] the reuse actually happened", flush=True)
    if on_counts is None:
        check(False, "the solver exposes num_symbolic_factorizations() (it does not -- old build?)")
        return 1
    sym_on, num_on = on_counts
    sym_off, num_off = off_counts if off_counts else (0, 0)
    print("      with reuse   : %d symbolic, %d numeric-only" % (sym_on, num_on), flush=True)
    print("      without reuse: %d symbolic, %d numeric-only" % (sym_off, num_off), flush=True)
    check(sym_on == 1, "exactly one symbolic factorization was taken (got %d)" % sym_on)
    check(num_on >= 1, "at least one numeric-only refactorization was taken (got %d)" % num_on)
    check(num_off == 0, "no numeric-only refactorization when reuse is off (got %d)" % num_off)
    check(sym_off > 1, "without reuse every step refactorizes symbolically (got %d)" % sym_off)

    print("\n[3] the reuse is given up when the pattern changes", flush=True)
    with _Bratu() as p:
        p.quiet()
        p.set_linear_solver("accelerate")
        p.initialise()
        la = p.get_la_solver()
        lam = p.get_global_parameter("lam")
        p.solve()
        sym_before = la.solver.num_symbolic_factorizations()
        # Nudge the parameter before each re-solve. Re-solving an already-converged problem can return
        # straight after the residual check without factorizing at all, which would make both checks
        # below pass or fail for reasons that have nothing to do with the reuse.
        lam.value = 3.2
        p.solve()                                    # same pattern: must NOT refactorize symbolically
        sym_same = la.solver.num_symbolic_factorizations()
        num_same = la.solver.num_numeric_refactorizations()
        check(sym_same == sym_before,
              "re-solving on an unchanged pattern takes no new symbolic factorization (%d -> %d)"
              % (sym_before, sym_same))
        check(num_same > 0, "the re-solve actually factorized something (%d numeric-only)" % num_same)
        # Renumbering the equations kills the pattern. Done explicitly rather than through a mesh
        # adaptation, which might legitimately decide to refine nothing and would then fail this check
        # for a reason that has nothing to do with the solver.
        p.assign_eqn_numbers()
        lam.value = 3.4
        p.solve()
        sym_after = la.solver.num_symbolic_factorizations()
        check(sym_after > sym_same,
              "a new symbolic factorization is taken after renumbering (%d -> %d)" % (sym_same, sym_after))

    print("")
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("All Accelerate reuse checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
