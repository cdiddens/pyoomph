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

"""Does the stripe scan find a Hopf pair that shift-invert misses, and merge it in cleanly?

Three ODEs whose eigenvalues are known exactly:

  * two weakly damped real modes at lambda = -0.01 and -0.02,
  * an oscillator with lambda = -0.5 +- i*omega, a pair sitting far up the imaginary axis.

pyoomph asks SLEPc for the eigenvalues whose REAL part is nearest the target, so a 2-eigenvalue solve
returns the two real ones and never the pair - the pair's real part is further from 0. That is exactly
the situation the stripe is for. (An earlier version of this probe put the pair at the LARGEST real
part, where a 2-eigenvalue solve returns it and there is nothing to demonstrate.)

Needs a COMPLEX PETSc; the caller must put its lib on PYTHONPATH.
"""
import sys

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits

OMEGA = 8.0
EPS = -0.5          # the pair's real part: further from 0 than the two real modes
SLOW = (-0.01, -0.02)


class Eqs(ODEEquations):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def define_fields(self):
        self.define_ode_variable("a", "b", "x", "y")

    def define_residuals(self):
        a, at = var_and_test("a")
        b, bt = var_and_test("b")
        x, xt = var_and_test("x")
        y, yt = var_and_test("y")
        # Two damped modes; mu shifts them so there is something to continue in.
        self.add_weak(partial_t(a) - SLOW[0]*a - 0*self.mu*a, at)
        self.add_weak(partial_t(b) - SLOW[1]*b - 0*self.mu*b, bt)
        # An oscillator: d/dt (x,y) = eps*(x,y) + omega*(-y,x), eigenvalues eps +- i*omega.
        self.add_weak(partial_t(x) - EPS*x + OMEGA*y, xt)
        self.add_weak(partial_t(y) - EPS*y - OMEGA*x, yt)


class Prob(Problem):
    def define_problem(self):
        eqs = Eqs(self.get_global_parameter("mu"))
        eqs += InitialCondition(a=0, b=0, x=0, y=0)
        self += eqs @ "ode"


# Run under a __main__ guard: this file lives in tests/ and the suite is invoked as
# `python -m pytest *.py` (see citools/nightly_develop.sh), which hands pytest every .py in
# the directory by name -- and a file named on the command line is imported regardless of
# python_files. Without the guard the whole problem below is built and solved at COLLECTION
# time, and pyoomph's own argv parsing then reads pytest's first filename argument as the
# output directory, so it tries to mkdir over a source file and collection dies. Every other
# worker in this directory is guarded the same way.
def main():
    with Prob() as problem:
        problem.set_output_directory(sys.argv[1])
        problem.set_eigensolver("slepc")
        problem.get_global_parameter("mu").value = 1.0
        problem.quiet()

        gui = BifurcationGUI(problem, "mu")
        gui.neigen = 2                      # deliberately too few to see everything
        c = gui.controller
        c.view = _FixedViewLimits(xlim=(-5, 40), ylim=(-5, 5))
        c.start(0.5)

        p = c.current_point
        print("expected eigenvalues: {:+g}, {:+g}, {:+g}+-{:g}i".format(SLOW[0], SLOW[1], EPS, OMEGA))
        print("shift-invert (neigen=2):")
        for v in p.eig_values:
            print("   {:+.4f}{:+.4f}i".format(v.real, v.imag))
        assert not any(abs(v.imag) > 1 for v in p.eig_values), \
            "the pair should NOT be in a 2-eigenvalue shift-invert result"
        print("   -> the pair is invisible to it, as intended")
        n_before = len(p.eig_values)

        c.stripe_re, c.stripe_im = 1.0, OMEGA + 2.0
        c.stripe_merge = True
        print("\nstripe |Re|<{:g}, |Im|<{:g}, merging ...".format(c.stripe_re, c.stripe_im))
        assert c.scan_stripe(p), "the scan failed"
        print("after the merge:")
        for v in p.eig_values:
            print("   {:+.4f}{:+.4f}i".format(v.real, v.imag))

        pair = [v for v in p.eig_values if abs(v.imag) > 1]
        assert len(pair) == 2, "the pair must be there now, got {:d}".format(len(pair))
        assert abs(abs(pair[0].imag) - OMEGA) < 1e-6, pair
        assert abs(pair[0].real - EPS) < 1e-6, pair
        assert len(p.eig_values) == n_before + 2, \
            "merging must add exactly the two it found and duplicate nothing: {:d} -> {:d}".format(
                n_before, len(p.eig_values))
        assert p.unstable_count == 0, p.unstable_count
        print("\nleading {:+.4f}, unstable {:d}".format(p.eig_value_Re, p.unstable_count))

        # The solver must be back to shift-invert, or every later solve becomes a region scan.
        assert problem.get_eigen_solver().eigenvalue_region is None
        print("STRIPE OK")
        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
