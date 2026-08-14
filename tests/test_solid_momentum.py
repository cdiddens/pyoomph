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

# A freely floating elastic solid with inertia must not pick up momentum from nowhere: the
# stress term of the weak form vanishes for a constant test function, so the center of mass
# stays put to solver precision.
#
# This once failed whenever initialise() was called before run(): node construction leaves
# arbitrary values in the deeper position-history rows (the Newmark2 velocity/acceleration and
# adaptive predictor slots of the MultiTimeStepper), and the initialise-time IC application
# runs while dt is still unset, so its Newmark repair of those slots cannot work - and the
# second application at the start of run() did not recover from that. The garbage acted as an
# initial velocity, and the body translated at a constant, dt-independent speed of the order of
# one body length per time unit. set_initial_condition now starts every dof impulsively before
# applying the explicit initial conditions, which is what this test pins down.

import numpy
from pyoomph import Problem, GmshTemplate, WeakContribution, ODEFileOutput
from pyoomph.expressions import var, testfunction
from pyoomph.equations.generic import ElementSpace
from pyoomph.equations.solid import DeformableSolidEquations, GeneralizedHookeanSolidConstitutiveLaw


class _CurvedBlobMesh(GmshTemplate):
    """Unstructured triangle mesh of a fish-like blob: two splines meeting in sharp cusps.
    The asymmetry in x makes the historic drift large (~0.7 body lengths over the first half
    time unit), so the test discriminates sharply."""

    def define_geometry(self):
        self.default_resolution = 0.1
        shape = lambda x: -10.55 * x**5 + 21.87 * x**4 - 15.73 * x**3 + 3.19 * x**2 + 1.22 * x
        ptail = self.point(0, 0)
        phead = self.point(1, 0)
        upper, lower = [ptail], [ptail]
        for l in numpy.linspace(1 / 30, 1, 30, endpoint=False):
            upper.append(self.point(l, shape(l) * 0.2))
            lower.append(self.point(l, -shape(l) * 0.2))
        upper.append(phead)
        lower.append(phead)
        self.spline(upper, name="upper")
        self.spline(lower, name="lower")
        self.plane_surface("upper", "lower", name="blob")


class _FreeSolidProblem(Problem):
    def define_problem(self):
        self += _CurvedBlobMesh()
        law = GeneralizedHookeanSolidConstitutiveLaw(E=1000, nu=0.4)
        eqs = DeformableSolidEquations(law, mass_density=1) + ElementSpace("C2")

        # Lagrangian-frame center of mass as global observables: conserved for a free body
        X, Xtest = self.add_global_dof("X")
        Y, Ytest = self.add_global_dof("Y")
        eqs += WeakContribution(X - var("coordinate_x"), Xtest, lagrangian=True)
        eqs += WeakContribution(Y - var("coordinate_y"), Ytest, lagrangian=True)
        self += ODEFileOutput() @ "globals"
        self += eqs @ "blob"


def test_free_solid_conserves_momentum(tmp_path):
    with _FreeSolidProblem() as problem:
        problem.set_output_directory(str(tmp_path / "free_solid"))
        # initialise() before run() is essential to reproduce the historic drift: it applies
        # the initial conditions while dt is still unset, and the second application at the
        # start of run() did not recover from that.
        problem.initialise()
        problem.run(0.5, outstep=0.1)
        odes = problem.get_ode("globals")
        X = odes.get_value("X", as_float=True)
        Y = odes.get_value("Y", as_float=True)
        problem.run(1.0, outstep=0.1)
        assert abs(odes.get_value("X", as_float=True) - X) < 1e-9
        assert abs(odes.get_value("Y", as_float=True) - Y) < 1e-9
