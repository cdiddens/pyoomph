#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================


from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.utils import DeterministicRandomField  # for the random initial condition


class DampedKuramotoSivashinskyEquation(Equations):
    def __init__(self, gamma=0.0, delta=0.0, space="C2"):
        super(DampedKuramotoSivashinskyEquation, self).__init__()
        self.gamma, self.delta, self.space = gamma, delta, space

    def define_fields(self):
        self.define_scalar_field("h", "C2")  # h
        self.define_scalar_field("lapl_h", "C2")  # projection of div(grad(h))

    def define_residuals(self):
        h, v = var_and_test("h")
        lapl_h, w = var_and_test("lapl_h")
        self.add_residual(weak(partial_t(h) + self.gamma * h + self.delta * h ** 2 - dot(grad(h), grad(h)), v))
        self.add_residual(-weak(grad(h) + grad(lapl_h), grad(v)))
        self.add_residual(weak(lapl_h, w) + weak(grad(h), grad(w)))


class KSEProblem(Problem):
    def __init__(self):
        super(KSEProblem, self).__init__()
        self.L = 50  # domain length
        self.N = 40  # number of elements
        self.gamma,self.delta=0.24,0.05 # parameters
        self.random_amplitude=0.01 # Initial random initial condition amplitude

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N, size=self.L))

        eqs = DampedKuramotoSivashinskyEquation(gamma=self.gamma, delta=self.delta)
        eqs += MeshFileOutput()
        # Adding periodic boundaries: nodes at "bottom" will be merged by the nodes at top (found by applying offset to the position)
        eqs += PeriodicBC("top", offset=[0, self.L]) @ "bottom"
        # Same for the left<->right connection
        eqs += PeriodicBC("right", offset=[self.L, 0]) @ "left"

        # Create a deterministic random field. We must pass the corners of the domain
        # All random values will be pre-allocated so that successive evaluations of
        # the functions at the same point yield the same value
        h_init = DeterministicRandomField(min_x=[0, 0], max_x=[self.L, self.L], amplitude=self.random_amplitude)
        x, y = var(["coordinate_x", "coordinate_y"])
        eqs += InitialCondition(h=h_init(x, y))

        self.add_equations(eqs @ "domain")  # adding the equation


if __name__ == "__main__":
    with KSEProblem() as problem:
        problem.run(2000, outstep=True, startstep=0.1, temporal_error=1, maxstep=50)
