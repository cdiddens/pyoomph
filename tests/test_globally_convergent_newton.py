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

# One solve with globally_convergent_newton=True used to arm a heap overflow for the rest of the
# session.
#
# Problem::newton_solve() switches the linear solver's gradient computation ON when the method is
# in use and nothing ever switched it off again, so every LATER solve kept computing the gradient
# into LinearSolver::Gradient_for_glob_conv_newton_solve -- while reset_gradient(), which is what
# resizes that vector, is only reached through the same branch that enables it.
# CRDoubleMatrix::multiply_transpose() sizes its output only when it is not already built, so the
# first solve after the dof count grew accumulated past the end of a buffer still sized for the
# problem as it was.
#
# The overflow is silent. What it produced was a crash somewhere else entirely and much later, in a
# different place on different runs -- inside MKL Pardiso's mkl_serv_free once, as a null
# oomph::Node::position() during the next residual assembly another time. So this test does not try
# to catch a particular crash: it does the sequence that corrupts the heap and then keeps using the
# problem, and asserts the results are still right. Under valgrind it is the "Invalid write of
# size 8" that names the bug directly.
#
# See src/thirdparty/INFO_oomph-lib, "globally convergent Newton heap overflow".

import numpy
import pytest

from pyoomph import Problem, DirichletBC
from pyoomph.expressions import var, exp, dot
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _Poisson(Problem):
    def __init__(self):
        super().__init__()
        # Everything adapts inside solve(), so the dof count grows exactly where we want it to.
        self.initial_adaption_steps = 0

    def define_problem(self):
        x = var("coordinate")
        eqs = PoissonEquation(source=exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.002))
        eqs += DirichletBC(u=0) @ ["bottom", "top", "left", "right"]
        eqs += SpatialErrorEstimator(u=1)
        self += RectangularQuadMesh(N=8)
        self += eqs @ "domain"


def _make(tmp_path):
    p = _Poisson()
    p.set_output_directory(str(tmp_path))
    p.quiet()
    return p


def test_growing_the_mesh_after_a_globally_convergent_solve(tmp_path):
    """The exact sequence that overflowed: gcn solve, then a solve that adds dofs."""
    with _make(tmp_path) as p:
        p.solve(spatial_adapt=1)
        ndof_gcn = p.ndof()
        p.solve(spatial_adapt=0, globally_convergent_newton=True)
        assert p.ndof() == ndof_gcn

        # The gradient vector is now sized for ndof_gcn. Growing the mesh is what used to write
        # past the end of it.
        p.solve(spatial_adapt=1)
        assert p.ndof() > ndof_gcn, "the mesh did not grow, so the overflow was never provoked"

        dofs = numpy.array(p.get_current_dofs()[0])
        assert numpy.all(numpy.isfinite(dofs))
        # Keep using the problem: with a corrupted heap this is where the damage used to surface.
        p.solve(spatial_adapt=1)
        p.output()
        assert numpy.all(numpy.isfinite(numpy.array(p.get_current_dofs()[0])))


def test_repeated_alternation(tmp_path):
    """Alternating the flag must not accumulate anything either."""
    with _make(tmp_path) as p:
        p.solve(spatial_adapt=1)
        for _ in range(3):
            p.solve(spatial_adapt=0, globally_convergent_newton=True)
            p.solve(spatial_adapt=1)
        assert numpy.all(numpy.isfinite(numpy.array(p.get_current_dofs()[0])))


def test_globally_convergent_solve_is_still_correct(tmp_path):
    """Switching the method on must not change the solution it converges to.

    Poisson is linear, so both routes must land on the same solution to round-off - which also
    shows that disabling the gradient computation afterwards did not break the method itself.
    """
    with _make(tmp_path) as p:
        p.solve(spatial_adapt=1)
        plain = numpy.array(p.get_current_dofs()[0])

    with _make(tmp_path) as p:
        p.solve(spatial_adapt=1, globally_convergent_newton=True)
        gcn = numpy.array(p.get_current_dofs()[0])

    assert len(gcn) == len(plain), "the two routes adapted to different meshes"
    assert numpy.allclose(gcn, plain, rtol=1e-10, atol=1e-14)
