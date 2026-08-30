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

"""Interface terms acting on gradients of the *bulk* mesh test function.

InterfaceMeshStiffening tests against ``grad(testfunction("mesh",domain=".."))``, which is the only
way an interface equation can reach the mesh dofs of the *interior* nodes of the attached bulk
elements: the shape function of such a node vanishes on the face, but its gradient does not.

That used to be silently lost. The bulk nodes' positions are registered as external data of the
interface element (and their local equations remapped) only if the generated code requires certain
bulk shapes, and the condition listed ``Pos.psi`` but neither ``Pos.dx_psi`` nor ``Pos.dX_psi``. A
term built purely from position *gradients* therefore assembled into nothing, unless it happened to
also use the normal or the element size, which are on that list. Hence the residual test below.
"""

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import InterfaceMeshStiffening, LaplaceSmoothedMesh, PinMeshCoordinates


class BulkTestFunctionGradient(InterfaceEquations):
    """Bare-bones version of what InterfaceMeshStiffening does, without any prefactor."""

    def define_residuals(self):
        x = var("mesh", domain="..")
        X = var("lagrangian", domain="..")
        self.add_weak(grad(x - X, lagrangian=True), grad(testfunction("mesh", domain=".."), lagrangian=True), lagrangian=True)


class SquashedSquare(Problem):
    """Unit square whose top boundary is pushed up by ``amp``, everything else pinned."""

    def __init__(self, interface_eqs=None):
        super().__init__()
        self.interface_eqs = interface_eqs

    def define_problem(self):
        self.amp = self.get_global_parameter("amp")
        self += RectangularQuadMesh(N=4)
        eqs = LaplaceSmoothedMesh() + ElementSpace("C2")
        eqs += PinMeshCoordinates() @ ["left", "bottom", "right"]
        # The mesh is built from macro elements, so the Lagrangian coordinates are reset to whatever
        # the initial condition produced. Driving the deformation by a parameter after initialise()
        # is what makes the displacement nonzero at all.
        eqs += DirichletBC(mesh_x=True, mesh_y=1 + self.amp * var("lagrangian_x") * (1 - var("lagrangian_x"))) @ "top"
        if self.interface_eqs is not None:
            eqs += self.interface_eqs @ "top"
        self += eqs @ "domain"


def _layer_thicknesses(problem):
    """Thickness of each element layer along the vertical line through the middle of the domain."""
    ys = [y for _, y in sorted((n.x_lagr(1), n.x(1)) for n in problem.get_mesh("domain").nodes() if abs(n.x_lagr(0) - 0.5) < 1e-8)]
    return numpy.diff(ys)


def _solve(interface_eqs, tmpdir):
    with SquashedSquare(interface_eqs) as problem:
        problem.set_output_directory(str(tmpdir))
        problem.initialise()
        problem.amp.value = 0.4
        problem.solve()
        return _layer_thicknesses(problem)


def test_position_gradient_reaches_interior_bulk_nodes(tmp_path):
    """The interface term must change the interior nodes, not just the (pinned) interface ones."""
    without = _solve(None, tmp_path / "without")
    with_it = _solve(BulkTestFunctionGradient(), tmp_path / "with")
    assert numpy.max(numpy.abs(with_it - without)) > 1e-3


@pytest.mark.parametrize("mode", ["normal", "laplace", "elastic"])
def test_stiffening_keeps_the_attached_layer_closer_to_undeformed(mode, tmp_path):
    """The stiffened layer at the interface must absorb less of the imposed stretch."""
    without = _solve(None, tmp_path / "without")
    with_it = _solve(InterfaceMeshStiffening(10, mode=mode), tmp_path / str(mode))
    # Undeformed thickness is 1/8; the top boundary is pushed up, so every layer is stretched.
    assert without[-1] > 1.2 * 0.125
    assert with_it[-1] < without[-1]
    # The stretch is not lost, it is handed over to the next layer inwards.
    assert with_it[-2] > without[-2]
