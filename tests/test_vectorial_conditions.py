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

# DirichletBC and InitialCondition given a whole vector field at once:
#
#     DirichletBC(velocity=vector(1,0))       instead of  DirichletBC(velocity_x=1, velocity_y=0)
#     InitialCondition(velocity=vector(u,v))  instead of  InitialCondition(velocity_x=u, velocity_y=v)
#
# Both classes act on the SCALAR fields the code generator knows about, so the vector has to be split
# into its components (BaseEquations.expand_vectorial_entries). The oracle throughout is that the two
# spellings are bit-identical -- same ndof, same initial dof vector, same solution -- which is a
# stronger statement than "it runs", and the only one that catches a component being written to the
# wrong slot.
#
# The axisymmetric case is the one that makes the positional mapping worth testing rather than
# assuming: there the components are not x/y, and nothing in the expansion knows about coordinate
# systems -- it relies on the field's own component order being the order var("velocity") is built
# with.

import numpy
import pytest

from pyoomph import Problem, DirichletBC, InitialCondition
from pyoomph.expressions import vector, var
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import AxisymmetryBC
from pyoomph.equations.additional import InactiveDirichletBC
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _Cavity(Problem):
    """Lid-driven cavity, with every condition stated either component-wise or as a vector."""

    def __init__(self, vectorial, axisymmetric=False, inactive=False, N=4):
        super().__init__()
        self.vectorial, self.axisymmetric, self.inactive, self.N = vectorial, axisymmetric, inactive, N

    def define_problem(self):
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
        self += RectangularQuadMesh(N=self.N)
        eqs = NavierStokesEquations(mode="TH", dynamic_viscosity=1, mass_density=1)
        lid = InactiveDirichletBC if self.inactive else DirichletBC
        walls = ["right", "bottom"] if self.axisymmetric else ["left", "right", "bottom"]
        if self.vectorial:
            eqs += lid(velocity=vector(1, 0)) @ "top"
            eqs += DirichletBC(velocity=vector(0, 0)) @ walls
            eqs += InitialCondition(velocity=vector(0.3 * var("coordinate_y"), 0.2))
        else:
            eqs += lid(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ walls
            eqs += InitialCondition(velocity_x=0.3 * var("coordinate_y"), velocity_y=0.2)
        if self.axisymmetric:
            eqs += AxisymmetryBC() @ "left"
        eqs += DirichletBC(pressure=0) @ ("bottom/right" if self.axisymmetric else "bottom/left")
        self += eqs @ "domain"


def _solve(tmp_path, **kw):
    prob = _Cavity(**kw)
    with prob as p:
        p.set_output_directory(str(tmp_path / ("v" if kw.get("vectorial") else "s")))
        p.initialise()
        ic = numpy.array(p.get_current_dofs()[0])
        p.solve(timestep=0.1)
        return ic, numpy.array(p.get_current_dofs()[0]), p.ndof()


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_vector_valued_conditions_match_the_component_form(axisymmetric, tmp_path):
    ic_v, sol_v, ndof_v = _solve(tmp_path / "vec", vectorial=True, axisymmetric=axisymmetric)
    ic_s, sol_s, ndof_s = _solve(tmp_path / "scal", vectorial=False, axisymmetric=axisymmetric)
    assert ndof_v == ndof_s, "vector form pinned a different number of dofs (%d vs %d)" % (ndof_v, ndof_s)
    # Guards against the whole comparison being vacuous: an initial condition of all zeros, or a
    # solution that never moved, would match no matter which component went where.
    assert numpy.max(numpy.abs(ic_s)) > 1e-3
    assert numpy.max(numpy.abs(sol_s)) > 1e-3
    assert numpy.max(numpy.abs(ic_v - ic_s)) == 0.0, "the vector-valued InitialCondition differs"
    assert numpy.max(numpy.abs(sol_v - sol_s)) == 0.0, "the vector-valued DirichletBC differs"


def test_inactive_dirichlet_bc_expands_its_vector(tmp_path):
    # InactiveDirichletBC deactivates its conditions through mesh.set_dirichlet_active(), which names
    # the SCALAR fields -- so the vector has to be split there too, not only when the residuals are
    # defined. If it were not, the components would stay pinned and ndof would not change.
    ic, _, ndof_off = _solve(tmp_path, vectorial=True, inactive=True)
    prob = _Cavity(vectorial=True, inactive=True)
    with prob as p:
        p.set_output_directory(str(tmp_path / "on"))
        p.initialise()
        p.get_mesh("domain/top").set_dirichlet_active(velocity_x=True, velocity_y=True)
        p.reapply_boundary_conditions()
        assert p.ndof() < ndof_off, \
            "activating the condition pinned nothing (%d dofs either way) -- the vector was not expanded" \
            % ndof_off


def test_a_vector_for_a_scalar_field_is_reported(tmp_path):
    class P(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(N=2)
            self += (PoissonEquation(name="u", source=1, space="C1")
                     + DirichletBC(u=vector(1, 2)) @ "top") @ "domain"
    with pytest.raises(RuntimeError, match="not a vector field"):
        with P() as p:
            p.set_output_directory(str(tmp_path / "scalar"))
            p.initialise()


def test_too_many_components_is_reported(tmp_path):
    # vector() pads to GiNaC_vector_dim(), so a 2d field is routinely handed a zero third component and
    # that must pass. A NON-zero one is a real mistake and must not be dropped in silence.
    class P(Problem):
        def __init__(self, third):
            super().__init__()
            self.third = third
        def define_problem(self):
            self += RectangularQuadMesh(N=2)
            eqs = NavierStokesEquations(mode="TH", dynamic_viscosity=1, mass_density=1)
            eqs += DirichletBC(velocity=vector(1, 2, self.third)) @ "top"
            eqs += DirichletBC(velocity=vector(0, 0)) @ ["left", "right", "bottom"]
            eqs += DirichletBC(pressure=0) @ "bottom/left"
            self += eqs @ "domain"

    with P(0) as p:  # the padding vector() itself produces
        p.set_output_directory(str(tmp_path / "pad"))
        p.initialise()
    with pytest.raises(RuntimeError, match="non-zero component"):
        with P(3) as p:
            p.set_output_directory(str(tmp_path / "three"))
            p.initialise()
