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
#
# The position fields ("mesh", "coordinate", "lagrangian") are vectors in the same sense, but the
# element gets them from the C++ side rather than from define_vector_field(), so they are in no
# _vectorfields dict and need their own registration -- and their natural value, var("lagrangian"),
# is a deferred symbol that is not a matrix until the code generator resolves it. Both halves are
# needed before DirichletBC(mesh=var("lagrangian")) works, and each is tested separately below.

import numpy
import pytest

from pyoomph import Problem, DirichletBC, InitialCondition
from pyoomph.expressions import vector, var
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import AxisymmetryBC, ElementSpace
from pyoomph.equations.ALE import PseudoElasticMesh
from pyoomph.equations.additional import InactiveDirichletBC
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh


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


class _MovingMesh(Problem):
    """A pseudo-elastic mesh whose position is prescribed either as a vector or component-wise.

    The top boundary is sheared by ``shift`` times the first Lagrangian coordinate, so which slot each
    component ends up in decides where the interior nodes go; ``shift=0`` leaves the mesh undeformed
    and is only there to show that a non-zero one moves it at all.
    """

    def __init__(self, vectorial, axisymmetric=False, threed=False, value="lagrangian", shift=1.0, N=3):
        super().__init__()
        self.vectorial, self.axisymmetric, self.threed = vectorial, axisymmetric, threed
        self.value, self.shift, self.N = value, shift, N

    def define_problem(self):
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
        self += CuboidBrickMesh(N=self.N) if self.threed else RectangularQuadMesh(N=self.N)
        comps = ["x", "y", "z"] if self.threed else ["x", "y"]
        X = [var(self.value + "_" + c) for c in comps]
        off = [self.shift * f for f in (0.1, 0.2, 0.3)[:len(comps)]]
        # PseudoElasticMesh defines no field of its own, so the coordinate space has to be named.
        eqs = PseudoElasticMesh() + ElementSpace("C2")
        if self.vectorial:
            eqs += DirichletBC(mesh=var(self.value) + vector(*off) * X[0]) @ "top"
            eqs += DirichletBC(mesh=var(self.value)) @ "bottom"
            eqs += InitialCondition(mesh=var(self.value))
        else:
            eqs += DirichletBC(**{"mesh_" + c: X[i] + off[i] * X[0] for i, c in enumerate(comps)}) @ "top" #type:ignore
            eqs += DirichletBC(**{"mesh_" + c: X[i] for i, c in enumerate(comps)}) @ "bottom" #type:ignore
            eqs += InitialCondition(**{"mesh_" + c: X[i] for i, c in enumerate(comps)}) #type:ignore
        self += eqs @ "domain"


def _solve_positions(tmp_path, **kw):
    with _MovingMesh(**kw) as p:
        p.set_output_directory(str(tmp_path))
        p.solve()
        m = p.get_mesh("domain")
        return p.ndof(), numpy.array([[n.x(i) for i in range(n.ndim())] for n in m.nodes()])


@pytest.mark.parametrize("case", ["cartesian", "axisymmetric", "threed", "coordinate"])
def test_position_field_as_a_vector_matches_the_component_form(case, tmp_path):
    # "mesh" is not registered by define_vector_field(), and var("lagrangian") is not a matrix when the
    # condition is built -- so this whole family used to fail with "is not defined in the element".
    # As above, the oracle is that the two spellings give the same node positions (bit-identical on
    # x86_64; see the tolerance at the bottom for why that is not asserted as such).
    kw = {"axisymmetric": case == "axisymmetric", "threed": case == "threed",
          "value": "coordinate" if case == "coordinate" else "lagrangian"}
    ndof_v, pos_v = _solve_positions(tmp_path / "vec", vectorial=True, **kw)
    ndof_s, pos_s = _solve_positions(tmp_path / "scal", vectorial=False, **kw)
    _, pos_flat = _solve_positions(tmp_path / "flat", vectorial=False, shift=0.0, **kw)
    assert ndof_v == ndof_s, "vector form pinned a different number of dofs (%d vs %d)" % (ndof_v, ndof_s)
    assert pos_v.shape == pos_s.shape
    # Guards against a vacuous comparison: if the condition moved nothing, every mix-up matches.
    assert numpy.max(numpy.abs(pos_s - pos_flat)) > 1e-3
    # Not == 0.0: the two spellings reach the same Newton solution through slightly differently
    # ordered generated code, and on arm64 the [threed] case lands one ULP apart (2.2e-16) where
    # x86_64 agrees exactly. The guard above is 1e-3, so anything this test is actually defending
    # against - a component mix-up, a condition applied to the wrong field - is thirteen orders of
    # magnitude larger than what is tolerated here.
    assert numpy.max(numpy.abs(pos_v - pos_s)) < 1e-13, "the vector-valued condition moved the mesh differently"


def test_a_deferred_symbol_is_resolved_before_it_is_split(tmp_path):
    # The value need not be an explicit vector(): anything that resolves to one has to be split too,
    # including an expression built around a deferred var(). Splitting the unresolved symbol instead
    # would put the whole vector into mesh_x.
    class _Split(PseudoElasticMesh):
        def define_residuals(self):
            super().define_residuals()
            got = self.expand_vectorial_entries({"mesh": 2 * var("lagrangian")}, "test")
            assert set(got.keys()) == {"mesh_x", "mesh_y"}, got
            assert "lagrangian_x" in str(got["mesh_x"]) and "lagrangian_y" not in str(got["mesh_x"])
            assert "lagrangian_y" in str(got["mesh_y"]) and "lagrangian_x" not in str(got["mesh_y"])
            # A plain scalar is not a vector, whatever the field is called: left untouched, so the code
            # generator reports it along with the fields it does have.
            assert self.expand_vectorial_entries({"mesh": 0}, "test") == {"mesh": 0}

    class P(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(N=2)
            self += (_Split() + ElementSpace("C2")) @ "domain"

    with P() as p:
        p.set_output_directory(str(tmp_path / "split"))
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
