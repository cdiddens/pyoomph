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

# Coverage gates for the single dof walk, Mesh::visit_global_dofs (src/mesh.cpp).
#
# Both per-dof descriptions pyoomph exposes are built from that one walk:
#
#   Problem.get_dof_description()              -> Mesh::describe_global_dofs
#   Problem._get_dof_to_global_field_index_mapping() -> Mesh::fill_dof_to_global_field_index_buffer
#
# They used to be two hand-written copies of the same nested loop over elements, nodes, DG spaces and
# internal data, and the copies had drifted in three places (the reserved coordinate slots are indexed
# in reverse in one of them, the interface branch is guarded differently, and the DG buffer index is
# computed differently on a facet element). Merging them onto one walk is only safe if something
# notices when a kind of dof stops being reached, and nothing did.
#
# What is asserted is COVERAGE, not particular index values: every dof of the problem must be
# described and field-mapped. That is the property a missed branch breaks -- an unvisited dof keeps the
# -1 both buffers are initialised with -- and it is the property the coming dof-ordering work depends
# on, since a reordering that cannot see a dof cannot place it.
#
# The shapes are chosen to hit every branch of the walk between them: nodal continuous values, nodal
# positions on a moving mesh, an ODE dof, interface-only continuous values, a C2TB bubble with a DL
# pressure, a DG space with a D0 pressure and its internal facet mesh, and a 1D mesh (where a DL Data
# carries two values rather than three).

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import EnforcedDirichlet, IntegralConstraint
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _Plain(Problem):
    """Nodal C2 values only, with boundary-specific Dirichlet dof types."""

    def define_problem(self):
        self += RectangularQuadMesh(N=4)
        eqs = PoissonEquation(source=1, space="C2")
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"


class _MovingWithInterfaces(Problem):
    """Nodal positions (moving mesh), an ODE dof from the integral constraint, an interface Lagrange
    multiplier, and a second one on a single point. The position dofs are the reversed-slot case: the
    Dirichlet buffer numbers coordinate_x as -1 and coordinate_z as -3, so with the +3 offset x lands
    in slot 2 and z in slot 0."""

    def define_problem(self):
        self += RectangularQuadMesh(N=[6, 2], size=[3, 1])
        eqs = PoissonEquation(source=1, space="C2")
        eqs += LaplaceSmoothedMesh()
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ ["left", "right", "top", "bottom"]
        eqs += DirichletBC(u=0) @ "bottom"
        eqs += EnforcedDirichlet(u=1) @ "top"
        eqs += EnforcedDirichlet(u=1) @ "top/left"
        eqs += IntegralConstraint(u=0.25)
        self += eqs @ "domain"


class _StokesCR(Problem):
    """C2TB velocity, whose bubble node belongs to exactly one element, plus a DL pressure: internal
    data with one value per DL node."""

    def define_problem(self):
        self += RectangularQuadMesh(N=3, split_in_tris="left")
        ns = StokesEquations(dynamic_viscosity=1, mode="CR")
        eqs = ns + ns.create_pressure_fixation(value=0)
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        self += eqs @ "domain"


class _StokesD1D0(Problem):
    """A DG velocity with a D0 pressure, and hence an internal facet mesh. The facet elements reach
    their neighbours' DG Data, which is what Mesh::DofVisit::dg_on_own_facet is about."""

    def define_problem(self):
        self += RectangularQuadMesh(N=3)
        eqs = StokesEquations(dynamic_viscosity=1, mode="D1D0", DG_alpha=10)
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        # Without this the pressure level is undetermined -- the mode refuses create_pressure_fixation
        # -- and the two runs of test_dof_ordering.py then differ by a constant pressure shift that has
        # nothing to do with the numbering.
        eqs += IntegralConstraint(pressure=0)
        self += eqs @ "domain"


class _Facet(Equations):
    def define_fields(self):
        self.define_scalar_field("uhat", "C2")

    def define_residuals(self):
        uhat, uhattest = var_and_test("uhat")
        u, utest = var_and_test("u", domain="..")
        self.add_residual(weak(uhat - u, uhattest) + weak(0.1 * uhat, utest))


class _InterfaceField(Problem):
    """An interface-only continuous field, i.e. the buffer_offset_interf branch."""

    def define_problem(self):
        self += RectangularQuadMesh(N=4)
        eqs = PoissonEquation(source=1, space="C2")
        eqs += DirichletBC(u=0) @ ["left", "right", "bottom"]
        eqs += _Facet() @ "top"
        self += eqs @ "domain"


class _Line1D(Problem):
    """1D, where a DL Data has two values rather than three."""

    def define_problem(self):
        self += LineMesh(N=6)
        eqs = PoissonEquation(source=1, space="C2")
        eqs += DirichletBC(u=0) @ ["left", "right"]
        self += eqs @ "domain"


_SHAPES = {
    "plain": _Plain,
    "moving_with_interfaces": _MovingWithInterfaces,
    "stokes_cr": _StokesCR,
    "stokes_d1d0": _StokesD1D0,
    "interface_field": _InterfaceField,
    "line1d": _Line1D,
}


@pytest.fixture(params=sorted(_SHAPES), ids=sorted(_SHAPES))
def shape(request, tmp_path):
    p = _SHAPES[request.param]()
    p.set_output_directory(str(tmp_path))
    p.quiet()
    p.set_linear_solver("superlu")
    p.initialise()
    return p


# ---------------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------------

def test_every_dof_is_described(shape):
    """describe_global_dofs must reach every dof. An unreached one keeps the -1 the buffer is filled
    with, and shows up in Problem.get_dof_description() as an out-of-range type index."""
    types, names = shape.get_dof_description()
    types = numpy.asarray(types)
    assert len(types) == shape.ndof()
    missing = numpy.flatnonzero(types < 0)
    assert missing.size == 0, "%d of %d dofs undescribed, first: %s" % (
        missing.size, shape.ndof(), missing[:10].tolist())
    assert types.max() < len(names)


def test_every_dof_has_a_global_field(shape):
    """The same for the field-index map, which the PETSc field split slices the matrix with: a dof with
    -1 there belongs to no split and would be silently dropped from the preconditioner."""
    mapping = numpy.asarray(shape._get_dof_to_global_field_index_mapping())
    names = shape._get_global_field_names()
    assert len(mapping) == shape.ndof()
    missing = numpy.flatnonzero(mapping < 0)
    assert missing.size == 0, "%d of %d dofs unmapped, first: %s" % (
        missing.size, shape.ndof(), missing[:10].tolist())
    assert mapping.max() < len(names)


def test_the_two_descriptions_agree_on_the_field(shape):
    """The two walks label a dof differently -- one by dof TYPE (which is per boundary, so
    'domain/left/u' and 'domain/u' are distinct) and one by global FIELD -- but the field named by the
    type must be the field named by the map, once the boundary prefix is stripped.

    This is what actually ties the two consumers of the walk together; the coverage tests above would
    both pass if the walk reported every dof under the wrong label."""
    types, tnames = shape.get_dof_description()
    mapping = numpy.asarray(shape._get_dof_to_global_field_index_mapping())
    fnames = shape._get_global_field_names()
    for eqn, (t, f) in enumerate(zip(numpy.asarray(types), mapping)):
        tfield = tnames[t].split("/")[-1]
        ffield = fnames[f].split("/")[-1]
        if tfield.startswith("mesh_"):
            # describe_global_dofs spells a position mesh_x; the field map spells it coordinate_x.
            tfield = "coordinate_" + tfield[len("mesh_"):]
        # Nothing is exempt, including the ODE dof an IntegralConstraint adds: both walks call it
        # "_meshwide__domain/u", so it compares like any other.
        assert tfield == ffield, \
            "dof %d: type says %r, field map says %r" % (eqn, tnames[t], fnames[f])
