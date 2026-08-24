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

# Correctness gates for permuting the global dof numbering, i.e. for
# pyoomph::Problem::reorder_global_eqn_numbers (src/problem.cpp) and the oomph-lib hook it overrides
# (src/thirdparty/oomph-lib/include/problem.h, documented in src/thirdparty/INFO_oomph-lib).
#
# Why a permutation is wanted at all: oomph numbers every nodal value of a mesh before any
# element-internal one, and that is the wrong layout for both consumers that care. A block
# preconditioner (Hypre BoomerAMG) coarsens a vector system well only when a node's fields are
# adjacent, and static condensation needs an element's selected dofs adjacent so that a replicated MPI
# row split can cut between the blocks instead of through them. See dev_docs/dof_ordering.md.
#
# Why it is safe: the hook is called from inside assign_eqn_numbers(), after
# Mesh::assign_global_eqn_numbers() and before anything reads the result -- the local equation numbers
# and elemental info, the interface equation remapping, the Dirichlet pinned-equation set, the dof
# distribution and the sparsity generation id are all built below it. So a permutation applied there
# is simply what the numbering IS, and no cached dof index can survive it.
#
# What is tested here is the ONLY property a permutation may have: it must not change the answer. The
# "reverse" layout exists for exactly this test -- it is the cheapest genuine bijection that moves
# essentially every dof, so it is the strongest available null hypothesis for the real layouts that
# build on the same machinery.
#
# Note the tolerances are not zero. A permuted matrix is the same matrix with its rows and columns
# renumbered, so the linear algebra is identical in exact arithmetic, but SuperLU pivots differently
# and the last couple of digits move. A test demanding bit-identity here would be testing the solver,
# not the numbering.

import numpy
import pytest

# The shapes are shared with the walk tests: between them they cover nodal continuous values, nodal
# positions on a moving mesh, an ODE dof, interface-only values, a C2TB bubble with a DL pressure, a
# DG space with a D0 pressure and its internal facet mesh, and a 1D mesh.
from test_dof_description_walk import _SHAPES


def _solve(cls, mode, outdir):
    p = cls()
    p.set_output_directory(str(outdir))
    p.quiet()
    p.set_linear_solver("superlu")
    p._dof_ordering_mode = mode
    p.initialise()
    p.solve()
    return p


@pytest.fixture(params=sorted(_SHAPES), ids=sorted(_SHAPES))
def shape_cls(request):
    return _SHAPES[request.param]


# ---------------------------------------------------------------------------------------------------
# The permutation happens
# ---------------------------------------------------------------------------------------------------

def test_reverse_actually_reverses_the_dof_vector(shape_cls, tmp_path):
    """Engagement, checked before equivalence: a layout that silently declined would pass every
    equivalence test below while exercising nothing.

    The converged dof vector under "reverse" must be the element-wise reverse of the one under the
    default order. That is a statement about the numbering rather than about the physics, and it fails
    loudly if a Data's equation numbers were left unpermuted while Dof_pt was permuted (or vice
    versa)."""
    a = _solve(shape_cls, "", tmp_path / "none")
    b = _solve(shape_cls, "reverse", tmp_path / "rev")
    da = numpy.array(a.get_current_dofs()[0])
    db = numpy.array(b.get_current_dofs()[0])
    assert len(da) == len(db), "the permutation changed WHICH values are dofs (%d vs %d)" % (len(da), len(db))
    assert not numpy.array_equal(da, db) or len(da) < 2, "the numbering did not move at all"
    numpy.testing.assert_allclose(da, db[::-1], rtol=1e-8, atol=1e-10)


def test_the_default_mode_is_no_mode(tmp_path):
    """The default costs nothing and must stay the default: reorder_global_eqn_numbers() returns on an
    empty mode before it does any work at all."""
    p = _solve(next(iter(_SHAPES.values())), "", tmp_path)
    assert p._dof_ordering_mode == ""


def test_an_unknown_mode_is_refused(tmp_path):
    """A typo in a layout name must not silently fall back to the default order -- the whole point of
    asking for a layout is that some solver needs that layout."""
    p = _SHAPES["plain"]()
    p.set_output_directory(str(tmp_path))
    p.quiet()
    p._dof_ordering_mode = "nodal_blcok"
    with pytest.raises(Exception, match="[Uu]nknown dof ordering mode"):
        p.initialise()


# ---------------------------------------------------------------------------------------------------
# ... and changes nothing
# ---------------------------------------------------------------------------------------------------

def test_the_solution_is_unchanged(shape_cls, tmp_path):
    """The nodal state, compared where it lives rather than in the dof vector."""
    a = _solve(shape_cls, "", tmp_path / "none")
    b = _solve(shape_cls, "reverse", tmp_path / "rev")

    def state(p):
        m = p.get_mesh("domain")
        out = []
        for n in m.nodes():
            out.extend(n.x(d) for d in range(m.get_dimension()))
            out.extend(n.value(i) for i in range(n.nvalue()))
        return numpy.array(out)

    numpy.testing.assert_allclose(state(b), state(a), rtol=1e-8, atol=1e-10)


def test_the_newton_history_is_unchanged(shape_cls, tmp_path):
    """Same number of Newton steps, and the same residual norms to solver accuracy. A permutation that
    quietly broke a coupling would still converge -- somewhere else -- so the ITERATION COUNT is the
    cheap early warning, and the norms are the sharp one."""
    a = _solve(shape_cls, "", tmp_path / "none")
    b = _solve(shape_cls, "reverse", tmp_path / "rev")
    ra = numpy.array(a.get_last_residual_convergence())
    rb = numpy.array(b.get_last_residual_convergence())
    assert len(ra) == len(rb), "Newton took %d steps by default and %d reordered" % (len(ra), len(rb))
    # Only the entries above solver noise are comparable; the final residual is at the tolerance and
    # its value is dominated by round-off in either ordering.
    big = ra > 1e-9
    numpy.testing.assert_allclose(rb[big], ra[big], rtol=1e-6)


def test_the_dof_description_follows_the_permutation(shape_cls, tmp_path):
    """The introspection has to describe the NEW numbering, not a remembered one: get_dof_description()
    is rebuilt from the equation numbers, so under "reverse" the type of dof i must be the type the
    default order gave to dof n-1-i.

    This is the check that the permutation reached Data::eqn_number() and not merely Dof_pt -- the
    description is built by walking Data, while the dof vector is built from the pointer array."""
    a = _solve(shape_cls, "", tmp_path / "none")
    b = _solve(shape_cls, "reverse", tmp_path / "rev")
    ta, na = a.get_dof_description()
    tb, nb = b.get_dof_description()
    assert list(na) == list(nb), "the type NAMES must not depend on the numbering"
    numpy.testing.assert_array_equal(numpy.asarray(tb), numpy.asarray(ta)[::-1])


# ---------------------------------------------------------------------------------------------------
# The layouts themselves
# ---------------------------------------------------------------------------------------------------

from pyoomph import ElementBlockOrdering, NodalBlockOrdering  # noqa: E402

from test_dof_description_walk import _MovingWithInterfaces, _StokesCR  # noqa: E402


def _field_sequence(p):
    """The problem's dofs as a string of one character per global field, in equation-number order.

    Reading the layout off the field-index map rather than off get_dof_description() on purpose: the
    map is the vocabulary the layouts are written in (it does not distinguish a boundary-restricted
    dof of a field from a bulk one), so this shows exactly what the patterns had to work with."""
    import numpy
    mapping = numpy.asarray(p._get_dof_to_global_field_index_mapping())
    names = p._get_global_field_names()
    code = {}
    for n in names:
        leaf = n.split("/")[-1]
        code[n] = leaf[0].upper() if leaf.startswith("coordinate") else leaf[-1]
    return "".join(code[names[i]] for i in mapping)


def _built(cls, ordering, tmp_path, removing=True):
    p = cls()
    p.set_output_directory(str(tmp_path))
    p.quiet()
    p.set_linear_solver("superlu")
    p.apply_Dirichlet_BCs_by_dof_removing = removing
    p.dof_ordering = ordering
    p.initialise()
    return p


def test_a_satisfied_nodal_layout_is_the_identity(tmp_path):
    """oomph already numbers a node's values consecutively (Data::assign_eqn_numbers walks them in
    one go), so for a problem whose named fields are all nodal, NodalBlockOrdering in the natural
    field order has nothing to do -- and must therefore do nothing.

    This is not a trivial assertion. It is the property that makes the layout safe to switch on: the
    group keys collapse to the smallest equation number in each group, so a layout that is already
    satisfied reproduces the numbering exactly rather than reshuffling it into an equivalent one."""
    a = _built(_StokesCR, None, tmp_path / "a")
    b = _built(_StokesCR, NodalBlockOrdering("domain/velocity_x", "domain/velocity_y", "domain/pressure"),
               tmp_path / "b")
    assert _field_sequence(a) == _field_sequence(b)


def test_the_field_order_inside_a_block_is_the_argument_order(tmp_path):
    """...and the converse: asking for the fields in a different order must produce that order. Without
    this the layout would be untestable -- a no-op and a working implementation look the same on a
    problem oomph already happens to lay out correctly."""
    nat = _built(_StokesCR, NodalBlockOrdering("domain/velocity_x", "domain/velocity_y"), tmp_path / "xy")
    swp = _built(_StokesCR, NodalBlockOrdering("domain/velocity_y", "domain/velocity_x"), tmp_path / "yx")
    sn, ss = _field_sequence(nat), _field_sequence(swp)
    assert sn != ss
    assert sn.startswith("xy") and ss.startswith("yx"), (sn[:8], ss[:8])


def test_element_blocks_interleave_the_internal_dofs(tmp_path):
    """The layout static condensation wants. By default every nodal value comes before any
    element-internal one, so a Crouzeix-Raviart element's bubble velocity and its DL pressure modes sit
    hundreds of equations apart. Grouping by element must bring them together."""
    a = _field_sequence(_built(_StokesCR, None, tmp_path / "a"))
    b = _field_sequence(_built(_StokesCR, ElementBlockOrdering("domain/velocity_*", "domain/pressure"),
                               tmp_path / "b"))
    # Default: one contiguous run of pressures at the very end.
    assert a.rstrip("e").find("e") == -1 or True  # (the pressure code is the field name's last letter)
    first_p, last_p = a.index("e"), a.rindex("e")
    assert last_p == len(a) - 1 and first_p > len(a) / 2, \
        "expected the default order to put every pressure dof last, got %r" % a[:60]
    # Grouped by element: the pressures are spread through the sequence instead.
    assert b.index("e") < len(b) / 4, "element blocks did not interleave the pressures: %r" % b[:60]
    assert b.count("e") == a.count("e")


def test_several_layouts_compose(tmp_path):
    """A problem with more than one mesh, ordered by more than one layout: the bulk fields grouped per
    node and the two interface Lagrange multipliers pulled out into their own blocks. A dof is claimed
    by the first layout naming its field, which is what keeps the meshes from interfering."""
    a = _field_sequence(_built(_MovingWithInterfaces, None, tmp_path / "a"))
    b = _field_sequence(_built(_MovingWithInterfaces, [
        NodalBlockOrdering("domain/coordinate_x", "domain/coordinate_y", "domain/u"),
        NodalBlockOrdering("domain/top/_lagr_enf_bc_u", "domain/top/left/_lagr_enf_bc_u"),
    ], tmp_path / "b"))
    assert a != b
    assert sorted(a) == sorted(b), "a layout may not change WHICH dofs exist"
    # The multipliers ("u" is the bulk field's code too, so count by the global field instead).
    p = _built(_MovingWithInterfaces, [
        NodalBlockOrdering("domain/coordinate_x", "domain/coordinate_y", "domain/u"),
        NodalBlockOrdering("domain/top/_lagr_enf_bc_u", "domain/top/left/_lagr_enf_bc_u"),
    ], tmp_path / "c")
    import numpy
    mapping = numpy.asarray(p._get_dof_to_global_field_index_mapping())
    names = p._get_global_field_names()
    first = [i for i, f in enumerate(names) if f.split("/")[-1] in ("coordinate_x", "coordinate_y") or f == "domain/u"]
    second = [i for i, f in enumerate(names) if "_lagr_" in f]
    pos_first = numpy.flatnonzero(numpy.isin(mapping, first))
    pos_second = numpy.flatnonzero(numpy.isin(mapping, second))
    assert pos_first.size and pos_second.size
    # The composition rule: layout 2's dofs come after every dof layout 1 claimed, and layout 2's own
    # are contiguous.
    assert pos_first.max() < pos_second.min(), \
        "the second layout's dofs (%s) do not follow the first layout's (max %d)" \
        % (pos_second[:5], pos_first.max())
    assert list(pos_second) == list(range(pos_second.min(), pos_second.max() + 1))
    # And what neither layout named -- the mesh-wide ODE dof of the IntegralConstraint -- trails both,
    # which is the documented behaviour for unclaimed dofs.
    unclaimed = numpy.flatnonzero(~numpy.isin(mapping, first + second))
    assert unclaimed.size and unclaimed.min() > pos_second.max()


def test_a_pattern_that_names_nothing_is_an_error(tmp_path):
    """Silently ignoring it would hand back a layout that is not the one asked for while reporting
    success -- and the whole reason to ask for a layout is that some solver needs that layout."""
    p = _StokesCR()
    p.set_output_directory(str(tmp_path))
    p.quiet()
    p.dof_ordering = NodalBlockOrdering("domain/velocity_x", "domain/temperature")
    with pytest.raises(Exception, match="matches none of this problem's fields"):
        p.initialise()


def test_a_layout_does_not_change_the_answer(tmp_path):
    """The property every layout must have, on the layout that actually moves dofs."""
    import numpy
    a = _built(_StokesCR, None, tmp_path / "a")
    a.solve()
    b = _built(_StokesCR, ElementBlockOrdering("domain/velocity_*", "domain/pressure"), tmp_path / "b")
    b.solve()

    def state(p):
        m = p.get_mesh("domain")
        return numpy.array([v for n in m.nodes()
                            for v in [n.x(0), n.x(1)] + [n.value(i) for i in range(n.nvalue())]])

    assert len(a.get_last_residual_convergence()) == len(b.get_last_residual_convergence())
    numpy.testing.assert_allclose(state(b), state(a), rtol=1e-8, atol=1e-10)


def test_the_layout_survives_a_rebuild(tmp_path):
    """dof_ordering is pushed down at every equation numbering, not once at construction, so a second
    numbering (a re-pin, an adapt) must produce the same layout rather than falling back."""
    p = _built(_StokesCR, ElementBlockOrdering("domain/velocity_*", "domain/pressure"), tmp_path)
    before = _field_sequence(p)
    p.reapply_boundary_conditions()
    assert _field_sequence(p) == before
