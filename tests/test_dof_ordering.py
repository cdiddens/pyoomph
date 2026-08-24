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
