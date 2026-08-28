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

"""Layouts for the global degree-of-freedom numbering.

oomph-lib numbers every nodal value of a mesh before any element-internal one, which is the wrong
layout for the two things that care about it:

* a block preconditioner -- Hypre's BoomerAMG above all -- coarsens a vector system well only when the
  unknowns of one node are adjacent and strided, i.e. (node1.ux, node1.uy, node1.p), (node2.ux, ...);
* static condensation needs the dofs it eliminates from one element to be adjacent, so that a
  replicated MPI row split can cut between the blocks instead of through them.

Assign a layout (or a list of them) to :py:attr:`~pyoomph.generic.problem.Problem.dof_ordering`::

    problem.dof_ordering = NodalBlockOrdering("domain/velocity_x", "domain/velocity_y", "domain/pressure")

Fields are named by glob patterns over the problem's global field names -- the same vocabulary as
``problem.petsc_fieldsplit`` -- so ``"domain/velocity_*"`` works, an interface-only field is named by
its interface's full path (``"domain/top/lambda"``), and a moving mesh position is
``"domain/coordinate_x"`` rather than ``mesh_x``. ``problem.get_global_field_names()`` lists what is
available.

Several layouts compose, which is how a problem with more than one mesh is handled: they are applied
in the order given and a dof is claimed by the FIRST layout that names its field, so different meshes
(and different layouts on different meshes) do not interfere::

    problem.dof_ordering = [
        NodalBlockOrdering("bulk/velocity_*", "bulk/pressure"),
        ElementBlockOrdering("bulk/pressure_dg"),
        NodalBlockOrdering("bulk/top/lambda"),
    ]

Dofs no layout names keep their original relative order and follow the ordered ones. A pattern that
matches nothing raises, rather than quietly producing a different layout from the one asked for.

The permutation is applied inside ``assign_eqn_numbers()``, before anything reads the numbering, and
it is rank-local under MPI so a distributed run keeps each rank's contiguous range. It changes the
numbering only -- never which values are dofs, and never the answer.
"""

from typing import List, Sequence, Union


class BaseDofOrdering:
    """One layout: a set of fields, and how to group their dofs into blocks."""

    #: Whether a block is an element's dofs (True) or a node's (False).
    by_element = False

    def __init__(self, *fields: str):
        if not fields:
            raise ValueError("A dof ordering must name at least one field, e.g. "
                             "%s(\"domain/velocity_x\", \"domain/velocity_y\")" % type(self).__name__)
        for f in fields:
            if not isinstance(f, str):
                raise TypeError("Dof ordering fields are strings like \"domain/velocity_x\", got %r" % (f,))
        self.fields: List[str] = list(fields)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(repr(f) for f in self.fields))

    def _apply_to(self, problem):
        problem._add_dof_ordering_spec(self.by_element, self.fields)


class NodalBlockOrdering(BaseDofOrdering):
    """Keep the named fields of one node adjacent, in the order the fields are given.

    This is the layout a block preconditioner wants. Note that a constant block size additionally
    requires ``problem.apply_Dirichlet_BCs_by_dof_removing = False``: with the default dof removal a
    constrained value is not a dof at all, so the boundary nodes' blocks are short and the stride
    breaks.
    """
    by_element = False


class ElementBlockOrdering(BaseDofOrdering):
    """Keep the named fields of one element adjacent.

    A dof reachable from several elements is claimed by the first that reports it; a cell-interior
    bubble node belongs to exactly one element, which is what makes a Crouzeix-Raviart block (bubble
    velocity plus the element's pressure gradient modes) come out contiguous.
    """
    by_element = True


DofOrderingLike = Union[BaseDofOrdering, Sequence[BaseDofOrdering], None]


def _normalise(value: DofOrderingLike) -> List[BaseDofOrdering]:
    """Accept a single layout, a list of them, or None."""
    if value is None:
        return []
    if isinstance(value, BaseDofOrdering):
        return [value]
    out = list(value)
    for o in out:
        if not isinstance(o, BaseDofOrdering):
            raise TypeError("dof_ordering takes NodalBlockOrdering/ElementBlockOrdering instances, got %r" % (o,))
    return out
