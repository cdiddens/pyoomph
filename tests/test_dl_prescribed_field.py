#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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


# A DL field prescribed by a DirichletBC or an InitialCondition must reproduce a field that DL
# represents exactly, i.e. any linear one, on every element shape and order.
#
# The DirichletBC used to be imposed as a value at the centre of [s_min,s_max]^dim plus one difference
# quotient per local direction. Every shape_at_s_DL is [1, s_0, s_1, ...], so that is right on quads,
# bricks and lines, whose reference element is centred on s=0, and wrong on simplices, where s=0 is a
# vertex: the constant was set from g on the hypotenuse and the slopes from points outside the element.
# Measured with the linear g below, relative L2 errors of 0.27 on triangles and 0.85 on tets. It is now
# fitted onto the basis exactly as the DL initial condition is (mesh.cpp, set_DL_Dirichlet_condition).
#
# Wedges and pyramids could not be prescribed at all: the fit's sample lattice asks
# local_coord_is_valid(), which those elements did not implement, so initialise() raised an
# OomphException for either condition. They implement it now (wedges_and_pyramids.hpp).
#
# Not covered here: carrying a DL field through a REFINEMENT. On every element without a tensor tree
# (triangles, tets, wedges, pyramids) further_build() still copies only the constant from the father.

import sys
import os

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, LineMesh

sys.path.insert(0, os.path.dirname(__file__))
from box_mesh_3d import MixedBoxMesh3D


class _Prescribed(Equations):
    """A DL field whose own equation (c = 0) only gives it a Jacobian row; the condition overrides it."""

    def define_fields(self):
        self.define_scalar_field("c", "DL")

    def define_residuals(self):
        c, ct = var_and_test("c")
        self.add_weak(c, ct)


class _PrescribedC2(_Prescribed):
    """The same, plus a C2 field, so that the elements are the quadratic ones of their family."""

    def define_fields(self):
        super().define_fields()
        self.define_scalar_field("w", "C2")

    def define_residuals(self):
        super().define_residuals()
        w, wt = var_and_test("w")
        self.add_weak(w, wt)


class _LinearDL(Problem):
    def __init__(self, mesh, dim, how, order="C1"):
        super().__init__()
        self.m, self.dim, self.how, self.order = mesh, dim, how, order

    def define_problem(self):
        self.add_mesh(self.m)
        x = var("coordinate")
        g = 0.3 + 1.7 * x[0]
        if self.dim > 1:
            g += -0.9 * x[1]
        if self.dim > 2:
            g += 0.6 * x[2]
        eqs = _PrescribedC2() if self.order == "C2" else _Prescribed()
        eqs += DirichletBC(c=g) if self.how == "dbc" else InitialCondition(c=g)
        eqs += IntegralObservables(err=(var("c") - g) ** 2, gnorm=g ** 2)
        self.add_equations(eqs @ "domain")


_CASES = {
    "line": (lambda: LineMesh(N=4), 1),
    "quad": (lambda: RectangularQuadMesh(N=3), 2),
    "tri": (lambda: RectangularQuadMesh(N=3, split_in_tris="alternate_left"), 2),
    "tri_crossed": (lambda: RectangularQuadMesh(N=2, split_in_tris="crossed"), 2),
}
for _kind in ("hex", "tet", "wedge", "pyr", "hex_pyr_wedge", "all_four"):
    _CASES[_kind] = ((lambda k=_kind: MixedBoxMesh3D(kind=k, N=2)), 3)


def _rel_error(tmp_path, case, how, order="C1"):
    mk, dim = _CASES[case]
    p = _LinearDL(mk(), dim, how, order)
    p.set_output_directory(str(tmp_path))
    p.quiet()
    p.initialise()
    m = p.get_mesh("domain")
    return abs(float(m.evaluate_observable("err")) / float(m.evaluate_observable("gnorm"))) ** 0.5


@pytest.mark.parametrize("how", ["dbc", "ic"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_linear_field_is_exact(tmp_path, case, how):
    rel = _rel_error(tmp_path, case, how)
    assert rel < 1e-7, "a linear %s on a DL field is off by %.3e (relative L2) on %s" % (how, rel, case)


@pytest.mark.parametrize("how", ["dbc", "ic"])
@pytest.mark.parametrize("case", ["hex", "tet", "wedge", "pyr"])
def test_linear_field_is_exact_on_quadratic_elements(tmp_path, case, how):
    """C2 wedges and pyramids have their own element classes and shape_at_s_DL."""
    rel = _rel_error(tmp_path, case, how, order="C2")
    assert rel < 1e-7, "a linear %s on a DL field is off by %.3e (relative L2) on C2 %s" % (how, rel, case)
