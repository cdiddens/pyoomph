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

# A multi-return callback is real by construction - its ABI takes a double* and fills a double*, so
# it can neither receive nor return an imaginary part. GiNaC has to be told, because its default for
# an unknown function is "possibly complex", and taking a real part DISTRIBUTES:
#     real_part(g*w) -> real_part(g)*real_part(w) - imag_part(g)*imag_part(w)
# Without the hooks the second term survives with a spurious imag_part(g) in it, so real_part could
# not be moved across a callback and imag_part(g) was not zero. That is what these tests pin.
#
# Scope, measured rather than assumed: no azimuthal or normal-mode stability result could be made to
# depend on this. The expansion never routes the residual through GiNaC's real_part - it derives two
# separate named residual contributions by expansion mode - and a user-written real_part() reaches
# the EXPANDED node (GiNaCMultiRetCallback, a pyginacstruct, whose real_part/imag_part/conjugate
# already answer itself/0/itself) before anything is printed. An azimuthal m=1 sweep over sqrt(g),
# subexpression(1/sqrt(g)), g**(3/2), absolute(g), real_part(g*u) and imag_part(g*u), with and
# without a moving mesh, gave byte-identical eigenvalues before and after the hooks were added. So
# these are consistency tests between the two stages of the same callback, not a repair of a wrong
# number - keep them cheap and symbolic.
#
# They are written as invariants of a REAL quantity rather than as a comparison against the same
# function spelled out symbolically: exp(u) and a callback computing exp(u) are NOT interchangeable
# under real_part, because GiNaC does not know that the field u is real either, so real_part of the
# symbolic one legitimately expands into something else.

import numpy

from pyoomph.expressions import var, real_part, imag_part
from pyoomph.expressions.cb import CustomMultiReturnExpression


class _CB(CustomMultiReturnExpression):
    """f(a) = (exp(a), a**3), in C and in Python."""

    def get_num_returned_scalars(self, nargs):
        return 2

    def generate_c_code(self):
        return """
            result_list[0]=exp(arg_list[0]);
            result_list[1]=arg_list[0]*arg_list[0]*arg_list[0];
            if (flag) { derivative_matrix[0]=exp(arg_list[0]); derivative_matrix[1]=3.0*arg_list[0]*arg_list[0]; }
        """

    def eval(self, flag, arg_list, result_list, derivative_matrix):
        a = arg_list[0]
        result_list[0] = numpy.exp(a)
        result_list[1] = a ** 3
        if flag:
            derivative_matrix[0, 0] = numpy.exp(a)
            derivative_matrix[1, 0] = 3 * a * a


# =============================================================================================
# Symbolic: what the node itself answers
# =============================================================================================

def test_the_callback_node_is_real():
    cb = _CB()
    g = cb(var("u"))[0]
    assert (real_part(g) - g).is_zero()
    assert imag_part(g).is_zero()


def test_real_part_passes_straight_through_a_product_with_the_callback():
    """The invariant that matters: g is real, so real_part may be pushed past it untouched.

    This is what fails without the hooks - the product rule leaves -imag_part(g)*imag_part(w)
    behind, and that term is neither zero nor printable as C.
    """
    cb = _CB()
    g, w = cb(var("u"))[0], var("w")
    assert (real_part(g * w) - g * real_part(w)).is_zero()
    assert (imag_part(g * w) - g * imag_part(w)).is_zero()
    assert (real_part(g + w) - g - real_part(w)).is_zero()
    assert (imag_part(g * g)).is_zero()
