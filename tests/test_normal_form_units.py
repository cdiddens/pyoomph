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

# The two pure-Python helpers of the normal form, tested without a solve.
#
# They are here rather than in tests/branch_switch_worker.py because they need no Problem at all, and
# because both of them exist to catch an input that a working solve never produces -- so a test that
# has to build a bifurcation first cannot reach them.

import numpy
import pytest

from pyoomph.generic.bifurcation_tools import _as_real_eigenvector, _fd_directional_step


def test_a_rotated_real_eigenvector_comes_back_real():
    """A real eigenvalue's eigenvector is only determined up to a scalar, and on a COMPLEX PETSc
    build that scalar is complex. numpy.real() of exp(i*phi)*v is cos(phi)*v -- a vector of roundoff
    for an unlucky phase, which the caller then normalises up into the direction the whole normal
    form is built out of."""
    v = numpy.array([1.0, -2.0, 0.5, 3.0])
    for phi in (0.0, 0.9, numpy.pi / 2 - 1e-9, 2.7, -1.3):
        got = _as_real_eigenvector(v * numpy.exp(1j * phi), "test")
        assert got.dtype == numpy.float64
        # Up to an overall sign, which is free for an eigenvector.
        assert min(float(numpy.linalg.norm(got - v)), float(numpy.linalg.norm(got + v))) < 1e-12, phi


def test_the_phase_near_pi_over_two_is_what_the_naive_real_part_loses():
    """The case the helper exists for: at phi = pi/2 the naive numpy.real() is exactly zero."""
    v = numpy.array([1.0, -2.0, 0.5, 3.0])
    rotated = v * numpy.exp(1j * (numpy.pi / 2))
    assert float(numpy.linalg.norm(numpy.real(rotated))) < 1e-15, "the naive real part is nothing"
    got = _as_real_eigenvector(rotated, "test")
    assert abs(float(numpy.linalg.norm(got)) - float(numpy.linalg.norm(v))) < 1e-12


def test_a_genuinely_complex_vector_is_refused():
    """Not real up to any phase -- that is a Hopf's eigenvector, not a real bifurcation's."""
    v = numpy.array([1.0 + 0j, 1.0j, -1.0 + 0j, 0.3j])
    with pytest.raises(RuntimeError, match="not real up to a phase"):
        _as_real_eigenvector(v, "the critical eigenvector")


def test_the_tolerance_is_a_relative_one():
    v = numpy.array([1.0, -2.0, 0.5, 3.0])
    n = float(numpy.linalg.norm(v))
    ok = v + 1j * numpy.array([0.0, 0.0, 0.0, 1e-10 * n])
    bad = v + 1j * numpy.array([0.0, 0.0, 0.0, 1e-4 * n])
    _as_real_eigenvector(ok, "test")                     # under tol, accepted
    with pytest.raises(RuntimeError):
        _as_real_eigenvector(bad, "test")


def test_a_real_input_is_returned_as_a_copy():
    """numpy.real() of a complex array is a VIEW, and normalising it in place used to rescale the
    problem's own stored eigenvector as a side effect."""
    v = numpy.array([1.0, 2.0, 3.0])
    got = _as_real_eigenvector(v, "test")
    got *= 7.0
    assert v[0] == 1.0, "the input must not be touched"


def test_the_fd_step_is_relative_to_both_the_state_and_the_direction():
    """The step used to be fd_eps*direction with direction unit-normalised in the EUCLIDEAN dof norm,
    i.e. fd_eps/sqrt(N) per dof -- below the roundoff floor of the dofs on any real mesh, and worse
    the finer the mesh."""
    eps = 1e-5
    for N in (10, 1000, 100000):
        d = numpy.ones(N) / numpy.sqrt(N)              # unit 2-norm, entries ~ N^-1/2
        u = numpy.full(N, 3.0)
        step = _fd_directional_step(u, d, eps)
        # The LARGEST dof moves by eps*|u|_inf, whatever N is.
        assert abs(step * float(numpy.max(numpy.abs(d))) - eps * 3.0) < 1e-18 * N
    # And it follows the scale of the state, not an absolute constant.
    d = numpy.array([1.0, 0.0])
    assert _fd_directional_step(numpy.array([1e3, 0.0]), d, eps) == pytest.approx(eps * 1e3)
    # ...with a floor of 1, so a state that is identically zero still gets a usable step.
    assert _fd_directional_step(numpy.zeros(2), d, eps) == pytest.approx(eps)
