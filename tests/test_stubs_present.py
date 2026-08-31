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

"""The wheels must actually carry _pyoomph_core.pyi - a cheap check that runs on every wheel.

The stub is what gives editors and type checkers the API of the compiled extension, and the wheel
ships ``py.typed`` next to it, which tells them that information is present. A wheel with the marker
and no stub is therefore worse than one with neither: Pyright and mypy stop looking and find a
binary they cannot introspect, so the user gets no completion at all rather than an obvious gap.

This exists because that is precisely what shipped. Every Windows wheel lacked the stub - the
extension could not be imported under ``nanobind.stubgen`` at build time (see
``cmake/stubgen_launcher.py``), generation was best-effort, and the failure was printed and
swallowed. Nothing else in the suite looks at the package's contents, so 2489 tests passed on that
wheel. ``PYOOMPH_REQUIRE_STUBS`` now fails the BUILD in that situation; this fails the wheel, which
is the guard that survives someone turning the option off.

Like test_openmp_present.py, the hard assertion is gated on an environment variable the wheel jobs
set, because a from-source build is entitled to have been configured without stubs.
"""

import os
import pathlib

import pytest

import pyoomph


def _stub_path() -> pathlib.Path:
    return pathlib.Path(pyoomph.__file__).parent / "_pyoomph_core.pyi"


def _require() -> None:
    if not os.environ.get("PYOOMPH_EXPECT_STUBS", ""):
        pytest.skip("PYOOMPH_EXPECT_STUBS is not set; this build may legitimately have no stub")


def test_the_stub_is_installed_beside_the_extension():
    _require()
    stub = _stub_path()
    assert stub.is_file(), (
        f"no _pyoomph_core.pyi in {stub.parent} - this build promised a stub (py.typed is shipped "
        f"there) and has none, so type checkers see an untyped extension")


def test_the_stub_describes_the_extension_rather_than_being_a_stump():
    # A stub can exist and still be useless: nanobind.stubgen writes the file before it walks the
    # module, so an import that dies midway can leave a header and nothing else. Ask for a handful
    # of the types the Python layer is built on rather than trusting the file's existence.
    _require()
    stub = _stub_path()
    assert stub.is_file(), f"no _pyoomph_core.pyi in {stub.parent}"
    text = stub.read_text(encoding="utf-8", errors="replace")

    expected = ["class Problem", "class Expression", "class FiniteElementCode", "class Mesh"]
    missing = [name for name in expected if name not in text]
    assert not missing, (
        f"_pyoomph_core.pyi is {len(text)} characters and does not declare {missing} - "
        f"stub generation produced a truncated file")


def test_the_py_typed_marker_and_the_stub_agree():
    # The two must ship together. The marker without the stub is the failure this file is here for;
    # the stub without the marker would make every type checker ignore the stub instead.
    _require()
    stub = _stub_path()
    marker = stub.parent / "py.typed"
    assert marker.is_file(), f"no py.typed in {stub.parent}, so nothing will read the stub"
    assert stub.is_file(), f"py.typed is present in {stub.parent} but the stub it promises is not"
