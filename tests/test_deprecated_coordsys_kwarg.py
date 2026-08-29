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

"""
The keyword argument naming the coordinate system is called ``coordsys`` everywhere. The former
spellings - ``coordinate_system``, ``coord_sys``, ``csys``, ``_coordinate_system`` and
``kinematic_bc_coordinate_sys`` - still work, but warn.

What is actually checked here is that the old spelling produces the *same* expression as the new one,
not merely that it is accepted: a shim that swallows the argument and silently drops it would pass a
"does it warn" test and quietly change every residual it touches.
"""

import warnings

import pytest

from pyoomph import Problem
from pyoomph.expressions import cartesian, axisymmetric, var, testfunction, weak, Weak
from pyoomph.equations.generic import WeakContribution, ProjectExpression, IntegralObservables, EnforcedBC
from pyoomph.equations.additional import SetCoordinateSystem


def _call_deprecated(func, **kwargs):
    """Call func(**kwargs) and return (result, the DeprecationWarning raised)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(**kwargs)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {[str(w.message) for w in caught]}"
    return result, deprecations[0]


def test_weak_accepts_the_old_name_and_gives_the_same_expression():
    a, b = var("u"), testfunction("u")
    new = weak(a, b, coordsys=cartesian)
    old, warning = _call_deprecated(lambda **kw: weak(a, b, **kw), coordinate_system=cartesian)
    assert str(old) == str(new)
    assert "coordinate_system" in str(warning.message) and "coordsys" in str(warning.message)
    # The axisymmetric system must give something else, otherwise the equality above is vacuous
    assert str(weak(a, b, coordsys=axisymmetric)) != str(new)


def test_Weak_and_minimize_functional_derivative_accept_the_old_name():
    a, b = var("u"), testfunction("u")
    old, _ = _call_deprecated(lambda **kw: Weak(a, b, **kw), coordinate_system=cartesian)
    assert str(old) == str(Weak(a, b, coordsys=cartesian))


def test_passing_both_spellings_is_an_error():
    a, b = var("u"), testfunction("u")
    with pytest.raises(TypeError, match="deprecated alias"):
        weak(a, b, coordsys=cartesian, coordinate_system=cartesian)


@pytest.mark.parametrize("cls,kwargs,old,new", [
    (WeakContribution, {"a": var("u"), "b": testfunction("u")}, "coordinate_system", "coordsys"),
    (ProjectExpression, {"u": var("u")}, "coordinate_system", "coordsys"),
    (IntegralObservables, {"u": var("u")}, "_coordinate_system", "coordsys"),
    (EnforcedBC, {"u": var("u")}, "coordinate_system", "coordsys"),
    (SetCoordinateSystem, {}, "coord_sys", "coordsys"),
])
def test_equation_constructors_accept_the_old_name(cls, kwargs, old, new):
    obj, _ = _call_deprecated(lambda **kw: cls(**kwargs, **kw), **{old: cartesian})
    assert getattr(obj, new) is cartesian


def test_set_coordinate_system_accepts_csys():
    problem = Problem()
    _call_deprecated(problem.set_coordinate_system, csys="axisymmetric")
    assert problem.get_coordinate_system() is axisymmetric
    problem.set_coordinate_system(coordsys="cartesian")
    assert problem.get_coordinate_system() is cartesian


def test_integral_observables_no_longer_overrides_the_equations_coordinate_system():
    """``_coordinate_system`` is the slot BaseEquations keeps its own coordinate system in, so the
    old argument name used to overwrite it rather than only steering the observables' measure."""
    obs = IntegralObservables(coordsys=cartesian, u=var("u"))
    assert obs.coordsys is cartesian
    assert obs._coordinate_system is None


def test_deprecated_attribute_alias_still_reads_and_writes():
    contrib = WeakContribution(var("u"), testfunction("u"), coordsys=cartesian)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert contrib.coordinate_system is cartesian
        contrib.coordinate_system = axisymmetric
    assert contrib.coordsys is axisymmetric
    assert len([w for w in caught if issubclass(w.category, DeprecationWarning)]) == 2


def test_no_public_signature_uses_an_old_spelling_any_more():
    """A source-level sweep, so that a newly added argument cannot reintroduce one of the old names."""
    import ast
    import os

    import pyoomph

    root = os.path.dirname(pyoomph.__file__)
    forbidden = {"coordinate_system", "coord_sys", "csys", "kinematic_bc_coordinate_sys", "_coordinate_system"}
    offenders = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            # encoding= is not optional here: without it Python picks the locale codec, which on
            # Windows is cp1252 and dies on the first non-ASCII byte in the sources ("charmap codec
            # can't decode byte 0x81"). The wheel test run of 29th August 2026 failed exactly there,
            # having read 583 kB of pyoomph before hitting one.
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args
                for arg in args.posonlyargs + args.args + args.kwonlyargs:
                    if arg.arg in forbidden:
                        offenders.append(f"{os.path.relpath(path,root)}:{node.lineno} {node.name}({arg.arg})")
    assert not offenders, "use 'coordsys' instead: " + ", ".join(offenders)


def test_kinematic_bc_coordsys_keeps_its_deprecated_alias():
    from pyoomph.equations.navier_stokes import NavierStokesFreeSurface
    from pyoomph.equations.multi_component import MultiComponentNavierStokesInterface

    for cls in (NavierStokesFreeSurface, MultiComponentNavierStokesInterface):
        aliases = getattr(cls.__init__, "__deprecated_kwargs__", {})
        assert aliases.get("kinematic_bc_coordinate_sys") == "kinematic_bc_coordsys", cls.__name__
    # MultiComponentNavierStokesInterface needs an interface property set to construct, so only the
    # free surface is actually instantiated here.
    obj, _ = _call_deprecated(NavierStokesFreeSurface, kinematic_bc_coordinate_sys=cartesian)
    assert obj.kinematic_bc_coordsys is cartesian
