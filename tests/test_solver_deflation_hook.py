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

# Every linear-solver backend must post-process the Newton increment.
#
# A deflation operator does not change the matrix or the right-hand side: it turns the solved
# increment into the deflated one by a scalar rescale, applied through
# GenericLinearSystemSolver._postprocess_newton_step (or _solve_newton_step, which wraps it). A
# backend that returns the raw solution therefore does not fail, or warn, or produce a wrong matrix -
# it silently takes the UNDEFLATED Newton step, and deflation does nothing at all.
#
# Which is exactly what the Accelerate backend did. On macOS arm64 - Apple silicon without
# PETSc/MUMPS, i.e. the default there - all three deflation tests failed while the same tests passed
# on Intel, which reaches MKL Pardiso. The wheel runs of 29th August 2026 established that W, U, R, J,
# M and G were identical to Linux's to the last digit, and the first deflated step still went to -41.6
# instead of +0.55: the ingredients were right and the rescale was never applied.
#
# The bug is invisible to any test that runs one backend, since the backends are chosen by what the
# machine has installed. It is equally invisible to one that IMPORTS them: pyoomph.solvers.accelerate
# does not import off a Mac, so a registry-based check ran green on Linux with the bug reintroduced -
# measured, not assumed. So this reads the source files instead, which every platform can do for every
# backend. Crude, but it is the shape of the defect: a missing call, not a wrong number.

"""Every linear-solver backend routes its solution through the deflation hook."""

import ast
import os

import pytest

import pyoomph.solvers


_HOOKS = ("_postprocess_newton_step", "_solve_newton_step")
# The entry points oomph calls to get a solved increment back.
_ENTRY_POINTS = ("solve_serial", "solve_distributed")
_SOLVER_DIR = os.path.dirname(os.path.abspath(pyoomph.solvers.__file__))
# generic.py is where the hooks are defined, so it names them for a different reason.
_SKIP_FILES = {"generic.py", "__init__.py"}


def _backends():
    """(file, class name, entry points it defines, its source) for every backend, by reading."""
    out = []
    for filename in sorted(os.listdir(_SOLVER_DIR)):
        if not filename.endswith(".py") or filename in _SKIP_FILES:
            continue
        path = os.path.join(_SOLVER_DIR, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            defines = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in _ENTRY_POINTS]
            if defines:
                out.append((filename, node.name, defines, ast.get_source_segment(text, node) or ""))
    return out


def test_there_are_backends_to_check():
    """Guards the guard: a wrong directory would otherwise make this file vacuously green."""
    found = _backends()
    assert len(found) >= 3, found
    names = {f for f, _, _, _ in found}
    assert "accelerate.py" in names and "pardiso.py" in names, sorted(names)


@pytest.mark.parametrize("filename,classname",
                         [(f, c) for f, c, _, _ in _backends()],
                         ids=[f"{f}:{c}" for f, c, _, _ in _backends()])
def test_the_backend_applies_the_deflation_rescale(filename, classname):
    entry = next(b for b in _backends() if b[0] == filename and b[1] == classname)
    _, _, defines, source = entry
    assert any(hook in source for hook in _HOOKS), (
        "%s in %s implements %s but never calls %s, so a deflated solve through this backend would "
        "silently take the undeflated Newton step - which is what Accelerate did on macOS arm64"
        % (classname, filename, "/".join(defines), " or ".join(_HOOKS)))
