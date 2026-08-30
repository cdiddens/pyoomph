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

"""Give cibuildwheel's Windows test interpreter a working ``tkinter``.

cibuildwheel tests Windows wheels with a CPython from the **nuget** packages
(``...\\cibuildwheel\\Cache\\nuget-cpython\\python.3.12.10\\tools``), and those deliberately ship
without Tcl/Tk: no ``_tkinter.pyd``, no ``Lib/tkinter``, no ``tcl/``. There is nothing to pip-install
instead - ``tkinter`` is a stdlib C extension that comes with the installer or not at all. So the one
test that imports pyoomph's bifurcation GUI in-process skipped on Windows and nowhere else, i.e. the
GUI went untested on the platform where its file handling is most likely to differ.

This copies the missing pieces out of a full CPython of the SAME minor version (the one
actions/setup-python installs in the job) into the nuget interpreter's own directory. Same minor
version is not a nicety: ``_tkinter.pyd`` is a version-specific extension module, and a 3.14 one
loaded by 3.12 would fail at import with a much less obvious message than "no module named tkinter".

Deliberately never fails: it is invoked from cibuildwheel's before-test hook, and a wheel that
otherwise works must not be held up because a test convenience could not be arranged. Every refusal
says why, and the test itself still carries ``pytest.importorskip("tkinter")``, so the outcome
without this script is a skip, exactly as before.
"""

import os
import shutil
import sys


def _say(message):
    print("windows_add_tkinter: " + message, flush=True)


def main():
    if os.name != "nt":
        _say("not Windows, nothing to do")
        return 0

    try:
        import tkinter  # noqa: F401
    except ImportError:
        pass
    else:
        _say("this interpreter already has tkinter, nothing to do")
        return 0

    # Argument first, environment second: cibuildwheel's before-test hook is where this runs, and
    # whether CIBW_TEST_ENVIRONMENT reaches that hook or only the test command itself is a detail of
    # cibuildwheel that should not decide whether the copy happens.
    source = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PYOOMPH_TK_SOURCE", "")
    if not source:
        _say("no donor directory given (argument or PYOOMPH_TK_SOURCE), so there is nothing to copy from")
        return 0
    if not os.path.isdir(source):
        _say("the donor directory %r does not exist" % (source,))
        return 0

    wanted = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PYOOMPH_TK_SOURCE_VERSION", "")
    running = "%d.%d" % sys.version_info[:2]
    if wanted and wanted != running:
        # A version-specific extension module in the wrong interpreter is worse than no tkinter: it
        # fails at import time, well away from anything that mentions Tk.
        _say("refusing to copy: the source is CPython %s but this is %s" % (wanted, running))
        return 0

    # A venv's stdlib is its base interpreter's, and before-test runs inside cibuildwheel's test
    # venv, so the files have to land in the nuget installation rather than in sys.prefix.
    target = sys.base_prefix
    _say("copying Tk from %s into %s" % (source, target))

    copied = []
    # (relative path in the source, relative path in the target, whether it is a directory)
    items = [(os.path.join("Lib", "tkinter"), os.path.join("Lib", "tkinter"), True),
             (os.path.join("DLLs", "_tkinter.pyd"), os.path.join("DLLs", "_tkinter.pyd"), False),
             ("tcl", "tcl", True)]
    # The Tcl/Tk runtime DLLs are named per release (tcl86t.dll, tk86t.dll, and on newer builds
    # tcl90.dll/tk90.dll), so they are matched rather than listed.
    dll_dir = os.path.join(source, "DLLs")
    if os.path.isdir(dll_dir):
        for name in os.listdir(dll_dir):
            low = name.lower()
            if low.endswith(".dll") and (low.startswith("tcl") or low.startswith("tk") or low.startswith("zlib")):
                items.append((os.path.join("DLLs", name), os.path.join("DLLs", name), False))

    for rel_src, rel_dst, is_dir in items:
        src = os.path.join(source, rel_src)
        dst = os.path.join(target, rel_dst)
        if not os.path.exists(src):
            _say("missing in the source, skipped: " + rel_src)
            continue
        if os.path.exists(dst):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if is_dir:
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(rel_dst)
        except Exception as e:
            _say("could not copy %s: %s" % (rel_src, e))

    _say("copied %d item(s): %s" % (len(copied), ", ".join(copied) or "none"))

    # Whether it worked is not a guess: the point of the exercise is that this import stops failing.
    import subprocess
    check = subprocess.run([sys.executable, "-c", "import tkinter; print(tkinter.TkVersion)"],
                           capture_output=True, text=True)
    if check.returncode == 0:
        _say("tkinter now imports, Tk version " + check.stdout.strip())
    else:
        _say("tkinter still does not import; the test will skip as before:\n"
             + (check.stderr or "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
