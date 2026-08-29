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

"""Find out WHICH module the tcc-compiled JIT DLL is missing on Windows.

tests/test_math_functions.py's tccbox arm fails on Windows with

    DLL ...\\ode.dll could not be loaded. Error code: 126

and 126 is ERROR_MOD_NOT_FOUND: the DLL was compiled and written, and then a module it *depends on*
could not be found. Which one is recorded in the DLL's own import table and nowhere else, so no
amount of reading the sources on another platform will answer it - hence this, which JITs the
smallest possible problem with tccbox and dumps the import table of whatever it produced.

Windows compiles the JIT code differently from the other platforms: no -nostdinc/-nostdlib, so tcc
brings its own headers and runtime and names its own CRT (see pyoomph/generic/ccompiler.py), while
jitbridge.h's hand-written prototypes are still in force via PYOOMPH_TCC_TO_MEMORY. That mismatch is
the suspect, and the import table is what confirms or clears it.

Diagnostic only: it prints, it never fails, and nothing depends on its exit status.
"""

import glob
import os
import subprocess
import sys


def _say(message):
    print("windows_tcc_diag: " + message, flush=True)


def _dump_imports(dll):
    """Print the DLLs `dll` imports, using whichever tool the runner happens to have."""
    for tool, args in (("objdump", ["-p", dll]), ("dumpbin", ["/DEPENDENTS", dll])):
        try:
            out = subprocess.run([tool] + args, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            _say("%s unavailable (%s)" % (tool, e))
            continue
        if out.returncode != 0:
            _say("%s failed: %s" % (tool, (out.stderr or "").strip()[:400]))
            continue
        _say("---- %s on %s ----" % (tool, dll))
        for line in (out.stdout or "").splitlines():
            # objdump prints "DLL Name: foo.dll"; dumpbin just lists them indented.
            if "DLL Name" in line or line.strip().lower().endswith(".dll"):
                print("    " + line.strip(), flush=True)
        return True
    _say("no tool available to read the import table")
    return False


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tccdiag_out"

    code = (
        # All FIVE functions the failing test uses, not just erf: the first version of this probe
        # used erf alone, and the DLL loaded and solved fine, importing nothing but msvcrt.dll. The
        # C99 inverse hyperbolics are the ones legacy msvcrt lacks, so they are the likeliest reason
        # the test's DLL and this one differ - and a probe that does not reproduce the failure proves
        # nothing at all.
        "from pyoomph import Problem, ODEEquations\n"
        "from pyoomph.expressions import var_and_test, erf, erfc, asinh, acosh, atanh\n"
        "class E(ODEEquations):\n"
        "    def define_fields(self):\n"
        "        self.define_ode_variable('u')\n"
        "    def define_residuals(self):\n"
        "        u, v = var_and_test('u')\n"
        "        f = erf(0.7) + erfc(0.7) + asinh(0.7) + acosh(2.0) + atanh(0.7)\n"
        "        self.add_residual((u - f) * v)\n"
        "class P(Problem):\n"
        "    def define_problem(self):\n"
        "        self.add_equations(E() @ 'ode')\n"
        "with P() as p:\n"
        "    p.set_c_compiler('tccbox')\n"
        "    p.set_output_directory(%r)\n"
        "    try:\n"
        "        p.solve()\n"
        "        print('DIAG: the tcc JIT DLL loaded and solved fine here')\n"
        "    except Exception as e:\n"
        "        print('DIAG: reproduced the failure:', e)\n" % (outdir,)
    )
    # A separate process on purpose: the failure under investigation is a DLL load, and a load that
    # goes wrong can take the interpreter with it.
    # cwd=outdir's parent, never the checkout: a repository copy of pyoomph/ on sys.path has no
    # compiled extension, so the child would fail with "No module named pyoomph._pyoomph_core"
    # before reaching the JIT at all - which is exactly how this probe first came back empty.
    workdir = os.path.dirname(os.path.abspath(outdir)) or "."
    os.makedirs(workdir, exist_ok=True)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800,
                         cwd=workdir)
    for stream in (run.stdout, run.stderr):
        for line in (stream or "").splitlines():
            if line.startswith("DIAG:") or "could not be loaded" in line or "Error code" in line:
                _say(line.strip())
    _say("the JIT run exited with %d" % run.returncode)

    # .so/.dylib too, so that a dry run of this probe on another platform reports what it really
    # found rather than "no DLL", which would read like the Windows failure it is looking for.
    produced = []
    for pattern in ("*.dll", "*.so", "*.dylib"):
        produced += glob.glob(os.path.join(outdir, "**", pattern), recursive=True)
    produced = sorted(produced)
    sources = sorted(glob.glob(os.path.join(outdir, "**", "*.c"), recursive=True))
    _say("produced %d shared librar(y/ies) and %d C file(s) under %s" % (len(produced), len(sources), outdir))
    if not produced:
        # Worth knowing on its own: if tcc never wrote a DLL, the failure is at COMPILE time and the
        # import-table theory is wrong from the start.
        _say("nothing was produced at all, so this is not a load-time dependency problem")
        return 0
    for dll in produced:
        _dump_imports(dll)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never fail the job over a probe
        _say("the probe itself failed: %r" % (e,))
        sys.exit(0)
