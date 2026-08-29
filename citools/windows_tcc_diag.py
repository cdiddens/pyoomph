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


def _rerun_the_compile(source):
    """Run the same tccbox command pyoomph runs, with its output shown and the result inspected."""
    try:
        import pyoomph
    except Exception as e:
        _say("cannot import pyoomph to locate the jit headers: %r" % (e,))
        return
    include = os.path.join(os.path.dirname(os.path.abspath(pyoomph.__file__)), "jitbridge")
    target = os.path.splitext(source)[0] + ".dll"
    cmd = [sys.executable, "-m", "tccbox", "-I", include, "-shared", "-rdynamic",
           "-DPYOOMPH_TCC_TO_MEMORY", "-Dsize_t=unsigned long long", source, "-o", target]
    _say("re-running: " + " ".join(cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        _say("could not run tccbox: %r" % (e,))
        return
    _say("tccbox exited %d" % out.returncode)
    for stream, name in ((out.stdout, "stdout"), (out.stderr, "stderr")):
        text = (stream or "").strip()
        _say("tccbox %s: %s" % (name, text if text else "(empty)"))
    _say("target exists afterwards: %s" % os.path.exists(target))
    if os.path.exists(target):
        _dump_imports(target)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tccdiag_out"

    # The functions are applied to an UNKNOWN, not to a number. erf(0.7) is folded to a literal at
    # code-generation time, so the first two versions of this probe produced a DLL with no libm call
    # in it at all - it loaded fine and imported nothing but msvcrt, which proved nothing about a
    # test whose DLL does call erf, erfc, asinh, acosh and atanh. This mirrors _Values in
    # tests/test_math_functions.py: one unknown per function, each written through a solved variable
    # so the calls and their derivatives both reach the generated code.
    code = (
        "from pyoomph import Problem, ODEEquations\n"
        "from pyoomph.expressions import var_and_test, erf, erfc, asinh, acosh, atanh\n"
        "FUNCS = {'erf': (erf, 0.0), 'erfc': (erfc, 0.0), 'asinh': (asinh, 0.0),\n"
        "         'acosh': (acosh, 2.0), 'atanh': (atanh, 0.0)}\n"
        "class E(ODEEquations):\n"
        "    def define_fields(self):\n"
        "        self.define_ode_variable('s')\n"
        "        for n in FUNCS:\n"
        "            self.define_ode_variable('u_' + n)\n"
        "    def define_residuals(self):\n"
        "        s, vs = var_and_test('s')\n"
        "        self.add_residual((s - 0.7) * vs)\n"
        "        for n, (f, shift) in FUNCS.items():\n"
        "            u, v = var_and_test('u_' + n)\n"
        "            self.add_residual((u - f(s + shift)) * v)\n"
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
        # import-table theory is wrong from the start. That is what happens here - and pyoomph does
        # not notice, because call_cmd only raises on a NONZERO exit and tcc exits 0 having written
        # nothing, so the failure surfaces much later as "DLL could not be loaded. Error code: 126",
        # which is Windows saying the file does not exist. So run the same compile again, by hand,
        # and show what tcc says when its output is not thrown away.
        _say("nothing was produced at all, so this is a COMPILE failure, not a load-time dependency")
        if sources:
            _rerun_the_compile(sources[0])
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
