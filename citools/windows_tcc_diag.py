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


_LIBM_PROBES = ("erf", "erfc", "asinh", "acosh", "atanh", "exp", "log", "sqrt", "pow",
                "sinh", "cosh", "tanh", "atan2", "fabs", "fmax", "fmin", "strcpy", "strlen")


def _which_libm_symbols_link(outdir):
    """Which of the functions the generated code may call can tcc actually LINK on this platform.

    tcc stops at the first undefined symbol, so one failing compile names one function and hides the
    rest. Each is tried on its own here, because the answer decides the fix: asinh, acosh and atanh
    have exact closed forms that could be supplied in the header, while erf and erfc to the 1e-14 the
    tests demand do not, and would need the symbols themselves - from the UCRT rather than the legacy
    msvcrt tcc links by default.
    """
    workdir = os.path.join(os.path.abspath(outdir), "symcheck")
    os.makedirs(workdir, exist_ok=True)
    ok, missing, skipped = [], [], []
    for name in _LIBM_PROBES:
        src = os.path.join(workdir, name + ".c")
        dst = os.path.splitext(src)[0] + (".dll" if os.name == "nt" else ".so")
        argument = '"x"' if name in ("strcpy", "strlen") else "0.5"
        if name == "strcpy":
            body = "char buf[8]; double f(void){ strcpy(buf, \"ab\"); return 0.0; }\n"
            decl = "char *strcpy(char *, const char *);\n"
        elif name == "strlen":
            body = "double f(void){ return (double)strlen(\"ab\"); }\n"
            decl = "unsigned long long strlen(const char *);\n"
        elif name in ("pow", "atan2", "fmax", "fmin"):
            body = "double f(double x){ return %s(x, 0.5); }\n" % name
            decl = "double %s(double, double);\n" % name
        else:
            body = "double f(double x){ return %s(x); }\n" % name
            decl = "double %s(double);\n" % name
        with open(src, "w") as f:
            f.write(decl + body)
        cmd = [sys.executable, "-m", "tccbox", "-shared", src, "-o", dst]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            skipped.append("%s (%s)" % (name, e))
            continue
        if os.path.exists(dst):
            ok.append(name)
        else:
            first = (out.stderr or out.stdout or "").strip().splitlines()
            missing.append("%s [%s]" % (name, first[0] if first else "no diagnostic"))
    _say("libm symbols tcc CAN link: " + (", ".join(ok) or "none"))
    _say("libm symbols tcc CANNOT link: " + (", ".join(missing) or "none"))
    if skipped:
        _say("not probed: " + ", ".join(skipped))


def _try_alternative_c_runtimes(outdir):
    """Can tcc be pointed at a C runtime that HAS the C99 functions?

    tcc links msvcrt by default, and msvcrt predates C99: erf, erfc, asinh, acosh, atanh, fmax and
    fmin are all absent from it, which is why the JIT compile fails. The host CPython is UCRT-based
    and ucrtbase.dll exports every one of them, so if tcc can be given that instead - it resolves
    imports through .def files in its own lib directory - the whole class of failure goes away with
    one flag and no numerics of our own. This reports which .def files exist and which -l name links.
    """
    try:
        import tccbox
    except Exception as e:
        _say("tccbox is not importable here (%r)" % (e,))
        return
    root = os.path.dirname(os.path.abspath(tccbox.__file__))
    defs = sorted(glob.glob(os.path.join(root, "**", "*.def"), recursive=True))
    _say("tccbox ships %d .def files: %s" % (len(defs), ", ".join(os.path.basename(d) for d in defs[:20]) or "none"))

    workdir = os.path.join(os.path.abspath(outdir), "runtimecheck")
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "probe.c")
    with open(src, "w") as f:
        # One function from each missing family, so a partial answer is still informative.
        f.write("double acosh(double);\ndouble erf(double);\ndouble fmax(double,double);\n"
                "double f(double x){ return acosh(x) + erf(x) + fmax(x, 0.5); }\n")
    for lib in ("ucrtbase", "ucrt", "api-ms-win-crt-math-l1-1-0", "msvcr120", "m"):
        dst = os.path.join(workdir, "with_%s%s" % (lib, ".dll" if os.name == "nt" else ".so"))
        cmd = [sys.executable, "-m", "tccbox", "-shared", src, "-l" + lib, "-o", dst]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            _say("-l%-28s could not run (%s)" % (lib, e))
            continue
        if os.path.exists(dst):
            _say("-l%-28s LINKS - this runtime has the C99 functions" % (lib,))
        else:
            first = (out.stderr or out.stdout or "").strip().splitlines()
            _say("-l%-28s no (%s)" % (lib, first[0] if first else "no diagnostic"))


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
        _which_libm_symbols_link(outdir)
        _try_alternative_c_runtimes(outdir)
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
