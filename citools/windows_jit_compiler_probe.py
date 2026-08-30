#!/usr/bin/env python3
"""Which C compiler does pyoomph pick on Windows, in this process and in a child - and why.

Written for one specific unexplained failure. On the Windows job of the full-suite run
(33309250214, 30th August 2026) the three tests/test_two_live_problems.py cases died in the CHILD
process they launch with

    jitbridge.h(31): fatal error C1083: Cannot open include file: 'math.h'

out of an MSVC whose include list held only the VC directories and no Windows SDK - while the very
same pytest process compiled the other 2485 tests' element code without trouble. So the parent and
the child disagreed about which compiler to use, and nothing in the log said what either of them
chose. get_ccompiler() ranks by BaseCCompiler.compiler_quality (SystemCCompiler 5 beats TCCBox 4)
among those whose check_avail() says yes, so the question is which environment makes MSVC claim to
be available when it cannot compile stdlib.h.

Run through .github/workflows/debug_wheel_case.yml's `command` input, which needs no quoting:

    python -P citools/windows_jit_compiler_probe.py

It reports the current process and then re-runs itself as a child under several environments, since
the parent/child difference is the whole point. Every line is prefixed PROBE and is JSON.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile


def _system_compiler_facts():
    """What SystemCCompiler thinks, plus the error a real #include <math.h> compile gives.

    check_avail() compiles abort()/stdlib.h and returns a bare bool, and toolchain_located() only
    asks whether cl.exe was FOUND - neither says why a compile fails, which is exactly what is
    missing from the CI log.
    """
    facts = {}
    try:
        from pyoomph.generic.ccompiler import SystemCCompiler
    except Exception as e:
        return {"import_error": repr(e)}
    try:
        facts["check_avail"] = SystemCCompiler.check_avail()
    except Exception as e:
        facts["check_avail"] = "raised " + repr(e)
    try:
        inst = SystemCCompiler()
        facts["compiler_type"] = getattr(inst.comp, "compiler_type", None)
        facts["toolchain_located"] = inst.toolchain_located()
        # The include dirs only exist after initialize(), which toolchain_located() ran for MSVC.
        facts["include_dirs"] = list(getattr(inst.comp, "include_dirs", []) or [])[:12]
        d = tempfile.mkdtemp()
        src = os.path.join(d, "probe.c")
        with open(src, "w") as f:
            f.write("#include <math.h>\n#include <stdlib.h>\ndouble f(double x){return sqrt(x);}\n")
        try:
            inst.comp.compile([src], output_dir=d)
            facts["math_h_compile"] = "ok"
        except Exception as e:
            facts["math_h_compile"] = repr(e)[:600]
    except Exception as e:
        facts["instantiation"] = repr(e)
    return facts


def report(tag):
    out = {"tag": tag, "executable": sys.executable}
    for var in ("CC", "CXX", "INCLUDE", "LIB", "WindowsSdkDir", "VCINSTALLDIR"):
        out[var] = os.environ.get(var)
    out["path_head"] = os.environ.get("PATH", "").split(os.pathsep)[:6]
    for exe in ("gcc", "cl", "cc", "tcc"):
        out["which_" + exe] = shutil.which(exe)
    try:
        import distutils.ccompiler as dcc
        out["distutils_default"] = dcc.get_default_compiler()
    except Exception as e:
        out["distutils_default"] = repr(e)
    try:
        from pyoomph.generic.ccompiler import BaseCCompiler
        out["available_compilers"] = BaseCCompiler.available_compilers()
    except Exception as e:
        out["available_compilers"] = repr(e)
    out["system"] = _system_compiler_facts()
    print("PROBE " + json.dumps(out, default=str), flush=True)


def child(tag, env_overrides):
    env = dict(os.environ)
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    subprocess.run([sys.executable, "-P", os.path.abspath(__file__), tag], env=env, check=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(sys.argv[1])
    else:
        report("parent")
        # The environment differences between the wheel job (cibuildwheel driven from the MSYS2
        # shell, CC=gcc exported by the workflow step, /ucrt64/bin on PATH) and this debug lane
        # (git-bash, hosted-tool-cache Python) - one variable at a time, so whichever one flips the
        # answer is named rather than guessed at.
        ucrt = r"C:\msys64\ucrt64\bin"
        child("child_inherited", {})
        child("child_cc_gcc", {"CC": "gcc", "CXX": "g++"})
        child("child_no_cc", {"CC": None, "CXX": None})
        child("child_ucrt_first", {"PATH": ucrt + os.pathsep + os.environ.get("PATH", "")})
        child("child_no_gcc", {"PATH": os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if not os.path.isfile(os.path.join(p, "gcc.exe")))})
