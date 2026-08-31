#!/usr/bin/env python3
"""Why does pyoomph's PETSc autodetection fail under the COMPLEX arch on macOS arm64?

The macOS arm64 tutorial job of 30th August 2026 handed
rayleigh_benard_azimuthal_stability.py the complex PETSc - the harness said so, and its own
check_petsc() probe had confirmed that petsc4py imports from there and reports complex scalars -
and the script still started with "falling back to the 'accelerate' solver, since no better solver
was found" and solved its eigenproblems with spectra. Accelerate's LU then took 55 s per
factorisation at ndof=10888 and the script hit the 1800 s timeout, while every other platform
finished it in 413-717 s on Pardiso.

So _have_petsc_mumps() returned False in that process. Which of its three parts said no is exactly
what a bare `except:` throws away (pyoomph/__init__.py), hence this script: it reproduces the
autodetection step by step, in a child process per environment, and prints the exception instead of
swallowing it.

Two environments per arch, because the difference between them is the leading suspect:

  "tutorial"  PYTHONPATH -> the chosen arch, DYLD_LIBRARY_PATH left at env.sh's - i.e. real/lib.
              That is all citools/test_all_tutorial_scripts.py's env_with_petsc() changes.
  "suite"     PYTHONPATH and DYLD_LIBRARY_PATH both -> the chosen arch, which is what
              test_wheel_suite.yml sets ("The COMPLEX arch, not env.sh's real one") and where the
              complex SLEPc is known to work.

If the complex/tutorial cell is the only red one, the bug is that env_with_petsc() moves PYTHONPATH
and not the dynamic loader path: libpetsc.dylib and libslepc.dylib exist under BOTH arches with the
same leaf names, and DYLD_LIBRARY_PATH is searched by leaf name before an install_name or an RPATH.
petsc4py alone survives that - which is precisely why the harness's probe passed.

Run it on the runner as:  python -P citools/petsc_arch_diag.py
(-P so the checkout's source pyoomph/ does not shadow the installed wheel.)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Run in the child, one line of JSON out. Everything is wrapped: the point of the exercise is the
# reason for a failure, so no single failing step may stop the ones after it.
_PROBE = r'''
import json, os, sys, traceback
out = {"steps": {}, "images": []}

def step(name, fn):
    try:
        out["steps"][name] = {"ok": True, "value": fn()}
    except BaseException as e:
        out["steps"][name] = {"ok": False,
                              "error": "%s: %s" % (type(e).__name__, e),
                              "traceback": traceback.format_exc().splitlines()[-6:]}
        return None
    return out["steps"][name]["value"]

def _petsc():
    from petsc4py import PETSc
    return PETSc

step("import petsc4py", lambda: __import__("petsc4py").__file__)

def _scalar():
    import numpy
    return "complex128" if _petsc().ScalarType is numpy.complex128 else str(_petsc().ScalarType.__name__)
step("PETSc.ScalarType", _scalar)

# hasExternalPackage is the question pyoomph actually asks; setFactorSolverType (what
# citools/build_petsc_slepc.sh checks with) only records a string and would pass either way.
step("PETSc.Sys.hasExternalPackage('mumps')", lambda: bool(_petsc().Sys.hasExternalPackage("mumps")))
step("import slepc4py", lambda: __import__("slepc4py").__file__)
step("from slepc4py import SLEPc", lambda: str(__import__("slepc4py.SLEPc", fromlist=["SLEPc"])))

# The three imports pyoomph's own probe does, in its order.
step("pyoomph.solvers.petsc import", lambda: str(__import__(
    "pyoomph.solvers.petsc", fromlist=["PETSc", "PETSCMUMPSSolver", "SlepcMUMPSEigenSolver"])))
step("pyoomph._have_petsc_mumps()", lambda: bool(__import__("pyoomph", fromlist=["_have_petsc_mumps"])._have_petsc_mumps()))
step("pyoomph._have_pardiso()", lambda: bool(__import__("pyoomph", fromlist=["_have_pardiso"])._have_pardiso()))
step("pyoomph._have_accelerate()", lambda: bool(__import__("pyoomph", fromlist=["_have_accelerate"])._have_accelerate()))

# What the autodetection settles on - the two lines the tutorial log prints as
# "LINEAR SOLVER WAS SET TO:" and "Generalized Eigen Solver:".
def _linsolver():
    import pyoomph  # noqa: F401  - the autodetection runs on import
    from pyoomph.solvers.generic import get_default_linear_solver
    return str(get_default_linear_solver())
step("default linear solver", _linsolver)
def _eigsolver():
    import pyoomph
    return str(pyoomph._autodetect_eigen_solver())
step("autodetected eigen solver", _eigsolver)

# Which libpetsc/libslepc the process actually mapped. The names, not the search path, are the
# evidence: a complex petsc4py that mapped real/lib/libslepc.dylib is the whole bug.
# Only the libraries whose ARCH is in question - a full image list is a few hundred lines of
# system frameworks. Matched on the leaf name, since that is what DYLD_LIBRARY_PATH matches on.
def _interesting(path):
    leaf = path.rsplit("/", 1)[-1]
    return any(k in leaf for k in ("petsc", "slepc", "mumps", "scalapack", "gfortran", "libmpi"))

try:
    import ctypes, ctypes.util
    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    for i in range(libc._dyld_image_count()):
        n = libc._dyld_get_image_name(i).decode("utf-8", "replace")
        if _interesting(n):
            out["images"].append(n)
except BaseException:
    # Not macOS: /proc/self/maps says the same thing, so the script is still usable for a
    # local sanity run on Linux before it is dispatched.
    try:
        seen = []
        for line in open("/proc/self/maps"):
            n = line.rstrip().rsplit(" ", 1)[-1]
            if n.startswith("/") and n not in seen and _interesting(n):
                seen.append(n)
        out["images"] = seen
    except BaseException as e:
        out["images"] = ["<could not list loaded images: %s>" % e]

out["env"] = {k: os.environ.get(k, "") for k in
              ("PYTHONPATH", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PETSC_DIR")}
print("PROBE_JSON:" + json.dumps(out))
'''


def arch_lib(petsc_dir, arch):
    """$PETSC_DIR/$arch/lib, or wherever petsc4py actually sits - as the harness resolves it."""
    root = Path(petsc_dir) / arch
    for cand in (root / "lib", root):
        if (cand / "petsc4py").is_dir():
            return cand
    raise FileNotFoundError("no petsc4py under %s" % root)


def run(label, env):
    print("=" * 96)
    print("== %s" % label)
    print("   PYTHONPATH        = %s" % env.get("PYTHONPATH", ""))
    print("   DYLD_LIBRARY_PATH = %s" % env.get("DYLD_LIBRARY_PATH", ""))
    # -P: the checkout's source pyoomph/ has no compiled extension, and would shadow the wheel.
    p = subprocess.run([sys.executable, "-P", "-c", _PROBE],
                       capture_output=True, text=True, env=env, timeout=600)
    report = None
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON:"):
            report = json.loads(line[len("PROBE_JSON:"):])
        else:
            print("   | %s" % line)
    for line in p.stderr.splitlines()[-15:]:
        print("   ! %s" % line)
    if report is None:
        print("   the probe produced no report at all (exit %d) - it died before printing" % p.returncode)
        return label, None
    for name, res in report["steps"].items():
        if res["ok"]:
            print("   ok   %-42s %s" % (name, res["value"]))
        else:
            print("   FAIL %-42s %s" % (name, res["error"]))
            for t in res["traceback"]:
                print("        %s" % t)
    for n in report["images"]:
        print("   img  %s" % n)
    return label, report


def main():
    petsc_dir = os.environ.get("PETSC_DIR")
    if not petsc_dir:
        print("PETSC_DIR is unset - dispatch this with with_petsc=true on macos-arm64", file=sys.stderr)
        return 2
    arches = {"real": os.environ.get("PETSC_ARCH_REAL", "real"),
              "complex": os.environ.get("PETSC_ARCH_COMPLEX", "complex")}
    print("PETSC_DIR=%s  real=%s  complex=%s" % (petsc_dir, arches["real"], arches["complex"]))
    print("inherited DYLD_LIBRARY_PATH=%s" % os.environ.get("DYLD_LIBRARY_PATH", ""))
    for name in ("real", "complex"):
        d = Path(petsc_dir) / arches[name] / "lib"
        libs = sorted(p.name for p in d.glob("lib*petsc*") ) + sorted(p.name for p in d.glob("lib*slepc*"))
        print("  %s/lib: %s" % (arches[name], ", ".join(libs) or "(none)"))

    # env.sh is what the tutorial job sources, and it points the loader at the REAL arch. Reconstruct
    # that rather than trusting whatever this job happens to have exported, so the comparison holds
    # even when dispatched from debug_wheel_case.yml, which already sets the loader path to complex.
    mpi_lib = str(Path(petsc_dir) / "mpi" / "lib")
    real_lib = str(arch_lib(petsc_dir, arches["real"]))

    results = []
    for name in ("real", "complex"):
        lib = str(arch_lib(petsc_dir, arches[name]))
        # "tutorial": exactly what env_with_petsc() does - PYTHONPATH prepended, nothing else.
        env = dict(os.environ)
        env["PYTHONPATH"] = lib
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join([mpi_lib, real_lib])
        results.append(run("%s arch, tutorial env (DYLD -> real)" % name, env))
        # "suite": the loader path moved with the arch.
        env = dict(os.environ)
        env["PYTHONPATH"] = lib
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join([mpi_lib, lib])
        results.append(run("%s arch, suite env (DYLD -> same arch)" % name, env))

    print("=" * 96)
    print("SUMMARY")
    print("%-44s %-10s %-10s %-9s %s" % ("environment", "scalars", "slepc4py", "mumps", "linear solver"))
    for label, rep in results:
        if rep is None:
            print("%-44s %s" % (label, "probe died"))
            continue
        s = rep["steps"]
        def val(k):
            r = s.get(k)
            return "-" if r is None else (str(r["value"]) if r["ok"] else "FAILED")
        print("%-44s %-10s %-10s %-9s %s" % (label, val("PETSc.ScalarType"),
                                             "ok" if s.get("import slepc4py", {}).get("ok") else "FAILED",
                                             val("PETSc.Sys.hasExternalPackage('mumps')"),
                                             val("default linear solver")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
