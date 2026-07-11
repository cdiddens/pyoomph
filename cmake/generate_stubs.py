#!/usr/bin/env python3
"""Generate .pyi type stubs for the pyoomph._core extension module.

This reproduces what the old top-level build script did with
`pybind11-stubgen` + `src/pybind/patch_stubs.py`, but is invoked as a
POST_BUILD step on the `_core` CMake target so it happens automatically as
part of `./configure && make` / `pip install .` via scikit-build-core.

Design goals, matching the old script's spirit:
  - Purely a developer convenience (IDE completion) - never required to
    actually build/install/import pyoomph.
  - Missing `pybind11-stubgen`, or any failure while generating/patching
    stubs, is reported to stderr but always exits 0 so it can never break
    an actual wheel build (e.g. in a minimal manylinux/cibuildwheel image
    that doesn't have pybind11-stubgen installed).

Usage (called from CMakeLists.txt):
    generate_stubs.py --module-dir DIR --module-name _core \
        --stage-dir DIR [--patch-script PATH] [--python EXE]

On success, `<stage-dir>` ends up containing either:
  - `<module-name>.pyi`               (flat module, the common case), or
  - `<module-name>/__init__.pyi` (+.pyi siblings) (module with submodules)
so that `install(DIRECTORY "<stage-dir>/" DESTINATION pyoomph OPTIONAL)`
in CMakeLists.txt can drop it straight next to the built extension.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", required=True,
                         help="Directory containing the built extension (added to PYTHONPATH)")
    parser.add_argument("--module-name", default="_core",
                         help="Import name of the extension module (default: _core)")
    parser.add_argument("--stage-dir", required=True,
                         help="Directory the final stub(s) are normalized into")
    parser.add_argument("--patch-script", default=None,
                         help="Optional patch_stubs.py-style script run on the generated stub")
    parser.add_argument("--python", default=sys.executable,
                         help="Python interpreter to use (default: this interpreter)")
    args = parser.parse_args()

    try:
        import pybind11_stubgen  # noqa: F401
    except ImportError:
        print("pybind11-stubgen not installed - skipping .pyi stub generation "
              "for pyoomph._core (pip install pybind11-stubgen to enable "
              "IDE completion stubs)", file=sys.stderr)
        return 0

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = args.module_dir + os.pathsep + env.get("PYTHONPATH", "")

    # pybind11-stubgen's CLI has changed across versions - some accept
    # --no-setup-py, others reject it. Try with it first (stderr suppressed,
    # like the old `2>/dev/null`), and fall back to without it, matching the
    # old bash script's two-attempt logic.
    base_cmd = [args.python, "-m", "pybind11_stubgen", "-o", str(stage_dir), args.module_name]

    result = run(base_cmd + ["--no-setup-py"], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        result = run(base_cmd, env=env)
        if result.returncode != 0:
            print("Error in stub generation", file=sys.stderr)
            return 0

    # A fallback run without --no-setup-py may drop a setup.py we don't want
    # installed into the wheel.
    stray_setup_py = stage_dir / "setup.py"
    if stray_setup_py.exists():
        stray_setup_py.unlink()

    flat_stub = stage_dir / f"{args.module_name}.pyi"
    pkg_init = stage_dir / args.module_name / "__init__.pyi"
    if pkg_init.exists():
        target = pkg_init
    elif flat_stub.exists():
        target = flat_stub
    else:
        print(f"pybind11-stubgen did not produce a stub for "
              f"{args.module_name!r} - skipping", file=sys.stderr)
        return 0

    if args.patch_script:
        patch_script = Path(args.patch_script)
        if patch_script.exists():
            patch_result = run([args.python, str(patch_script), str(target)])
            if patch_result.returncode != 0:
                print("Error while patching generated stubs (continuing anyway)",
                      file=sys.stderr)
        else:
            print(f"patch script {patch_script} not found - skipping patch step",
                  file=sys.stderr)

    print(f"Generated stub: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
