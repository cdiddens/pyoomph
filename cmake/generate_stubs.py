#!/usr/bin/env python3
"""Generate .pyi type stubs for the pyoomph._core extension module.

This reproduces what the old top-level build script did with
`pybind11-stubgen` + `src/nanobind/patch_stubs.py`, but is invoked as a
POST_BUILD step on the `_core` CMake target so it happens automatically as
part of `./configure && make` / `pip install .` via scikit-build-core.
Since the switch to nanobind, stub generation uses nanobind's own bundled
`python -m nanobind.stubgen` (always available - nanobind is a hard build
dependency, unlike the old optional `pybind11-stubgen`).

Design goals, matching the old script's spirit:
  - Purely a developer convenience (IDE completion) - never required to
    actually build/install/import pyoomph.
  - Any failure while generating/patching stubs is reported to stderr but
    always exits 0 so it can never break an actual wheel build.

Usage (called from CMakeLists.txt):
    generate_stubs.py --module-dir DIR --module-name _core \
        --stage-dir DIR [--extra-copy-dir DIR] [--patch-script PATH] [--python EXE]

On success, `<stage-dir>` ends up containing either:
  - `<module-name>.pyi`               (flat module, the common case), or
  - `<module-name>/__init__.pyi` (+.pyi siblings) (module with submodules)
so that `install(DIRECTORY "<stage-dir>/" DESTINATION pyoomph OPTIONAL)`
in CMakeLists.txt can drop it straight next to the built extension.

`--extra-copy-dir` additionally mirrors the same stub into a second
location - normally the source-tree `pyoomph/` package directory - so that
static analyzers (Pylance/Pyright/mypy) editing the checked-out source can
resolve `pyoomph._core` even without a full `pip install`, since they never
see the build/install directory.
"""
import argparse
import os
import shutil
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
    parser.add_argument("--extra-copy-dir", action="append", default=[],
                         help="Additional directory (e.g. the source-tree pyoomph/ "
                              "package) to mirror the final stub(s) into, so editors "
                              "like Pylance can resolve pyoomph._core without needing "
                              "a full `pip install`. May be given multiple times.")
    parser.add_argument("--patch-script", default=None,
                         help="Optional patch_stubs.py-style script run on the generated stub")
    parser.add_argument("--python", default=sys.executable,
                         help="Python interpreter to use (default: this interpreter)")
    args = parser.parse_args()

    try:
        import nanobind.stubgen  # noqa: F401
    except ImportError:
        print("nanobind.stubgen not importable - skipping .pyi stub generation "
              "for pyoomph._core", file=sys.stderr)
        return 0

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = args.module_dir + os.pathsep + env.get("PYTHONPATH", "")

    # nanobind.stubgen always writes a single flat "<module>.pyi" file (no
    # "<module>/__init__.pyi" package-directory form the way pybind11-stubgen
    # could produce for modules with submodules).
    # -P/--include-private: pyoomph's Python layer calls a number of leading-underscore
    # methods directly (e.g. _set_current_codegen, _resolve_based_on_domain_name), which
    # nanobind.stubgen omits by default (unlike the old pybind11-stubgen, which always
    # included them) - keep them in the stub so editors/type-checkers can resolve them.
    base_cmd = [args.python, "-m", "nanobind.stubgen",
                "-m", args.module_name, "-O", str(stage_dir), "-P"]

    result = run(base_cmd, env=env)
    if result.returncode != 0:
        print("Error in stub generation", file=sys.stderr)
        return 0

    flat_stub = stage_dir / f"{args.module_name}.pyi"
    if flat_stub.exists():
        target = flat_stub
    else:
        print(f"nanobind.stubgen did not produce a stub for "
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

    for extra_dir in args.extra_copy_dir:
        dest_root = Path(extra_dir)
        dest_root.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            dest = dest_root / target.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(target, dest)
        else:
            dest = dest_root / target.name
            shutil.copy2(target, dest)
        print(f"Mirrored stub into: {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
