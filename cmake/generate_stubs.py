#!/usr/bin/env python3
"""Generate .pyi type stubs for the pyoomph._core extension module.

This reproduces what the old top-level build script did with
`pybind11-stubgen` + `src/nanobind/patch_stubs.py`, but is invoked as a
POST_BUILD step on the `_core` CMake target so it happens automatically as
part of `./configure && make` / `pip install .` via scikit-build-core.
Since the switch to nanobind, stub generation uses nanobind's own bundled
`python -m nanobind.stubgen` (always available - nanobind is a hard build
dependency, unlike the old optional `pybind11-stubgen`).

The stub is REQUIRED, not a convenience. It is what gives editors and type
checkers the API of the compiled extension, and the wheel also ships
`py.typed`, which tells them the information is there - a wheel without the
stub therefore promises typing and delivers an unintrospectable binary,
i.e. no completion at all rather than obviously none.

This script used to exit 0 unconditionally, on the reasoning that stubs are
a developer convenience and must never break a wheel build. That is exactly
how the Windows wheel shipped without a `.pyi` for as long as it did: the
extension failed to import under `nanobind.stubgen` (see
`cmake/stubgen_launcher.py`), the failure was printed and swallowed, and the
build stayed green. With `--required` (which CMake passes unless
PYOOMPH_REQUIRE_STUBS=OFF) any failure to produce the stub is fatal, so the
next such breakage stops the build that would have shipped it.

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
    parser.add_argument("--required", action="store_true",
                         help="Fail (exit non-zero) if the stub cannot be generated, instead of "
                              "reporting it and exiting 0. Passed by CMake unless the build sets "
                              "PYOOMPH_REQUIRE_STUBS=OFF.")
    args = parser.parse_args()

    # Every "cannot produce a stub" path funnels through this, so that --required cannot be
    # honoured in some of them and forgotten in others.
    def give_up(message: str) -> int:
        print(message, file=sys.stderr)
        if args.required:
            print("Stub generation is REQUIRED for this build and it failed. Re-run the build "
                  "with -DPYOOMPH_REQUIRE_STUBS=OFF to downgrade this to a warning.",
                  file=sys.stderr)
            return 1
        return 0

    try:
        import nanobind.stubgen  # noqa: F401
    except ImportError:
        return give_up("nanobind.stubgen not importable - cannot generate the .pyi stub "
                       "for pyoomph._core")

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
    # Routed through stubgen_launcher.py rather than "-m nanobind.stubgen" directly: on Windows
    # the extension is imported straight out of the build tree, where its MSYS2/UCRT64
    # dependencies are only on PATH - which CPython >= 3.8 does not search for extension DLLs.
    # See the module docstring there. On the other platforms the launcher is a passthrough.
    launcher = Path(__file__).with_name("stubgen_launcher.py")
    if launcher.exists():
        base_cmd = [args.python, str(launcher),
                    "-m", args.module_name, "-O", str(stage_dir), "-P"]
    else:
        base_cmd = [args.python, "-m", "nanobind.stubgen",
                    "-m", args.module_name, "-O", str(stage_dir), "-P"]

    result = run(base_cmd, env=env)
    if result.returncode != 0:
        return give_up("Error in stub generation")

    flat_stub = stage_dir / f"{args.module_name}.pyi"
    if flat_stub.exists():
        target = flat_stub
    else:
        return give_up(f"nanobind.stubgen did not produce a stub for {args.module_name!r}")

    if args.patch_script:
        patch_script = Path(args.patch_script)
        if patch_script.exists():
            patch_result = run([args.python, str(patch_script), str(target)])
            if patch_result.returncode != 0:
                # patch_stubs.py raises when a pattern it is replacing is not in the stub, which
                # means the binding it corrects has changed shape - the stub is then wrong about
                # a nullable return or a numpy-accepting parameter rather than merely unpatched.
                return give_up("Error while patching the generated stub")
        else:
            return give_up(f"patch script {patch_script} not found")

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
