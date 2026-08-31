#!/usr/bin/env python3
"""Run ``nanobind.stubgen`` with the build tree's DLL directories on the search path.

Only Windows needs this, and only at BUILD time. ``nanobind.stubgen`` works by importing
the extension and reading its docstrings, so the freshly built ``_pyoomph_core.pyd`` has to
load out of the CMake build tree - before delvewheel has vendored anything next to it. Since
CPython 3.8 the loader no longer searches ``PATH`` for an extension module's dependencies
(bpo-36085); only the directories handed to ``os.add_dll_directory()`` are consulted. The
MSYS2/UCRT64 DLLs the extension links against - ``libgomp-1.dll`` above all, since the wheel
is built with ``PYOOMPH_USE_OPENMP=ON`` - live in ``/ucrt64/bin``, which is on ``PATH`` and
therefore exactly where the loader will not look. The import failed, stub generation is
deliberately non-fatal, and the Windows wheel shipped without a ``.pyi`` while every other
platform had one; it also shipped ``py.typed``, so type checkers stopped looking and silently
saw an untyped module rather than an obviously missing one.

Restoring the pre-3.8 behaviour for this one subprocess is enough: every existing directory on
``PATH`` is registered, plus anything named in ``PYOOMPH_STUB_DLL_DIRS``. Nothing else in the
build inherits the widened search path, since this runs in its own process and exits.

On non-Windows this is a plain passthrough - ``os.add_dll_directory`` does not exist there and
the platform loaders use RPATH/``LD_LIBRARY_PATH`` as before.
"""
import os
import runpy
import sys


def _add_dll_directories() -> "list[str]":
    """Register the toolchain's DLL directories; return the ones that took."""
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return []

    # PYOOMPH_STUB_DLL_DIRS first, so an explicit answer wins over whatever PATH happens to
    # hold. Both are searched in order by the loader.
    raw = [
        *os.environ.get("PYOOMPH_STUB_DLL_DIRS", "").split(os.pathsep),
        *os.environ.get("PATH", "").split(os.pathsep),
    ]

    added, seen = [], set()
    for entry in raw:
        entry = entry.strip().strip('"')
        if not entry:
            continue
        try:
            key = os.path.normcase(os.path.abspath(entry))
        except (OSError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        if not os.path.isdir(entry):
            continue
        try:
            # The handle is deliberately leaked: the directory has to stay searchable for as
            # long as this process lives, and the process is this one command.
            add(entry)
        except (OSError, ValueError):
            continue
        added.append(entry)
    return added


def main() -> int:
    added = _add_dll_directories()
    if added and os.environ.get("PYOOMPH_STUB_VERBOSE"):
        print(f"stubgen_launcher: {len(added)} DLL directories registered", file=sys.stderr)

    # argv[0] is replaced so nanobind's own argparse usage message reads sensibly.
    sys.argv = ["nanobind.stubgen", *sys.argv[1:]]
    runpy.run_module("nanobind.stubgen", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
