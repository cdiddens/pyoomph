#!/usr/bin/env python3
"""Why the Windows wheel ships no ``_pyoomph_core.pyi``, and whether the launcher fixes it.

Background. ``nanobind.stubgen`` generates the stub by IMPORTING the extension and reading its
docstrings, and CMake runs it as a POST_BUILD step - so the import happens against the ``.pyd``
sitting in the build tree, before delvewheel has vendored anything beside it. On the Windows CI
that import dies with ``ImportError: DLL load failed while importing _pyoomph_core``. Stub
generation is deliberately non-fatal, so the wheel builds anyway and every test passes; it just
has no ``.pyi`` in it, while carrying ``py.typed`` and thus promising one.

The hypothesis under test: the dependencies are perfectly present on ``PATH`` (that is where
``delvewheel repair --add-path C:/msys64/ucrt64/bin`` finds them later), but since CPython 3.8
the loader no longer searches ``PATH`` for an extension module's dependencies - only directories
registered with ``os.add_dll_directory()``. If that is right, then putting the directory on
``PATH`` changes nothing and registering it fixes everything, and the four stages below say so
in as many words.

The lane this runs in (debug_wheel_case.yml) installs a REPAIRED wheel, so the extension it can
reach has had its dependencies vendored into ``pyoomph.libs`` under delvewheel's mangled names.
Copying the ``.pyd`` alone into an empty directory reproduces the build-time situation exactly in
shape - an extension whose dependencies are not beside it - which is what the four stages need.
The DLL names differ from the build tree's; the loader rule being tested does not.

Stages, all four generating the same stub from the same isolated ``.pyd``:

  A  plain stubgen, dependency directory NOT reachable      expect FAIL   (baseline)
  B  plain stubgen, dependency directory on PATH            expect FAIL   (the root cause)
  C  launcher,      dependency directory on PATH            expect OK     (the fix)
  D  launcher,      dependency directory via the env var    expect OK     (explicit override)

Exits non-zero if the outcomes disagree with that, so the job goes red on a wrong diagnosis.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# A minimal PE import-table reader. objdump would do it, but that would make the
# diagnosis depend on an MSYS2 that may or may not be on the runner; the import
# directory is about forty lines to walk and needs nothing but the file itself.
# ---------------------------------------------------------------------------
def dll_imports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"{path} is not a PE image")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path} has no PE signature")

    coff = pe + 4
    n_sections, = struct.unpack_from("<H", data, coff + 2)
    opt_size, = struct.unpack_from("<H", data, coff + 16)
    opt = coff + 20
    magic, = struct.unpack_from("<H", data, opt)
    # The data directories start after the optional header's fixed part, which is the only
    # thing that differs between PE32 (0x10b) and PE32+ (0x20b).
    dirs = opt + (112 if magic == 0x20B else 96)
    import_rva, = struct.unpack_from("<I", data, dirs + 8)
    if not import_rva:
        return []

    sections = []
    sec = opt + opt_size
    for i in range(n_sections):
        off = sec + i * 40
        va, = struct.unpack_from("<I", data, off + 12)
        raw_size, = struct.unpack_from("<I", data, off + 16)
        raw_ptr, = struct.unpack_from("<I", data, off + 20)
        sections.append((va, raw_size, raw_ptr))

    def to_offset(rva: int) -> int | None:
        for va, raw_size, raw_ptr in sections:
            if va <= rva < va + raw_size:
                return raw_ptr + (rva - va)
        return None

    def cstring(off: int) -> str:
        end = data.index(b"\0", off)
        return data[off:end].decode("ascii", "replace")

    names, entry = [], to_offset(import_rva)
    if entry is None:
        return []
    while True:
        chunk = data[entry:entry + 20]
        if len(chunk) < 20 or chunk == b"\0" * 20:
            break
        name_rva, = struct.unpack_from("<I", chunk, 12)
        if not name_rva:
            break
        off = to_offset(name_rva)
        if off is not None:
            names.append(cstring(off))
        entry += 20
    return names


def stage(label: str, expect_ok: bool, cmd: list[str], env: dict, out_dir: Path) -> bool:
    """Run one stubgen attempt; return True when the outcome was the expected one."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    stub = out_dir / "_pyoomph_core.pyi"
    produced = stub.exists() and stub.stat().st_size > 0
    ok = proc.returncode == 0 and produced

    verdict = "OK  " if ok else "FAIL"
    detail = f"{stub.stat().st_size} bytes" if produced else "no stub"
    agrees = ok == expect_ok
    print(f"  [{verdict}] {label:<52} {detail:>14}   "
          f"{'as expected' if agrees else '*** UNEXPECTED ***'}")
    if not ok:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        for line in tail[-3:]:
            print(f"         | {line}")
    return agrees


def main() -> int:
    if os.name != "nt":
        print("Windows only - nothing to diagnose here.")
        return 0

    import pyoomph
    pkg = Path(pyoomph.__file__).parent
    pyd = next(iter(pkg.glob("_pyoomph_core*.pyd")), None)
    if pyd is None:
        print(f"No _pyoomph_core*.pyd in {pkg}", file=sys.stderr)
        return 1

    print(f"extension : {pyd}  ({pyd.stat().st_size} bytes)")
    print("imports   :")
    for name in sorted(dll_imports(pyd)):
        print(f"            {name}")

    # Where the repaired wheel's vendored dependencies live. delvewheel puts them in a
    # "<distribution>.libs" directory beside the package and renames them with a content hash.
    libs = next((d for d in pkg.parent.glob("pyoomph.libs") if d.is_dir()), None)
    print(f"vendored  : {libs}")
    if libs is not None:
        for dll in sorted(libs.glob("*.dll")):
            print(f"            {dll.name}")
    if libs is None:
        print("No pyoomph.libs directory - cannot stage the experiment", file=sys.stderr)
        return 1

    launcher = Path(__file__).resolve().parent.parent / "cmake" / "stubgen_launcher.py"
    print(f"launcher  : {launcher}  ({'present' if launcher.exists() else 'MISSING'})")
    if not launcher.exists():
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="stubgen-diag-"))
    isolated = tmp / "isolated"
    isolated.mkdir()
    # The .pyd ALONE: no vendored directory beside it, and no pyoomph/__init__.py, so
    # delvewheel's injected load-order patch does not run either. This is the build tree's
    # situation - an extension whose dependencies are somewhere else entirely.
    shutil.copy2(pyd, isolated / "_pyoomph_core.pyd")

    base = dict(os.environ)
    base["PYTHONPATH"] = str(isolated)
    # A PATH with nothing useful on it, so stage A cannot succeed by accident, and stages B/C
    # differ from it by exactly one entry.
    bare_path = os.pathsep.join(
        p for p in base.get("PATH", "").split(os.pathsep)
        if p and Path(p).name.lower() not in {"pyoomph.libs"}
    )

    without = dict(base, PATH=bare_path)
    without.pop("PYOOMPH_STUB_DLL_DIRS", None)
    with_path = dict(base, PATH=str(libs) + os.pathsep + bare_path)
    with_path.pop("PYOOMPH_STUB_DLL_DIRS", None)
    with_var = dict(base, PATH=bare_path, PYOOMPH_STUB_DLL_DIRS=str(libs))

    py = sys.executable
    plain = [py, "-m", "nanobind.stubgen", "-m", "_pyoomph_core", "-P", "-O"]
    fixed = [py, str(launcher), "-m", "_pyoomph_core", "-P", "-O"]

    print("\nstages:")
    outs = [tmp / f"out{i}" for i in "abcd"]
    results = [
        stage("A  plain stubgen, dependencies unreachable", False,
              plain + [str(outs[0])], without, outs[0]),
        stage("B  plain stubgen, dependencies on PATH", False,
              plain + [str(outs[1])], with_path, outs[1]),
        stage("C  launcher, dependencies on PATH", True,
              fixed + [str(outs[2])], with_path, outs[2]),
        stage("D  launcher, dependencies via PYOOMPH_STUB_DLL_DIRS", True,
              fixed + [str(outs[3])], with_var, outs[3]),
    ]

    if all(results):
        print("\nDiagnosis confirmed: PATH is not searched, os.add_dll_directory is, "
              "and the launcher generates the stub.")
        return 0
    print("\nThe outcomes do not match the hypothesis - see the stages above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
