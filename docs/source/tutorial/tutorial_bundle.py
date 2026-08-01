"""Assemble the tutorial example scripts into ``tutorial_example_scripts.zip``.

This replaces the former ``generate_tutorial_zip.sh``, which had to be run by hand and whose
output was committed to git. The bundle is now built from the tutorial sources on demand:
``docs/source/conf.py`` calls :func:`build_zip` on ``builder-inited``, so every documentation
build (including Read the Docs, which runs ``conf.py`` on a plain repository checkout) produces
a bundle that matches the scripts the surrounding text shows. Nothing here needs ``zip``,
``find`` or a shell, so it works on Windows and inside the Read the Docs container alike.

``citools/test_all_tutorial_scripts.py`` uses :func:`export_tree` instead of unpacking the zip,
which means the pipeline tests the scripts as they are in the tree rather than as they were
when someone last remembered to regenerate the archive.

Run this file directly to write the zip, or with ``--check`` to only run the consistency checks.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# Tutorial source directory -> folder name inside the bundle. Everything reachable below the
# source directory is flattened into that one folder, i.e. the sub-chapter structure of the
# documentation is dropped - which is why the file name checks below matter.
CHAPTERS = [
    ("temporal", "Temporal_ODEs"),
    ("spatial", "Spatial_PDEs"),
    ("pde", "SpatioTemporal_PDEs"),
    ("ale", "Moving_Mesh"),
    ("multidom", "Multiple_Domains"),
    ("mcflow", "Multicomponent_Flow"),
    ("dg", "Discontinuous_Galerkin"),
    ("advstab", "Advanced_Linear_Dynamics"),
    ("plotting", "Plotting_Interface"),
    ("precice", "PreCICE_Coupling"),
]

# Non-python files that the examples need to run.
EXTRA_FILES = [
    ("precice/*.xml", "PreCICE_Coupling"),
]

BUNDLE_ROOT = "pyoomph_tutorial_scripts"

README_TEXT = """Here, you find the python scripts used and explained throughout the tutorial of pyoomph.
You can find pyoomph here: https://github.com/pyoomph/pyoomph
The tutorial is hosted here: https://pyoomph.readthedocs.io
"""

# Several scripts deliberately exist twice: a chapter that builds on an earlier example keeps a
# local copy so that its ``from <example> import *`` works when the reader only has that one
# folder. Those copies must stay byte-identical, and check_consistency() enforces it.
#
# These names, however, belong to genuinely different examples that happen to share a name - the
# mcflow versions use the multi-component material library, the pde/navier ones spell the same
# physics out by hand. They end up in different bundle folders, so they do not clash.
KNOWN_DIFFERENT_NAMESAKES = {
    "marangoni_instability.py",
    "rayleigh_taylor_instability.py",
}

# Fixed timestamp for every archive member: the bundle is rebuilt on every documentation build,
# and without this it would differ each time even though nothing changed.
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def tutorial_dir() -> Path:
    """Directory holding the tutorial chapters (the one this file lives in)."""
    return Path(__file__).resolve().parent


def collect(root: Optional[Path] = None) -> Dict[str, Path]:
    """Map each path inside the bundle to the tutorial file it is taken from."""
    root = tutorial_dir() if root is None else Path(root)
    files: Dict[str, Path] = {}
    for source, target in CHAPTERS:
        for path in sorted((root / source).rglob("*.py")):
            files[f"{BUNDLE_ROOT}/{target}/{path.name}"] = path
    for pattern, target in EXTRA_FILES:
        subdir, glob = pattern.split("/", 1)
        for path in sorted(root.joinpath(subdir).iterdir()):
            if path.is_file() and fnmatch.fnmatch(path.name, glob):
                files[f"{BUNDLE_ROOT}/{target}/{path.name}"] = path
    return files


def check_consistency(root: Optional[Path] = None) -> List[str]:
    """Report scripts that are duplicated inconsistently, or that shadow each other.

    Returns a list of human-readable problems; empty means everything is fine.
    """
    root = tutorial_dir() if root is None else Path(root)
    problems: List[str] = []

    # Files of the same name from different sub-chapters collapse onto the same bundle path.
    # The shell version copied them one over the other and silently shipped whichever came last.
    by_bundle_path: Dict[str, List[Path]] = {}
    for source, target in CHAPTERS:
        for path in sorted((root / source).rglob("*.py")):
            by_bundle_path.setdefault(f"{BUNDLE_ROOT}/{target}/{path.name}", []).append(path)
    for bundle_path, sources in by_bundle_path.items():
        if len(sources) > 1:
            listed = ", ".join(str(p.relative_to(root)) for p in sources)
            problems.append(f"{bundle_path} would be written from several files: {listed}")

    # Copies of the same example in different chapters must not drift apart.
    by_name: Dict[str, List[Path]] = {}
    for source, _ in CHAPTERS:
        for path in sorted((root / source).rglob("*.py")):
            by_name.setdefault(path.name, []).append(path)
    for name, sources in sorted(by_name.items()):
        if len(sources) < 2 or name in KNOWN_DIFFERENT_NAMESAKES:
            continue
        contents = {p.read_bytes() for p in sources}
        if len(contents) > 1:
            listed = ", ".join(str(p.relative_to(root)) for p in sources)
            problems.append(
                f"copies of {name} have drifted apart: {listed} "
                f"(make them identical, or add the name to KNOWN_DIFFERENT_NAMESAKES in "
                f"{Path(__file__).name} if they are meant to be different examples)"
            )
    return problems


def _zip_bytes(root: Optional[Path] = None) -> bytes:
    files = dict(sorted(collect(root).items()))
    files_data = {path: source.read_bytes() for path, source in files.items()}
    files_data[f"{BUNDLE_ROOT}/README.txt"] = README_TEXT.encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Explicit directory entries: not needed by unzip, but some GUI extractors rely on them.
        for folder in sorted({f"{Path(p).parent.as_posix()}/" for p in files_data} | {f"{BUNDLE_ROOT}/"}):
            info = zipfile.ZipInfo(folder, date_time=_ZIP_TIMESTAMP)
            info.external_attr = (0o755 << 16) | 0x10
            zf.writestr(info, b"")
        for bundle_path, data in sorted(files_data.items()):
            info = zipfile.ZipInfo(bundle_path, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return buffer.getvalue()


def build_zip(dest: Optional[Path] = None, root: Optional[Path] = None) -> bool:
    """Write the bundle to *dest*. Returns True if the file changed.

    An unchanged archive is left untouched so that repeated documentation builds do not keep
    rewriting it.
    """
    root = tutorial_dir() if root is None else Path(root)
    dest = root / "tutorial_example_scripts.zip" if dest is None else Path(dest)
    data = _zip_bytes(root)
    if dest.exists() and dest.read_bytes() == data:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def export_tree(dest: Path, root: Optional[Path] = None) -> Path:
    """Write the same content as the zip into a directory tree, replacing what is there.

    *dest* is the directory that will contain the ``pyoomph_tutorial_scripts`` folder; the path
    of that folder is returned.
    """
    root = tutorial_dir() if root is None else Path(root)
    dest = Path(dest)
    bundle = dest / BUNDLE_ROOT
    shutil.rmtree(bundle, ignore_errors=True)
    for bundle_path, source in sorted(collect(root).items()):
        target = dest / bundle_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (bundle / "README.txt").write_text(README_TEXT)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Only run the consistency checks, do not write anything")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Where to write the zip (default: next to the tutorial chapters)")
    args = parser.parse_args()

    problems = check_consistency()
    for problem in problems:
        print("PROBLEM:", problem, file=sys.stderr)
    if args.check:
        if not problems:
            print(f"{len(collect())} tutorial files, no inconsistencies found")
        return 1 if problems else 0

    dest = args.output if args.output is not None else tutorial_dir() / "tutorial_example_scripts.zip"
    changed = build_zip(dest)
    print(f"{'Wrote' if changed else 'Unchanged'}: {dest} ({len(collect())} files)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
