from __future__ import annotations
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
#
# Content-addressed cache for JIT-compiled element code shared libraries.
#
# pyoomph's code generation (src/codegen.cpp) is deterministic across process
# runs (see branch deterministic_codegen): the same equations/settings always
# produce byte-identical generated C code. That means the compiled shared
# library for a given piece of generated C code (plus the compiler/flags used
# to build it) can be reused across runs/output directories/machines instead
# of recompiling from scratch every time.
#
# This is "Tier 1" caching: it keys on the *generated* C code text (i.e. it
# still requires running codegen to produce that text), and only saves the
# compilation step. See jit_cache_fingerprint.py (Tier 2, shadow mode) for the
# more ambitious (and riskier) idea of also skipping codegen itself.

import hashlib
import os
import platform
import shutil
import sys
import tempfile
from ..typings import *

# Bump this whenever codegen.cpp's generated-code format changes in a way that
# could make an old cached .so incompatible with a newer pyoomph build, or
# whenever this cache's key composition changes. Old cache entries simply
# become permanent misses (never read, eventually evicted) after a bump -
# there is no migration.
FORMAT_VERSION = 3


def _platform_runtime_tag() -> str:
    """A string identifying the C runtime / OS version the compiled .so will be
    loaded against, folded into the cache key (see compute_key()) alongside
    platform.system()/machine().

    The cache is content-addressed on the *generated C source* plus the compiler
    identity, and is explicitly meant to be reusable across machines (see the
    module docstring). But a shared library is linked against a C runtime, and
    that runtime is NOT captured by the compiler version string (gcc --version
    says nothing about glibc). On a single machine updated in place this is
    harmless - glibc/libSystem/UCRT all keep backward compatibility, so an .so
    built before an update still loads after it. The unsafe case is cross-machine
    reuse (shared/NFS cache dir, copied cache) or a downgrade: an .so built
    against a newer runtime can reference versioned symbols (e.g. memcpy@GLIBC_2.14)
    absent on an older one, and dlopen() then fails at load time. Mixing the
    runtime version into the key makes a mismatched host recompile instead.

    Reports the runtime pyoomph is *running* on (the conservative, load-time
    view). Degrades cleanly to "" on platforms/libcs it cannot identify (e.g.
    musl, where platform.libc_ver() typically yields nothing) - that just
    reproduces today's behaviour of not discriminating on the runtime there,
    never a wrong reuse beyond what already happens now."""
    try:
        if sys.platform.startswith("linux"):
            # ('glibc', '2.39') on a glibc host; ('', '') on musl / when it
            # cannot tell - both hash cleanly.
            name, ver = platform.libc_ver()
            return "linux|" + name + "|" + ver
        elif sys.platform == "darwin":
            # Running macOS version; a slightly-too-strict proxy for the binary's
            # MACOSX_DEPLOYMENT_TARGET (worst case: an occasional needless recompile).
            return "mac|" + platform.mac_ver()[0]
        elif sys.platform == "win32":
            # Windows build number - tracks UCRT/vcruntime movement.
            return "win|" + platform.win32_ver()[1]
    except Exception:
        # platform.* helpers shell out / read files on some hosts; a caching
        # convenience must never crash a run, so fall back to no discrimination.
        return ""
    return ""


def _default_cache_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "pyoomph", "jit_cache")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Caches")
        return os.path.join(base, "pyoomph", "jit_cache")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.join(base, "pyoomph", "jit_cache")


_runtime_disabled = False


def set_enabled(enabled: bool) -> None:
    """Runtime on/off switch, e.g. driven by Problem's --no-cache command-line
    flag - takes effect immediately for every Problem/compiler instance in this
    process (there is deliberately no per-instance override: --no-cache is a
    process-wide command-line flag, so a process-wide switch is the correct
    scope for it, unlike the per-Problem suppress_code_writing/suppress_compilation
    checks, which are applied at the call site instead - see problem.py's
    compile_bulk_element_code())."""
    global _runtime_disabled
    _runtime_disabled = not enabled


_ginac_deterministic_warned = False


def _ginac_hash_is_deterministic() -> bool:
    """Whether the GiNaC linked into this pyoomph build is known to have
    deterministic term/hash ordering across process runs (see CMakeLists.txt's
    PYOOMPH_GINAC_HASH_PATCHED and citools/patches/ginac-deterministic-*.patch).
    Without that, the same equations can print out as textually different
    (but semantically identical) generated C code from one run to the next, so
    the entire premise of this content-addressed cache - identical content
    implies safe to reuse - no longer holds. Fails safe (assumes False,
    disabling the cache) if this can't be determined at all."""
    from .. import _pyoomph_core as _pyoomph
    try:
        return bool(_pyoomph.ginac_hash_is_deterministic())
    except AttributeError:
        return False


_cache_io_broken = False


def _report_cache_io_failure(cache_dir: str, exc: BaseException) -> None:
    """Called wherever cache I/O (creating the cache dir, writing an entry, ...)
    fails - e.g. a read-only filesystem, a permission-locked home directory, a
    full disk, or some sandboxed/CI environment where the default location
    just isn't writable. A caching convenience feature must never be allowed to
    crash an actual simulation run over this, so every write path catches
    OSError and reports it here instead of letting it propagate; this disables
    the cache for the rest of the process (after a single warning) so a
    persistently broken location doesn't just fail silently on every single
    compile call for no benefit."""
    global _cache_io_broken
    if not _cache_io_broken:
        _cache_io_broken = True
        print("pyoomph JIT cache: disabling itself for the rest of this process after a filesystem "
              "error accessing " + repr(cache_dir) + " (" + str(exc) + "). Set PYOOMPH_JIT_CACHE_DIR "
              "to a writable location (or PYOOMPH_JIT_CACHE=0 to silence this) to use caching.")


def _jit_cache_enabled_at_build_time() -> bool:
    """PYOOMPH_ENABLE_JIT_CACHE in CMakeLists.txt: a build-time kill switch, independent of
    (and checked before) any runtime flag/env var - those can only narrow this down further,
    never override an OFF set here. Defaults to True (matches the CMake option's own ON
    default) if the compiled extension predates this check entirely, for backward
    compatibility with already-built binaries."""
    from .. import _pyoomph_core as _pyoomph
    try:
        return bool(_pyoomph.jit_cache_enabled_at_build_time())
    except AttributeError:
        return True


def is_enabled() -> bool:
    if not _jit_cache_enabled_at_build_time():
        return False
    if _runtime_disabled or _cache_io_broken:
        return False
    if os.environ.get("PYOOMPH_JIT_CACHE", "1") in ("0", "false", "False", ""):
        return False
    if not _ginac_hash_is_deterministic():
        global _ginac_deterministic_warned
        if not _ginac_deterministic_warned:
            _ginac_deterministic_warned = True
            print("pyoomph JIT cache: disabled, because the linked GiNaC is not known to have "
                  "deterministic term/hash ordering across process runs (see PYOOMPH_GINAC_HASH_PATCHED "
                  "in CMakeLists.txt/cmake/ThirdPartyGiNaC.cmake). Generated code for the same equations "
                  "cannot be assumed identical from one run to the next, so caching it would be unsafe.")
        return False
    return True


def get_cache_dir() -> str:
    return os.environ.get("PYOOMPH_JIT_CACHE_DIR") or _default_cache_dir()


def get_max_size_bytes() -> int:
    mb = int(os.environ.get("PYOOMPH_JIT_CACHE_MAX_MB", "2048"))
    return mb * 1024 * 1024


def get_max_fingerprint_entries() -> int:
    # Tier-2 shadow-mode bookkeeping entries are tiny (a couple hundred bytes
    # each: one hex hash), so a byte-size cap like the compiled-.so cache above
    # would allow an impractically large *file count* before ever triggering -
    # cap by entry count directly instead, since count (inodes, directory listing
    # cost), not bytes, is what actually matters for this directory.
    return int(os.environ.get("PYOOMPH_JIT_CACHE_MAX_FINGERPRINTS", "100000"))


def tier2_shadow_enabled() -> bool:
    """Tier 2 (see FiniteElementCode::get_precodegen_fingerprint_text() in
    src/codegen.cpp): predicts, from a cheap pre-codegen fingerprint, whether
    write_code() would reproduce a previously seen generated-code hash - and
    logs loudly if that prediction ever turns out wrong. This NEVER skips
    codegen; write_code() always still runs in full. It exists purely to build
    up evidence (across real usage and the tutorial suite) of whether the
    fingerprint's coverage is complete enough to ever trust for real, before
    any future mode is added that would actually skip codegen on a hit."""
    return is_enabled() and os.environ.get("PYOOMPH_JIT_CACHE_TIER2", "1") not in ("0", "false", "False", "")


# Bump whenever get_precodegen_fingerprint_text()'s coverage/format changes (keep
# in sync with the "FMTn" tag inside that C++ function).
FINGERPRINT_FORMAT_VERSION = 5


class JITCache:
    def __init__(self, cache_dir: str | None = None, max_size_bytes: int | None = None, max_fingerprint_entries: int | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else get_cache_dir()
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else get_max_size_bytes()
        self.max_fingerprint_entries = max_fingerprint_entries if max_fingerprint_entries is not None else get_max_fingerprint_entries()
        self._objects_dir = os.path.join(self.cache_dir, "objects")
        self._fingerprints_dir = os.path.join(self.cache_dir, "fingerprints")
        # Tier-1 hit/miss counters, purely for diagnostics - e.g. `python -m pyoomph check
        # compiler` uses these to verify the cache is actually being read from, not just
        # written to (see try_restore() below).
        self.hits = 0
        self.misses = 0

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self._objects_dir, key[:2], key[2:])

    def _fingerprint_path(self, key: str) -> str:
        return os.path.join(self._fingerprints_dir, key[:2], key[2:])

    def check_fingerprint_shadow(self, fingerprint_text: str, actual_code_text: str, label: str) -> None:
        """Tier 2 shadow-mode check (see tier2_shadow_enabled()): record which
        generated-code hash this pre-codegen fingerprint produced, and if it was
        seen before, verify it still produces the same code. Always logs a
        mismatch (regardless of the `quiet` flag elsewhere) since it signals an
        incomplete fingerprint - a real correctness gap worth surfacing, not
        routine progress output. codegen has already run in full by the time
        this is called; nothing here ever gets skipped."""
        fp_key = hashlib.sha256((str(FINGERPRINT_FORMAT_VERSION) + "\0" + fingerprint_text).encode("utf-8")).hexdigest()
        code_key = hashlib.sha256(actual_code_text.encode("utf-8")).hexdigest()
        path = self._fingerprint_path(fp_key)
        try:
            with open(path, "r") as f:
                recorded = f.read().strip()
        except OSError:
            recorded = None
        if recorded is not None and recorded != code_key:
            print("*** JIT cache Tier-2 shadow-mode MISMATCH for " + label + ": the same pre-codegen "
                  "fingerprint previously produced different generated code than this run. The fingerprint's "
                  "coverage (see get_precodegen_fingerprint_text() in src/codegen.cpp) is missing something - "
                  "Tier 2 is NOT yet safe to trust for skipping codegen in this case. Codegen ran in full "
                  "regardless; this is reported for investigation only.")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
        except OSError as e:
            _report_cache_io_failure(self.cache_dir, e)
            return
        try:
            with os.fdopen(fd, "w") as f:
                f.write(code_key)
            os.replace(tmp_path, path)
        except OSError:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return
        self._prune_fingerprints_if_needed()

    def compute_key(self, code_text: str, compiler_fingerprint: str, extra_flags: Sequence[str], flag_state: str, header_text: str = "") -> str:
        h = hashlib.sha256()
        h.update(str(FORMAT_VERSION).encode("utf-8"))
        h.update(b"\0")
        h.update(platform.system().encode("utf-8"))
        h.update(b"\0")
        h.update(platform.machine().encode("utf-8"))
        h.update(b"\0")
        h.update(_platform_runtime_tag().encode("utf-8"))
        h.update(b"\0")
        h.update(compiler_fingerprint.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(repr(list(extra_flags)).encode("utf-8"))
        h.update(b"\0")
        h.update(flag_state.encode("utf-8"))
        h.update(b"\0")
        h.update(header_text.encode("utf-8"))
        h.update(b"\0")
        h.update(code_text.encode("utf-8"))
        return h.hexdigest()

    def try_restore(self, key: str, dest_path: str) -> bool:
        src = self._path_for_key(key)
        if not os.path.isfile(src):
            self.misses += 1
            return False
        try:
            shutil.copy2(src, dest_path)
            os.utime(src, None)  # bump mtime for LRU purposes
            self.hits += 1
            return True
        except OSError:
            self.misses += 1
            return False

    def store(self, key: str, src_path: str) -> None:
        if not os.path.isfile(src_path):
            return
        dest = self._path_for_key(key)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), prefix=".tmp_")
        except OSError as e:
            _report_cache_io_failure(self.cache_dir, e)
            return
        try:
            os.close(fd)
            shutil.copy2(src_path, tmp_path)
            os.replace(tmp_path, dest)
        except OSError:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return
        self._prune_if_needed()

    def _list_shard_entries(self, base_dir: str) -> list[tuple[float, int, str]]:
        entries: list[tuple[float, int, str]] = []
        if not os.path.isdir(base_dir):
            return entries
        try:
            shards = os.listdir(base_dir)
        except OSError:
            return entries
        for shard in shards:
            shard_dir = os.path.join(base_dir, shard)
            if not os.path.isdir(shard_dir):
                continue
            try:
                names = os.listdir(shard_dir)
            except OSError:
                continue
            for name in names:
                if name.startswith(".tmp_"):
                    continue
                p = os.path.join(shard_dir, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, p))
        return entries

    def _prune_if_needed(self) -> None:
        # Housekeeping only, run right after a successful store() - never allowed to
        # raise and take the actual compile down with it if the filesystem misbehaves.
        try:
            entries = self._list_shard_entries(self._objects_dir)
            total = sum(size for _mtime, size, _p in entries)
            if total <= self.max_size_bytes:
                return
            # Evict oldest-accessed entries first (LRU) down to 90% of the cap, to
            # avoid pruning on almost every single store once near the limit.
            target = int(self.max_size_bytes * 0.9)
            entries.sort(key=lambda e: e[0])
            for _mtime, size, p in entries:
                if total <= target:
                    break
                try:
                    os.remove(p)
                    total -= size
                except OSError:
                    pass
        except OSError:
            pass

    def _prune_fingerprints_if_needed(self) -> None:
        # Same idea as _prune_if_needed(), but capped by entry COUNT: fingerprint
        # entries are tiny (a hex hash each), so a byte-size budget would let the
        # file/inode count grow far too large before ever tripping - see
        # get_max_fingerprint_entries().
        try:
            entries = self._list_shard_entries(self._fingerprints_dir)
            if len(entries) <= self.max_fingerprint_entries:
                return
            target = int(self.max_fingerprint_entries * 0.9)
            entries.sort(key=lambda e: e[0])
            for _mtime, _size, p in entries[:len(entries) - target]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        except OSError:
            pass

    def get_usage_stats(self) -> dict[str, int]:
        """Read-only directory inspection for `python -m pyoomph cache usage` - deliberately
        does not go through is_enabled()/get_jit_cache(), so usage of a cache directory that
        was populated earlier can still be reported even if caching is currently disabled
        (e.g. after switching to an unpatched GiNaC, or with PYOOMPH_JIT_CACHE=0 set)."""
        objects = self._list_shard_entries(self._objects_dir)
        fingerprints = self._list_shard_entries(self._fingerprints_dir)
        return {
            "objects_count": len(objects),
            "objects_bytes": sum(size for _mtime, size, _p in objects),
            "objects_max_bytes": self.max_size_bytes,
            "fingerprints_count": len(fingerprints),
            "fingerprints_max_count": self.max_fingerprint_entries,
        }


_global_cache: JITCache | None = None
_global_cache_dir_used: str | None = None


def get_jit_cache() -> JITCache | None:
    """Returns the process-wide JITCache instance, or None if caching is disabled
    via PYOOMPH_JIT_CACHE=0."""
    global _global_cache, _global_cache_dir_used
    if not is_enabled():
        return None
    cache_dir = get_cache_dir()
    if _global_cache is None or _global_cache_dir_used != cache_dir:
        _global_cache = JITCache(cache_dir)
        _global_cache_dir_used = cache_dir
    return _global_cache


def clear_cache(cache_dir: str | None = None) -> bool:
    """Removes the entire on-disk cache directory (compiled objects, trust-mode entries if
    any, and Tier-2 fingerprint bookkeeping alike) - for `python -m pyoomph cache clear`.
    Returns whether a directory was actually found and removed. Resets the process-wide
    JITCache singleton if it pointed at the same directory, so hit/miss counters etc. don't
    keep referring to a now-deleted location."""
    global _global_cache, _global_cache_dir_used
    target = cache_dir if cache_dir is not None else get_cache_dir()
    existed = os.path.isdir(target)
    if existed:
        shutil.rmtree(target, ignore_errors=True)
    if _global_cache_dir_used == target:
        _global_cache = None
        _global_cache_dir_used = None
    return existed
