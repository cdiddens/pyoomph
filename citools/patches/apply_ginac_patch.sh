#!/usr/bin/env bash
# Applies pyoomph's determinism patches to a GiNaC source tree (run with cwd =
# the extracted GiNaC source root, i.e. the directory containing ginac/).
#
# Idempotent by design: CMake's ExternalProject_Add can re-invoke
# PATCH_COMMAND on a reconfigure even after the patches were already applied to
# that source tree (observed while developing this on branch
# deterministic_codegen), and a plain `patch -p1 -i ...` fails loudly the
# second time ("Reversed (or previously applied) patch detected!"), breaking
# the build. So for each patch: try applying forward; if that fails, check
# whether it's because the patch is already applied (the *reverse* applies
# cleanly) rather than assuming so, and only then treat it as success - a
# genuine mismatch (e.g. GiNaC source changed unexpectedly) still aborts the
# build loudly.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Both fix the same class of bug (GiNaC hashing the ASLR-dependent address of
# an RTTI type-name string instead of something deterministic) in two
# different spots - see each patch's own header comment for the full
# rationale. Applied in order; add further patches to this list as found.
patches=(
    "ginac-deterministic-hash-seed.patch"
    "ginac-deterministic-constant-hash.patch"
)

for patch_name in "${patches[@]}"; do
    patch_file="$script_dir/$patch_name"
    if patch -p1 -N --dry-run -s -i "$patch_file" >/dev/null 2>&1; then
        patch -p1 -N -i "$patch_file"
    elif patch -p1 -R --dry-run -s -i "$patch_file" >/dev/null 2>&1; then
        echo "$patch_name already applied, skipping."
    else
        echo "ERROR: $patch_name does not apply (forward or reverse) against $(pwd) - GiNaC source may have changed unexpectedly." >&2
        exit 1
    fi
done
