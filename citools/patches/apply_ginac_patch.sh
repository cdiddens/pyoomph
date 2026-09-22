#!/usr/bin/env bash
# Applies pyoomph's GiNaC patches to a GiNaC source tree (run with cwd = the
# extracted GiNaC source root, i.e. the directory containing ginac/).
#
# Idempotent by design: cmake/ThirdPartyGiNaC.cmake makes the ExternalProject
# patch step depend on this script and on every *.patch beside it, so touching
# any of them re-runs the step against a tree that is already (partly) patched.
# A plain `patch -p1 -i ...` fails loudly the second time ("Reversed (or
# previously applied) patch detected!"), so each patch has to be recognised as
# already applied and skipped - while a genuine mismatch (e.g. GiNaC source
# changed unexpectedly) still aborts the build loudly.
#
# That recognition used to be "reverse-dry-run the whole patch; if it applies,
# it is already applied". That is wrong for patches that touch the same region
# of the same file, and it broke the 2026-09-22 nightly: patch 5 inserts
# exact_whole_number_exponent() and the real_part()/imag_part() preamble
# exactly into patch 4's trailing context, so against a tree carrying both,
# patch 4 no longer reverses (its hunk 1 wants power::real_part() to follow
# binomial_expansion_pow(), and now it does not) while its hunk 1 still
# *forward*-matched with fuzz somewhere else. Neither branch held and the
# script aborted on a correctly patched tree.
#
# So record what was applied instead of re-deriving it: a stamp per patch under
# $stamp_dir, holding that patch's checksum. The stamp lives inside the source
# tree, so a re-extracted tarball correctly starts over, and editing a patch
# invalidates only its own stamp. The two fallbacks below adopt a tree that was
# patched before stamps existed - including the tree the 2026-09-22 failure
# left behind - and write its stamps, so this heals in place with no need to
# blow away build/.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stamp_dir=".pyoomph-ginac-patches"

# The first two fix the same class of bug (GiNaC hashing the ASLR-dependent
# address of an RTTI type-name string instead of something deterministic) in
# two different spots; the third stops power::to_polynomial() recursing
# forever on a negative integer power of a power with a symbolic exponent; the
# fourth stops power::real_part()/imag_part() raising pow(0,0) on the end terms
# of their binomial expansion; the fifth lets those two use that binomial branch
# for an inexact whole-number exponent (2.0) as well, instead of falling through
# to a polar form that buries the basis in an atan2.
# See each patch's own header comment for the full rationale. Applied in
# order; add further patches to this list as found.
patches=(
    "ginac-deterministic-hash-seed.patch"
    "ginac-deterministic-constant-hash.patch"
    "ginac-to-polynomial-symbolic-exponent.patch"
    "ginac-binomial-real-imag-pow00.patch"
    "ginac-inexact-whole-number-exponent.patch"
)

# Every non-blank line a patch adds, already present in the file it adds it to?
# The last-resort "already applied" test, and the only one that survives a
# later patch rewriting an earlier one's context: it asks about the state of
# the tree rather than about whether patch(1) can still line the hunks up.
# Exact whole-line matching, so it does not confuse an added line with a
# similar one elsewhere; context and removed lines are deliberately ignored.
added_lines_all_present() {
    awk '
        /^\+\+\+ / { target = $2; sub(/^b\//, "", target); next }
        /^--- /    { next }
        /^\+/ {
            if (target == "")
                next
            line = substr($0, 2)
            if (line ~ /^[[:space:]]*$/)
                next
            if (!(target in loaded)) {
                loaded[target] = 1
                while ((getline l < target) > 0)
                    present[target SUBSEP l] = 1
                close(target)
            }
            if (!((target SUBSEP line) in present))
                ok = 1
        }
        END { exit ok }
    ' "$1"
}

mkdir -p "$stamp_dir"

for patch_name in "${patches[@]}"; do
    patch_file="$script_dir/$patch_name"
    stamp_file="$stamp_dir/$patch_name.applied"
    checksum="$(sha256sum "$patch_file" | cut -d' ' -f1)"

    if [ -f "$stamp_file" ] && [ "$(cat "$stamp_file")" = "$checksum" ]; then
        echo "$patch_name already applied (stamp), skipping."
        continue
    fi

    # --fuzz=0: a hunk must match exactly. With patch(1)'s default fuzz of 2 an
    # already-applied patch can half-match somewhere else entirely - which is
    # how patch 4 reported "Hunk #1 succeeded at 698 with fuzz 1 (offset 36
    # lines)" against a tree that already had it - and applying that would
    # duplicate the addition rather than fail.
    if patch -p1 -N --fuzz=0 --dry-run -s -i "$patch_file" >/dev/null 2>&1; then
        patch -p1 -N --fuzz=0 -i "$patch_file"
    elif patch -p1 -R --fuzz=0 --dry-run -s -i "$patch_file" >/dev/null 2>&1; then
        echo "$patch_name already applied (reverses cleanly), recording stamp."
    elif added_lines_all_present "$patch_file"; then
        echo "$patch_name already applied (every added line present), recording stamp."
    else
        echo "ERROR: $patch_name does not apply to $(pwd) and is not already applied - GiNaC source may have changed unexpectedly." >&2
        exit 1
    fi

    printf '%s\n' "$checksum" > "$stamp_file"
done
