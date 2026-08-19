#!/usr/bin/env python3
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

"""Over-request detector for the generated code's `shapes_required_*` flags.

For every generated .c file it collects

  * which required-shapes flags the code SETS (per attached-element chain: the code's own
    shapeinfo, its bulk's, the opposite interface's, and one level deeper), and
  * which shape buffers the code actually READS (`shapeinfo->...`, `...->bulk_shapeinfo->...`,
    `...->opposite_shapeinfo->...`),

and reports the flags that are set but whose buffers never appear on the read side.

    python3 dev_docs/scripts/scan_required_shapes.py ~/code/pyoomph_runs
    python3 dev_docs/scripts/scan_required_shapes.py DIR --only-b1      # the bulk/opposite Pos.psi case
    python3 dev_docs/scripts/scan_required_shapes.py DIR --verbose      # one line per file

IMPORTANT, and the whole reason this is a *detector* and not a proof: the absence of a read is
CORPUS EVIDENCE, never proof that a flag is dead. Nothing here knows what a residual could have
looked like -- see dev_docs/code_generation.md, "corpus evidence" caveat. A flag reported here is a
candidate to be understood, and then to be gated at the code generator with an argument, not
something to delete because the scan said so. Conversely a flag whose buffer IS read is proof that
it is needed, which is the direction this tool can actually settle.

Two known non-shape flags are reported separately rather than as over-requests:
  * `D0.psi` is an external-data signal (it makes the element attach the source element's D0 internal
    data), not a shape-buffer flag; there is no `shape_D0` to read.
  * a bulk/opposite chain whose only read is `hanginfo_Pos` / `hanginfo` is using the buffer as the
    local-equation REMAP channel, which is what a moving mesh needs and what a static one does not.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

# The chain by which the generated code reaches an attached element's shape info. The keys are the
# accessor prefixes as they appear in the SETTER (`....bulk_shapes->`) and the values the prefix of
# the corresponding READ (`shapeinfo->bulk_shapeinfo->`).
CHAINS = {
    "": "own",
    "bulk_shapes->": "bulk",
    "opposite_shapes->": "opposite",
    "bulk_shapes->bulk_shapes->": "bulk/bulk",
    "opposite_shapes->bulk_shapes->": "opposite/bulk",
}
CHAIN_READ_DEPTH = {"own": 0, "bulk": 1, "opposite": 1, "bulk/bulk": 2, "opposite/bulk": 2}

# flag name (as written in the generated setter) -> buffer identifiers that flag authorises.
# A read of ANY of them counts as "the flag is used".
FLAG_TO_BUFFERS = {
    "Pos.psi": ["shape_Pos"],
    "Pos.dx_psi": ["dx_shape_Pos"],
    "Pos.dX_psi": ["dX_shape_Pos"],
    "Pos.d2x_psi": ["d2x_shape_Pos", "d2S_shape_Pos"],
    "DL.psi": ["shape_DL"],
    "DL.dx_psi": ["dx_shape_DL"],
    "DL.dX_psi": ["dX_shape_DL"],
    "DL.d2x_psi": ["d2x_shape_DL", "d2S_shape_DL"],
    "normal": ["normal"],
    "normal_deriv": ["dnormal_dx", "d_normal_dcoord", "d_dnormal_dx_dcoord"],
    "elemsize_Eulerian": ["elemsize_Eulerian", "elemsize_d_coords", "elemsize_d2_coords"],
    "elemsize_Eulerian_cartesian": ["elemsize_Eulerian_cartesian", "elemsize_Cart_d_coords",
                                    "elemsize_Cart_d2_coords"],
    "elemsize_Lagrangian": ["elemsize_Lagrangian"],
    "elemsize_Lagrangian_cartesian": ["elemsize_Lagrangian_cartesian"],
    "history_integral_dx1": ["int_pt_weight"],
    "history_integral_dx2": ["int_pt_weight"],
    "history_geometry1": ["dx_shapes", "dx_shape_Pos", "normal", "elemsize_Eulerian"],
    "history_geometry2": ["dx_shapes", "dx_shape_Pos", "normal", "elemsize_Eulerian"],
}
# Per-space flags carry the space index along, so their buffer read has to match the index too.
SPACE_FLAG_TO_BUFFERS = {
    "psi": ["shapes"],
    "dx_psi": ["dx_shapes"],
    "dX_psi": ["dX_shapes"],
    "d2x_psi": ["d2x_shapes", "d2S_shapes"],
}
# Not shape flags at all; reported in their own bucket so they do not drown the real findings.
NON_SHAPE_FLAGS = {"D0.psi", "D0.dx_psi", "D0.dX_psi", "D0.d2x_psi"}

SET_RE = re.compile(
    r"functable->shapes_required_(\w+?)(?:\[\d+\])?\.((?:\w+_shapes->)*)"
    r"(?:continuous_spaces\[(\w+)\]\.)?([A-Za-z_0-9.]+)\s*=\s*true\s*;")

# `shapeinfo`, `bulk_shapeinfo`, `opposite_shapeinfo` chains on the read side. The generated code
# always spells the full chain out, so the depth can be counted from the identifier itself.
READ_RE = re.compile(r"((?:\w*shapeinfo(?:->\w*shapeinfo)*))->(\w+)((?:\[[^\]\[]*\])*)")
INDEX_RE = re.compile(r"\[([^\]\[]*)\]")


def read_map(text):
    """{(depth, buffer name): set(index expressions)} of everything the code reads.

    ALL index expressions are collected, not just the first: dx_shapes is indexed
    [history][space], so keying on the leading [0] would make every dx_psi flag look unread.
    """
    out = defaultdict(set)
    for m in READ_RE.finditer(text):
        chain, buf, idx = m.group(1), m.group(2), m.group(3)
        depth = chain.count("shapeinfo") - 1
        idxs = INDEX_RE.findall(idx)
        out[(depth, buf)].update(idxs if idxs else [None])
    return out


def scan_file(path):
    text = open(path, "r", errors="replace").read()
    reads = read_map(text)
    findings, non_shape, remap_only = [], [], []
    seen = set()
    for m in SET_RE.finditer(text):
        which, chain, space, flag = m.group(1), m.group(2), m.group(3), m.group(4)
        cname = CHAINS.get(chain)
        if cname is None:
            continue  # an accessor chain this scan does not model; report nothing rather than guess
        depth = CHAIN_READ_DEPTH[cname]
        if space:
            bufs = SPACE_FLAG_TO_BUFFERS.get(flag)
            key = (which, cname, "%s[%s]" % (flag, space))
        else:
            bufs = FLAG_TO_BUFFERS.get(flag)
            key = (which, cname, flag)
        if key in seen:
            continue
        seen.add(key)
        if flag in NON_SHAPE_FLAGS or (not space and flag.startswith("D0.")):
            non_shape.append(key)
            continue
        if bufs is None:
            continue  # unmodelled flag
        used = False
        for b in bufs:
            idxs = reads.get((depth, b))
            if idxs is None:
                continue
            if space is None or space in idxs or None in idxs:
                used = True
                break
        if used:
            continue
        # Distinguish "reaches the attached element only through the equation remap channel"
        if depth > 0 and any(reads.get((depth, h)) for h in ("hanginfo_Pos", "hanginfo")):
            remap_only.append(key)
        else:
            findings.append(key)
    return findings, non_shape, remap_only


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="directories to walk for generated .c files")
    ap.add_argument("--verbose", action="store_true", help="one line per file with a finding")
    ap.add_argument("--only-b1", action="store_true",
                    help="only the bulk/opposite Pos.psi over-request")
    args = ap.parse_args()

    files = []
    for root in args.roots:
        if os.path.isfile(root):
            files.append(root)
            continue
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                if n.endswith(".c"):
                    files.append(os.path.join(dirpath, n))

    tally = Counter()
    remap_tally = Counter()
    nonshape_tally = Counter()
    nfiles_with_finding = 0
    b1_files, b1_remap_files = set(), set()
    for f in sorted(files):
        try:
            findings, non_shape, remap_only = scan_file(f)
        except Exception as e:  # a half-written .c from an interrupted run is not a finding
            print("SKIP %s (%s)" % (f, e), file=sys.stderr)
            continue
        is_b1 = [k for k in findings if k[1] in ("bulk", "opposite", "bulk/bulk", "opposite/bulk")
                 and k[2] == "Pos.psi"]
        if is_b1:
            b1_files.add(f)
        if [k for k in remap_only if k[2] == "Pos.psi"]:
            b1_remap_files.add(f)
        if args.only_b1:
            findings = is_b1
        if findings:
            nfiles_with_finding += 1
            if args.verbose:
                print("%s:" % f)
                for k in sorted(findings):
                    print("    set but never read: %s %s %s" % k)
        for k in findings:
            tally[k[1:]] += 1
        for k in remap_only:
            remap_tally[k[1:]] += 1
        for k in non_shape:
            nonshape_tally[k[1:]] += 1

    print("scanned %d generated .c files, %d have at least one set-but-never-read flag"
          % (len(files), nfiles_with_finding))
    print("--- set but never read (candidates, NOT proof of deadness) ---")
    for k, n in tally.most_common():
        print("  %-14s %-34s %d files" % (k[0], k[1], n))
    if remap_tally and not args.only_b1:
        print("--- set, no shape read, but the attached element IS reached through the equation "
              "remap channel (hanginfo) ---")
        for k, n in remap_tally.most_common():
            print("  %-14s %-34s %d files" % (k[0], k[1], n))
    if nonshape_tally and not args.only_b1:
        print("--- non-shape flags (D0.psi is an external-data signal, there is no shape_D0) ---")
        for k, n in nonshape_tally.most_common():
            print("  %-14s %-34s %d files" % (k[0], k[1], n))
    print("B1 (bulk/opposite Pos.psi with no shape_Pos read anywhere): %d files" % len(b1_files))
    print("   of which reached only through the remap channel: %d files" % len(b1_remap_files))


if __name__ == "__main__":
    main()
