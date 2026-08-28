#!/usr/bin/env python3
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
"""
Regenerate pyoomph's AIOMFAC parameter tables from the AIOMFAC Fortran source, and report what
changed.

    git clone --depth 1 https://github.com/andizuend/AIOMFAC.git
    python3 citools/generate_aiomfac_parameters.py AIOMFAC/FortranCode --audit
    python3 citools/generate_aiomfac_parameters.py AIOMFAC/FortranCode --write

``--audit`` compares the current tables against the source and prints the differences by category;
``--write`` emits ``pyoomph/materials/UNIFAC/aiomfac.py`` and ``aiomfac_electrolyte.py``. Run the
audit first and keep its output: it is the record of what a parameter update actually changed, and
some of those changes move computed activity coefficients.

The parameters are AIOMFAC's, distributed under GPL v3. Cite the AIOMFAC publications
(https://aiomfac.lab.mcgill.ca/citation.html) when publishing results based on them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aiomfac_param_import import (AIOMFACParameters, FIRST_CATION, MR_DEFAULTS,
                                  NMAINGROUPS, OMEGA_NEUTRAL_ION)

#: Above this, an entry of the interaction matrix is one of AIOMFAC's "not determined" markers
#: rather than a parameter: the largest real one is 1e4.
MARKER_MAGNITUDE = 1.0e5

PYOOMPH_ROOT = Path(__file__).parent.parent
UNIFAC_DIR = PYOOMPH_ROOT / "pyoomph" / "materials" / "UNIFAC"

_LICENSE = (UNIFAC_DIR / "aiomfac.py").read_text().split('"""')[0].split("from ")[0]


def _num(v: float) -> str:
    """A float that reads back exactly. repr() is shortest-round-trip in Python 3."""
    return repr(float(v))


def _safe(name: str) -> str:
    """AIOMFAC's own name, usable as a Python string literal."""
    return name.replace('"', "'")


# =================================================================================================
#  Audit
# =================================================================================================

def audit(p: AIOMFACParameters) -> None:
    import pyoomph.materials.UNIFAC.aiomfac  # noqa: F401  (registers the model)
    from pyoomph.materials.activity import ActivityModel, UNIFACLikeActivityModel

    m = ActivityModel.get_activity_model_by_name("AIOMFAC")
    assert isinstance(m, UNIFACLikeActivityModel)
    by_index = {sg.index: sg for sg in m.subgroups.values() if sg.index is not None}
    idx_of_main = {g.name: i for i, g in m.maingroup_by_index.items()}

    print("=" * 90)
    print("AIOMFAC parameter audit: pyoomph's current tables against the AIOMFAC source")
    print("=" * 90)
    print("main groups:  pyoomph %d, AIOMFAC %d" % (len(m.maingroups), NMAINGROUPS))
    print("subgroups:    pyoomph %d, AIOMFAC %d (named or with a molar mass)"
          % (len(by_index), len([i for i in range(1, 201) if i in p.molar_mass]) + len(p.name)))

    added = [i for i in sorted(p.R) if i not in by_index and (i < FIRST_CATION or i in p.name)
             and p.R[i] > 0.0 and p.main_group_of.get(i, 0) > 0]
    reserved = [i for i in sorted(p.R) if i not in by_index and (i < FIRST_CATION or i in p.name)
                and i not in added]
    print("\n-- subgroups AIOMFAC has and pyoomph does not (%d) --" % len(added))
    print("   " + (", ".join("%d (%s)" % (i, p.name.get(i, "?")) for i in added) or "none"))
    if reserved:
        print("   (%d further slots exist in AIOMFAC with no volume or no main group, i.e. reserved"
              % len(reserved))
        print("    but unparameterised, and are deliberately not imported: " +
              ", ".join(str(i) for i in reserved) + ")")
    removed = [i for i in sorted(by_index) if i not in p.R]
    if removed:
        print("\n-- subgroups pyoomph has and AIOMFAC does not (%d) --" % len(removed))
        print("   " + ", ".join(str(i) for i in removed))

    print("\n-- subgroup R/Q that changed --")
    for i in sorted(by_index):
        if i not in p.R:
            continue
        dr, dq = abs(by_index[i].R - p.R[i]) > 1e-9, abs(by_index[i].Q - p.Q[i]) > 1e-9
        if dr or dq:
            print("   %3d %-14s R %-10s -> %-10s   Q %-10s -> %-10s"
                  % (i, p.name.get(i, by_index[i].name), by_index[i].R, p.R[i],
                     by_index[i].Q, p.Q[i]))

    print("\n-- ions whose identity pyoomph has wrong --")
    for i in sorted(by_index):
        if i >= FIRST_CATION and i in p.name and by_index[i].name != p.name[i]:
            print("   %3d pyoomph calls it %-10s AIOMFAC calls it %-10s (M = %s kg/mol)"
                  % (i, by_index[i].name, p.name[i], p.molar_mass.get(i)))

    diffs = 0
    for iname, row in m.interaction_table.items():
        for jname, entry in row.items():
            i, j = idx_of_main[iname], idx_of_main[jname]
            for key, tab in (("A", "ARR"), ("B", "BRR"), ("C", "CRR")):
                if key in entry and abs(entry[key] - p.interaction[tab][i][j]) > 1e-6 * max(1.0, abs(p.interaction[tab][i][j])):
                    diffs += 1
    present = {(idx_of_main[i], idx_of_main[j]) for i, row in m.interaction_table.items() for j in row}
    missing, undetermined = 0, 0
    for i in range(1, NMAINGROUPS + 1):
        for j in range(1, NMAINGROUPS + 1):
            a = p.interaction["ARR"][i][j]
            if i == j or (i, j) in present or abs(a) < 1e-12:
                continue
            if abs(a) > MARKER_MAGNITUDE:
                undetermined += 1
            else:
                missing += 1
    print("\n-- main group interactions --")
    print("   %d entries pyoomph has differ from AIOMFAC" % diffs)
    print("   %d parameters AIOMFAC has and pyoomph does not" % missing)
    print("   %d pairs AIOMFAC marks as never determined, which are listed rather than imported"
          % undetermined)

    print("\n-- subgroups that regenerating drops --")
    for i in sorted(by_index):
        if i in p.R and p.R[i] > 0 and p.main_group_of.get(i, 0) > 0 and (i < FIRST_CATION or i in p.name):
            continue
        why = ("AIOMFAC gives it no name, so it has no parameters either" if i >= FIRST_CATION
               else "AIOMFAC reserves the slot but gives it no volume")
        print("   %3d %-14s %s" % (i, by_index[i].name, why))

    usable = [i for i in p.ion_ids() if p.is_usable_ion(i)]
    print("\n-- electrolyte parameters (the middle-range tables) --")
    print("   %d ions with middle-range parameters: %s"
          % (len(usable), ", ".join(p.name[i] for i in usable)))
    print("   %d cation-anion pairs" % sum(len(v) for v in p.cation_anion["b"].values()))
    print("=" * 90)


# =================================================================================================
#  Emitters
# =================================================================================================

def current_subgroup_names() -> dict[int, str]:
    """The names the current tables give each subgroup index.

    They are pyoomph's own -- AIOMFAC names only about half of its subgroups in the source, and
    where it does the spelling often differs ("CH3CO (ketone)" against "CH3CO"). Material
    definitions reference these names, so regenerating must not rename anything that already
    exists; new subgroups take AIOMFAC's name where there is one.
    """
    import pyoomph.materials.UNIFAC.aiomfac  # noqa: F401
    from pyoomph.materials.activity import ActivityModel, UNIFACLikeActivityModel
    m = ActivityModel.get_activity_model_by_name("AIOMFAC")
    assert isinstance(m, UNIFACLikeActivityModel)
    return {sg.index: sg.name for sg in m.subgroups.values() if sg.index is not None}


def emit_short_range(p: AIOMFACParameters, keep_names: dict[int, str]) -> str:
    """``aiomfac.py``: the subgroups and the main-group interaction matrix."""
    out = [_LICENSE.rstrip("\n"), '''

##########################################################################################
# These parameters stem from the AIOMFAC activity model
#\t\thttp://www.aiomfac.caltech.edu/
#
# For the source code and parameter tables (GPL v3), see
#\t\thttps://github.com/andizuend/AIOMFAC
#
# Cite the relevant publications when publishing results based on AIOMFAC
#\t\tsee: https://aiomfac.lab.mcgill.ca/citation.html
#
# GENERATED FILE -- do not edit by hand.
# Regenerate with citools/generate_aiomfac_parameters.py, which parses the AIOMFAC Fortran source.
##########################################################################################


from ..activity import UNIFACLikeActivityModel,ActivityModel

@ActivityModel.register_activity_model()
class AIOMFAC(UNIFACLikeActivityModel):
    """
    AIOMFAC activity model (http://www.aiomfac.caltech.edu/), parameters taken from the AIOMFAC
    source code (https://github.com/andizuend/AIOMFAC).

    Cite the publications when using this activity model (https://aiomfac.lab.mcgill.ca/citation.html).

    The ion subgroups (201 and up) carry a charge and are what makes this the only model in pyoomph
    that can say anything about a salt; the middle- and long-range parts that go with them are in
    :py:mod:`pyoomph.materials.UNIFAC.aiomfac_electrolyte`.
    """
    name="AIOMFAC"
    def __init__(self):
        super(AIOMFAC, self).__init__()
        self.define_groups()
        self.define_interaction_table()

    def define_groups(self):''']

    subs_by_main: dict[int, list[int]] = {}
    for sub in sorted(p.R):
        if sub >= FIRST_CATION and sub not in p.name:
            continue      # an unnamed ion slot: AIOMFAC reserves the range, it does not fill it
        if p.R[sub] <= 0.0:
            continue      # a reserved subgroup slot with no volume, i.e. no parameters at all
        main = p.main_group_of.get(sub, 0)
        if main == 0:
            continue      # NKTAB marks these "not assigned to a main group"
        subs_by_main.setdefault(main, []).append(sub)

    # Every main group, including the ones with no subgroup of their own: the interaction matrix is
    # indexed by main group, and set_interaction can only name a group that has been defined.
    for main in range(1, NMAINGROUPS + 1):
        out.append('        with self.define_main_group("%s", index=%d):'
                   % (_safe(p.main_group_name.get(main, "maingroup_%d" % main)), main))
        if not subs_by_main.get(main):
            out.append("            pass   # no subgroup of its own; it appears in the interaction matrix")
        for sub in subs_by_main.get(main, []):
            # AIOMFAC's name wins for an ion: its identity is AIOMFAC's to state, and the old tables
            # had two of them attached to the wrong species (247 was called SCN- and is OH-). For a
            # neutral the existing name wins, because material definitions reference it.
            name = (p.name[sub] if sub >= FIRST_CATION else
                    keep_names.get(sub) or p.name.get(sub) or ("subgroup_%d" % sub))
            args = ['"%s"' % _safe(name),
                    "R=" + _num(p.R[sub]), "Q=" + _num(p.Q[sub])]
            if sub in p.molar_mass:
                args.append("molar_mass=" + _num(p.molar_mass[sub]))
            if sub >= FIRST_CATION:
                args.append("charge=" + str(p.charge[sub]))
            args.append("index=" + str(sub))
            out.append("            self.define_sub_group(" + ", ".join(args) + ")")

    out.append("")
    out.append("    def define_interaction_table(self):")
    undetermined = sorted((i, j) for i in range(1, NMAINGROUPS + 1) for j in range(1, NMAINGROUPS + 1)
                          if abs(p.interaction["ARR"][i][j]) > MARKER_MAGNITUDE)
    out.append('        # Pairs AIOMFAC never determined. Its tables hold a marker there'
               ' (-8.89e5 and')
    out.append('        # relatives); a real interaction parameter is of order 1e3. Left out of the'
               ' table and')
    out.append('        # listed instead, so that a mixture needing one is refused rather than'
               ' quietly')
    out.append('        # treated as ideal -- AIOMFAC stops with an error on exactly these.')
    out.append("        self.undetermined_interactions = {" + ", ".join(
        "(%d, %d)" % (i, j) for i, j in undetermined) + "}")
    for i in range(1, NMAINGROUPS + 1):
        for j in range(1, NMAINGROUPS + 1):
            if i == j:
                continue
            # A only, deliberately. AIOMFAC also carries BRR and CRR, but they belong to the
            # three-parameter temperature form of Ganbavale et al. (2015),
            #   Psi = exp(-A/T + B(1/T0 - 1/T) + C((T0-T)/T + ln(T/T0))),
            # which ModSRunifac.f90 selects only for particular fit datasets; every ordinary
            # AIOMFAC calculation takes the default branch, Psi = exp(-A/T). pyoomph's B and C mean
            # something else again (exp(-(A/T + B + C*T))), so importing AIOMFAC's into them would
            # not be a different parameterisation but a wrong one -- exp(-986) instead of exp(-3).
            a = p.interaction["ARR"][i][j]
            if abs(a) > MARKER_MAGNITUDE or a == 0.0:
                continue
            out.append("        self.set_interaction(%d, %d, Aij=%s)" % (i, j, _num(a)))
    out.append("")
    return "\n".join(out)


def emit_electrolyte(p: AIOMFACParameters) -> str:
    """``aiomfac_electrolyte.py``: everything the middle- and long-range parts need."""
    usable = [i for i in p.ion_ids() if p.is_usable_ion(i)]
    out = [_LICENSE.rstrip("\n"), '''
"""
AIOMFAC's electrolyte parameters: the middle-range ion interactions.

GENERATED FILE -- do not edit by hand. Regenerate with citools/generate_aiomfac_parameters.py.

The short-range part of an ion (its `R` and `Q`) lives with the other subgroups in
:py:mod:`pyoomph.materials.UNIFAC.aiomfac`; what is here is what has no counterpart in plain UNIFAC.

  * `MR_B_ION` / `MR_C_ION`: b and c of a neutral **main group** against an ion, entering as
    `B_ki(I) = b + c*exp(-omega*sqrt(I))` with `omega` = `OMEGA_NEUTRAL_ION`.
  * `MR_CATION_ANION`: b, c, c1, c2 and the two omegas of a cation-anion pair. Missing entries take
    `MR_CATION_ANION_DEFAULTS`, which is what AIOMFAC initialises its tables to.

A pair that AIOMFAC never determined is simply absent here rather than present as its
-7.777778E+04 marker, so a lookup that misses is a mixture AIOMFAC cannot do either.
"""

from __future__ import annotations
''']

    out.append("\n#: Ion subgroup id -> charge number.")
    out.append("ION_CHARGE = {" + ", ".join("%d: %+d" % (i, p.charge[i]) for i in usable) + "}")
    out.append("\n#: Ion subgroup id -> molar mass in kg/mol.")
    out.append("ION_MOLAR_MASS = {" + ", ".join("%d: %s" % (i, _num(p.molar_mass[i])) for i in usable) + "}")
    out.append("\n#: Ion subgroup id -> AIOMFAC's name for it.")
    out.append("ION_NAME = {" + ", ".join('%d: "%s"' % (i, _safe(p.name[i])) for i in usable) + "}")
    out.append("\n#: Molar mass of each neutral subgroup in kg/mol, needed by the middle-range part,")
    out.append("#: which weights a main group's contribution by its mass.")
    out.append("SUBGROUP_MOLAR_MASS = {" + ", ".join(
        "%d: %s" % (i, _num(p.molar_mass[i])) for i in sorted(p.molar_mass) if i < FIRST_CATION) + "}")

    out.append("\n#: The exponent in B_ki(I) = b + c*exp(-omega*sqrt(I)) for a neutral main group")
    out.append("#: against an ion. AIOMFAC uses one value throughout.")
    out.append("OMEGA_NEUTRAL_ION = " + _num(OMEGA_NEUTRAL_ION))

    for pyname, key in (("MR_B_ION", "b"), ("MR_C_ION", "c")):
        out.append("\n#: {ion id: {main group: %s}} for a neutral main group against an ion."
                   % key)
        out.append(pyname + " = {")
        for ion in usable:
            row = p.b_ion[key].get(ion, {})
            if not row:
                continue
            out.append("    %d: {%s}," % (ion, ", ".join(
                "%d: %s" % (mg, _num(v)) for mg, v in sorted(row.items()))))
        out.append("}")

    out.append("\n#: What AIOMFAC initialises the cation-anion tables to before overriding entries.")
    out.append("MR_CATION_ANION_DEFAULTS = {" + ", ".join(
        '"%s": %s' % (k, _num(v)) for k, v in MR_DEFAULTS.items()) + "}")
    out.append("\n#: {(cation id, anion id): {b, c, cn1, cn2, omega, omega2}}.")
    out.append("MR_CATION_ANION = {")
    pairs: set[tuple[int, int]] = set()
    for kind, table in p.cation_anion.items():
        for c, row in table.items():
            for a in row:
                pairs.add((c, a))
    for c, a in sorted(pairs):
        if c not in usable or a not in usable:
            continue
        entry = {k: p.cation_anion[k].get(c, {}).get(a) for k in MR_DEFAULTS}
        if all(v is None or v == MR_DEFAULTS[k] for k, v in entry.items()):
            continue     # nothing but defaults: AIOMFAC has no parameters for this pair
        out.append("    (%d, %d): {%s}," % (c, a, ", ".join(
            '"%s": %s' % (k, _num(v)) for k, v in entry.items() if v is not None)))
    out.append("}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fortran_dir", help="the FortranCode directory of an AIOMFAC clone")
    ap.add_argument("--audit", action="store_true", help="report what differs from the current tables")
    ap.add_argument("--write", action="store_true", help="write the regenerated parameter modules")
    args = ap.parse_args()

    p = AIOMFACParameters(args.fortran_dir)
    if args.audit or not args.write:
        audit(p)
    if args.write:
        keep_names = current_subgroup_names()
        (UNIFAC_DIR / "aiomfac.py").write_text(emit_short_range(p, keep_names))
        (UNIFAC_DIR / "aiomfac_electrolyte.py").write_text(emit_electrolyte(p))
        print("wrote aiomfac.py and aiomfac_electrolyte.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
