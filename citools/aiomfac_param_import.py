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
Reader for the AIOMFAC Fortran parameter tables.

pyoomph's AIOMFAC tables were imported once, by a script nobody kept, from a version nobody recorded.
This is that script, written down: it parses the Fortran source of
https://github.com/andizuend/AIOMFAC directly, so the next AIOMFAC release is a rerun rather than an
archaeology exercise, and so a claim that pyoomph "has the AIOMFAC parameters" can be checked.

Used by ``generate_aiomfac_parameters.py``, which does the comparing and the code emitting. This file
only reads.

The parameters are AIOMFAC's and are distributed under GPL v3; cite the AIOMFAC publications
(https://aiomfac.lab.mcgill.ca/citation.html) when publishing results based on them.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Number of SR main groups, ``Nmaingroups`` in ModSystemProp.f90.
NMAINGROUPS = 76
#: Index of the last subgroup, ``topsubno``. Ions occupy 201-265.
TOPSUBNO = 265
#: The first cation and the first anion subgroup id.
FIRST_CATION, FIRST_ANION = 201, 241
#: AIOMFAC's marker for "this parameter was never determined". Reading one is an error there, and
#: here it means the pair must not be offered to a user at all.
UNDETERMINED = -7.777778e4
#: The same marker in the molar mass tables, where it is written with fewer digits.
UNDETERMINED_MW = -8.8888e5

#: Defaults AIOMFAC initialises the middle-range tables with before overriding individual entries
#: (ModMRpart.f90). ``omega_n_ion`` is never overridden outside the PEG systems, which are out of
#: scope here.
MR_DEFAULTS = {"b": 0.0, "c": 0.0, "cn1": 0.0, "cn2": 0.0, "omega": 0.8, "omega2": 0.6}
OMEGA_NEUTRAL_ION = 1.2


def _fortran_floats(text: str) -> list[float]:
    """Every real literal in a Fortran array body, in order.

    Handles the ``D``/``E`` exponent forms and the ``_wp`` kind suffix, and drops the ``kind=wp``
    trailer that would otherwise read as a bare word.
    """
    text = re.sub(r",\s*kind\s*=\s*wp", "", text)
    out: list[float] = []
    for m in re.finditer(r"[-+]?\d*\.\d+(?:[DdEe][-+]?\d+)?|[-+]?\d+\.(?:[DdEe][-+]?\d+)?", text):
        out.append(float(m.group(0).replace("D", "E").replace("d", "e")))
    return out


def _strip_comments(text: str) -> str:
    """Fortran line comments. The tables carry a ``! 12 (CHn)`` label on nearly every line, and
    those labels contain digits that would otherwise be read as data."""
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def _balanced_bracket_body(text: str, start: int) -> tuple[str, int]:
    """The body of the ``[ ... ]`` beginning at ``start``, and the index just past it."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced [ in the Fortran source at offset " + str(start))


def _find_assignments(text: str, name: str) -> list[tuple[str, str]]:
    """Every ``name(<subscript>) = ... [ body ]`` in the text, as (subscript, body) pairs.

    The subscript is empty for a plain ``name = [...]``. Both the ``reshape(real([...]),...)`` and
    the bare ``real([...], kind=wp)`` spellings end up here, since only the bracket body is taken.
    """
    res: list[tuple[str, str]] = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*(\(([^)=]*)\))?\s*=", text):
        subscript = (m.group(2) or "").strip()
        open_bracket = text.find("[", m.end())
        if open_bracket < 0:
            continue
        body, _ = _balanced_bracket_body(text, open_bracket)
        res.append((subscript, body))
    return res


class AIOMFACParameters:
    """Everything pyoomph needs out of the AIOMFAC Fortran source."""

    def __init__(self, fortran_dir: str | Path):
        self.dir = Path(fortran_dir)
        self._srparam = _strip_comments((self.dir / "ModSRparam.f90").read_text())
        self._subgroup = _strip_comments((self.dir / "ModSubgroupProp.f90").read_text())
        # Not comment-stripped: the ion names live *in* the code as string literals, and the file's
        # own comments are harmless to the regex that reads them.
        self._subgroup_raw = (self.dir / "ModSubgroupProp.f90").read_text()
        self._mrpart = _strip_comments((self.dir / "ModMRpart.f90").read_text())
        self._read_subgroups()
        self._read_interactions()
        self._read_middle_range()

    # ---- short range ---------------------------------------------------------------------------

    def _read_subgroups(self) -> None:
        self.main_group_of: dict[int, int] = {}
        # NKTAB is written as bare integers ("01, 01, 03"), so the real-literal reader does not see
        # them at all -- it needs the decimal point that a Fortran real always has.
        nktab = [int(v) for v in re.findall(r"\d+", _find_assignments(self._subgroup, "NKTAB")[0][1])]
        for i, v in enumerate(nktab[:200], start=1):
            self.main_group_of[i] = v
        # The ions are written as an implied-do "(51, i = 201,topsubno)", which contributes a single
        # literal rather than one per ion.
        for sub in range(FIRST_CATION, TOPSUBNO + 1):
            self.main_group_of[sub] = 51

        self.R: dict[int, float] = {}
        self.Q: dict[int, float] = {}
        for name, target in (("SR_RR", self.R), ("SR_QQ", self.Q)):
            vals = _fortran_floats(_find_assignments(self._srparam, name)[0][1])
            if len(vals) != TOPSUBNO:
                raise ValueError(name + " has " + str(len(vals)) + " entries, expected " + str(TOPSUBNO))
            for i, v in enumerate(vals, start=1):
                target[i] = v

        # Molar masses, all in g/mol in the Fortran and all converted to kg/mol here: pyoomph is an
        # SI codebase, and mixing the two is exactly the bug this import found in the old tables.
        self.molar_mass: dict[int, float] = {}
        for i, v in enumerate(_fortran_floats(_find_assignments(self._subgroup, "GroupMW")[0][1]), start=1):
            if v > UNDETERMINED_MW / 2:
                self.molar_mass[i] = v * 1e-3
        for name, first in (("SMWC", FIRST_CATION), ("SMWA", FIRST_ANION)):
            for i, v in enumerate(_fortran_floats(_find_assignments(self._subgroup, name)[0][1])):
                if v > UNDETERMINED_MW / 2:
                    self.molar_mass[first + i] = v * 1e-3

        # Charges follow the id ranges rather than a table: 201-220 are +1, 221-240 +2, 241-260 -1,
        # 261- -2 (Ioncharge in ModSubgroupProp.f90).
        self.charge: dict[int, int] = {}
        for sub in range(FIRST_CATION, TOPSUBNO + 1):
            self.charge[sub] = 1 if sub < 221 else 2 if sub < 241 else -1 if sub < 261 else -2

        self.name: dict[int, str] = {}
        for m in re.finditer(r"subgrname\((\d+)\)\s*=\s*\"([^\"]*)\"", self._subgroup_raw):
            self.name[int(m.group(1))] = m.group(2).strip("()")
        self.main_group_name: dict[int, str] = {}
        for m in re.finditer(r"maingrname\((\d+)\)\s*=\s*\"([^\"]*)\"", self._subgroup_raw):
            self.main_group_name[int(m.group(1))] = m.group(2).strip("()")

    def _read_interactions(self) -> None:
        """ARR/BRR/CRR, as ``[i][j]`` dictionaries over main groups.

        Each is a 76x76 matrix written column-major, one Fortran continuation line per column, so
        the k-th value of the flat list is ARR(i,j) with j = k//76 and i = k%76. AIOMFAC ships two
        ARR tables and selects between them with ``use_latest_param``, whose default is ``.false.``;
        the later one is marked "not yet ready for AIOMFAC-web" and is empty. Taking the one the
        model actually uses is the whole point of the exercise, so this reads the *last* assignment
        of each name, which is the one inside the ``else`` branch.
        """
        self.interaction: dict[str, dict[int, dict[int, float]]] = {}
        for name in ("ARR", "BRR", "CRR"):
            bodies = [b for _, b in _find_assignments(self._srparam, name)]
            vals: list[float] = []
            for body in bodies:
                v = _fortran_floats(body)
                if len(v) == NMAINGROUPS * NMAINGROUPS:
                    vals = v      # keep the last complete table, i.e. the one that is actually used
            if not vals:
                raise ValueError("no complete " + name + " table found")
            table: dict[int, dict[int, float]] = {i: {} for i in range(1, NMAINGROUPS + 1)}
            for k, v in enumerate(vals):
                j, i = k // NMAINGROUPS + 1, k % NMAINGROUPS + 1
                table[i][j] = v
            self.interaction[name] = table

    # ---- middle range --------------------------------------------------------------------------

    def _read_middle_range(self) -> None:
        """The MR tables: main group <-> ion, and cation <-> anion.

        ``bTABnc(1:Nmaingroups,NC)`` is one statement per cation, listing the 76 main groups; the
        anion tables are the same with NA = id-240. The cation-anion tables are indexed the other
        way round, ``bTABAC(NC,NA)``, one statement per cation listing the anions.
        """
        self.b_ion: dict[str, dict[int, dict[int, float]]] = {}
        for tabname, kind, first in (("bTABnc", "b", FIRST_CATION), ("cTABnc", "c", FIRST_CATION),
                                     ("bTABna", "b", FIRST_ANION), ("cTABna", "c", FIRST_ANION)):
            target = self.b_ion.setdefault(kind, {})
            for subscript, body in _find_assignments(self._mrpart, tabname):
                m = re.search(r",\s*(\d+)\s*$", subscript)
                if m is None:
                    continue
                ion = first + int(m.group(1)) - 1
                vals = _fortran_floats(body)
                if len(vals) != NMAINGROUPS:
                    continue
                target.setdefault(ion, {})
                for mg, v in enumerate(vals, start=1):
                    if v > UNDETERMINED / 2:
                        target[ion][mg] = v

        # The cation-anion parameters are written one scalar assignment at a time,
        # "bTABAC(1,2) = 0.1065...E0_wp", rather than as array literals.
        self.cation_anion: dict[str, dict[int, dict[int, float]]] = {}
        for tabname, kind in (("bTABAC", "b"), ("cTABAC", "c"), ("cn1TABAC", "cn1"),
                              ("cn2TABAC", "cn2"), ("omegaTAB", "omega"), ("omega2TAB", "omega2")):
            target = self.cation_anion.setdefault(kind, {})
            for m in re.finditer(r"\b" + tabname + r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*"
                                 r"([-+]?[\d.]+(?:[DdEe][-+]?\d+)?)", self._mrpart, re.IGNORECASE):
                cation = FIRST_CATION + int(m.group(1)) - 1
                anion = FIRST_ANION + int(m.group(2)) - 1
                target.setdefault(cation, {})[anion] = float(m.group(3).replace("D", "E").replace("d", "e"))

        self.lambda_co2: dict[int, float] = {}
        for _, body in _find_assignments(self._mrpart, "lambdaIN"):
            for i, v in enumerate(_fortran_floats(body)):
                self.lambda_co2[FIRST_CATION + i] = v
            break

    # ---- convenience ----------------------------------------------------------------------------

    def ion_ids(self) -> list[int]:
        """Every ion subgroup that AIOMFAC gives a name, in id order."""
        return sorted(i for i in self.name if i >= FIRST_CATION)

    def is_usable_ion(self, ion: int) -> bool:
        """Whether the ion has the data an actual calculation needs: a molar mass, and at least one
        determined middle-range interaction. F- is named but has neither, and AIOMFAC's own comment
        says it "is not yet supported for calculations"."""
        if ion not in self.molar_mass:
            return False
        if self.R.get(ion, 0.0) <= 0.0:
            # Middle-range parameters but no volume: AIOMFAC reserves the slot without being able to
            # put the ion in the short-range part, so it cannot be used at all.
            return False
        if ion < FIRST_ANION:
            return bool(self.cation_anion.get("b", {}).get(ion))
        return any(ion in per_cation for per_cation in self.cation_anion.get("b", {}).values())
