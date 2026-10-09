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

"""Numerical validation of the tutorial scripts, for citools/test_all_tutorial_scripts.py.

The harness runs 141 tutorial scripts and, on its own, only asks whether they exit 0. A script that
computes the wrong answer - a flipped sign in a weak form, a regressed time integrator, an
eigensolver that converges to the wrong mode - passes just as happily as a correct one. This module
is the post-run step that compares what a script produced against reference numbers committed to the
repository.

The tutorial scripts themselves stay untouched. They are documentation, and an assertion in the
middle of one is noise to a reader, so nothing here is visible from inside a script. Three kinds of
artefact are read instead:

  * the text files a script writes anyway (ODEFileOutput, IntegralObservableOutput, TextFileOutput,
    Problem.create_text_file_output), parsed with pyoomph's own LoadedTextDataFile;
  * the stdout the harness already has in hand;
  * the state fingerprint Problem writes when $PYOOMPH_VALIDATION_DUMP is set (see
    Problem._write_validation_dump) - for 29 of the scripts that is the only artefact there is, and
    it is the one that carries the global parameter values, i.e. the answer of every fold, Hopf,
    pitchfork and arclength script.

What is checked per script lives in specs.py; the numbers live in data/<Folder>/<stem>/reference.json
and are regenerated with the harness's --update-validation.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
from pathlib import Path

# Tolerances. Named here rather than spelled out per check, the way tests/ names its own
# (_CROSS_RTOL in tests/test_adapt_while_tracking.py). The default is deliberately not as tight as
# a single machine can hold: the same reference data has to pass on linux-x86_64, macOS-arm64 and
# Windows, against three BLAS implementations and two PETSc builds, and a Newton-converged solution
# agrees to roughly this much across those, not to 1e-12.
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-10

#: Rows kept in a reference time series. The full evolution is compared - every reference row is
#: looked up in the produced file - but storing all 1001 rows of a numouts=1000 run would put tens
#: of megabytes of recorded output into the repository for nothing: a time integrator that drifts
#: does not do so between two neighbouring outputs only.
MAX_REFERENCE_ROWS = 64

_FINGERPRINT_FILE = "_pyoomph_validation.jsonl"  # Problem._write_validation_dump writes this
_KEYFILE = "_pyoomph_run_.txt"                   # ...and this marks a pyoomph output directory


# ----------------------------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------------------------

class Check:
    """Base class: the tolerances, and the two ways a check can be excused.

    Args:
        rtol, atol: passed to _close(). Per-check, because what is reproducible differs by orders
            of magnitude between a stationary solve and a long transient.
        reason: why this check is weaker than the default. Required by review, not by the code: a
            relaxed tolerance with no reason next to it is the thing nobody can judge later.
        skip_under: pass labels ("mpi") under which this check cannot hold and is not run.
    """

    kind = "check"

    def __init__(self, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
                 reason: "str | None" = None, skip_under: "tuple[str,...]" = ()):
        self.rtol, self.atol = rtol, atol
        self.reason = reason
        self.skip_under = tuple(skip_under)

    def slug(self) -> str:
        """The key this check's reference numbers are stored under. Stable across runs."""
        raise NotImplementedError

    def label(self) -> str:
        return self.slug()


class Evolution(Check):
    """A whole time series, from any of pyoomph's line-per-row text writers.

    This is the "for ODEs the full evolution can be compared" case: every stored reference row is
    compared, not just the final state.

    Args:
        file: the file to read, as a glob against its path relative to the directory the script ran
            in (so "harmonic_oscillator.txt" finds it inside whatever output directory the script
            chose, and "diffusion/*/ode.txt" picks one of several output directories apart).
        columns: which columns to compare, by name as the header spells them. None means all of
            them except the abscissa.
        abscissa: the column rows are matched by. None means the first column, which is "time" for
            every writer pyoomph has.
        until_time: ignore the part of the evolution beyond this abscissa value. For the chaotic
            scripts, where a comparison is only meaningful early on.
        from_time: ignore the part before it. For a quantity that starts from a transient nobody
            should pin down - a Lyapunov exponent is meaningless until it has converged, and its
            early values grow exponentially, so no tolerance could hold them across platforms.
        match: how a stored row is found again in a later run.

            "exact" (the default) looks the stored abscissa value up in the produced file and wants
                a row at that very instant. Right for everything that writes at prescribed times,
                which is every run(endtime=..., numouts=...): those instants are a property of the
                script, not of the machine. A time stepper that lands elsewhere is then reported,
                rather than papered over.
            "interp" interpolates the produced series onto the stored instants. For a script whose
                output times are themselves adaptive and so differ from run to run. Note what it
                costs: linear interpolation over a step h is wrong by about h^2*y''/8, which for a
                coarse step is far more than DEFAULT_RTOL, so a check using it has to carry an rtol
                that admits the interpolation error and a reason= saying so. It will not catch a
                small drift.
            "rows" matches by row index and insists on the same number of rows. For a file whose
                first column does not grow at all - an arclength continuation writing the parameter
                there, a sweep that turns around.
    """

    kind = "evolution"

    def __init__(self, file: str, columns: "list[str] | None" = None, abscissa: "str | None" = None,
                 until_time: "float | None" = None, from_time: "float | None" = None,
                 match: str = "exact", **kw):
        super().__init__(**kw)
        if match not in ("exact", "interp", "rows"):
            raise ValueError("match= is 'exact', 'interp' or 'rows', not %r" % match)
        if match == "rows" and (until_time is not None or from_time is not None):
            # Row matching insists on the same total row count, and a window would make that count
            # depend on values rather than on the file - two different ways of lining rows up.
            raise ValueError("match=\"rows\" cannot be combined with until_time/from_time")
        self.file, self.columns, self.abscissa = file, columns, abscissa
        self.until_time, self.from_time = until_time, from_time
        self.match = match

    def window(self, data, xi):
        """*data* restricted to [from_time, until_time] on column *xi*."""
        if self.from_time is not None:
            data = data[data[:, xi] >= self.from_time]
        if self.until_time is not None:
            data = data[data[:, xi] <= self.until_time]
        return data

    def slug(self) -> str:
        return "evolution:" + self.file

    def label(self) -> str:
        window = ""
        if self.from_time is not None or self.until_time is not None:
            window = " [%s:%s]" % ("" if self.from_time is None else "%g" % self.from_time,
                                   "" if self.until_time is None else "%g" % self.until_time)
        return ("evolution of " + self.file + window
                + ("" if self.match == "exact" else " (%s)" % self.match))


class FinalState(Check):
    """The last spatial output of a PDE script.

    Args:
        file: glob for the numbered files TextFileOutput writes
            (``<trunk>/<trunk>_%06d.txt``); the highest-numbered match is taken.
        reduce: compare per-column n/min/max/mean/l2 instead of node by node. The default, and what
            makes the check survive a node ordering that depends on the mesh partitioning or on an
            adaptation. Set False only where the mesh is fixed and the ordering is not in doubt.
        columns: which columns to compare, by name. None means all of them.
    """

    kind = "finalstate"

    def __init__(self, file: str, reduce: bool = True, columns: "list[str] | None" = None, **kw):
        super().__init__(**kw)
        self.file, self.reduce, self.columns = file, reduce, columns

    def slug(self) -> str:
        return "finalstate:" + self.file

    def label(self) -> str:
        return ("reduced " if self.reduce else "nodal ") + "final state of " + self.file


class Fingerprint(Check):
    """The state fingerprint Problem wrote at teardown.

    Args:
        index: which problem, for a script that builds several. 0 is the first one torn down.
        outdir: restrict to the fingerprint written in this output directory (a glob). For the
            scripts that set their own, several times per run.
        only / skip: dotted keys of the fingerprint to compare or to leave out, e.g.
            only=["params.r"], or skip=["ndof", "dofs.*.n"] for an adaptive script whose element
            count is not reproducible across platforms.
    """

    kind = "fingerprint"

    def __init__(self, index: int = 0, outdir: "str | None" = None,
                 only: "list[str] | None" = None, skip: "list[str] | None" = None, **kw):
        super().__init__(**kw)
        self.index, self.outdir, self.only, self.skip = index, outdir, only, list(skip or [])

    def slug(self) -> str:
        return "fingerprint:%s:%d" % (self.outdir or "", self.index)

    def label(self) -> str:
        return "state fingerprint" + (" of %s" % self.outdir if self.outdir else "") + \
               ("" if self.index == 0 else " (problem %d)" % self.index)


class Stdout(Check):
    """Numbers a script only prints.

    The last resort, for a script whose answer is neither in a file nor a global parameter. Brittle
    against a reworded print by construction, so prefer a Fingerprint wherever the number is a
    parameter or a dof.

    Args:
        pattern: a regular expression whose capture groups are read as floats, searched over the
            whole stdout. Every match is compared, in order.
    """

    kind = "stdout"

    def __init__(self, pattern: str, **kw):
        super().__init__(**kw)
        self.pattern = pattern

    def slug(self) -> str:
        return "stdout:" + self.pattern

    def label(self) -> str:
        return "printed numbers matching /%s/" % self.pattern


# ----------------------------------------------------------------------------------------------
# Comparing
# ----------------------------------------------------------------------------------------------

def _close(ref, got, rtol, atol) -> bool:
    if ref is None or got is None:
        return ref is got
    if isinstance(ref, bool) or isinstance(got, bool):
        return bool(ref) == bool(got)
    if isinstance(ref, str) or isinstance(got, str):
        return ref == got
    rf, gf = float(ref), float(got)
    if math.isnan(rf) or math.isnan(gf):
        # NaN is a value here, not a failure: the 1d text writers separate line segments with NaN
        # rows, and a reference NaN has to be matched by a produced NaN.
        return math.isnan(rf) and math.isnan(gf)
    if math.isinf(rf) or math.isinf(gf):
        return rf == gf
    return abs(gf - rf) <= atol + rtol * abs(rf)


def _deviation(ref, got) -> str:
    try:
        rf, gf = float(ref), float(got)
    except (TypeError, ValueError):
        return "expected %r, got %r" % (ref, got)
    rel = "" if rf == 0.0 else " (%+.3g relative)" % ((gf - rf) / rf)
    return "expected %.12g, got %.12g%s" % (rf, gf, rel)


class Outcome:
    """What one script's validation came to."""

    def __init__(self):
        self.ran: "list[str]" = []        # labels of the checks that compared something
        self.skipped: "list[str]" = []    # labels excused by skip_under
        self.problems: "list[str]" = []   # one line per mismatch or missing artefact

    @property
    def status(self) -> str:
        if self.problems:
            return "mismatch"
        if self.ran:
            return "ok"
        # Nothing compared. Either there is no reference data for this script yet, or every one of
        # its checks said skip_under this pass - a distinction worth keeping, since the first is a
        # gap to fill and the second is a deliberate statement about the script.
        return "skipped" if self.skipped else "no-reference"

    def summary(self) -> str:
        if self.problems:
            return "%d problem(s) in %d check(s)" % (len(self.problems), len(self.ran) + len(self.problems))
        parts = ["%d check(s)" % len(self.ran)] if self.ran else ["nothing to check"]
        if self.skipped:
            parts.append("%d skipped" % len(self.skipped))
        return ", ".join(parts)


# ----------------------------------------------------------------------------------------------
# Finding the artefacts a run left behind
# ----------------------------------------------------------------------------------------------

class RunArtifacts:
    """Everything one finished script left in the directory it ran in.

    Only what THIS run wrote: the harness runs every script of a folder with that folder as the
    working directory, and the output directories of earlier scripts are still lying around (only
    the one named after the script is deleted, and --keep-outdirs keeps even that). The mtime filter
    is the same one simulation_seconds() uses in the harness, and for the same reason.
    """

    def __init__(self, rundir: "str | Path", started_at: float, stdout: bytes = b""):
        self.rundir = Path(rundir)
        self.started_at = started_at
        self.stdout = stdout
        self.outdirs = self._find_outdirs()

    def _fresh(self, path: Path) -> bool:
        """Whether *path* was written after the script started.

        No slack at all, unlike simulation_seconds() in the harness, which allows a second: here
        a second is enough to read the PREVIOUS script's output as this one's. Several tutorials
        set their own output directories (time_stepping_schemes.py writes seven), and the harness
        only ever deletes the one named after the script, so those directories survive into the
        next script's run - which starts a fraction of a second later. With a second of slack,
        bifurcation_transcritital_arclength_eigen.py was credited with
        time_stepping_schemes.py's anharmonic_oscillator.txt and with its second problem's
        fingerprint. started_at is taken immediately before the subprocess starts, so anything this
        run wrote is strictly newer.
        """
        try:
            return path.stat().st_mtime >= self.started_at
        except OSError:
            return False

    def _find_outdirs(self) -> "list[Path]":
        found = []
        for keyfile in self.rundir.rglob(_KEYFILE):
            if self._fresh(keyfile):
                found.append(keyfile.parent)
        return sorted(found)

    def text_files(self) -> "list[Path]":
        """The data files of this run, as paths relative to the run directory.

        Housekeeping is left out the way tests/test_text_output_headers.py leaves it out: anything
        whose name starts with an underscore (_pyoomph_logfile.txt, _numerical_factors.txt,
        _pyoomph_run_.txt) and anything inside a _-prefixed directory (_ccode, _states, _plots).
        """
        out = []
        for outdir in self.outdirs:
            for path in sorted(outdir.rglob("*.txt")):
                rel = path.relative_to(self.rundir)
                if any(part.startswith("_") for part in rel.parts):
                    continue
                if self._fresh(path):
                    out.append(rel)
        return out

    def match_files(self, pattern: str) -> "list[Path]":
        """The data files matching a spec's glob, sorted by name.

        Matched against the path relative to the run directory, so a pattern can name the output
        directory when it has to; a pattern without a slash is matched against the basename alone,
        which is the common case and keeps the script's own stem out of the spec.
        """
        files = self.text_files()
        if "/" in pattern:
            return [f for f in files if fnmatch.fnmatch(f.as_posix(), pattern)
                    or fnmatch.fnmatch(f.as_posix(), "*/" + pattern)]
        return [f for f in files if fnmatch.fnmatch(f.name, pattern)]

    def fingerprints(self, outdir_pattern: "str | None" = None) -> "list[dict]":
        """The fingerprint records of this run, in the order the problems were torn down."""
        records = []
        for outdir in self.outdirs:
            path = outdir / _FINGERPRINT_FILE
            if not path.is_file() or not self._fresh(path):
                continue
            rel = outdir.relative_to(self.rundir).as_posix()
            if outdir_pattern is not None and not fnmatch.fnmatch(rel, outdir_pattern):
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records


# ----------------------------------------------------------------------------------------------
# Reading a pyoomph text file
# ----------------------------------------------------------------------------------------------

def _load_text(path: Path):
    """(column names, data) of a pyoomph text file.

    Spelled out here rather than delegating to pyoomph.utils.num_text_out.LoadedTextDataFile, which
    does exactly this and more, because **this module runs inside the harness process and that
    process must stay free of MPI**. Importing pyoomph initialises MPI, and MPI_Init sets some
    twenty PMIX_*/OMPI_* variables through C setenv() that Python's os.environ never sees but that
    exec() hands to every tested script - where the mpirun of --mpirun finds PMIX_NAMESPACE, decides
    it is already inside an MPI job and exits 1 without printing anything. See the note above
    _PETSC_PROBE in citools/test_all_tutorial_scripts.py, which keeps petsc4py out of this process
    for the same reason.

    The parsing rules are LoadedTextDataFile's: split the "#" header on TABS, which is what every
    pyoomph writer joins it with, falling back to whitespace for a file that has none (a column name
    written before UNIT_SEPARATOR_IN_FILES existed can contain a space, "power[kg m^2/s^3]"), and
    treat the trailing "@key=value" entries - which TextFileOutput appends for the time and the
    global parameters - as parameters rather than as columns.
    """
    import warnings

    import numpy
    with open(path) as f:
        header = f.readline().strip()
    headerless = not header.startswith("#")
    with warnings.catch_warnings():
        # A header and no rows is a legitimate state - a script that opened an output file and then
        # took a branch which writes nothing into it - and numpy warns about it. Every caller here
        # handles an empty table, and says something more useful about it than numpy does.
        warnings.simplefilter("ignore", UserWarning)
        data = numpy.loadtxt(path, ndmin=2)
    if data.size == 0:
        data = data.reshape((0, 0))
    if headerless:
        # Not every writer emits one: NumericalTextOutputFile takes header=None, and
        # utils/lyapunov.py's exponent file is written that way. Name the columns by position then,
        # so a spec can still select them - it just has to say column_2 rather than "z".
        return ["column_%d" % i for i in range(data.shape[1])], data
    body = header.strip("#").strip()
    names = [n.strip() for n in (body.split("\t") if "\t" in body else body.split()) if n.strip()]
    return names[:data.shape[1]], data


def _last_numbered(files: "list[Path]") -> Path:
    """The highest-numbered of TextFileOutput's <trunk>_%06d.txt files."""
    def number(path: Path) -> int:
        m = re.search(r"_(\d+)\.txt$", path.name)
        return int(m.group(1)) if m else -1
    return sorted(files, key=lambda f: (number(f), f.as_posix()))[-1]


def _reduce_column(values) -> dict:
    import numpy
    col = numpy.asarray(values, dtype=float)
    finite = col[~numpy.isnan(col)]
    if not len(finite):
        return {"n": int(len(col)), "n_finite": 0}
    return {"n": int(len(col)), "n_finite": int(len(finite)),
            "min": float(numpy.min(finite)), "max": float(numpy.max(finite)),
            "mean": float(numpy.mean(finite)),
            "l2": float(numpy.sqrt(numpy.sum(finite * finite)))}


# ----------------------------------------------------------------------------------------------
# Per-check generation and comparison
#
# Each check has a _gen_* that turns a finished run into the numbers that go into the repository,
# and a _cmp_* that compares a finished run against those numbers. They are deliberately a pair:
# whatever the generator decided (which rows, which matching mode) is recorded, so the comparison
# is not free to decide it differently on another machine.
# ----------------------------------------------------------------------------------------------

def _pick_rows(count: int) -> "list[int]":
    """At most MAX_REFERENCE_ROWS indices, spread over 0..count-1, always including the last."""
    if count <= MAX_REFERENCE_ROWS:
        return list(range(count))
    step = (count - 1) / float(MAX_REFERENCE_ROWS - 1)
    picked = sorted({int(round(i * step)) for i in range(MAX_REFERENCE_ROWS)} | {count - 1})
    return [i for i in picked if i < count]


def _abscissa_index(check, columns: "list[str]") -> int:
    if check.abscissa is None:
        return 0
    for i, name in enumerate(columns):
        if name == check.abscissa or name.startswith(check.abscissa):
            return i
    raise KeyError("no column '%s' in %s" % (check.abscissa, ", ".join(columns)))


def _compared_columns(check, columns: "list[str]", exclude: "set[int]" = frozenset()) -> "list[int]":
    if check.columns is None:
        return [i for i in range(len(columns)) if i not in exclude]
    out = []
    for want in check.columns:
        hits = [i for i, name in enumerate(columns) if name == want]
        if not hits:
            hits = [i for i, name in enumerate(columns) if name.startswith(want)]
        if len(hits) != 1:
            raise KeyError("'%s' matches %d columns of %s" % (want, len(hits), ", ".join(columns)))
        out.append(hits[0])
    return out


def _gen_evolution(check: Evolution, art: RunArtifacts) -> dict:
    import numpy
    files = art.match_files(check.file)
    if len(files) != 1:
        raise FileNotFoundError("'%s' matched %d files (%s)"
                                % (check.file, len(files), ", ".join(f.as_posix() for f in files) or "none"))
    columns, data = _load_text(art.rundir / files[0])
    xi = _abscissa_index(check, columns)
    data = check.window(data, xi)
    if not len(data):
        raise ValueError("no rows left to compare in " + files[0].as_posix()
                         + " once from_time/until_time are applied")
    cols = _compared_columns(check, columns, exclude={xi})
    if not cols:
        raise ValueError("nothing left to compare in %s once the abscissa '%s' is excluded"
                         % (files[0].as_posix(), columns[xi]))
    x = data[:, xi]
    monotonic = bool(numpy.all(numpy.diff(x) >= 0.0)) and len(numpy.unique(x)) > 1
    if check.match != "rows" and not monotonic:
        # Looking a value up, or interpolating on it, both need an abscissa that only grows.
        raise ValueError("the column '%s' of %s does not grow monotonically, so match='%s' cannot "
                         "work - use match=\"rows\""
                         % (columns[xi], files[0].as_posix(), check.match))
    rows = _pick_rows(len(data))
    if check.match == "exact":
        # Duplicated instants are what output_every_step=True produces, and a stored row at one of
        # them would be ambiguous on the way back. Prefer a neighbour that is unique.
        unique = set(numpy.flatnonzero(_unique_abscissa_mask(x)).tolist())
        rows = [r for r in rows if r in unique] or rows
    ref = {"kind": "evolution", "file": files[0].as_posix(),
           "abscissa": columns[xi], "columns": [columns[i] for i in cols],
           "match": check.match,
           "nrows": int(len(data)),
           "rows": [[float(data[r, xi])] + [float(data[r, i]) for i in cols] for r in rows]}
    if check.match == "rows":
        ref["row_indices"] = rows
    return ref


def _unique_abscissa_mask(x):
    """Rows whose abscissa value occurs exactly once in *x* (which must be sorted)."""
    import numpy
    mask = numpy.ones(len(x), dtype=bool)
    mask[:-1] &= x[1:] != x[:-1]
    mask[1:] &= x[1:] != x[:-1]
    return mask


def _cmp_evolution(check: Evolution, art: RunArtifacts, ref: dict, out: Outcome) -> None:
    import numpy
    files = art.match_files(check.file)
    if len(files) != 1:
        out.problems.append("%s: '%s' matched %d files, expected 1"
                            % (check.label(), check.file, len(files)))
        return
    columns, data = _load_text(art.rundir / files[0])
    try:
        xi = columns.index(ref["abscissa"])
        cols = [columns.index(want) for want in ref["columns"]]
    except ValueError:
        out.problems.append("%s: the columns changed - the reference wants %s, the file has %s"
                            % (check.label(), ", ".join([ref["abscissa"]] + ref["columns"]),
                               ", ".join(columns)))
        return
    data = check.window(data, xi)
    if not len(data):
        out.problems.append("%s: no rows left in %s once from_time/until_time are applied"
                            % (check.label(), files[0].as_posix()))
        return
    refrows = numpy.asarray(ref["rows"], dtype=float)
    mode = ref.get("match", "exact")

    if mode == "rows":
        if len(data) != ref["nrows"]:
            out.problems.append("%s: %d rows, the reference was taken from %d - with a "
                                "non-monotonic abscissa the rows have to line up exactly"
                                % (check.label(), len(data), ref["nrows"]))
            return
        got = numpy.column_stack([data[ref["row_indices"], i] for i in cols])
    else:
        x = data[:, xi]
        order = numpy.argsort(x, kind="stable")
        x, rows_sorted = x[order], data[order]
        span = max(float(numpy.max(x) - numpy.min(x)), 1.0)
        if mode == "exact":
            # Looked up, not interpolated: the instants a script writes at are prescribed by the
            # script (run(endtime=..., numouts=...) hits them exactly), so a row that is not there
            # any more means the time stepping changed, which is itself worth reporting.
            idx = numpy.searchsorted(x, refrows[:, 0])
            idx = numpy.clip(idx, 0, len(x) - 1)
            left = numpy.clip(idx - 1, 0, len(x) - 1)
            take = numpy.where(numpy.abs(x[left] - refrows[:, 0])
                               <= numpy.abs(x[idx] - refrows[:, 0]), left, idx)
            off = numpy.abs(x[take] - refrows[:, 0])
            bad = numpy.flatnonzero(off > 1e-9 * span)
            if len(bad):
                out.problems.append(
                    "%s: %d of %d stored instants do not occur in this run (worst: %s=%.12g is "
                    "%.3g away from the nearest row). The time stepping changed; if the output "
                    "times of this script are themselves adaptive, the check needs "
                    "match=\"interp\" and a tolerance that admits the interpolation error"
                    % (check.label(), len(bad), len(refrows), ref["abscissa"],
                       refrows[bad[int(numpy.argmax(off[bad]))], 0], float(numpy.max(off))))
                return
            got = numpy.column_stack([rows_sorted[take, i] for i in cols])
        else:
            lo, hi = float(x[0]), float(x[-1])
            want_lo, want_hi = float(numpy.min(refrows[:, 0])), float(numpy.max(refrows[:, 0]))
            if want_lo < lo - 1e-9 * span or want_hi > hi + 1e-9 * span:
                # numpy.interp clamps, so without this a run that stopped early would have its last
                # value compared against the whole tail of the reference, and pass.
                out.problems.append("%s: the evolution covers %.12g to %.12g, the reference needs "
                                    "%.12g to %.12g"
                                    % (check.label(), lo, hi, want_lo, want_hi))
                return
            # numpy.interp wants a strictly increasing abscissa, and output_every_step=True
            # produces the same instant twice. Keep the last row of each run of equal values: that
            # is the one written after the step completed.
            keep = numpy.ones(len(x), dtype=bool)
            keep[:-1] = x[1:] != x[:-1]
            got = numpy.column_stack([numpy.interp(refrows[:, 0], x[keep], rows_sorted[keep, i])
                                      for i in cols])

    worst = []
    for r in range(len(refrows)):
        for c, name in enumerate(ref["columns"]):
            if not _close(refrows[r, c + 1], got[r, c], check.rtol, check.atol):
                worst.append("at %s=%.12g, %s: %s"
                             % (ref["abscissa"], refrows[r, 0], name,
                                _deviation(refrows[r, c + 1], got[r, c])))
    if worst:
        out.problems.append("%s: %d of %d values differ\n        %s"
                            % (check.label(), len(worst), len(refrows) * len(ref["columns"]),
                               "\n        ".join(worst[:8])
                               + ("\n        ... and %d more" % (len(worst) - 8) if len(worst) > 8 else "")))


def _gen_finalstate(check: FinalState, art: RunArtifacts) -> dict:
    files = art.match_files(check.file)
    if not files:
        raise FileNotFoundError("'%s' matched no file" % check.file)
    path = _last_numbered(files)
    columns, data = _load_text(art.rundir / path)
    cols = _compared_columns(check, columns)
    ref = {"kind": "finalstate", "file": path.as_posix(), "reduce": bool(check.reduce),
           "columns": [columns[i] for i in cols], "nrows": int(len(data))}
    if check.reduce:
        ref["stats"] = {columns[i]: _reduce_column(data[:, i]) for i in cols}
    else:
        ref["values"] = [[float(data[r, i]) for i in cols] for r in range(len(data))]
    return ref


def _cmp_finalstate(check: FinalState, art: RunArtifacts, ref: dict, out: Outcome) -> None:
    files = art.match_files(check.file)
    if not files:
        out.problems.append("%s: '%s' matched no file" % (check.label(), check.file))
        return
    path = _last_numbered(files)
    columns, data = _load_text(art.rundir / path)
    missing = [name for name in ref["columns"] if name not in columns]
    if missing:
        out.problems.append("%s: the file no longer has the column(s) %s"
                            % (check.label(), ", ".join(missing)))
        return
    if ref.get("reduce", True):
        for name in ref["columns"]:
            got = _reduce_column(data[:, columns.index(name)])
            for stat, want in sorted(ref["stats"][name].items()):
                if stat not in got:
                    out.problems.append("%s: %s has no %s any more" % (check.label(), name, stat))
                elif not _close(want, got[stat], check.rtol, check.atol):
                    out.problems.append("%s: %s %s: %s" % (check.label(), name, stat,
                                                           _deviation(want, got[stat])))
    else:
        if len(data) != ref["nrows"]:
            out.problems.append("%s: %d rows, the reference has %d"
                                % (check.label(), len(data), ref["nrows"]))
            return
        bad = 0
        for r, row in enumerate(ref["values"]):
            for c, name in enumerate(ref["columns"]):
                if not _close(row[c], data[r, columns.index(name)], check.rtol, check.atol):
                    bad += 1
                    if bad <= 8:
                        out.problems.append("%s: row %d, %s: %s"
                                            % (check.label(), r, name,
                                               _deviation(row[c], data[r, columns.index(name)])))
        if bad > 8:
            out.problems.append("%s: ... and %d further nodal values" % (check.label(), bad - 8))


# These two are recorded in a fingerprint so that a reader can tell which mode produced it, not so
# that they can be compared: the MPI pass runs the same script over four ranks on purpose, and the
# whole point of the per-dof-type reductions is that its answer is the serial one.
_FINGERPRINT_NEVER_COMPARED = ("mpi_size", "distributed")


def _flatten_fingerprint(record: dict) -> "dict[str,object]":
    """A fingerprint as dotted keys, which is what only=/skip= select on."""
    flat: "dict[str,object]" = {}
    for key in ("ndof", "time"):
        if key in record:
            flat[key] = record[key]
    for name, value in (record.get("params") or {}).items():
        flat["params." + name] = value
    for i, ev in enumerate(record.get("eigenvalues") or []):
        flat["eigenvalues.%d.re" % i] = ev[0]
        flat["eigenvalues.%d.im" % i] = ev[1]
    for name, stats in (record.get("dofs") or {}).items():
        for stat, value in stats.items():
            flat["dofs.%s.%s" % (name, stat)] = value
    return flat


def _fingerprint_record(check: Fingerprint, art: RunArtifacts) -> dict:
    records = art.fingerprints(check.outdir)
    if len(records) <= check.index:
        raise FileNotFoundError(
            "the run left %d fingerprint record(s)%s, so there is no number %d. Is "
            "$PYOOMPH_VALIDATION_DUMP set for the script?"
            % (len(records), " for '%s'" % check.outdir if check.outdir else "", check.index))
    return records[check.index]


def _gen_fingerprint(check: Fingerprint, art: RunArtifacts) -> dict:
    return {"kind": "fingerprint", "record": _fingerprint_record(check, art)}


def _cmp_fingerprint(check: Fingerprint, art: RunArtifacts, ref: dict, out: Outcome) -> None:
    try:
        record = _fingerprint_record(check, art)
    except FileNotFoundError as e:
        out.problems.append("%s: %s" % (check.label(), e))
        return
    want = _flatten_fingerprint(ref["record"])
    got = _flatten_fingerprint(record)

    def selected(key: str) -> bool:
        if key in _FINGERPRINT_NEVER_COMPARED:
            return False
        if check.only is not None and not any(fnmatch.fnmatch(key, p) for p in check.only):
            return False
        return not any(fnmatch.fnmatch(key, p) for p in check.skip)

    keys = sorted(set(want) | set(got))
    problems = []
    for key in keys:
        if not selected(key):
            continue
        if key not in got:
            problems.append("%s is gone (the reference has %r)" % (key, want[key]))
        elif key not in want:
            problems.append("%s appeared (%r), which the reference does not have" % (key, got[key]))
        elif not _close(want[key], got[key], check.rtol, check.atol):
            problems.append("%s: %s" % (key, _deviation(want[key], got[key])))
    if problems:
        out.problems.append("%s: %d difference(s)\n        %s"
                            % (check.label(), len(problems), "\n        ".join(problems[:12])
                               + ("\n        ... and %d more" % (len(problems) - 12) if len(problems) > 12 else "")))


def _stdout_matches(check: Stdout, art: RunArtifacts) -> "list[list[float]]":
    text = art.stdout.decode("utf-8", errors="replace")
    found = []
    for m in re.finditer(check.pattern, text):
        groups = m.groups() if m.groups() else (m.group(0),)
        found.append([float(g) for g in groups])
    return found


def _gen_stdout(check: Stdout, art: RunArtifacts) -> dict:
    found = _stdout_matches(check, art)
    if not found:
        raise ValueError("/%s/ matched nothing in the output" % check.pattern)
    return {"kind": "stdout", "pattern": check.pattern, "matches": found}


def _cmp_stdout(check: Stdout, art: RunArtifacts, ref: dict, out: Outcome) -> None:
    found = _stdout_matches(check, art)
    if len(found) != len(ref["matches"]):
        out.problems.append("%s: matched %d time(s), the reference %d - a reworded print, or a "
                            "run that did something else"
                            % (check.label(), len(found), len(ref["matches"])))
        return
    for i, (want, got) in enumerate(zip(ref["matches"], found)):
        for j, (w, g) in enumerate(zip(want, got)):
            if not _close(w, g, check.rtol, check.atol):
                out.problems.append("%s: match %d, number %d: %s"
                                    % (check.label(), i + 1, j + 1, _deviation(w, g)))


_GENERATORS = {"evolution": _gen_evolution, "finalstate": _gen_finalstate,
               "fingerprint": _gen_fingerprint, "stdout": _gen_stdout}
_COMPARERS = {"evolution": _cmp_evolution, "finalstate": _cmp_finalstate,
              "fingerprint": _cmp_fingerprint, "stdout": _cmp_stdout}


# ----------------------------------------------------------------------------------------------
# Reference files
# ----------------------------------------------------------------------------------------------

def data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


def reference_path(key: str) -> Path:
    """Where one script's reference numbers live. *key* is "Folder/script.py"."""
    folder, script = key.split("/", 1)
    return data_root() / folder / Path(script).stem / "reference.json"


def load_reference(key: str) -> "dict | None":
    path = reference_path(key)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _dump_reference(path: Path, payload: dict) -> None:
    """Write a reference file that can be read in a review.

    json.dump's own indentation puts every number of a time series on a line of its own, which
    turns a 64-row table into 400 lines that no reviewer reads. Each innermost list of numbers -
    one row of a table, one eigenvalue, one list of row indices - is therefore rendered on a single
    line, by substituting it in after the fact under a token that cannot occur in the data.
    """
    rendered: "dict[str,str]" = {}

    def mark(obj):
        if isinstance(obj, list) and obj and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            token = "@@row%d@@" % len(rendered)
            rendered[token] = "[" + ", ".join(json.dumps(v) for v in obj) + "]"
            return token
        if isinstance(obj, list):
            return [mark(v) for v in obj]
        if isinstance(obj, dict):
            return {k: mark(v) for k, v in obj.items()}
        return obj

    text = json.dumps(mark(payload), indent=1, sort_keys=True)
    for token, line in rendered.items():
        text = text.replace('"' + token + '"', line)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


# ----------------------------------------------------------------------------------------------
# What the harness calls
# ----------------------------------------------------------------------------------------------

def checks_for(key: str) -> "list[Check] | None":
    """The checks specified for one script, or None when it has no spec entry yet."""
    from . import specs
    return specs.VALIDATION.get(key)


def validate(key: str, art: RunArtifacts, pass_labels: "tuple[str,...]" = ()) -> Outcome:
    """Compare one finished run against the committed reference numbers."""
    out = Outcome()
    checks = checks_for(key)
    if not checks:
        return out
    reference = load_reference(key)
    if reference is None:
        return out
    for check in checks:
        if any(label in check.skip_under for label in pass_labels):
            out.skipped.append(check.label())
            continue
        ref = reference.get(check.slug())
        if ref is None:
            out.problems.append("%s: no reference numbers stored - regenerate with "
                                "--update-validation" % check.label())
            continue
        try:
            _COMPARERS[check.kind](check, art, ref, out)
        except Exception as e:
            out.problems.append("%s: could not be compared (%s: %s)"
                                % (check.label(), type(e).__name__, e))
        out.ran.append(check.label())
    return out


def update(key: str, art: RunArtifacts) -> "tuple[Outcome, Path | None]":
    """Regenerate one script's reference file from a finished run."""
    out = Outcome()
    checks = checks_for(key)
    if not checks:
        return out, None
    payload = {"_about": "Reference numbers for " + key + ", regenerated with "
                         "citools/test_all_tutorial_scripts.py --update-validation. "
                         "What is checked, and why each tolerance is what it is, lives in "
                         "citools/tutorial_validation/specs.py."}
    for check in checks:
        try:
            payload[check.slug()] = _GENERATORS[check.kind](check, art)
        except Exception as e:
            out.problems.append("%s: nothing recorded (%s: %s)" % (check.label(), type(e).__name__, e))
            continue
        out.ran.append(check.label())
    if not out.ran:
        return out, None
    path = reference_path(key)
    _dump_reference(path, payload)
    return out, path


def propose(key: str, art: RunArtifacts) -> str:
    """A ready-to-paste specs.py entry for a script that has none yet.

    This is what makes adding the remaining chapters cheap: it says which data files the run
    actually left behind and what the fingerprint has to offer, so the only judgement left is which
    of them is the meaningful quantity and how tight the tolerance can be.
    """
    lines = ['  "%s": [' % key]
    for rel in art.text_files():
        columns, data = [], None
        try:
            columns, data = _load_text(art.rundir / rel)
        except Exception as e:
            lines.append("      # %s could not be read (%s)" % (rel.as_posix(), e))
            continue
        name = rel.name if len([f for f in art.text_files() if f.name == rel.name]) == 1 else rel.as_posix()
        if not len(data):
            lines.append("      # %s: a header and no rows in this run, so there is nothing in it "
                         "to compare" % rel.as_posix())
            continue
        lines.append("      # %d row(s): %s" % (len(data), ", ".join(columns)))
        numbered = re.search(r"_\d+\.txt$", rel.name)
        if numbered:
            lines.append('      FinalState("%s"),' % re.sub(r"_\d+\.txt$", "_*.txt", name))
            continue
        import numpy
        if not bool(numpy.all(numpy.diff(data[:, 0]) >= 0.0)):
            lines.append('      Evolution("%s", match="rows", reason="the first column does not '
                         'grow"),' % name)
            continue
        lines.append('      Evolution("%s"),' % name)
        # A row count that is not one more than a round number of outputs is the signature of
        # output_every_step=True on an ADAPTIVE time stepper, whose instants differ from one run to
        # the next. Say so here rather than let the recorded reference fail on the next machine.
        if len(data) > 1 and (len(data) - 1) % 10 and (len(data) - 1) % 100:
            lines.append("      #   ^ %d rows is not a round number of outputs. If this script's "
                         "output times are adaptive, this needs match=\"interp\" and an rtol "
                         "that admits the interpolation error" % len(data))
    records = art.fingerprints()
    for i, record in enumerate(records):
        flat = _flatten_fingerprint(record)
        params = [k for k in flat if k.startswith("params.")]
        lines.append("      # fingerprint %d: ndof=%s, %d dof type(s), %d parameter(s)%s, "
                     "%d eigenvalue(s)"
                     % (i, record.get("ndof"), len(record.get("dofs") or {}),
                        len(record.get("params") or {}),
                        (" (" + ", ".join(sorted(p[len("params."):] for p in params)) + ")") if params else "",
                        len(record.get("eigenvalues") or [])))
        lines.append("      Fingerprint(%s)," % ("" if i == 0 else "index=%d" % i))
    if not records:
        lines.append("      # no fingerprint: the script built no Problem, or it was not run with "
                     "$PYOOMPH_VALIDATION_DUMP")
    lines.append("  ],")
    return "\n".join(lines)
