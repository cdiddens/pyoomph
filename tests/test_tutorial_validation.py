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

"""The comparison logic of citools/tutorial_validation, against synthetic runs.

The tutorial validation is only worth having if a wrong answer actually fails. That is hard to see
from the tutorial pass itself - it takes over an hour, and a check that silently stopped comparing
anything looks exactly like a check that passed. So the logic is exercised here instead, on
hand-written output directories: no pyoomph, no solver, a fraction of a second.

What each test pins down is the thing that would otherwise rot unnoticed:

  * a run compared against itself passes, and a run that is one percent off does not;
  * a changed time grid is REPORTED rather than absorbed, which is what the default
    match="exact" is for;
  * match="interp" lives with an adaptive grid but still catches a drift;
  * a run that stopped early cannot pass by numpy.interp clamping its last value;
  * missing reference data is not a failure, because rolling the reference data out chapter by
    chapter has to be possible.

This module imports citools/tutorial_validation, which deliberately does not import pyoomph (see
the note on _load_text there), so it is safe to collect in the same session as everything else.
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import pytest

_CITOOLS = Path(__file__).resolve().parent.parent / "citools"
sys.path.insert(0, str(_CITOOLS))

import tutorial_validation as tv  # noqa: E402
from tutorial_validation import Evolution, FinalState, Fingerprint, Stdout  # noqa: E402

KEY = "Temporal_ODEs/synthetic.py"
DEFAULT_CHECKS = [Evolution("oscillator.txt"), FinalState("domain_*.txt"),
                  Fingerprint(), Stdout(r"omega=([0-9.eE+-]+)")]
STDOUT = b"the result: omega=1\n"


def _fixed_times():
    return [i * 0.1 for i in range(101)]


def _adaptive_times():
    """A grid no second run would reproduce, and coarse enough that interpolation is visibly wrong."""
    times = [0.0]
    while times[-1] < 10.0:
        times.append(round(times[-1] + 0.037, 6))
    return times


def _make_run(root, name, *, scale=1.0, times=None, duplicate_last=0):
    """A directory shaped like one a tutorial script leaves behind.

    The keyfile is what marks it as a pyoomph output directory (Problem.initialise writes it), the
    two text files are shaped like ODEFileOutput's and TextFileOutput's, and the .jsonl is shaped
    like Problem._write_validation_dump's.
    """
    out = Path(root) / name / "synthetic"
    (out / "domain").mkdir(parents=True, exist_ok=True)
    (out / "_pyoomph_run_.txt").write_text("")
    ts = _fixed_times() if times is None else times
    with open(out / "oscillator.txt", "w") as f:
        f.write("#time\ty\tydot\n")
        for t in ts:
            f.write("%r\t%r\t%r\n" % (t, scale * math.cos(t), -scale * math.sin(t)))
        for _ in range(duplicate_last):
            # output_every_step=True writes the same instant twice: once after the step, once as
            # the output. The comparison has to cope with that, not trip over it.
            f.write("%r\t%r\t%r\n" % (ts[-1], scale * math.cos(ts[-1]), -scale * math.sin(ts[-1])))
    for step in (0, 1):
        with open(out / "domain" / ("domain_%06d.txt" % step), "w") as f:
            f.write("# coordinate_x\tu\t@time=%g\n" % step)
            for i in range(11):
                f.write("%.18e\t%.18e\n" % (i / 10.0, scale * step * (i / 10.0) ** 2))
            f.write("nan\tnan\n")  # the separator the 1d writers put between line segments
    with open(out / "_pyoomph_validation.jsonl", "w") as f:
        f.write(json.dumps({"ndof": 3, "time": ts[-1], "params": {"omega": 1.0 * scale},
                            "eigenvalues": [[-0.5, 1.0], [-0.5, -1.0]],
                            "dofs": {"osc/y": {"n": 2, "min": -scale, "max": scale,
                                               "mean": 0.0, "l2": 1.41 * scale}},
                            "mpi_size": 1, "distributed": False}) + "\n")
    return out.parent


@pytest.fixture
def validation(tmp_path, monkeypatch):
    """The module with its reference data redirected into tmp_path, and a settable spec."""
    data = tmp_path / "data"
    monkeypatch.setattr(tv, "data_root", lambda: data)
    state = {"checks": DEFAULT_CHECKS}
    monkeypatch.setattr(tv, "checks_for", lambda key: state["checks"] if key == KEY else None)

    class Harness:
        root = tmp_path

        def set_checks(self, checks):
            state["checks"] = checks

        def run(self, name, **kw):
            return _make_run(tmp_path, name, **kw)

        def artifacts(self, rundir, stdout=STDOUT):
            # started_at in the past: everything the fake run "wrote" has to look fresh.
            return tv.RunArtifacts(rundir, time.time() - 5.0, stdout)

        def record(self, rundir, stdout=STDOUT):
            shutil.rmtree(data, ignore_errors=True)
            outcome, path = tv.update(KEY, self.artifacts(rundir, stdout))
            assert outcome.problems == [], outcome.problems
            return outcome, path

        def check(self, rundir, stdout=STDOUT, labels=()):
            return tv.validate(KEY, self.artifacts(rundir, stdout), labels)

    return Harness()


def test_a_run_validates_against_itself(validation):
    run = validation.run("a")
    outcome, path = validation.record(run)
    assert len(outcome.ran) == len(DEFAULT_CHECKS)
    assert path.is_file()
    result = validation.check(run)
    assert (result.status, result.problems) == ("ok", [])


def test_reference_file_stays_reviewable(validation):
    """One table row per line: a reference nobody can read in a diff is not a reference."""
    _, path = validation.record(validation.run("a"))
    text = path.read_text()
    assert len(text.splitlines()) < 200, "a 64-row table should not be hundreds of lines of diff"
    assert json.loads(text)["_about"].startswith("Reference numbers for ")


@pytest.mark.parametrize("scale", [1.01, 0.999])
def test_a_drifted_run_fails_every_check(validation, scale):
    validation.record(validation.run("ref"))
    drifted = validation.run("drifted", scale=scale)
    result = validation.check(drifted, stdout=("the result: omega=%r\n" % scale).encode())
    assert result.status == "mismatch"
    # Each of the four kinds has to notice on its own: a check that stopped comparing anything is
    # the failure mode this test exists for.
    assert len({p.split(":")[0] for p in result.problems}) == len(DEFAULT_CHECKS)


def test_duplicated_output_instants_are_not_a_mismatch(validation):
    validation.record(validation.run("ref"))
    assert validation.check(validation.run("dup", duplicate_last=3)).status == "ok"


def test_a_changed_time_grid_is_reported(validation):
    """The default must not paper an adaptive grid over by interpolating onto it."""
    validation.record(validation.run("ref"))
    result = validation.check(validation.run("adaptive", times=_adaptive_times()))
    assert result.status == "mismatch"
    assert any("do not occur in this run" in p for p in result.problems)


def test_interp_mode_lives_with_an_adaptive_grid_but_still_catches_a_drift(validation):
    # rtol has to admit the interpolation error itself: h^2*y''/8 is ~1.7e-4 for this grid.
    validation.set_checks([Evolution("oscillator.txt", match="interp", rtol=1e-3,
                                     reason="the output times of this script are adaptive")])
    validation.record(validation.run("ref"))
    assert validation.check(validation.run("adaptive", times=_adaptive_times())).status == "ok"
    assert validation.check(validation.run("drifted", scale=1.01,
                                           times=_adaptive_times())).status == "mismatch"


def test_a_run_that_stopped_early_cannot_pass_by_clamping(validation):
    """numpy.interp clamps, so the last value of a short run would match the whole tail."""
    validation.set_checks([Evolution("oscillator.txt", match="interp", rtol=1e-3, reason="test")])
    validation.record(validation.run("ref"))
    result = validation.check(validation.run("short", times=_fixed_times()[:51]))
    assert result.status == "mismatch"
    assert any("the reference needs" in p for p in result.problems)


def test_a_non_monotonic_abscissa_needs_row_matching(validation):
    """An arclength continuation writes the parameter in the first column, and it turns around."""
    run = validation.run("sweep", times=_fixed_times()[:40] + _fixed_times()[:40][::-1])
    validation.set_checks([Evolution("oscillator.txt")])
    outcome, _ = tv.update(KEY, validation.artifacts(run))
    assert outcome.problems and "does not grow monotonically" in outcome.problems[0]
    validation.set_checks([Evolution("oscillator.txt", match="rows", reason="a parameter sweep")])
    validation.record(run)
    assert validation.check(run).status == "ok"
    assert validation.check(validation.run("sweep2", scale=1.01,
                                           times=_fixed_times()[:40] + _fixed_times()[:40][::-1])
                            ).status == "mismatch"


def test_fingerprint_selection(validation):
    validation.record(validation.run("ref"))
    drifted = validation.run("drifted", scale=1.01)
    # only= narrows the fingerprint to the quantity that is the answer - the global parameter, for
    # every fold and Hopf script.
    validation.set_checks([Fingerprint(only=["params.*"])])
    validation.record(validation.run("ref"))
    assert validation.check(drifted).status == "mismatch"
    # ...and skip= is how an adaptive script excuses a dof count that is not reproducible.
    validation.set_checks([Fingerprint(skip=["params.*", "dofs.*", "ndof"])])
    validation.record(validation.run("ref"))
    assert validation.check(drifted).status == "ok"


def test_mpi_size_is_recorded_but_never_compared(validation):
    """The MPI pass runs the same scripts over four ranks on purpose."""
    validation.record(validation.run("ref"))
    run = validation.run("on_four_ranks")
    path = run / "synthetic" / "_pyoomph_validation.jsonl"
    record = json.loads(path.read_text())
    record["mpi_size"], record["distributed"] = 4, True
    path.write_text(json.dumps(record) + "\n")
    assert validation.check(run).status == "ok"


def test_skip_under_excuses_a_check(validation):
    validation.set_checks([Evolution("oscillator.txt", skip_under=("mpi",)), Fingerprint()])
    validation.record(validation.run("ref"))
    result = validation.check(validation.run("ref"), labels=("mpi",))
    assert (len(result.skipped), len(result.ran)) == (1, 1)


def test_missing_reference_data_is_not_a_failure(validation):
    """Rolling the reference data out chapter by chapter has to be possible."""
    run = validation.run("a")
    result = validation.check(run)
    assert (result.status, result.problems) == ("no-reference", [])
    validation.set_checks(None)
    assert tv.validate("Spatial_PDEs/unknown.py", validation.artifacts(run)).status == "no-reference"


def test_a_spec_asking_for_a_file_that_is_not_there_is_reported(validation):
    validation.set_checks([Evolution("not_written_by_anybody.txt")])
    outcome, path = tv.update(KEY, validation.artifacts(validation.run("a")))
    assert path is None
    assert outcome.problems and "matched 0 files" in outcome.problems[0]


def test_stale_output_of_an_earlier_script_is_ignored(validation):
    """The harness runs every script of a folder in the same directory, and the output directories
    of earlier scripts are still lying around."""
    run = validation.run("a")
    import os
    stale = run / "synthetic"
    old = time.time() - 3600.0
    for path in list(stale.rglob("*")) + [stale]:
        os.utime(path, (old, old))
    art = validation.artifacts(run)
    assert art.outdirs == [] and art.text_files() == [] and art.fingerprints() == []


def test_propose_names_the_files_and_the_fingerprint(validation):
    text = tv.propose(KEY, validation.artifacts(validation.run("a")))
    assert '"%s": [' % KEY in text
    assert 'Evolution("oscillator.txt")' in text
    assert "FinalState(" in text and "domain_*.txt" in text
    assert "Fingerprint()" in text
    assert "1 parameter(s) (omega)" in text


def test_a_headerless_file_gets_positional_column_names(validation, tmp_path):
    """NumericalTextOutputFile takes header=None, and utils/lyapunov.py uses it that way."""
    run = validation.run("a")
    bare = run / "synthetic" / "bare.txt"
    bare.write_text("".join("%g\t%g\n" % (i * 0.5, i * i) for i in range(20)))
    validation.set_checks([Evolution("bare.txt")])
    validation.record(run)
    columns = json.loads((tv.data_root() / "Temporal_ODEs" / "synthetic"
                          / "reference.json").read_text())["evolution:bare.txt"]["columns"]
    assert columns == ["column_1"]
    assert validation.check(run).status == "ok"


def test_from_time_skips_a_transient_nobody_should_pin(validation):
    """A Lyapunov exponent is meaningless before it converges, and grows exponentially until then."""
    run = validation.run("ref")
    series = run / "synthetic" / "transient.txt"

    def write(path, tail):
        with open(path, "w") as f:
            f.write("#time\tvalue\n")
            for i in range(21):  # a wild, machine-dependent transient...
                f.write("%r\t%r\n" % (i * 0.5, 1e9 * (1.0 + i)))
            for i in range(21, 61):  # ...settling onto the number that means something
                f.write("%r\t%r\n" % (i * 0.5, tail))

    write(series, 0.9056)
    validation.set_checks([Evolution("transient.txt", from_time=15.0,
                                     reason="the exponent is still converging before that")])
    validation.record(run)
    other = validation.run("other")
    write(other / "synthetic" / "transient.txt", 0.9056)
    # The transient differs by a factor of two and must not matter; the converged value must.
    with open(other / "synthetic" / "transient.txt") as f:
        lines = f.readlines()
    lines[1:21] = ["%s\t%r\n" % (l.split("\t")[0], 2e9 * (1.0 + i)) for i, l in enumerate(lines[1:21])]
    (other / "synthetic" / "transient.txt").write_text("".join(lines))
    assert validation.check(other).status == "ok"
    drifted = validation.run("drifted")
    write(drifted / "synthetic" / "transient.txt", 0.9056 * 1.01)
    assert validation.check(drifted).status == "mismatch"


def test_a_fully_skipped_script_is_not_mistaken_for_an_uncovered_one(validation):
    """"no reference data yet" is a gap to fill; "every check is skip_under here" is a decision."""
    validation.set_checks([Evolution("oscillator.txt", skip_under=("mpi",)),
                           Fingerprint(skip_under=("mpi",))])
    validation.record(validation.run("ref"))
    assert validation.check(validation.run("ref"), labels=("mpi",)).status == "skipped"


def test_the_previous_scripts_output_is_not_credited_to_this_one(validation):
    """Several tutorials write into directories of their own choosing, and the harness deletes only
    the one named after the script - so the next script, starting a fraction of a second later,
    finds them. A second of mtime slack was enough to read them as its own."""
    import os
    leftover = validation.run("leftover")
    just_before = time.time() - 0.2
    for path in list(leftover.rglob("*")) + [leftover]:
        os.utime(path, (just_before, just_before))
    art = tv.RunArtifacts(leftover, time.time(), b"")
    assert art.outdirs == [], "output written before the script started is not its output"
    assert art.text_files() == [] and art.fingerprints() == []


def _write_fingerprint(run, dofs):
    """Overwrite the fake run's fingerprint with one carrying the given dof statistics."""
    path = run / "synthetic" / "_pyoomph_validation.jsonl"
    record = json.loads(path.read_text())
    record["dofs"] = dofs
    path.write_text(json.dumps(record) + "\n")


def test_a_near_zero_extremum_is_judged_against_the_fields_scale(validation):
    """min, max and mean carry the field's units, so comparing one against its OWN magnitude is
    hypersensitive wherever a field passes through zero - a pressure referenced to an arbitrary
    level, a symmetric velocity component. Measured on stokes_flow_around_object.py over four ranks:
    a pressure minimum of 9.6e-4 against a maximum of 3.0 moved by 1.9e-8 of solver round-off, which
    is 2e-5 of the value and six parts in a billion of the field."""
    validation.set_checks([Fingerprint()])
    ref = validation.run("ref")
    _write_fingerprint(ref, {"liquid/pressure": {"n": 64, "min": 0.000959557419398,
                                                 "max": 2.99808, "mean": 1.4, "l2": 14.8424}})
    validation.record(ref)

    noisy = validation.run("noisy")
    _write_fingerprint(noisy, {"liquid/pressure": {"n": 64, "min": 0.000959576099052,
                                                   "max": 2.99808, "mean": 1.4, "l2": 14.8424}})
    assert validation.check(noisy).status == "ok", "round-off on a near-zero minimum is not a drift"

    # ...but a minimum that moves by a real fraction of the field still fails: 1 % of the scale.
    moved = validation.run("moved")
    _write_fingerprint(moved, {"liquid/pressure": {"n": 64, "min": 0.000959557419398 + 0.03,
                                                   "max": 2.99808, "mean": 1.4, "l2": 14.8424}})
    assert validation.check(moved).status == "mismatch"


def test_l2_is_still_compared_relatively(validation):
    """The scale rule deliberately leaves l2 out: its magnitude IS the field's scale, and it is the
    statistic that notices the field changing as a whole."""
    validation.set_checks([Fingerprint()])
    ref = validation.run("ref")
    base = {"liquid/pressure": {"n": 64, "min": 0.0, "max": 3.0, "mean": 1.4, "l2": 14.8424}}
    _write_fingerprint(ref, base)
    validation.record(ref)
    drifted = validation.run("drifted")
    _write_fingerprint(drifted, {"liquid/pressure": dict(base["liquid/pressure"], l2=14.8424 * 1.001)})
    assert validation.check(drifted).status == "mismatch", "a 0.1 % change in l2 must be caught"
