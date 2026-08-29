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

"""Turn the --report-json of citools/test_all_tutorial_scripts.py into a GitHub job summary.

The tutorial harness prints a report meant for a human reading a terminal, and always exits 0 -
the nightly greps its output. On GitHub neither is any use: the run's log is 127 scripts long and
nobody scrolls it, and a workflow needs an exit code to go red. So the harness also writes a JSON
record per script, and this turns that into the markdown table that appears on the run's summary
page (and returns 1 if anything failed).

    python3 citools/tutorial_report_to_summary.py report.json --title linux-x86_64 \
        >> "$GITHUB_STEP_SUMMARY"

Written to stdout, so it works just as well outside CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MARK = {"passed": ":white_check_mark:", "failed": ":x:", "skipped": ":fast_forward:"}


def seconds(value):
    return "-" if value is None else "%.1f" % value


def row(rec):
    name = rec["folder"] + "/" + rec["script"]
    note = rec.get("note") or ""
    return "| %s %s | `%s` | %s | %s | %s |" % (
        _MARK.get(rec["status"], ""), rec["status"], name,
        seconds(rec.get("wall_seconds")), seconds(rec.get("sim_seconds")), note)


HEADER = ["| Result | Script | Wall (s) | Simulation (s) | Note |",
          "| --- | --- | ---: | ---: | --- |"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="the --report-json file the harness wrote")
    parser.add_argument("--title", default="Tutorial scripts", help="heading for this platform's section")
    parser.add_argument("--out", default=None, help="write here instead of stdout (appends)")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    scripts = report["scripts"]
    failed = [r for r in scripts if r["status"] == "failed"]
    skipped = [r for r in scripts if r["status"] == "skipped"]
    passed = [r for r in scripts if r["status"] == "passed"]

    opts = report.get("options", {})
    flags = [name for name, on in (("--quick-test", opts.get("quick_test")),
                                   ("--tcc", opts.get("tcc")),
                                   ("--no-petsc", opts.get("no_petsc")),
                                   ("--distribute", opts.get("distribute"))) if on]
    if opts.get("mpirun"):
        flags.append("--mpirun %d" % opts["mpirun"])
    if opts.get("omp"):
        flags.append("--omp %d" % opts["omp"])

    out = []
    out.append("## %s - %d passed, %d failed, %d skipped" % (args.title, len(passed), len(failed), len(skipped)))
    out.append("")
    out.append("Python %s on %s%s" % (report.get("python", "?"), report.get("platform", "?"),
                                      (", run with `%s`" % " ".join(flags)) if flags else ""))
    total_wall = sum(r.get("wall_seconds") or 0.0 for r in scripts)
    total_sim = sum(r.get("sim_seconds") or 0.0 for r in scripts)
    out.append("")
    out.append("Total %.0f s wall, of which %.0f s inside the simulations "
               "(the rest is interpreter start-up and imports, once per script)." % (total_wall, total_sim))

    for problem in report.get("bundle_problems", []):
        out.append("")
        out.append(":warning: bundle inconsistency: %s" % problem)

    # Failures first and uncollapsed: they are the reason anybody opens this page.
    if failed:
        out += ["", "### Failed", ""] + HEADER + [row(r) for r in failed]
    if skipped:
        out += ["", "### Skipped", ""] + HEADER + [row(r) for r in skipped]

    # Everything else, slowest first, folded away - 100+ green rows would bury the two above.
    if passed:
        by_time = sorted(passed, key=lambda r: -(r.get("sim_seconds") or r.get("wall_seconds") or 0.0))
        out += ["", "<details><summary>Passed (%d), slowest first</summary>" % len(passed), ""]
        out += HEADER + [row(r) for r in by_time]
        out += ["", "</details>"]

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    return 1 if (failed or not report.get("all_okay", False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
