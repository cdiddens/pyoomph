# Validating the tutorial scripts

`citools/test_all_tutorial_scripts.py` runs all 141 tutorial scripts. On its own it only asks
whether they exit 0, which a script computing the wrong answer does just as happily as a correct
one. This directory holds the numbers it compares them against, and the judgement of which number
is worth comparing for which script.

The tutorial scripts themselves contain nothing of this. They are documentation: an assertion in
the middle of one is noise to a reader, and the validation would have to be maintained in the
chapter sources. Everything happens in the harness after the script has finished.

```
specs.py                               what is checked per script, and why
data/<Folder>/<stem>/reference.json    the numbers
__init__.py                            the check types and the comparison
```

## Running it

Validation is **on by default**, so an ordinary pass already does it:

```sh
python3 -u citools/test_all_tutorial_scripts.py --only Temporal_ODEs
```

It is off, and says so in its first lines, under `--quick-test` (the script leaves from inside its
first Newton solve, so there is no result), under `--extra-arg` (forcing a solver changes what is
computed, not only how fast), and under `--no-validation`.

A mismatch fails that script exactly the way a crash does - the same `================= FAILED`
line, the same per-script `.log` with the numeric diff appended - so `nightly_develop.sh` and the
GitHub job summary need no changes to report it. A script with **no** reference data is not a
failure: it is listed under `NOT VALIDATED` at the end of the pass.

## Adding a script

```sh
python3 -u citools/test_all_tutorial_scripts.py --only <script> --propose-validation
```

prints a ready-to-paste `specs.py` entry: the data files the run left behind with their columns and
row counts, and what its state fingerprint has to offer. Keep the quantity that means something for
that script, drop the rest, then generate the numbers:

```sh
python3 -u citools/test_all_tutorial_scripts.py --only <script> --update-validation
```

and read the diff before committing it.

### Choosing the quantity

* **ODE scripts** write their whole evolution to one text file, so compare it: `Evolution(...)`
  stores up to 64 rows spread over the run and compares every one of them.
  Its `match=` decides how a stored row is found again, and the default is the one to want:
  * `match="exact"` looks the stored instant up in the produced file. Right for everything that
    writes at prescribed times, i.e. every `run(endtime=..., numouts=...)`: those instants belong
    to the script, not to the machine. A time stepper that lands elsewhere is then *reported*.
  * `match="rows"` matches by row index and insists on the same row count. For a file whose first
    column does not grow - an arclength continuation writing the parameter there, a transient
    restarted from several initial conditions. Most of the bifurcation chapter needs this.
  * `match="interp"` interpolates onto the stored instants, for a script whose output times are
    themselves adaptive. It costs accuracy: linear interpolation over a step `h` is wrong by about
    `h^2*y''/8`, which is percent-level on a Lorenz trajectory, so such a check needs an `rtol`
    that admits that and will not catch a small drift. Use it only where `"exact"` cannot work.

  `from_time=` and `until_time=` cut the comparison window: `until_time` for a chaotic script,
  where only the first stretch is reproducible, and `from_time` for a quantity that starts from a
  transient nobody should pin down (a Lyapunov exponent is meaningless until it has converged, and
  grows exponentially before that).
* **PDE scripts** cannot be compared node by node across platforms and mesh adaptations.
  `FinalState(...)` reduces each column of the last output to `n/min/max/mean/l2`, which is
  invariant under the node ordering. An `IntegralObservableOutput` time series, where the script has
  one, is better still: it is `Evolution(...)` over quantities that mean something physically.
* **Bifurcation, fold, Hopf, pitchfork and arclength scripts** mostly write nothing at all, and
  their answer is a global parameter value. `Fingerprint()` records every global parameter, the
  sorted eigenvalues and the per-dof-type reductions, so for these it is the whole check.
* `Stdout(...)` is the last resort, for a number that is neither in a file nor a parameter. It
  breaks when somebody rewords a `print`, so prefer anything else.

### Tolerances

The default is `rtol=1e-5`, `atol=1e-10`. That is deliberately not as tight as one machine can
hold: the same committed numbers have to pass on linux-x86_64, macOS-arm64 and Windows, against
three BLAS implementations and two PETSc builds.

Two independent generations of the whole `Temporal_ODEs` chapter on one machine came out
**bit-identical**, every number of all 37 reference files. That is what `match="exact"` rests on,
and it is also the reason the default tolerance can be as tight as it is: what the tolerance has to
absorb is the difference between *platforms*, not between two runs.

Where a check needs more room, say why in `reason=`. A relaxed tolerance with no reason beside it
is the one nobody can judge a year later, and the chaotic scripts are the clear case:
`adaptive_lorenz_attractor.py` cannot be compared beyond the first few time units at all, because
two runs of it separate exponentially - what stays reproducible is the extent of the attractor.

`skip_under=("mpi",)` excuses a check in the `--mpirun` pass (also `"distribute"`, `"omp"`,
`"tcc"`). Needing it is a result, not an annoyance: it says that this script's answer depends on the
number of ranks, which is worth knowing. A script whose every check is excused is reported as
`skipped here` rather than as validated.

### Files without a header

Not every writer emits one - `Problem.create_text_file_output()` takes `header=None`, and
`utils/lyapunov.py` writes its exponents that way. Those files are read all the same; their columns
are then named `column_0`, `column_1`, ... by position, which is what a spec selecting them has to
say. Several of the bifurcation scripts write their `(r, x, eigenvalue)` branch files by hand like
this, and those are the strongest checks in that chapter.

## When a number legitimately changes

A changed number is either a bug or a deliberate change in what the tutorial computes. Regenerate
with `--update-validation` and **review the diff** - that review is the only thing standing between
a silent regression and the repository. Widening a tolerance until the comparison passes is not an
answer to either case.

## Testing the validation itself

`tests/test_tutorial_validation.py` exercises the comparison against hand-written output
directories - no pyoomph, no solver, a fraction of a second - and it is the thing to run after
touching `__init__.py`:

```sh
python3 -m pytest tests/test_tutorial_validation.py -q
```

It is there because a check that has quietly stopped comparing anything looks exactly like a check
that passed, and the tutorial pass itself is far too slow to notice. What it pins down: a run
compared against itself passes and a one-percent drift fails *every* kind of check; a changed time
grid is reported rather than absorbed; a run that stopped early cannot pass by `numpy.interp`
clamping; the previous script's leftover output directories are not credited to this script; and
missing reference data is not a failure.

## The fingerprint

`Problem._write_validation_dump()` in `pyoomph/generic/problem.py` writes it, only when
`$PYOOMPH_VALIDATION_DUMP` is set, as one JSON record per problem appended to
`_pyoomph_validation.jsonl` in that problem's own output directory:

```json
{"ndof": 199, "time": 0.0, "params": {"r": 0.0}, "eigenvalues": [[0.0, 0.0]],
 "dofs": {"domain/u": {"n": 199, "min": 0.00995, "max": 0.5, "mean": 0.335, "l2": 5.16398}},
 "mpi_size": 1, "distributed": false}
```

Reductions, never the dof vector: `min/max/mean/l2` per dof *type* do not change when the dofs are
reordered, which is what lets one reference serve a serial run, an `mpirun -n 4` run and a run whose
mesh adaptation renumbered everything. `mpi_size` and `distributed` are recorded so a reader can
tell which mode produced a record; they are never compared.

## Why this module does not import pyoomph

It runs inside the harness process, and that process has to stay free of MPI. Importing pyoomph
calls `MPI_Init`, which sets `PMIX_NAMESPACE` and some twenty other variables through C `setenv()`;
Python's `os.environ` never sees them, but `exec()` hands them to every tested script, where the
`mpirun` of `--mpirun` concludes it is already inside an MPI job and exits 1 without printing
anything - i.e. every single script "failing" with an empty log. That is why `_load_text()` here
repeats what `pyoomph.utils.num_text_out.LoadedTextDataFile` already does, and it is the same reason
the harness keeps petsc4py out of its own process.
