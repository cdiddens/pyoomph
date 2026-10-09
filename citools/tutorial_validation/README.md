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

### What a tolerance is measured against

For the per-field statistics - a `Fingerprint`'s dof groups and a reduced `FinalState`'s columns -
`min`, `max` and `mean` are compared against the **field's own scale**, `max(|min|, |max|)` over
that group or column, rather than against each statistic's own magnitude:

    tolerance = atol + rtol * max(|reference value|, scale)

Without that, any field passing through zero becomes hypersensitive: a pressure is referenced to an
arbitrary level and a symmetric component has `min = -max`, so whichever statistic straddles zero
gets a tolerance near zero with it. Two measured cases:
`stokes_flow_around_object.py`'s `liquid_sphere/pressure` has a minimum at 0.03 % of the field's
range which moved by six parts in a billion of the scale over four MPI ranks, and
`beads_on_string.py`'s `normal_y` has a mean of 7e-8 against an extent of order 1 which moved by
115 % *of itself* between two runs.

`l2` is deliberately left out of that rule in both places: its magnitude **is** the field's scale,
so a relative comparison on it is already the right question, and it is the statistic that notices a
field changing as a whole. Counts are exact.

### Reducing a time series

`FinalState` over a line-per-output file compares its columns' statistics instead of the nodes of a
spatial field, which is the right check for an observable written on an **adaptive** time grid: the
instants are a property of the machine, so matching rows would fail for no reason, and `match="interp"`
would only approximate them at a tolerance wide enough to hide a real drift.

Pass `stats=("min", "max")` when you do. An adaptive run writes a machine-dependent *number* of
rows, and `mean`, `l2` and `n` all go with it - measured on `beads_on_string.py`, where min and max
of `r_min` and `z_min` were bit-identical across two runs while the l2 of `z_min` moved by 3.3 %.
The extremes are the physics anyway: the deepest pinch, the largest fragment count, the bounds on a
conserved volume.

### Scripts that are not reproducible run to run

Most are, to the last bit - two independent generations of `Temporal_ODEs` and of `Spatial_PDEs`
came out byte-identical. Some are not, and no tolerance fixes that; find out which before writing
the entry.

**Two runs do not bound a spread.** Where a quantity varies at all, its run-to-run difference is a
draw from a distribution, and two draws can land arbitrarily close. This has now been wrong twice
here, in both directions:

* `kuramoto_sivanshinsky.py`: the first pair agreed to 4.4e-05 on `h`'s l2, which looked like room
  to pin it at `rtol=1e-3`. Eight runs put the spread at **23 %** - four orders of magnitude wider.
* `beads_on_string.py`: two runs agreed on `max(z_min)`, so it went into the entry. It is bimodal
  (21.0198 or 18.8496) and failed 3 of 5 repeats.

So two runs are enough to prove a script *is* irreproducible, and never enough to prove it is
reproducible. If a check is going to rest on a measured spread, measure it five to eight times -
the scripts here cost seconds to tens of seconds each, which is cheaper than a nightly that fails
intermittently a month later.

**An unseeded random initial condition is its own category.** `DeterministicRandomField` is
deterministic only *within* a run: without `seed=`, the cloud is redrawn per run, so the script
genuinely starts somewhere else each time and no amount of tolerance makes its state comparable.
Only structural quantities (`ndof`, the per-group node counts) can be pinned. In the tutorial set,
`SpatioTemporal_PDEs/kuramoto_sivanshinsky.py` and `Plotting_Interface/kuramoto_sivanshinsky.py`
are unseeded; `Multicomponent_Flow/marangoni_instability.py` passes a seed and is fully
reproducible. Check for this first - it looks exactly like chaos in a diff, and the entry it calls
for is the same, but the reason belongs in the comment.

`rayleigh_plateau.py` is the clearest case. Four runs on one machine agree bit-for-bit for 148 rows
of `minimum.txt`, differ by **one ULP** (2.44e-16) at t=8.6259, exceed 1e-6 one row later and 1e-3
by t=8.76, and finish with a deepest neck radius spread over 0.000379..0.000400 - 5 %. Two of them
remeshed differently, 299 dofs of `mesh_y` against 433. Pinch-off is a finite-time singularity, so
it amplifies round-off the way a chaotic trajectory does. The entry therefore checks the trajectory
*before* the singularity, exactly, and records nothing after it.

`droplet_spread_marangoni_and_gravity.py` is a different shape: its parameters are bit-identical and
everything geometric agrees to 1e-10 or better, but every pressure group's min, max and mean shifted
by an identical -0.414154 between runs, and the volume Lagrange multiplier by -0.414155. The
pressure level and that multiplier are one free direction, so the entry skips both.

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

Two causes of rank-dependence showed up, and they want different entries:

* **The mesh itself differs.** `moffatt_eddies.py` stops its adaptation at 73740 dofs over four
  ranks against 73705 serially, and `heated_cylinder.py` at 74167 against 74146; every field
  statistic then follows the mesh. A different mesh is a different discretisation, so these are
  excused rather than loosened. They are the exception, not the rule - `convdiffu_simple`,
  `marangoni_instability`, `navier_stokes`, `lubrication_coalescence`, `laplace_smoothed_mesh` and
  `cantilever` adapt too and came through the same pass untouched, which is why the dof counts are
  still compared everywhere else.
* **The instants differ.** `rayleigh_plateau.py`'s adaptive stepper lands ~6e-7 away from the
  stored instants over four ranks, so `match="exact"`'s lookup misses. The fix is *not*
  `match="interp"`: that would weaken the serial comparison, which is the strong one, to buy a
  weaker MPI one.

A reduced series often survives where a final state does not, and that is the argument for
preferring one. `rayleigh_plateau_pinchoff.py` is the clean demonstration: 76 fingerprint entries
move over four ranks (up to 0.44 % on `mesh_x`'s l2, because the state follows a topological
surgery), while the reduced `max(fragments)` and the bounds on volume come through untouched.

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
