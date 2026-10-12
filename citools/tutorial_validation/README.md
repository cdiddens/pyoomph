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
reproducible.

**And a verification pass that passes does not establish margin.** `--update-validation` compares a
run against itself, so generation can never reveal a spread; the verification pass that follows is
the first honest test, but it is still only one draw. The Kuramoto-Sivashinsky fold family is the
case that proves the point: its fingerprint passed the serial verification in one batch and the MPI
verification in another, and a later run deviated by 1.11e-05 against a default tolerance of 1e-05.
Measured over four runs, the spread is the continuation's own and has nothing to do with MPI - a
fresh serial run was the *worst* of the four, while three four-rank runs sat at 2.7e-06 to 6.0e-06
and differed among themselves by 9.9e-06. A tolerance sitting on a script's spread fails about half
the time, which in a nightly means a flake weeks later with no change to blame.

The practical rule: if a check's observed deviation is within a factor of a few of its tolerance,
that is not a pass, it is a warning. Measure the spread and set the tolerance from it. If a check is going to rest on a measured spread, measure it five to eight times -
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

### Scripts that are not reproducible across machines

A separate axis from the one above, and the one this set got wrong first. Several entries were
written with `skip_under=("mpi",)` and a comment explaining a genuine degeneracy - an orbit's
phase, the Lorenz symmetry, a continuation's stopping point - as though running over four ranks
were what caused it. It is not. **MPI is just the first thing that changes the arithmetic**, and
every other platform changes it too. The first cross-platform run of the tutorial set, on 12th
October 2026, failed six scripts whose degeneracy was already described in this file under that
heading.

So when a quantity turns out to be fixed only up to some freedom, say so unconditionally.

But do not then assume the MPI excuse was *wrong* - the two axes can be different sizes, and a
check may need both. `langford_floquet.py` is the worked example. Cross-platform and serial, `mu`
agrees and only the orbit's phase moves, so skipping `x` and `y` is enough and `z`, the Floquet
multipliers and `mu` stay pinned. Over four ranks it is `mu` itself that moves, by 4.34e-05, and
everything evaluated at that `mu` moves with it - `z` by 1.53e-04 and the multipliers by 8e-03.
Skipping the phase does not cover that, and a tolerance loose enough to admit it (1e-02) would
throw away the resolution the check has serially. The entry therefore carries the `skip=` *and*
`skip_under=("mpi",)`, with a comment for each.

The test for whether `skip_under` is still earning its place: run the script under `--mpirun 4`
after relaxing it for the cross-platform reason. If it still fails, the two reasons are different
and both belong in the entry.

What the freedoms look like in practice, and what to keep instead of the quantity itself:

* **A point on a periodic orbit.** The phase is free unless the script pins it. `langford_floquet.py`
  stopped at four different phases on four platforms while `sqrt(x^2+y^2)` agreed to nine decimals,
  so the entry skips `x` and `y` and keeps `z`, which the phase does not move. Look for a
  coordinate like that before dropping the whole check - and if the orbit is chaotic rather than a
  limit cycle, as in `manual_orbit.py`, there is nothing phase-invariant left and only the
  parameter survives.
* **An eigenvector, or a null vector.** Defined up to a scalar; a normalisation usually fixes the
  magnitude and not the sign. In `hanging_droplet.py` the fold tracker's null vector shares a dof
  group with the parameter dof, so even `gauge_free`'s `max - min` is not invariant and only `l2`
  and `n` are.
* **The order of a spectrum**, and **its length**. See `eigenvalues_by_value` and
  `eigenvalue_max_abs` below.
* **Where a continuation stops.** One step more or less, which is a relative difference in the
  parameter of the order of the last step size - 3.0e-4 for `hopf_switch.py`. Give the parameter a
  tolerance that admits one step and say which step size it came from.

### An adaptive mesh: `skip=ADAPTIVE_MESH`

A script whose mesh adapts refines on a **threshold crossing**, and round-off decides which side of
it the error estimator lands on. So the dof count is not reproducible once the arithmetic changes -
measured over four platforms, from +0.04 % on `stokes_flow_around_object.py` to -3.5 % on
`remeshing.py`, where the remesher triggers on geometric quality instead. A tolerance cannot fix a
count, and widening one until a count passes would wreck every other key in the same check.

`skip=ADAPTIVE_MESH` (defined at the top of `specs.py`) drops the four keys that describe the
*discretisation* rather than the solution:

| key | why it goes |
| --- | --- |
| `ndof`, `*.n` | they **are** the count |
| `*.l2` | a norm of the dof **vector**, so it scales with the count: `sqrt(2618/2576)` alone is 0.8 % of `rayleigh_plateau_pinchoff.py`'s 3.3 % |
| `*.mean` | an average over the dof **set**, i.e. weighted by dof density rather than by volume |

`min` and `max` stay. They are the field's extremes, which converge as the mesh refines, so they
remain a statement about the solution. But note they are sampled **at a node**: two meshes catch a
sharp peak differently, so they still need a tolerance above round-off - 7.3e-04 to 6.2e-03 on the
scripts here.

How much this is worth: dropping those four alone was enough for `moffatt_eddies.py` and
`viscoelastic_cylinder.py`, which then passed at the **default** `rtol=1e-5` on four platforms and
over four ranks. That is the test of whether this is a structural statement or a tolerance in
disguise. `moffatt_eddies.py` also shows why `mean` had to go: the script prescribes velocity on
every boundary and pins the pressure with `AverageConstraint(pressure=0)`, so the *volume* average
is zero by construction while the mean over dofs is a mesh-weighted number hovering near it - it
moved "by a factor of four" and was never a physical quantity at all.

This is the same argument the `FinalState` entries already made with `stats=("min","max")`. It had
simply never been applied to a `Fingerprint`.

**Confine a loose tolerance to the group that earns it.** Twice here one group wanted an order of
magnitude more than everything else: `rayleigh_plateau_pinchoff.py`'s `_kin_bc` (1.6e-02 against
4.7e-03 for the rest - it is the kinematic multiplier on the interface the topological surgery
cuts) and `falling_droplet_with_surfactants.py`'s `droplet_outside` velocities (2.9e-02 against
6.2e-03 - Marangoni stress concentrates the tangential flow into a thinner layer than in the
surfactant-free twin, which needs only 4.8e-03 on the same quantity). Adding those to `skip=` kept
the rest of each check an order of magnitude tighter than a single per-check `rtol` would have.

### Eigenvalues: `eigenvalues_by_value` and `eigenvalue_max_abs`

Neither the order of a spectrum nor its length is a property of the problem, so a `Fingerprint`
that compares `eigenvalues.N.re` index by index is pinning down the eigensolver as much as the
physics.

* `eigenvalues_by_value=True` moves each produced eigenvalue to the index of the reference
  eigenvalue it lies nearest to, then compares as usual. Only the order is forgiven; every value
  still has to meet the tolerance. Use it on anything with a conjugate pair -
  `bifurcation_hopf_tracking_lorenz.py` reported `+12.0557990943` on one platform and
  `-12.0557990943` on another, the same pair the other way round, which index matching called a
  200 % error.
* `eigenvalue_max_abs=X` drops `|lambda| > X` from both sides first. For a differential-algebraic
  system, whose constraints show up as "infinite" eigenvalues rendered as some huge finite number:
  they are a property of the conditioning, and **whether the solver returns them at all varies**.
  On `pendulum_lagrange_eigenvalues.py` SLEPc returns all five and the built-in fallback returns
  only the two finite ones, so the physical pair is not at a stable index in any sense. Choose X
  where the two groups are orders of magnitude apart (there, `10` against artefacts at 4e5), not
  by tuning until it passes.

* `abs_keys=["eigenvalues.*.im"]` compares the named dotted keys by magnitude - the `Fingerprint`
  analogue of `Evolution`'s `abs_columns`. **This, not `eigenvalues_by_value`, is what a lone
  conjugate needs.** `bifurcation_hopf_tracking_lorenz.py` records the single tracked eigenvalue,
  so there is no second entry to permute it with and reordering changes nothing; the tracker
  converges with `omega` or `-omega` and both describe the same Hopf point. Check what the
  reference actually holds before reaching for one or the other - that mistake cost a CI round
  here.

All three are applied at comparison time, to the reference and the produced record alike, so adding
any of them is a spec-side change like `skip=` and needs no reference regenerated.

A warning on the second one: it can only ever remove eigenvalues, so a run whose spectrum has
nothing left inside the window fails as "is gone" rather than passing quietly. Keep it that way -
a window wide enough to admit the artefacts, or an `only=` that excludes the eigenvalues
altogether, is how this stops checking anything.

### An adaptive output grid: `time_tol`, not `match="interp"`

`match="exact"` looks a stored instant up and insists it recur to within `1e-9 * span`, i.e.
bit-for-bit. An adaptive stepper does that on one machine and not on another, and the failure is
then pure lookup - no value is compared at all, so the check reports nothing about the physics.
`rayleigh_plateau.py` failed on all four platforms that way, the worst instant 3.58e-06 off.

`time_tol=X` lets the lookup slip by `X` in abscissa units. **Set it from the slope**, since the
cost is `|dy/dt| * time_tol`:

| | measured on `minimum.txt` |
| --- | --- |
| `max |dy/dt|` over the window | 0.3301 |
| column extent | 0.7337 |
| cost of a `1e-5` slip | 4.5e-06 of the extent, i.e. inside the `rtol=1e-5` already there |
| cost of `match="interp"` over one output interval | `h^2 |y''|/8` = 1.0e-03 of the extent |

So the slip is 630 times cheaper than interpolating, which is why this exists rather than switching
such checks to `interp`. Keep `time_tol` well under half the output spacing (0.0172 there); the
comparer refuses a larger one, because the "nearest" row could then be the wrong row.

Beware what fixing a lookup exposes. Once `rayleigh_plateau.py` could compare values over four
ranks, 11 of 128 differed - every one of them `z_min`, the *position* of the neck, which is read at
a mesh node and so is quantised by the mesh. `r_min`, the radius, agreed throughout. A lookup
failure can be hiding a column that was never comparable.

### A column whose error is not set by its own range: `column_atol`

The per-check `atol` applies to every column, which is wrong when one column is far less
reproducible than its neighbours. In `kuramoto_sivanshinsky_arclength_eigen.py`, `Re(lambda)` of a
7614-dof system at `shift=0` agrees between SLEPc and the built-in eigensolver only to 6.3e-04
while `h_rms` beside it reproduces bit-for-bit. Raising the check's `atol` would spend 0.29 % of
`h_rms`'s range and 4.9 % of `Im(lambda)`'s on a column that needs nothing.

`column_atol={"column_2": 2e-3}` confines it. Two things to check before using it:

* that an **absolute** tolerance is the right shape. Here it is, and the obvious guess was wrong:
  the deviations run 7.0e-06 to 6.3e-04 and are *not* monotonic in `|Re|` - the smallest `|Re|` in
  the file (5.1e-04) agrees while 1.0e-02 does not. That is solver accuracy on the operator, not a
  near-zero relative blow-up, so a bigger `rtol` would have been the wrong instrument.
* that the column still says something afterwards. 4 % of that column's range still separates the
  two branches the file interleaves (-0.031 against +0.003..+0.012), still catches a sign error and
  still catches the fold moving.

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

  The mechanism, measured on `Multicomponent_Flow/nacl_capillary_evaporation.py`: **adaptive time
  stepping is not rank-invariant, because the temporal error norm is a parallel reduction.** Its
  summation order differs with the rank count, so the estimate differs in its last bits, and that
  can tip a step-size decision. There it ends the run 4.0e-08 (relative) later at four ranks, which
  is enough to move the evaporative flux at that instant by 0.43 % - an amplification of about 1e5,
  because the capillary is drying out and the rate is changing steeply exactly where the run stops.
  Its other problem, in the same script and the same pass, ends at a bit-identical instant; this is
  one decision tipping, not a systematic offset, which is why it cannot be predicted from the
  script and has to be measured.

  Two consequences for writing an entry. A quantity sampled at the END of an adaptive run is worth
  less than it looks, and the steeper the quantity is there, the less it is worth. And a reduced
  series survives where that final state does not - the same script's three reduced series,
  `N_salt` conservation included, came through the same pass untouched.

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

### Do not edit the spec while a pass is running

`checks_for()` does `from . import specs`, and Python caches that in `sys.modules`, so a pass reads
the spec once - at the first script it validates, which is a minute or so after launch because the
harness regenerates the tutorial bundle first. Editing `specs.py` during a pass therefore produces a
report against an unknown mixture of the old and the new spec, and the ambiguity is invisible in the
output. If a correction cannot wait, re-check the affected scripts afterwards rather than reading
that pass's verdict for them.

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
