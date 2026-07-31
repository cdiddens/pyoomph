# Where code-generation time actually goes

Investigation on branch `codegen` into whether pyoomph's symbolic code generation can be made
faster, and what was changed as a result.

## The two phases, and which one is expensive

"Code generation" is not one step. A residual passes through:

1. **Ingestion** - `FiniteElementCode::add_residual()` (src/codegen.cpp). Runs when the equations
   are defined, i.e. inside `Problem.initialise()` well before any C is written. It expands
   placeholders (`field(...)`, `scale(...)`, `testfunction(...)`, ...) into shape expansions,
   checks the result is non-dimensional, and accumulates it into `residual[]`.
2. **Emission** - `FiniteElementCode::write_code()`. Differentiates the accumulated residual to
   get the Jacobian/mass matrix/Hessian and prints everything as C.

The intuition that `write_code()` is the expensive half is wrong. On a deliberately heavy element
(axisymmetric moving-mesh Navier-Stokes + N species with full cross-diffusion, analytic Hessian on),
timed per phase in-process:

| species | `add_residual` | `write_code` + tcc |
|---------|---------------|--------------------|
| 2       |  0.30 s       | 0.60 s             |
| 3       |  4.29 s       | 0.99 s             |
| 4       | 57.16 s       | 1.53 s             |

Emission grows roughly linearly with the size of the problem. Ingestion grew by a factor of ~13 per
added species and, at 5 species, did not finish within eight minutes.

## Cause 1: a normalisation whose result is thrown away

`add_residual()` computed

    GiNaC::ex expa = repl.expand().normal();

and then read `expa` **only** to decide whether any base unit (metre, second, ...) survived. The
expression actually stored is `repl.subs(sublist)`; `expa` is never used for anything else.

`normal()` puts a rational expression over a common denominator. A residual with several rational
nonlinearities - concentration-dependent diffusivities, Arrhenius laws, activity coefficients -
therefore gets all of its denominators multiplied together, at a cost of seconds per contribution,
and the swollen result is then discarded. Measured per contribution at 3 species: 1.4 s each, 4.15 s
of the 4.29 s total.

Neither `expand()` nor `normal()` can *introduce* a base unit that was not already somewhere in
`repl` - they only ever cancel them (`m/m -> 1`). So when a cheap single traversal finds no base
unit at all, the check that follows is guaranteed to find none either, and the normalisation can be
skipped entirely. Dimensional problems take exactly the old path.

Effect on the benchmark above, A/B'd inside one binary with the arms interleaved, 4 species:

| arm            | `add_residual` | `initialise()` total |
|----------------|---------------|----------------------|
| prescan off    | 56.01 / 56.37 s | 57.58 / 57.96 s    |
| prescan on     |  0.025 / 0.025 s |  1.58 / 1.59 s    |

5 species, which previously did not finish within eight minutes, takes 2.8 s.

Two escape hatches were added while validating this, and kept:

- `PYOOMPH_DISABLE_UNIT_PRESCAN=1` restores the unconditional `expand().normal()`, so a suspected
  behaviour change can be bisected inside one binary.
- `PYOOMPH_PARANOID_UNIT_PRESCAN=1` does the skipped work anyway and raises if it disagrees with the
  prescan. Run over the dimensional tutorial scripts, it never fired.

### The one visible consequence

`normal()` allocates temporary GiNaC symbols, so not calling it shifts the global symbol serial
counter, which is what GiNaC uses to order factors canonically when printing a product. On the
benchmark this changed **2 of 2537 lines** of generated C, each a pure permutation of the factors of
one product - same operands, different multiplication order. Results can therefore differ in the
last floating-point bits, exactly as a GiNaC version bump would, and existing Tier-1 JIT cache
entries (keyed on the generated text) miss once and repopulate. Fully dimensional models are
unaffected: they take the unchanged path and their generated code is byte-identical.

### What about problems that really do have units?

The prescan does nothing for them - they mention base units, so they take the full
`expand().normal()`. A dimensional counterpart of the benchmark (same rational cross-diffusion, but
the diffusivities carry m^2/s) spends 0.763 s of its 0.815 s of ingestion in that one call at 4
species.

There is a shortcut, and it comes straight out of the existing code: when the check below *does*
find a surviving base unit, it already asks `collect_base_units()` for the verdict and accepts the
contribution if the units it collects multiply out to 1. So ask that question first, on the
un-normalised expression. `collect_base_units()` is a structural dimensional analysis - it expands
and collects common factors as it recurses, but never performs a rational normalisation - so it does
not pay the common-denominator cost that makes `expand().normal()` expensive. If it proves the units
cancel, the normalisation is skipped; if it cannot, control falls through to exactly the old code,
which decides. The check can therefore only ever turn a rejection into an acceptance, never the
reverse, and the residual that gets stored is the same either way.

It works, and it is **opt-in** (`PYOOMPH_UNIT_FASTCHECK=1`), because the payoff is narrow and the
perturbation is not:

- On the dimensional benchmark, `initialise()` goes from 1.168 s to 0.400 s at 4 species.
- Across twelve dimensional tutorial scripts it is a wash. Interleaved repeats:
  `gcl_glycerol_water_capillary` 1.715/1.737/1.696 s off vs 1.702/1.689/1.701 s on;
  `solid_oscillations` 21.02/20.64 s off vs 20.82/20.96 s on. Their contributions are simply not
  rational enough for `normal()` to hurt (at most 0.14 s in any of the 31 scripts swept), and their
  ingestion time goes to `expand_placeholders` instead - which is the next section.
  `collect_base_units()` itself costs 0.03-0.16 s on those scripts, so it neither helps nor hurts.
- Skipping `normal()` changes the order in which subexpressions get registered, so the generated
  code comes out with its CSE temporaries renumbered and some products reassociated - on
  `marangoni_instability` that is 100+ lines across three files.

Correctness was checked three ways and all three agree:

1. `PYOOMPH_PARANOID_UNIT_PRESCAN=1` (extended to cross-check this decision too: run the
   authoritative `expand().normal()` and insist it would also have accepted) passed on all twelve
   dimensional scripts.
2. A deliberately inconsistent residual - the working dimensional benchmark plus a stray
   `weak(1*meter*c, test)` - is still rejected with "Found a dimensional contribution".
3. Every differing generated line was checked mechanically rather than by eye: canonicalise
   `subexpr_N` by the content of its definition, then bind every identifier to random values and
   evaluate both sides. All differences came out numerically equal, and what survived
   canonicalisation was two independent temporaries defined in swapped order.

Still, "verified equivalent on what I ran" is not a reason to reassociate every dimensional Jacobian
in the framework by default, for a win that only materialises on residuals with heavy rational
nonlinearity. Turn it on if that describes your model.

## Cause 2: the placeholder expander walked a DAG as a tree

On the actual tutorial scripts, cause 1 is small (at most 0.14 s in any of the 31 scripts swept).
There the ingestion cost is `expand_placeholders()` - 16.0 s of `solid_oscillations`' 16.4 s of
`add_residual`, 2.8 s of `marangoni_instability`' 3.3 s.

That is not the fixpoint loop iterating (it converges in one or two passes) and not the
`(old - repl).is_zero()` stuck-check (below the 1 ms reporting threshold). It is the single
`ReplaceFieldsToNonDimFields` pass itself. Counting entries into the mapper against the number of
distinct subexpressions it was entered on:

| script                  | mapper entries | distinct subexpressions |
|-------------------------|----------------|-------------------------|
| `solid_oscillations`    | 1,533,930      | ~395                    |
| `marangoni_instability` |    98,819      | ~1,077                  |

GiNaC expressions are reference-counted DAGs, but `ex::map` traverses them as trees: a subexpression
shared k times is rewritten k times. Hyperelasticity, where the same Cauchy-Green invariants appear
throughout the residual, hits this hardest.

`ReplaceFieldsToNonDimFields::operator()` is now a memoising wrapper around the old body
(`do_replace`), keyed on GiNaC's cached expression hash with `ex::is_equal` confirming the bucket
(that comparison short-circuits on pointer identity, which is the common case for a shared subtree).

**On by default**; `PYOOMPH_DISABLE_EXPAND_MEMO=1` turns it off. See "what it changes" below.

Two pieces of state have to survive a cache hit:

- `repl_count`, which the fixpoint loop reads as "did this pass change anything". The delta the first
  expansion produced is replayed, so a cached replacement still counts as progress and the loop
  cannot terminate a pass early with placeholders left unexpanded.
- `code->expanded_scales`, which is only ever assigned and holds the same value for the same field,
  so skipping the repeat assignment leaves it unchanged.

Nothing whose expansion went through a Python-overridable hook (`expand_additional_field`,
`expand_additional_testfunction`, `GiNaCDelayedPythonCallbackExpansion`, `python_multi_cb_function`)
is cached. There is already a disabled cache for exactly those
(`expanded_additional_field_cache`, marked "Do not use the cache for the moment"), and this change
should not quietly re-enable a caching decision somebody else backed out of. The exclusion propagates
to ancestors via a hook-call counter, since caching a *parent* freezes the hook's result just as
effectively - a first attempt that only excluded the calling node was not enough, and the
`marangoni_instability` generated code changed because of it.

### What it changes

Measured on setup only (a wrapper that stops right after `Problem.initialise()`, so the numbers are
code generation rather than simulation):

| script                  | memo off | memo on | generated C |
|-------------------------|----------|---------|-------------|
| `solid_oscillations`    | 20.65 s  |  5.58 s | identical   |
| `marangoni_instability` |  4.75 s  |  4.19 s | **differs** |

`solid_oscillations` is a 3.7x speedup of the whole setup phase with byte-identical output. But on
`marangoni_instability` - which drives the material library hard - the generated C still differs,
even with ancestor propagation. Inspection of the 32 differing lines shows the operands are the same
and only GiNaC's canonical sign placement and factor order move (`-a*(-X)` where the other arm has
`a*X`), which is the same class of difference the unit prescan produces. But that was checked by
reading the diff, not proved, and it was not possible to settle it numerically here: the one script
that exhibits it is itself non-deterministic run to run (a random perturbation feeding
`RefineToLevel`, so `ndof` differed 22991 vs 23186 between two runs of the *same* arm), which rules
out comparing assembled residuals across arms. Its generated code, unlike its ndof, *is* reproducible
per arm - so the difference is genuinely caused by the memo, not by the randomness.

The differing lines were later checked mechanically rather than by eye - canonicalise `subexpr_N` by
the content of its definition, then bind every identifier to random values and evaluate both sides -
and all of them compute the same thing. On that basis the memo is enabled by default. The open
question it does *not* answer is why `expanded_additional_field_cache` was disabled; the memo simply
never caches anything that went through those hooks.

## Smaller items in the emission path

Found while reading `write_code()`; each is dead or redundant work rather than an algorithmic change.
All three are on by default and all three are code-neutral: with the unit prescan disabled and the
memo off, the benchmark's generated C is byte-identical to what the pre-change build produced.

- **`print_simplest_form` archived every expression it printed.** `FiniteElementCode::archive`
  accumulated every residual/Jacobian/mass/Hessian expression - a full recursive walk plus a
  structure-sharing hash lookup per node, retained until the next `write_code()`. Nothing reads it;
  the only consumer was the `.gar` dump in `generate_and_compile_bulk_element_code()`, which is
  commented out. Now opt-in via `PYOOMPH_ARCHIVE_EXPRESSIONS`. Interleaved A/B at 4 species:
  `write_code` + tcc goes from 1.586 / 1.534 s to 1.326 / 1.348 s, i.e. about 14% of emission time
  spent filling a structure nobody reads.
- **`GiNaCShapeExpansion::derivative` stringified its symbol up front.** It built an `ostringstream`
  and a `std::string` on entry, for a value consulted only in two nodal-position branches. This
  function runs once per shape-expansion node per `GiNaC::diff()`, i.e. once per (residual term,
  field) pair for the Jacobian and again per field for the Hessian. Now computed on demand.
- **The Hessian double loop re-scanned every subexpression per field pair.** The harvest of shape
  expansions out of the registered subexpressions sat in the innermost `(f, f2)` loop, but the
  subexpression list only grows and set insertion is idempotent, so every revisit was a full tree
  walk contributing nothing. Now only newly appended entries are scanned.

## Things deliberately left alone

- **`expanded_additional_field_cache` is written but never read** (`if (false && ...)` in both the
  `field` and `nondimfield` branches). Today it costs memory and buys nothing. Re-enabling it is the
  obvious next lever for models that lean on Python-defined pseudo-fields, but the "for the moment"
  comment suggests it was backed out for a reason that is not recorded, so it wants its own
  investigation rather than being flipped back on here.
- **`resolve_corresponding_code` builds an `ostringstream` per placeholder** to recover the field
  name, and `get_scaling` is a virtual dispatched into Python. Both are on the expansion path, but
  after the memo above they run once per distinct subexpression rather than once per visit.
- **The `operator<` implementations for `ShapeExpansion`/`SpatialIntegralSymbol`/... re-evaluate every
  earlier comparison in each disjunct**, so a comparison costs O(n^2) field tests instead of O(n).
  They are cheap integer comparisons and did not show up in any measurement.

## Environment switches introduced

All default to off; none of them change what pyoomph computes, only how it gets there or what it
reports.

| variable | effect |
|----------|--------|
| `PYOOMPH_DISABLE_UNIT_PRESCAN` | restore the unconditional `expand().normal()` in `add_residual` |
| `PYOOMPH_UNIT_FASTCHECK` | ask `collect_base_units` before normalising, for dimensional contributions (off by default, see above) |
| `PYOOMPH_PARANOID_UNIT_PRESCAN` | do the skipped normalisation anyway and raise if it disagrees - covers both the prescan and the fast check |
| `PYOOMPH_DISABLE_EXPAND_MEMO` | turn off the placeholder-expansion memo (on by default) |
| `PYOOMPH_ARCHIVE_EXPRESSIONS` | fill `FiniteElementCode::archive` again |
| `PYOOMPH_TIME_ADD_RESIDUAL` | per-phase timing of `add_residual`/`expand_placeholders` on stderr |

## Reproducing

The benchmark and probes live in the session scratchpad, not in the repo. The shape is:

- build a heavy element, patch `Problem.compile_bulk_element_code` to time itself, and patch
  `_pyoomph_core.FiniteElementCode._add_residual` to time ingestion separately;
- `PYOOMPH_TIME_ADD_RESIDUAL=1` prints a per-phase breakdown of `add_residual`/`expand_placeholders`
  on stderr, including mapper entries, memo hits and distinct-subexpression counts;
- A/B the two expansion paths inside one binary with `PYOOMPH_DISABLE_EXPAND_MEMO` /
  `PYOOMPH_DISABLE_UNIT_PRESCAN`, interleaving the arms, and compare the generated `.c` files byte
  for byte.
