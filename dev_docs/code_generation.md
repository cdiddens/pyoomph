# Code generation: the time to produce the C, and the time to run it

Two investigations on the same pipeline, merged here because they keep pointing at each other.

* **Producing the C** (§1–§4) — branch `codegen`. Ingestion, not emission, was the expensive half,
  and two causes account for essentially all of it. Both fixes landed.
* **Running the C** (§5–§9) — whether the emitted code can be made faster. Against the *compiler* the
  answer is mostly no: pyoomph emits deliberately redundant C and the C compiler cleans it up well
  enough that three implemented, measured changes were all reverted. The one thing a C compiler cannot
  do is differentiate, so the symbolic derivative cache behind `subexpression()` is where the wins are —
  hence the advice in §7, and hence §9.3, which extends that cache to the Hessian for −91%.

§4 is the part to read even if none of the performance matters to you — a caching change silently
emitted a wrong residual, through a GiNaC printing bug that is still upstream.

Run-time numbers are in-process timings of `problem._benchmark_elemental_assembly()`, arms built up
front and timed interleaved, min over rounds, nothing else running.

---

## 1. The two phases, and which one is expensive

"Code generation" is not one step. A residual passes through:

1. **Ingestion** — `FiniteElementCode::add_residual()` (`src/codegen.cpp`). Runs when the equations
   are defined, i.e. inside `Problem.initialise()` well before any C is written. It expands
   placeholders (`field(...)`, `scale(...)`, `testfunction(...)`, ...) into shape expansions, checks
   the result is non-dimensional, and accumulates it into `residual[]`.
2. **Emission** — `FiniteElementCode::write_code()`. Differentiates the accumulated residual to get
   the Jacobian/mass matrix/Hessian and prints everything as C.

The intuition that `write_code()` is the expensive half is wrong. On a deliberately heavy element
(axisymmetric moving-mesh Navier-Stokes + N species with full cross-diffusion, analytic Hessian on),
timed per phase in-process:

| species | `add_residual` | `write_code` + tcc |
|---------|---------------|--------------------|
| 2       |  0.30 s       | 0.60 s             |
| 3       |  4.29 s       | 0.99 s             |
| 4       | 57.16 s       | 1.53 s             |

Emission grows roughly linearly with problem size. Ingestion grew by ~13x per added species and, at
5 species, did not finish within eight minutes.

## 2. Cause 1: a normalisation whose result is thrown away

`add_residual()` computed

    GiNaC::ex expa = repl.expand().normal();

and then read `expa` **only** to decide whether any base unit (metre, second, ...) survived. The
expression actually stored is `repl.subs(sublist)`; `expa` was never used for anything else.

`normal()` puts a rational expression over a common denominator. A residual with several rational
nonlinearities — concentration-dependent diffusivities, Arrhenius laws, activity coefficients —
therefore gets all of its denominators multiplied together, at seconds per contribution, and the
swollen result is then discarded. Measured at 3 species: 1.4 s per contribution, 4.15 s of the
4.29 s total.

Neither `expand()` nor `normal()` can *introduce* a base unit that was not already somewhere in
`repl` — they only ever cancel them (`m/m -> 1`). So when a cheap single traversal finds no base
unit at all, the check that follows is guaranteed to find none either, and the normalisation can be
skipped. Dimensional problems take exactly the old path.

A/B'd inside one binary with the arms interleaved, 4 species:

| arm            | `add_residual` | `initialise()` total |
|----------------|---------------|----------------------|
| prescan off    | 56.01 / 56.37 s | 57.58 / 57.96 s    |
| prescan on     |  0.025 / 0.025 s |  1.58 / 1.59 s    |

5 species, which previously did not finish within eight minutes, takes 2.8 s.

### 2.1 The one visible consequence

`normal()` allocates temporary GiNaC symbols, so not calling it shifts the global symbol serial
counter, which is what GiNaC uses to order factors canonically when printing a product. On the
benchmark this changed **2 of 2537 lines** of generated C, each a pure permutation of the factors of
one product — same operands, different multiplication order. Results can therefore differ in the last
floating-point bits, exactly as a GiNaC version bump would, and existing Tier-1 JIT cache entries
(keyed on the generated text) miss once and repopulate. Fully dimensional models are unaffected:
they take the unchanged path and their generated code is byte-identical.

### 2.2 Problems that really do have units — `PYOOMPH_UNIT_FASTCHECK`, opt-in

The prescan does nothing for them: they mention base units, so they take the full
`expand().normal()`. A dimensional counterpart of the benchmark spends 0.763 s of its 0.815 s of
ingestion in that one call at 4 species.

The shortcut comes straight out of the existing code: when the check *does* find a surviving base
unit, it already asks `collect_base_units()` for the verdict and accepts the contribution if the
units multiply out to 1. So ask that question first, on the un-normalised expression.
`collect_base_units()` is a structural dimensional analysis — it expands and collects common factors
as it recurses but never performs a rational normalisation — so it does not pay the common-denominator
cost. If it proves the units cancel, the normalisation is skipped; if it cannot, control falls through
to exactly the old code. The check can therefore only ever turn a rejection into an acceptance, never
the reverse, and the stored residual is the same either way.

It works, and it is nevertheless **opt-in**, because the payoff is narrow and the perturbation is not:

- On the dimensional benchmark, `initialise()` goes 1.168 s -> 0.400 s at 4 species.
- Across twelve dimensional tutorial scripts it is a wash — their contributions are not rational
  enough for `normal()` to hurt (at most 0.14 s in any of the 31 scripts swept), and their ingestion
  time goes to `expand_placeholders` instead (§3). `collect_base_units()` itself costs 0.03–0.16 s
  there, so it neither helps nor hurts.
- Skipping `normal()` renumbers CSE temporaries and reassociates some products — on
  `marangoni_instability`, 100+ lines across three files.

Correctness was checked three ways, all agreeing: `PYOOMPH_PARANOID_UNIT_PRESCAN=1` (extended to
cross-check this decision too) passed on all twelve dimensional scripts; a deliberately inconsistent
residual is still rejected; and every differing generated line was checked mechanically — canonicalise
`subexpr_N` by the content of its definition, bind every identifier to random values, evaluate both
sides — and came out numerically equal.

Still, "verified equivalent on what I ran" is not a reason to reassociate every dimensional Jacobian
in the framework by default, for a win that only materialises on heavy rational nonlinearity. Turn it
on if that describes your model.

## 3. Cause 2: the placeholder expander walked a DAG as a tree

On the actual tutorial scripts, cause 1 is small. There the ingestion cost is `expand_placeholders()`
— 16.0 s of `solid_oscillations`' 16.4 s of `add_residual`, 2.8 s of `marangoni_instability`'s 3.3 s.

That is not the fixpoint loop iterating (it converges in one or two passes) and not the
`(old - repl).is_zero()` stuck-check (below the 1 ms reporting threshold). It is the single
`ReplaceFieldsToNonDimFields` pass. Mapper entries against distinct subexpressions:

| script                  | mapper entries | distinct subexpressions |
|-------------------------|----------------|-------------------------|
| `solid_oscillations`    | 1,533,930      | ~395                    |
| `marangoni_instability` |    98,819      | ~1,077                  |

GiNaC expressions are reference-counted DAGs, but `ex::map` traverses them as trees: a subexpression
shared k times is rewritten k times. Hyperelasticity, where the same Cauchy-Green invariants appear
throughout the residual, hits this hardest.

`ReplaceFieldsToNonDimFields::operator()` is now a memoising wrapper around the old body
(`do_replace`), keyed on GiNaC's cached expression hash with `ex::is_equal` confirming the bucket
(that comparison short-circuits on pointer identity, the common case for a shared subtree). **On by
default.**

Two pieces of state have to survive a cache hit:

- `repl_count`, which the fixpoint loop reads as "did this pass change anything". The delta the first
  expansion produced is replayed, so a cached replacement still counts as progress and the loop cannot
  terminate a pass early with placeholders left unexpanded.
- `code->expanded_scales`, which is only ever assigned and holds the same value for the same field.

Nothing whose expansion went through a Python-overridable hook (`expand_additional_field`,
`expand_additional_testfunction`, `GiNaCDelayedPythonCallbackExpansion`, `python_multi_cb_function`)
is cached. There is already a disabled cache for exactly those (`expanded_additional_field_cache`,
marked "Do not use the cache for the moment"), and this change should not quietly re-enable a caching
decision somebody else backed out of. The exclusion propagates to ancestors via a hook-call counter,
since caching a *parent* freezes the hook's result just as effectively — a first attempt that only
excluded the calling node was not enough, and `marangoni_instability`'s generated code changed because
of it.

Measured on setup only (a wrapper stopping right after `Problem.initialise()`):

| script                  | memo off | memo on | generated C |
|-------------------------|----------|---------|-------------|
| `solid_oscillations`    | 20.65 s  |  5.58 s | identical   |
| `marangoni_instability` |  4.75 s  |  4.19 s | **differs** |

`solid_oscillations` is a 3.7x speedup of the whole setup phase with byte-identical output. On
`marangoni_instability` the 32 differing lines have the same operands with only GiNaC's canonical sign
placement and factor order moving; that was confirmed by the mechanical random-evaluation check, not
by eye. It could not be settled by comparing assembled residuals, because that script is itself
non-deterministic run to run (a random perturbation feeding `RefineToLevel`; `ndof` differed 22991 vs
23186 between two runs of the *same* arm). Its generated code, unlike its `ndof`, is reproducible per
arm — so the difference is caused by the memo, not by the randomness.

## 4. The memo did change what was computed, once

"Only canonical sign placement and factor order move" was not the whole story, and the mechanical
check above could not have caught the rest: the difference is one of *representation*, so both sides
evaluate to the same number and only the generated C source is wrong.

GiNaC compares and hashes numbers by value. An exact `-1` and a floating point `-1.0` are therefore
`is_equal` and land in the same memo bucket, and the memo hands back whichever it met first. Harmless
for every later computation — and not harmless at all for code emission, because
`mul::do_print_csrc` *does* distinguish them, and inconsistently: it decides whether to write `1.0/`
or `/` from `info_flags::negint`, which an inexact `-1` does not satisfy, but decides whether to leave
the exponent out altogether from `is_equal(-1)`, which it does. A factor `x^(-1.0)` is printed as a
multiplication by `x`. No error, no warning, a residual wrong by `x^2`.

Found through `ViscoelasticInflowBC`, which builds a matrix logarithm — hence a reciprocal
`(lambda_+ - lambda_- + eps)^(-1)` with an exact `-1` — out of a velocity profile whose symbolic
derivative carries a floating point `-1.0`.

Both halves are fixed:

- `ReplaceFieldsToNonDimFields::operator()` never memoises a bare `numeric`. A leaf costs nothing to
  expand, and this is the only position from which the memo can substitute one number for another.
- `print_simplest_form` makes any inexact whole-number exponent exact before printing
  (`ExactifyWholeNumberExponents`), which also covers an `x**(-1.0)` written by hand in Python. It is
  guarded by a scan that allocates nothing and normally finds nothing.

`tests/test_generated_code_expressions.py` pins what the compiled element computes for both. It cannot
separate the two fixes, because the printer guard masks the memo defect: reverting the memo fix alone
still passes, reverting the printer guard alone fails the hand-written `x**(-1.0)` test, reverting both
fails both.

**The lesson for any future cache in this path:** `ex::is_equal` is the wrong equality to key one on
if what comes out is going to be printed rather than only evaluated.

The GiNaC half is a plain upstream bug, not something our patches or our build cause.
[examples/ginac_csrc_inexact_negative_exponent.cpp](examples/ginac_csrc_inexact_negative_exponent.cpp)
reproduces it with no pyoomph involved and exits nonzero while it is present; it is what to send
upstream. It behaves the same on the 1.8.10 we build from source and on the distribution's stock 1.8.7
(`libginac-dev`), the two outputs differing in one line only, where canonical factor order puts the
affected factor first rather than second — which between them covers both branches of
`mul::do_print_csrc` that ought to emit a division. The one-line fix it suggests was verified by
compiling a patched `mul.cpp` and linking it ahead of the stock `libginac.a`:

    g++ -c -DHAVE_CONFIG_H -I$G -I$G/config -I$G/ginac -I$P/include -o mul_patched.o mul_patched.cpp
    g++ -o csrc_bug_patched ginac_csrc_inexact_negative_exponent.cpp mul_patched.o \
        -I$P/include -L$P/lib -lginac -lcln -lgmp

with `$G` the GiNaC source tree under `build/*/ginac_build/src/ginac_external` and `$P` the
third-party install prefix. `3*(1+x)^(-1.0)` then prints as `3.0*1.0/( x+1.0)`, and every case that was
already correct is unchanged. Should that land upstream, `ExactifyWholeNumberExponents` can go — the
memo fix has to stay either way.

## 5. Smaller items in the emission path

Each is dead or redundant work rather than an algorithmic change. All three are on by default and all
three are code-neutral: with the unit prescan disabled and the memo off, the benchmark's generated C
is byte-identical to what the pre-change build produced.

- **`print_simplest_form` archived every expression it printed.** `FiniteElementCode::archive`
  accumulated every residual/Jacobian/mass/Hessian expression — a full recursive walk plus a
  structure-sharing hash lookup per node, retained until the next `write_code()`. Nothing reads it;
  the only consumer was the `.gar` dump in `generate_and_compile_bulk_element_code()`, which is
  commented out. Now opt-in via `PYOOMPH_ARCHIVE_EXPRESSIONS`. Interleaved A/B at 4 species:
  `write_code` + tcc goes 1.586 / 1.534 s -> 1.326 / 1.348 s, i.e. ~14% of emission time spent filling
  a structure nobody reads.
- **`GiNaCShapeExpansion::derivative` stringified its symbol up front**, for a value consulted only in
  two nodal-position branches. It runs once per shape-expansion node per `GiNaC::diff()`. Now computed
  on demand.
- **The Hessian double loop re-scanned every subexpression per field pair.** The subexpression list
  only grows and set insertion is idempotent, so every revisit was a full tree walk contributing
  nothing. Now only newly appended entries are scanned.

## 6. What the generator emits, and what it already optimizes

`FiniteElementCode::write_generic_RJM` (`src/codegen.cpp`) produces, per element:

```
for ipt < n_int_pt {                     // quadrature
   fill_shape_buffer_for_point(...)
   <field interpolation over l_shape>
   <subexpression block>                 // write_code_subexpressions
   for l_test < nnode {                  // test function
      BEGIN_RESIDUAL_*(eqn, <one flat expression>)
      if (flag) for l_shape < nnode {    // trial function, innermost
         BEGIN_JACOBIAN_*(eqn, <one flat expression>)   // one per (test field, unknown field)
      }
   }
}
```

Macros in `src/jitbridge.h`. Each entry is a single flat C expression on one source line, printed by
`print_simplest_form`, which by default does `expr.evalf()` and hands the result to
`print_csrc_FEM : public GiNaC::print_csrc_double`. Lines of 15 000 to 350 000 characters are normal
in production elements.

The only CSE is what the user asks for. `subexpression()` (`pyoomph/expressions/generic.py`) is the
sole mechanism: `SubExpressionsToStructs` replaces each marker with a `GiNaCSubExpression`,
de-duplicating by `is_equal`; `write_code_subexpressions` emits `double subexpr_N = ...;` and declares
`d_subexpr_N_d_<field>` derivatives filled once per integration point under `if (flag)`; and
`GiNaCSubExpression::derivative` applies the chain rule using the *cached* scalar rather than
re-differentiating the body. That last part is what actually wins.

## 7. `subexpression()` beats every compiler flag

`--fast-math` (`Problem.parse_cmd_line`, adds `-ffast-math` in `ccompiler.py`) was the starting point.
Four arms compiled from byte-identical generated C, elemental residual+Jacobian relative to the
`-O3 -march=native` default:

| case | ndof | `-fno-math-errno` | fast-math minus `-ffinite-math-only` | `-ffast-math` |
|---|---|---|---|---|
| poisson3d | 4 624 | +0.1% | −1.0% | −0.6% |
| ns2d | 20 450 | +0.2% | −0.4% | −1.4% |
| coupled | 20 642 | −0.2% | −0.5% | −0.7% |
| ns3d | 7 102 | +0.2% | −4.9% | −4.7% |
| transcendental 2D | 18 430 | **−86.5%** | −89.2% | −89.6% |

On polynomial weak forms the flag is noise. The one case where it is worth 7x is a weak form full of
`exp`/`log`/non-integer `pow`, and there the win is almost entirely `-fno-math-errno`, which changes no
arithmetic at all — residuals came out **bit-identical**, unlike the reassociating arms (~1e-16
relative). The mechanism is visible in the `.so`; libm call sites in the same element:

| | `pow` | `exp` | `log` | total |
|---|---|---|---|---|
| default | 14 | 11 | 2 | 27 |
| `-fno-math-errno` | 3 | 2 | 1 | 6 |
| `subexpression()`, default flags | 3 | 4 | 2 | 9 |
| both | 2 | 2 | 1 | 5 |

Without `-fno-math-errno` a libm call is impure — it may set `errno` — so GCC may neither CSE it nor
hoist it out of the `l_shape` loop.

And that is the finding that redirected the whole investigation. Same element, same clock:

| | elemental res+jac |
|---|---|
| default flags | 0.4397 s |
| `-fno-math-errno` | 0.0594 s (−86.5%) |
| `-ffast-math` | 0.0459 s (−89.6%) |
| **`subexpression()`, default flags** | **0.0392 s (−91.1%)** |
| `subexpression()` + `-ffast-math` | 0.0360 s (−91.8%) |

`subexpression()` does the same CSE at code-generation time *and* caches the symbolic derivatives,
which is why its advantage is much larger in the Jacobian (−91.1% vs −86.5%) than in the residual
alone (−46.5% vs −42.9%). Once it is used, the flags are worth ~0.7%. `-fno-math-errno` was therefore
not added to the defaults.

> Note for anyone re-adding a hardcoded flag: the compile flags are **not** fully covered by the JIT
> cache key. `SystemCCompiler.get_cache_flag_state()` covers `PYOOMPH_DEBUG`, `_optimize_full_speed`
> and `compile_args`, from all of which today's flags are derivable; a flag hardcoded into the defaults
> is not, so it needs an explicit epoch in that string or every previously cached `.so` is silently
> reused.

## 8. What `-O3` rescues, and what it does not

Every item below was tested by rewriting the *emitted C* and recompiling, so no C++ was needed to get
the number.

### 8.1 `pow(x,2.0)` for every square — real, but only for TCC. Reverted.

GiNaC's `power::do_print_csrc` only takes its `x*x` fast path when the basis `is_a<symbol>`; pyoomph
bases are `GiNaC::structure` subclasses, so it never fires and every square is printed as a `pow`
call. The tutorial's torsioned hyperelastic beam emits 210 `pow(`, all with exponent exactly `2.0`.
Rewriting all 210 to `(x*x)`:

| compiler | `pow@plt` in the `.so` | before | after | |
|---|---|---|---|---|
| gcc `-O3 -march=native` | 45 → 0 | 0.3095 s | 0.3062 s | **−1.1%** |
| tccbox | 210 → 0 | 3.4888 s | 1.6423 s | **−52.9%** |

GCC folds 165 of the 210 by itself. Note the other number: **TCC is 11x slower than gcc `-O3`** on this
element, and still 5.3x slower after the fix.

A real printer replacement was implemented and was bit-exact (max abs difference exactly 0 on residual
*and* Jacobian). It was reverted for three reasons: printing is compiler-blind, so it cannot be scoped
to tcc without threading the target compiler through code generation; turning a square into a product
duplicates the basis, so any loosening of the "atomic operand" classifier makes `pow(exp(x),2)` a
straight loss on the one compiler without CSE; and tcc's selling point (compile speed) is largely
removed by the JIT cache, so the honest advice is to use the system compiler.

Two things from that exercise are worth keeping. The GiNaC limitation above will be met again by
anyone touching the printer. And a near-miss: the first classifier tested only for the absence of `(`,
on the theory that duplicating anything without a call or a parenthesised group is free. That let the
basis `coordinate_x-0.3` through and emitted

    coordinate_x-0.3*coordinate_x-0.3      instead of   (coordinate_x-0.3)*(coordinate_x-0.3)

— silently wrong by operator precedence. It survived a bit-identical residual *and* Jacobian check on
the solid element, whose bases are all plain variable references, and was caught only by `tests/`: 12
failures in `test_adaptivity.py` and `test_globally_convergent_newton.py`, showing as an initial Newton
residual of 2.4e+66. GiNaC's own `print_sym_pow` needs no parentheses because it only ever receives a
`symbol`; generalising it to arbitrary printed strings is what creates the obligation.

An `-O` sweep on a larger production element (1.94 MB of C, 2917 `pow(`, of which 1501 integer-exponent
and 1416 cube-root-family fractional):

| | compile | `.so` | `pow@plt` |
|---|---|---|---|
| `-O0` | 3.1 s | 1 373 048 B | 2491 |
| `-O1` | 9.0 s | 459 576 B | 1454 |
| `-O2` | 24.3 s | 443 192 B | 1454 |
| `-O3` | 29.7 s | 521 016 B | 1454 |

Everything integer-exponent is folded by `-O1`, and the 1454 survivors are essentially the 1416
genuinely fractional powers. What `-O3` buys over `-O1` here is a 3x longer compile and no fewer libm
calls.

### 8.2 The dead subexpression derivative cache — a correctness smell, not a cost

On a moving mesh, `GiNaCSubExpression::derivative` detects a position symbol and takes an escape hatch:
it differentiates the body on the spot and inlines the result at every use site, ignoring the cached
scalar. The cache is still emitted. In the tutorial solid element, 225 `d_subexpr_*` are declared and
assigned per integration point and **171 are never referenced**; in an archived production element
(3.7 MB) the count is 153 declared, 153 assigned, **0 read anywhere in the file**.

Deleting the 90 provably dead assignments and re-measuring: gcc −0.1%, tcc +0.6%. Nothing — they are
computed once per integration point against `nnode² = 729` Jacobian entries per point. So this is worth
fixing for clarity and for what it reveals (§9.2), not for speed.

The Hessian now has its own instance of the same thing (§9.3): the *original* subexpression's
`d_subexpr_*` are often unread there, because the outer index goes through a nested copy. Same ratio,
same verdict — with the caveat that an unreferenced `exp()` is not free to the compiler unless
`-fno-math-errno` is on (§7).

### 8.3 Repeated subtrees in the Jacobian entries — the real one, and it still did not pay

Each `BEGIN_JACOBIAN_*` entry is an independent `GiNaC::diff` of the whole residual, printed flat, so a
factor shared between entries is re-expanded into every one. Within a statement the same subtree also
appears repeatedly, because GiNaC does not factor `A*X + B*X` when `A` and `B` differ.

Modelled on the emitted C (collect entry expressions per `l_shape` block, find balanced sub-expressions
occurring more than once, hoist each to the outermost loop level whose variable it does not mention).
Naming a value cannot change it, so residuals must stay bit-identical — they did, on both residual and
Jacobian, which is what makes the numbers trustworthy:

| case | hoists | C size | gcc `-O3` | tccbox |
|---|---|---|---|---|
| hyperelastic solid, 3D | 135 | −12% | **−10.8%** | **−24.9%** |
| Navier-Stokes 2D (control) | 1 | −0% | −3.9% | +1.3% |

Insensitive to the size threshold (15/20/30/45/80 characters give −10.5% to −11.3%): the win comes from
a handful of large repeated groups, not from aggressive micro-CSE. The Navier-Stokes control confirms
the shape of the result — a polynomial weak form has almost nothing to factor.

**A real GiNaC pass was then written, validated and measured at −1.9%**, against the −10.6% the textual
model achieves on the same element with the same compiler. It was reverted. It sat after each
`GiNaC::diff` and before printing — the cheap place, because nothing differentiates a Jacobian
expression again, so a named temporary can be a plain symbol needing no new node type, no derivative
rule, and no interaction with the `subexpression()` cache or the Hessian. It was correct (residuals
exactly unchanged, Jacobian moved ~3e-14 relative from `subs` reassociating, 245 tests passing). It
simply was not fast:

| | distinct temporaries | generated C | elemental res+jac |
|---|---|---|---|
| baseline | — | 320 420 B | 0.3046 s |
| textual model | 45 | 283 208 B | 0.2719 s (**−10.6%**) |
| GiNaC pass, `min_savings=96` (best) | 27 | 266 763 B | 0.2983 s (**−1.9%**) |
| GiNaC pass, `min_savings=32` | 117 | 294 929 B | 0.3048 s (+0.2%) |
| GiNaC pass, maximal coverage | 504 | 292 853 B | 0.3070 s (+0.9%) |

Threshold sweeps across 2–512 find no better regime; everything more aggressive is neutral or worse,
naming something cheap costing a store and a live register. Selecting on cost alone rather than on
saved work was one real mistake found along the way (243 temporaries, *slower* than doing nothing, and
11% slower under tcc for want of a register allocator); the criterion is `(occurrences - 1) x cost`.

Three hypotheses were tested; two are dead and the gap survives all three.

* **`subs` re-canonicalisation — disproven.** Every `_cse_N` is used exactly 16 times in the emitted C,
  precisely the occurrence count the pass predicted.
* **The cost model — genuinely broken, and fixing it changed nothing.** Summing interior-node weights
  and treating leaves as free costed a 10-operand node at 31 and a two-operand sum at 3, so selection
  systematically preferred the small one. But a leaf is a shape expansion or a `subexpr_N` reference,
  i.e. a memory load, and the loads dominate. Charging 1 per leaf moved the sweet spot from 32 to 96
  and left the result exactly at −1.9%.
* **Scope — real, quantified, not sufficient.** The pass runs per
  `write_generic_RJM_jacobian_contribution` call, so it sees 16 occurrences of a subtree that occurs 48
  times across the block's three sibling `l_shape` loops. That explains a factor of three in the
  counters and nothing in the arithmetic.

What the evidence points at is **coverage**. Bytes of C inside the Jacobian entry statements versus
bytes moved into temporaries:

| | entry statements | temporaries | total |
|---|---|---|---|
| baseline | 152 049 B | 0 | 152 049 B |
| GiNaC pass, best setting | 97 213 B | 27 707 B | 124 920 B |
| GiNaC pass, maximal coverage | 63 262 B | 54 164 B | 117 426 B |
| textual model (45 temporaries) | **18 162 B** | 94 785 B | 112 947 B |

The model collapses the entries by a factor of eight; the GiNaC pass floors at 63 kB no matter how low
the threshold goes. They are not selecting the same things at different thresholds — there is structure
the subtree-based pass cannot reach at all. The leading explanation, untested: GiNaC's `add` and `mul`
are **flat n-ary** nodes, so a repeated *partial* sum or product — a subset of one node's operands — is
not a subtree and can never be a CSE candidate. The textual model, working on printed and explicitly
parenthesised output, factors exactly such groupings. Matching it would mean frequent-subset mining over
operand multisets: a materially harder algorithm, and a different piece of work.

**Why not to return to it:** this is work the C compiler is built to do and does well. `-O3` folds 165
of 210 `pow(x,2)` calls unaided, collapses 3.7 MB of generated C into a 118 kB shared library, and on
the transcendental case needs only permission (`-fno-math-errno`) to CSE and hoist libm calls out of the
innermost loop. GCC's GCSE/PRE see the whole basic block, including everything pyoomph's DAG cannot
express, and pay no cost at code-generation time. The entry condition for a return should be evidence
that `-O3` has actually given up — a production element where compile time or `pow@plt` counts show the
optimizer bailing out on 350 kB statements — not a general belief that emitted redundancy must cost
something.

That verdict is about *this* pass. It says nothing about §9, which is a different kind of thing: the
symbolic derivative cache is the one place code generation demonstrably beats the compiler, because a C
compiler cannot differentiate anything.

### 8.4 Ranked conclusions

| item | gcc `-O3` | tccbox | verdict |
|---|---|---|---|
| CSE with loop-level placement in the emitter | −11% achievable, **−1.9% achieved** | −25% achievable | tried, **reverted** (§8.3) |
| `pow(x,2)` → `x*x` in the C printer | 0 | −53% | tried, **reverted** (§8.1) |
| `-fno-math-errno` in the default flags | −86% on transcendental forms, 0 elsewhere | n/a | tried, **reverted** (§7) |
| **`subexpression()` in user code** | **−91% on transcendental forms** | — | **the recommendation** |
| drop the `IGNORED RESIDUAL` comment payload | 0 | 0 | **done** (§8.5) — compile time only, but 66% of one 5.8 MB file |
| remove the dead `d_subexpr_*` stores | ~0 | ~0 | not attempted; symptom of §9.2 |
| automatic `subexpression()` injection | unknown | unknown | not attempted; narrower than it looks |
| **`subexpression()` reaching the Hessian** | **−91% code, −97% Hessian assembly** | — | **done (§9.3)** |
| index-aware coordinate derivative cache | superseded | superseded | **not built** — §9.4 removes the cost instead of memoising it |
| **hoist the Jacobian entry's coefficients out of the trial loop** | **−63.6% on a 3D moving-mesh solid**, ~0 on static meshes | — | **done (§9.4.5)** |
| **split the assembly function by its flag** | **−7.8% residual-only**, and it is what makes the row above free of a residual regression | n/a (no `always_inline`) | **done (§9.4.6)**, at +114% `.so` and +116% compile |

On automatic injection: marker injection *before* differentiation is the obvious idea and the riskier
half — it changes what gets differentiated symbolically, it is blocked from anything containing a test
function (`src/codegen.cpp` throws), and on `coordinates_as_dofs` codes the escape hatch of §9.2 means
it can make the Jacobian larger, not smaller. Post-differentiation CSE needs no new GiNaC machinery at
all; that is the half that was built, and it did not pay off.

### 8.5 Dropping the `IGNORED RESIDUAL` comment payload — the one change kept

`set_ignore_residual_assembly()` switches off residual assembly for a contribution that exists only to
supply Jacobian and mass-matrix entries — the azimuthal and Cartesian normal-mode stability
contributions and the pitchfork mass matrix all do this. The entry was emitted as `0` followed by the
whole printed residual wrapped in a `/* IGNORED RESIDUAL ... */` comment. On an archived production
element that comment was **66.4% of a 5.81 MB file** (5 806 110 → 1 950 436 B over 43 ignored
residuals), handed to the C compiler on every JIT cache miss for no purpose.

`print_residual_entry` now prints the expression into a stream that is thrown away and emits only
`0 /* IGNORED RESIDUAL */`. The print is *kept* deliberately, because it is not side-effect free:
`GiNaCGlobalParameterWrapper::print` allocates this code's local slot for a global parameter on first
encounter, and that order decides both which `dResidual<N>dParameter_<i>` routines get written and how
`functable->global_parameters` is laid out. Skipping the print would renumber those, and would drop
entirely a parameter occurring only in a field-independent term of an ignored residual — such a term
survives in no Jacobian derivative either, so nothing else would ever register it.

Verified on the Rayleigh-Bénard azimuthal stability tutorial (39 ignored residuals): all eight generated
files byte-identical after normalising the comment down to the short marker; the compiled library the
same size with its **disassembly differing in exactly two instructions**, both `mov $imm,%edx` carrying
a `__LINE__` value for `__assert_fail`, plus the embedded source path; the element shrinking
291 086 → 262 853 B and compiling in 4.83 s instead of 5.09 s.

## 9. The subexpression machinery beyond the Jacobian

Three leads, recorded here when they were still open. §9.1 and §9.3 are now done — §9.1 turned out not
to be the bug it looked like, §9.3 was worth −91% on the generated Hessian. §9.2 named the moving-mesh
coordinate derivative as the largest remaining lever and left it unmeasured; **§9.4 settles it**, and
supersedes the fix §9.2 proposed — the cost it wanted to memoise per `l_shape` is removed instead, by
hoisting it out of the loop entirely (§9.4.5) and by splitting the assembly function so each mode
compiles only the code it runs (§9.4.6). §9.4.7 lists what is still open.

### 9.1 A swallowed `throw_runtime_error` — resolved, and it was not a bug

`GiNaCSubExpression::derivative` builds its result in two stages. First it walks the subexpression's
`req_fields` and accumulates the cached chain rule, `d_subexpr_N_d_<field> * <derived shape>`, into
`res`, setting `found`. Then, on a `coordinates_as_dofs` code, if the symbol being differentiated by is
a nodal coordinate, it takes the escape hatch and computes `deriv = diff(expr, s)` directly.

If that direct derivative was non-zero **and** `found` was already true, the code assembled a detailed
message — *"subexpression derivative wrto ... is non-zero, but we already have a contribution before"*,
printing `deriv` as "should be 0" — and then did not throw it (the `throw_runtime_error` was commented
out). It returned `deriv`, discarding `res`, which read like a silently dropped Jacobian term.

It is not one. `deriv = diff(body, s)` is the **complete** derivative: `GiNaCShapeExpansion::derivative`
already converts every position expansion inside the body to its derived form, so `res` is the same
contribution re-expressed through the cache. `res + deriv` would double-count; returning `deriv` alone
was correct all along, and the emitted `d_subexpr_N_d_<coordinate>` variables are then dead, which is
exactly what §8.2 counts.

The branch is also reachable, so a throw would have been wrong: `SubExpressionsToStructs` erases position
expansions from `req_fields` for this code, its bulk and its bulk's bulk, but **not** for the opposite
interface code — which the `is_coordinate` test does cover.

The message and the dead throw are gone; the reasoning is now in the comment.

### 9.2 Derivatives with respect to the moving-mesh coordinates

The escape hatch exists because a coordinate derivative of a subexpression genuinely cannot be a single
precomputed scalar: terms like `d(dpsi/dx * u)/dX^l_j` depend on the `l_shape` loop index, which the
cached `d_subexpr_N_d_<field>` variables do not carry. `SubExpressionsToStructs` therefore erases every
position shape expansion from the subexpression's required fields up front, so no cached coordinate
derivative is even emitted.

The consequence is that on a moving mesh `subexpression()` delivers only half of what it delivers
elsewhere: the value is hoisted, the derivative is re-expanded inline at every use site. That is exactly
the configuration in which the Jacobian statements reach hundreds of kilobytes — the archived 3.7 MB
hyperelastic element has nine single C statements of ~348 000 characters each, and none of its 153
cached derivatives is read.

Two directions, in increasing order of ambition:

- **Unwrap instead of half-wrapping.** If a subexpression's only remaining consumers on a moving mesh
  are coordinate derivatives, wrapping it may be a net loss: the value is hoisted once per integration
  point, but the derivative is inlined `nnode²` times per point *and* is now the derivative of a tree
  the user chose to make large. Cheap to measure first: take a moving-mesh element and compare generated
  size with the `subexpression()` calls removed.
- **Make the cache index-aware.** `d_subexpr_N_d_coordinate_x[l_shape]`, filled once per integration
  point in its own loop, would restore the chain rule for coordinates. Real work: the fill loop, the
  interaction with `is_derived_other_index`, and the Hessian's second index all have to be thought
  through.

Any change here changes which branch of `GiNaCSubExpression::derivative` fires. §9.1 has been settled,
so that is no longer a blocker; the Hessian's second index (§9.3) now runs through the same escape
hatch and has to be kept in view.

### 9.3 Subexpressions in the Hessian — done

They used to be stripped outright: `write_generic_Hessian` ran `RemoveSubexpressionsByIndentity` over
the residual before differentiating twice, and `write_code_subexpressions` threw
`"Hessian subexpressions!"` if any survived. So on a problem with `generate_hessian` — every azimuthal
or Cartesian normal-mode stability analysis with an analytic Hessian — `subexpression()` bought nothing
at all in what is routinely the *largest* generated function.

There is **no** `d2_subexpr_N_d_f_d_g` cache, and none is needed. The `__in_hessian` branch of
`GiNaCSubExpression::derivative` wraps `d(body)/d(field)` in a **nested** subexpression for the outer
index; that nested subexpression's own first-derivative cache, filled by `write_code_subexpressions`
like any other, *is* the second derivative. The list is in dependency order for free, because
`SubExpressionsToStructs` maps a body before pushing its parent.

Four things had to be fixed to switch it on:

- The nested body was wrapped without `__deriv_subexpression_wrto` and without
  `DerivedShapeExpansionsToUnity`, so the derived shape expansion stayed *inside* a value hoisted above
  the shape loop (printing as `psi[l_shape]` where `l_shape` does not exist) and every `req_fields` entry
  of the same field added the whole sum again. Both now mirror the non-Hessian fill. The saved pointer is
  restored rather than nulled: the branch re-enters itself through nested subexpressions.
- The fill's own `diff` ran with `__in_hessian` still set, so it appended new subexpressions *after* the
  list was snapshotted and the declarations emitted. It is cleared for the fill, which is also what the
  chain rule wants — read the nested cache, do not re-wrap.
- The fill used `get_derive_jacobian_by_expansion_mode()`. Every `d_subexpr_*` the Hessian reads is a
  *second*-index derivative (the outer index never touches the cache), and the second index is
  differentiated under `get_derive_hessian_by_expansion_mode()`. This is load-bearing, not cosmetic:
  with the Jacobian mode, the augmented Jacobian of an azimuthal `m=1` tracker came out wrong by 8.7e-5
  relative — small enough to pass a loose tolerance and still ruin Newton convergence.
- The shape-expansion harvest that told `write_spatial_interpolation` what to interpolate sat inside the
  `(f, f2)` loop, where the `assemble_hessian_by_symmetry` `continue` skips it and the last pair's
  subexpressions are never seen. `write_generic_Hessian` now sweeps the finished list once. A miss here
  is an undeclared identifier in the emitted C, not a wrong number.

Measured on a three-species transcendental element (`exp`/`log`/rational activity law shared by every
equation), same script, only the C++ differing:

| | before | after |
|---|---|---|
| `HessianVectorProduct0` | 809 029 B | 72 375 B (**−91%**) |
| `_assemble_hessian_tensor` | 0.1958 s | 0.0054 s (**−97%**) |
| `ResidualAndJacobian0` | 24 528 B | unchanged |

Correctness was gated on the analytic Hessian tensor and the augmented Jacobian matching a reference
built from the *identical* residual written without `subexpression()`, which never touches this
machinery: agreement to 1e-16 relative on a plain nonlinearity, on `partial_t` inside the wrapper (the
mass-matrix Hessian), on an axisymmetric `m=1` azimuthal tracker, and on a `coordinates_as_dofs` mesh
with `grad(u)` inside the wrapper.

It is a trade, not a free win, and it goes the wrong way on cheap bodies. Hoisting costs a declaration
and a fill per (subexpression, field) pair whether or not the inlined text it replaces was large.
`HessianVectorProduct` sizes against the same residual written without the wrapper:

| wrapped body | before | after |
|---|---|---|
| 3-species `exp`/`log` activity law | 809 029 B | 72 375 B |
| `exp(u)` | 4 497 B | 4 676 B |
| `exp(u)*u`, azimuthal `m=1` | 5 228 B | 5 683 B |
| SUPG `tau` on a **linear** residual | 10 045 B | 12 920 B |

The last row is the worst case: a linear residual has an identically zero Hessian, so every hoisted
scalar there is dead. This is the same trade `subexpression()` already makes in the Jacobian and it is
the user's explicit choice to wrap — but the Hessian used to be exempt from it, and is not any more.

Still open, and not made worse: on a moving mesh §9.2 still applies inside the Hessian — the coordinate
index takes the escape hatch and inlines. And the *original* subexpression's `d_subexpr_*` are now
often dead in the Hessian, since the outer index goes through the nested copy; that is §8.2's ratio
again (once per integration point against `nnode²` entries), except that an unreferenced `exp()` is not
free to the compiler without `-fno-math-errno`. Neither was measurable against the numbers above.

## 9.4 The moving-mesh shape sensitivities are closed forms, and the entries are exactly linear

§9.2 named the coordinate derivative "the largest remaining lever" and left it unmeasured. It is
measured now, and the lever turns out to be a different one from the cache it proposed.

### 9.4.1 The identities

Everything pyoomph fills for a moving mesh is a product of quantities the shape buffer already holds.
With `D` the Eulerian (on an interface: surface) gradient and `Psi` the geometry space's shapes,

```
d( D_i psi_l ) / dX^q_j  =  - (D_j psi_l)(D_i Psi_q)  +  N_ij ( D psi_l . D Psi_q )
d( dx )       / dX^q_j  =  dx * (D_j Psi_q)
d( n_i )      / dX^q_j  =  - n_j (D_i Psi_q)
```

`N = I - P` projects onto the normal space and is identically zero on a bulk element. In generated-code
names the first is `d_dx_shape_dcoord[S][l][i][q][j] == -dx_shapes[0][S][l][j]*dx_shape_Pos[0][q][i]`
plus the `N` term, so that whole rank-4 array is a rank-1 outer product; the second is already literally
what `src/elements_shapeinfo.cpp:106` computes, which is why the moving-measure family of Jacobian terms
is exactly the residual integrand times `dx_shape_Pos`; and the `COORDDIFF` pre-pass of
`write_spatial_interpolation` is the first identity contracted with nodal values.

Verified by `PYOOMPH_PARANOID_ALE_IDENTITY` (§11), an opt-in read-only check that rebuilds each filled
entry from closed form: **~265 million comparisons, zero violations, worst relative deviation 8.5e-13**,
over bulk 1D/2D/3D, codim-1 interfaces in 2D and 3D, a codim-2 line in 3D, C1/C2/C2TB/C1TB and `_DL`,
axisymmetric, deformed and undeformed, and including 249M from the 3D contact-line tutorial
`docs/source/tutorial/ale/spread/droplet_spread_3d.py`.

Two things that check will tell you, learned the hard way. An undeformed axis-aligned mesh satisfies the
identity trivially - the metric is diagonal and constant - so it proves nothing; deform first. And a
**collapsed** interface element is invisible to `set_detect_inverted_elements`, which only checks the
sign of a *square* mapping, while an interface's `J = sqrt(det g_ab)` is non-negative by construction:
such an element yields `J = 0`, infinite `g^{ab}`, and silently NaNs every residual on that interface.
The check reports it as `DEGENERATE MAPPING` with the offending node positions.

### 9.4.2 What follows: every Jacobian entry is linear in the trial basis

Each geometric primitive's sensitivity is linear in `{psi_q, D_i psi_q}`, so the chain rule forces

```
dR_l/dX^q_j = A_j(l_test) * shape_Pos[q] + sum_i B_ji(l_test) * dx_shape_Pos[0][q][i]
```

with `A`, `B` independent of `q`. The generator prints one flat expression *inside* the `l_shape` loop,
so those coefficients are recomputed `nnode` times per (integration point, `l_test`). Unlike §8.3's
subtree CSE, hoisting them is exact rather than heuristic - and it was confirmed on the emitted C:
reconstructing every entry from its `K` basis evaluations reproduces the assembled Jacobian to 2.5e-12
(`ale2d`) and 3.0e-9 (`solid3d`, where extracting coefficients by basis evaluation loses cancellation
structure a symbolic emitter would keep).

### 9.4.3 Measured, and it depends entirely on `nnode/K`

Two new `tests/benchmarks/bench_assembly.py` cases, both moving-mesh, timed in-process with the arms
interleaved at the process level. `zero` replaces every position-dof Jacobian entry with a constant
(wrong results; it bounds what any of this could win). `hoist` rewrites each entry into the exactly
equivalent hoisted form. `nocdiff` deletes the `COORDDIFF` accumulation.

| case | ndof | elemental res | res+jac | `zero` | `hoist` | `nocdiff` |
|---|---|---|---|---|---|---|
| `solid3d` (3D hyperelastic, C2, nnode 27) | 24 336 | 0.083 s | 4.575 s | 0.896 s (**-80.4%**) | 1.800 s (**-60.7%**) | n/a |
| `ale2d` (2D free-surface NS, C2, nnode 9) | 9 695 | 0.0071 s | 0.0403 s | 0.0297 s (**-26.4%**) | 0.0404 s (**+0.2%**) | 0.0395 s (-2.2%) |

Atoms per entry: 3-4 for `solid3d` against `nnode` 27; up to 7 for `ale2d` against `nnode` 9. **That
ratio is the whole result.** The win is roughly `1 - K/nnode` of the position-entry share, so a
high-order 3D element with few atoms wins enormously and a low-order 2D element with several coupled
fields wins nothing at all - the same shape of answer as §8.3's Navier-Stokes control, for the same
reason.

`solid3d`'s `hoist` number is pessimistic: the model hands the compiler `K` copies of each entry text
(320 kB of C becomes 508 kB) and its residual-only arm slows from 0.083 s to 0.52 s, i.e. ~0.44 s of
overhead that has nothing to do with the Jacobian entries. A real emitter prints each coefficient once -
they partition the entry, so the C would *shrink* - which puts the achievable figure between -61% and
the -80% ceiling.

### 9.4.4 What this says about the staged plan

* `solid3d` never touches the identities at all. Its position Jacobian is expressed in `dX_shape_Pos`,
  the **Lagrangian** shape derivative, which does not move with the mesh; the atoms are already minimal.
  So the coefficient hoisting is the entire win there, and the identity substitution is irrelevant.
* `ale2d` is the opposite: it is full of `d_dx_shape_dcoord`, `COORDDIFF`, `int_pt_weights_d_coords` and
  `d_normal_dcoord`, and the identities would collapse those to `dx_shape_Pos`. But the direct runtime
  saving is small - the `COORDDIFF` pre-pass is worth 2.2% - so their value is in reducing `K`, and in
  2D that only takes `K` from 7 to about 5 against `nnode` 9.
* Therefore: **the coefficient hoisting is worth building and the identity substitution is not, on
  runtime grounds alone.** The identities remain worth having for what they remove structurally (the
  rank-4 and rank-6 sensitivity arrays and the `COORDDIFF` pre-pass, i.e. memory and fill work that
  grows as `nnode^2` and `nnode^3`), and that case should be made on memory, not on these numbers.
* This supersedes §9.2's proposal. An index-aware `d_subexpr_N_d_coordinate_x[l_shape]` cache addresses
  the same cost by memoising per `l_shape`; hoisting removes the per-`l_shape` work entirely and needs
  no new cache.

### 9.4.5 Implemented: the coefficients are hoisted out of the trial loop

`FiniteElementSpace::write_generic_RJM_jacobian_contribution` now buffers the loop body, splits each
Jacobian (and mass-matrix) entry into `sum_k coeff_k * atom_k` over the `l_shape`-indexed atoms, and
emits `const double _jcN = <coeff_k>;` **before** the `for (l_shape...)` header. The split is
structural, not `expand()`-based - an add distributes, a mul must have exactly one factor containing
atoms, and anything else fails - because expanding a 350 kB entry is precisely the cost being removed.
Any failure returns the empty string and the entry is printed exactly as before, so a case the
mathematics did not anticipate costs performance, never correctness.

**Coefficients below `PYOOMPH_JACOBIAN_HOIST_MIN` (default 32) expression nodes are left inline.** This
is not a micro-optimisation, it is the difference between two regimes. Naming a coefficient makes it one
more value live across the whole trial loop, and the register allocator works on the whole generated
function - including the residual-only path, which never enters the `if (flag)` block the coefficients
live in and yet gets measurably slower. On the 3D solid element:

| `PYOOMPH_JACOBIAN_HOIST_MIN` | elemental residual | residual+jacobian |
|---|---|---|
| hoisting off | 0.0832 s | 4.5792 s |
| 1 | 0.4348 s | 1.6820 s (-63.2%) |
| 8 | 0.2981 s | 1.6828 s (-63.2%) |
| **32 (default)** | **0.2985 s** | **1.6652 s (-63.6%)** |
| 48 | 0.2997 s | 1.6862 s (-63.2%) |
| 64 | 0.0867 s | 2.4161 s (-47.2%) |
| 96 | 0.0872 s | 2.4227 s (-47.1%) |

The transition between 48 and 64 is a single coefficient: hoisting it is worth 16 points of Jacobian
time and costs 0.21 s of residual-only time. Per Newton step (one residual evaluation plus one
residual+Jacobian) 32 still wins clearly, 4.66 s -> 1.96 s, i.e. **-58%**; a workload doing many more
residual evaluations than Jacobians would prefer 64, which is what the switch is for.

Measured, all arms interleaved in one binary via `PYOOMPH_DISABLE_JACOBIAN_HOIST`:

| case | elemental res+jac, off | on | generated C | `.so` |
|---|---|---|---|---|
| `solid3d` | 4.5792 s | 1.6652 s (**-63.6%**) | 320 534 -> 359 634 B (+12%) | 61 272 -> 73 560 B |
| `ale2d` | 0.0404 s | 0.0388 s (**-3.8%**) | 55 308 -> 55 232 B | unchanged |
| `ns2d` / `ns3d` / `coupled` / `poisson3d` | | -0.1% / -0.1% / -0.5% / -0.2% | | |

The static-mesh cases confirm the shape of the result: this applies to every Jacobian block, not only
the moving-mesh ones, and where the entry is already cheap it is neutral rather than harmful. `ale2d`
gains 3.8% where §9.4.3's textual model predicted +0.2%, exactly as that section warned - the model
evaluated the whole entry `K` times, whereas the emitter prints coefficients that partition it.

Correctness: the assembled Jacobian is unchanged to **2.3e-12** relative on `ale2d`; 258 targeted tests
pass at the final setting (206 + 52, the latter being `test_tensor_index_conventions.py` and `test_bifurcation_tracker_jacobians.py`), including every azimuthal and normal-mode path. `solid3d` is not a conditioning witness -
its torsioned unsolved state has `max|A| = 1.9e13` and individual entries move by up to 5e-5 relative
through pure reassociation.

### 9.4.6 Splitting the assembly function by its flag

§9.4.5 left a wart: hoisting made the **residual-only** path 3.6x slower even though it never enters the
`if (flag)` block the coefficients live in. The cause is that there was only ever one generated
function. `flag` is a runtime parameter, so the Jacobian code sits in the same body as the residual
code, and both the register allocator and the instruction cache pay for it whether or not it runs.

So the body is now emitted once as `PYOOMPH_RJM_IMPL <name>_impl(..., const unsigned flag)` -
`static inline __attribute__((always_inline))` on GCC - behind a three-line entry point keeping the
registered name:

```c
static void ResidualAndJacobian0(..., unsigned flag)
{
  if (flag == 0) ResidualAndJacobian0_impl(..., 0);
  else if (flag == 1) ResidualAndJacobian0_impl(..., 1);
  else ResidualAndJacobian0_impl(..., 2);
}
```

`flag` is then a compile-time constant in each copy, so `if (flag)` and `if (flag == 2)` fold away and
each mode gets a body containing only the code it runs. Only 0, 1 and 2 are ever passed
(`src/elements_assembly.cpp:177-190`). Nothing outside the generated file changes: same name, same
function table, same ABI, same C++ callers. A compiler without the attribute (tcc) gets one body with a
runtime flag, exactly as before - correct, just not specialised.

**It is bitwise identical**: the arithmetic is untouched, only which copy of it exists. Verified on
`ale2d` with hoisting off, every Jacobian entry equal bit for bit.

Measured four ways in one binary (`PYOOMPH_DISABLE_RJM_SPLIT`, `PYOOMPH_DISABLE_JACOBIAN_HOIST`):

| `solid3d` | residual | res+jac |
|---|---|---|
| neither | 0.0825 s | 4.5569 s |
| split only | 0.0761 s (**-7.8%**) | 4.5448 s (-0.3%) |
| hoist only | 0.2906 s (**+252%**) | 1.6560 s (-63.7%) |
| **both** | **0.0761 s (-7.8%)** | **1.6848 s (-63.0%)** |

The split cures the hoisting's residual regression completely and costs 0.7 points of Jacobian time.
Per Newton step (one residual evaluation plus one residual+Jacobian) that is 4.639 s -> 1.761 s,
**-62%**, against -58% for hoisting alone. On `ale2d` all four arms are within ~1% of each other, so
nothing there argues either way.

**What it costs.** The specialised bodies are real code:

| | generated C | `.so` | gcc `-O3 -march=native` |
|---|---|---|---|
| neither | 320 534 B | 61 272 B | 2.02 s |
| both | 360 183 B (+12%) | 130 960 B (**+114%**) | 4.36 s (**+116%**) |

The C grows only 12% (the dispatcher is 549 B; the rest is §9.4.5's coefficients) but the binary and the
compile time roughly double, because three bodies get optimised instead of one. That is paid once per
JIT cache miss, and the cache key already covers it - but it is the reason to keep the switch.

Not yet established: whether the **third** specialisation earns its share. `solid3d` has no mass matrix
at all, and on `ale2d` the mass-matrix arm tracks the Jacobian arm to within noise. A two-way split
(`flag == 0` specialised, `flag != 0` left runtime) would halve the extra compile time and binary size
and is a one-line change to the dispatcher; it wants a case where mass-matrix assembly is actually hot,
i.e. an eigenproblem, before being decided. §9.4.7 item 4 is the cheaper half of the same idea: two of
the cases where the third body is provably dead can be decided at generation time without measuring
anything.

### 9.4.7 Outlook - what is open on this, in order

Both changes of §9.4.5 and §9.4.6 are in and measured, but they were built in that order and each
invalidates assumptions the other made. The list below is roughly in the order it should be worked
through, because the early items change what the later ones should be measured against.

**1. Re-check the hoist limit.** `PYOOMPH_JACOBIAN_HOIST_MIN` was set to 32 by the sweep in §9.4.5, and
that sweep was run *before* the function split existed. Its whole shape came from one effect: naming a
cheap coefficient slowed the residual-only path. §9.4.6 removed that effect entirely - the residual-only
body no longer contains the Jacobian code, so there is nothing for a coefficient to be live across. The
sweep should therefore be repeated on the split build, and the optimum is expected to move **down**
(hoist more, possibly everything), because the reason to hold back is gone. Note the two settings are
not independent: with the split off, 32 remains right.

**2. Inline directives beyond GCC - DONE.** `PYOOMPH_RJM_IMPL` now reads
`#if defined(_MSC_VER)` -> `static __forceinline void`, `#elif defined(__GNUC__) && !defined(__TINYC__)
&& defined(__OPTIMIZE__)` -> `always_inline`, else plain `static`. From the vendor documentation, since
only GCC can be tested here: MSVC has no `__attribute__` syntax and needs the `__forceinline` KEYWORD,
honoured under the `/O2` `ccompiler.py` passes but not under `/Ob0`, where it degrades silently with no
diagnostic; **clang-cl defines `_MSC_VER` and accepts `__forceinline`**, which is why `_MSC_VER` is
tested first; clang defines `__GNUC__` and takes the GCC branch, attempting the inline at any
optimisation level; and **GCC diagnoses a failed `always_inline` as a hard ERROR**, which on a JIT path
turns a missed optimisation into a failed compile - none of the documented causes (target-option
mismatch, definition unavailable, varargs, recursion) apply to a function defined immediately above its
only three callers in one translation unit. The `__OPTIMIZE__` term keeps a `PYOOMPH_DEBUG=1` build on
the single out-of-line body, which is worth 16% of debug compile time and is much easier to step
through; it is NOT true, as this file briefly claimed, that forcing it at -O0 triples the code - GCC
folds the constant `flag` branches while inlining even with no optimiser running, and the three copies
cost +5%. The original item text follows.

**2. Inline directives beyond GCC.** `PYOOMPH_RJM_IMPL` (`src/jitbridge.h`) currently reads
`#if defined(__GNUC__) && !defined(__TINYC__)`. Clang defines `__GNUC__` and accepts
`__attribute__((always_inline))`, so it *should* be covered - but that is an assumption, not a
measurement, and clang has its own inlining thresholds that an attribute does not always override.
MSVC is definitely **not** covered and silently falls back to `static`, i.e. no specialisation at all;
it needs `__forceinline`, and jitbridge.h already has `_MSC_VER` branches (lines 26, 1079) to follow.
Verify per compiler that the three bodies actually exist - `nm --size-sort` on the `.so` and the
residual-only timing are both sufficient tells, and a build where the attribute was quietly ignored
looks exactly like a build where the split does nothing.

**3. Hoisting in the Hessian - DONE, see §9.4.8 (-7.9%, and it corrected how the ceiling is measured).** `write_generic_Hessian_contribution` has the same structure with two
trial loops, `l_shape` and `l_shape2`, and a Hessian entry is *bilinear* in their atoms rather than
linear. So the same argument applies twice: coefficients independent of both indices hoist above both
loops, and coefficients depending only on `l_shape` hoist between them. The pay-off should be larger
than in the Jacobian because the inner work is `nnode^2` rather than `nnode` - but §9.3 warns that the
Hessian's second index runs through `GiNaCSubExpression::derivative`'s escape hatch, and
`is_l_shape_atom` currently classifies `is_derived_other_index` atoms together with the others, which
is right for detection and not obviously right for a two-level split. This is the largest remaining
item and deserves its own measurement rather than being assumed from §9.4.5's numbers.

**4. Do not specialise a mass matrix that cannot exist.** The third body is emitted unconditionally, and
it is the expensive one: §9.4.6's +114% binary and +116% compile time are three bodies where two would
often do. Two cases can be decided at generation time. If the residual contributes **no** mass-matrix
entry at all - already known, since `mark_mass_matrix_contribution_for_code` is called exactly when
`mass_part` is non-zero - then `flag == 2` and `flag == 1` produce identical code and the dispatcher
should fold them. And the **steady** routine (`ResidualAndJacobianSteady<N>`, emitted alongside the
unsteady one at `src/codegen.cpp:6895`) has no time derivatives by construction, so its mass-matrix
specialisation is dead weight whenever a separate steady routine is written at all. Both are cheap
tests and together they should recover most of the compile-time cost on the elements that do not need
it, without touching the ones that do.

**5 and 6. Per-element properties and the cache - DONE.** `FiniteElementCode` gained
`jacobian_hoist_min_cost` (-1: follow the global setting) and `split_rjm_by_flag`, bound through
nanobind and reachable from `EquationCompilationFlags`, with the environment switches acting as the
global default a per-element value overrides. Both are written into the debug info file next to
`coordinates_as_dofs=`, so a generated file can be traced back to the settings that produced it. They
need **no cache epoch**: the JIT key is the contents of the generated `.c` plus `jitbridge.h`
(`pyoomph/generic/ccompiler.py`), and both properties change that text - which is exactly the property a
future per-element setting must preserve, and why the comment saying so sits at the emission site.
Verified end to end on the emitted C rather than on the binding: default gives 3 dispatcher calls and 60
hoisted coefficients, `split_rjm_by_flag=False` gives 0 and 60, `jacobian_hoist_min_cost=999` gives 3
and 0.

Note the case for this weakened while it waited. It was written when the split cost +116% compile time
and +114% `.so`; item 4 brought that to +29% and +53%, so per-element control is now a diagnostic
convenience - bisecting a regression to one element - rather than a way to avoid a serious bill. The
original item text follows.

**5. Per-element properties, not just environment variables.** Hoisting and splitting are currently only
reachable through `PYOOMPH_JACOBIAN_HOIST_MIN`, `PYOOMPH_DISABLE_JACOBIAN_HOIST` and
`PYOOMPH_DISABLE_RJM_SPLIT`, which are process-wide. They should become `FiniteElementCode` properties
alongside `analytical_position_jacobian` (`src/codegen.hpp:1087`), exposed through nanobind and
reachable from `EquationCompilationFlags` (`pyoomph/equations/additional.py:162`), so a problem with one
huge element and several trivial ones can pay the compile time only where it buys something. The
environment variables should stay as the global default the properties start from - that is the pattern
`analytical_position_jacobian` already follows, and it keeps the A/B sweeps of §12 working.

**6. Get the settings into the JIT cache key - and check they already are.** The key is built from the
*contents* of the generated `.c` plus `jitbridge.h` (`pyoomph/generic/ccompiler.py:107-138`), and both
switches change that text, so today they are covered for free and no epoch is needed. Item 5 must not
break that: a per-element property that changes emission changes the text and stays covered, but a
property that changes anything *not* visible in the text would not be, and neither would a change to
`PYOOMPH_RJM_IMPL`'s definition if it ever moved out of `jitbridge.h`. The settings should also be
written into the debug info file next to `coordinates_as_dofs=` (`src/codegen.cpp:6435`), so that a
`.so` can be traced back to the settings that produced it.

**8. Can the COORDDIFF terms go away entirely? - MEASURED, and split in two: the pre-passes are worth
4.0% and are not worth doing, the rank-6 arrays are worth 24.4% and are. See §9.4.11.** The original
item text follows.

**8. Can the COORDDIFF terms go away entirely?** §9.4.4 concluded, on runtime grounds, that the identity
substitution was not worth building: the `COORDDIFF` pre-pass measured 2.2% on `ale2d` and that was the
whole direct saving. **That conclusion was reached before hoisting existed and should be revisited,
because hoisting gives the substitution a second, different pay-off.** Every distinct `l_shape`-indexed
atom in an entry is now one more hoisted coefficient, i.e. one more `const double _jcN = ...` in the
generated C; and §9.4.6 has just made generated code expensive (+116% compile time). Collapsing
`intrp_..._COORDDIFF_i_field[l_shape]`, `d_dx_shape_dcoord[...][l_shape][...]`,
`int_pt_weights_d_coords[j][l_shape]` and `d_normal_dcoord[i][l_shape][j]` into multiples of
`dx_shape_Pos[0][l_shape][i]` and `shape_Pos[l_shape]` takes `K` on `ale2d`'s bulk entries from 7 to
about 5 in 2D and, more to the point, makes several coefficients merge instead of standing separately.
So the question is no longer "is 2.2% worth it" but "does a smaller atom basis pay for itself in
generated code, compile time and hoisted-coefficient count". Measure `K` per entry, the number of `_jcN`
emitted, and the compile time - not only the runtime. The substitution itself belongs at the
`derivative()` sites (`GiNaCShapeExpansion::derivative` 11601/11648/11692,
`GiNaCTestFunction::derivative` 11862, `GiNaCNormalSymbol::derivative` 10901) so GiNaC sees a product of
two existing shape structures, and `set_ignore_dpsi_coord_diffs_in_jacobian` must keep working; the
interface case additionally needs the normal-projector term of §9.4.1, which is not a plain product.

**9. Which shape-info buffers are now dead weight?** Two independent questions, both in
`src/elements.cpp:839-890` and `:1085-1099`, and both worth asking with the identity check of §11
switched on. First, if item 8 lands, `d_dx_shape_dcoord` (`MAX_NODES x MAX_NODAL_DIM x MAX_NODES x
MAX_NODAL_DIM` per continuous space, plus the `_DL` twin), `d_normal_dcoord`, and above all the rank-6
`d2_dx2_shape_dcoord` / `d2_d2x2_shape_dcoord` are never read by the generated code any more, and both
their allocation and their per-integration-point fill (`elements_shapeinfo.cpp:1346-1384`, the
`require_dxdshape` and `require_hessian` blocks) can go. That is the memory argument §9.4.4 said would
have to be made on its own terms, and the rank-6 arrays are `(nnode*dim)^3` doubles per space per point,
so it is not a small one. Second, and independent of item 8: whether anything there is *already* filled and never read. **This
half is done** - see below.

#### 9a. The already-dead shape-buffer fields, and a warning about how to look for them

Method: take every field of `JITShapeInfo_t`, count the generated `.c` files on disk that reference it
(1171 of them, all untracked build artefacts - `grep -r` silently skips them, `find -exec grep` does
not), count references from the macros in `jitbridge.h`/`jitbridge_hang.h`, and count references from
`src/*.cpp`. 22 fields are referenced by no generated file and no macro.

**Only three of the 22 are actually dead**, and the other nineteen are the warning:

* `nodal_shapes` (one per continuous space) and `nodal_shape_DL` - allocated at
  `src/elements.cpp:860,868` at `MAX_NODES x MAX_NODES` each, **never written, never read, and the
  generator has no path that could emit them**. About 29 kB per shape buffer, allocated and freed for
  nothing. Their jitbridge comment - "In principle just delta_{i,j}" - reads like the note of someone
  who already suspected it.
* `opposite_node_index` - *is* written (`src/elements.hpp:1301`, copied out of the element's own member
  of the same name every time the opposite side is set up) and then read by nobody: the code that wants
  it uses the element member directly (`elements.hpp:1413`), not the shape-buffer copy.

The other nineteen are **live but simply absent from the corpus**, which is the trap. `elemsize_d_coords`,
`elemsize_d2_coords` and `elemsize_Cart_d2_coords` are emitted at `src/codegen.cpp:10974,10978`;
`d_dnormal_dx_dcoord` and `d2_dnormal_dx_d2coord` at `:11151,11155`; and the whole `_DL`/`_Pos`
sensitivity family comes out of `dcoord_shape_array`. Every one of them is built by *concatenation*, so
grepping the generator for the field name finds nothing, and none of them appears in the corpus because
no archived run happened to combine (say) an element-size symbol with a moving mesh. A field's absence
from generated code is evidence about the corpus, not about the field - the only sound test is whether
the generator has a path that can emit it.

Removing the three is safe but changes the `JITShapeInfo_t` layout, which every cached `.so` was compiled
against; that invalidates itself correctly, since the cache key includes the contents of `jitbridge.h`.

**7. Can the hoisted coefficients be simplified? - MEASURED, and no.** On the 3D solid element the 57
hoisted coefficients hold 182 880 bytes of right-hand side and contain **9** repeated balanced groups of
20 characters or more, worth 4 212 bytes - **2.3%**, by the same textual model that overstated §8.3's
real pass by a factor of five. The expected win is therefore well under 1%, against the -1.9% that got
that pass reverted, and the largest repeats are `pow(x, 2.0)` argument lists which §8.1 established GCC
folds unaided. There is a reason for the emptiness: **the hoist already partitioned the entry**, so
sibling coefficients share little by construction - the CSE opportunity was consumed by the thing that
created the coefficients. Not attempted. The original item text follows.

**7. Can the hoisted coefficients be simplified?** They are extracted structurally - `accumulate_linear_in`
multiplies out `factor * rest` as it descends - so nothing simplifies them afterwards, and sibling
coefficients of the same entry visibly share factors (`dx`, the test-function values, whole subexpression
references). Three things to try, cheapest first: run the existing `ccode_expression_mode` machinery
(`collect_common_factors`) over the coefficients only, where the expressions are now small enough that
it is affordable and the §8.3 objection about expanding 350 kB entries no longer applies; hoist factors
common to *several* coefficients of one entry into a further temporary; and check whether the
coefficients duplicate work already available as a `subexpression()` value. This is also the natural
place to revisit §8.3's conclusion, because that pass failed on entries that were flat and enormous,
and the coefficients are neither.

### 9.4.8 The Hessian: hoisted, -7.9%, and the ceiling measurement was wrong

A Hessian entry is a second derivative, hence **bilinear** in the two trial indices. Splitting it by the
atoms carrying `l_shape2` leaves coefficients that still depend on `l_shape`, and those are emitted
between the two loops - so `nnode*nnode2` evaluations of the whole entry become `nnode` evaluations plus
`nnode*nnode2` multiply-adds. Atoms carrying **both** indices join the inner set, because the entry is
linear in them too and their coefficients carry no index at all. It is the same linear splitter as
§9.4.5, applied with a class filter; `hoist_hessian_entry` is a dozen lines on top of it.

`l_shape_atom_class()` decides which index an atom carries. Two things there are easy to get wrong and
both were got wrong first:

* a **derived** `ShapeExpansion` with `nodal_coord_dir >= 0` prints as
  `d_dx_shape_dcoord[l_shape2][dir][l_shape][cdir]` - the trial function of one index differentiated by
  the nodal coordinate of the other - so it carries BOTH indices even though only the *first* coordinate
  direction is set. Classifying it by `nodal_coord_dir2` alone misses every bulk entry;
* the classes are BITS (1 = `l_shape`, 2 = `l_shape2`, 3 = both). `2 | 3` is `3`, which matches
  everything, so the mask pulled `l_shape`-only atoms into the substituted set and every bulk entry then
  failed the split as a "product of two atoms".

Neither was found by reasoning. `PYOOMPH_DEBUG_HOIST` - which reports why a split was declined, and dumps
the atoms with their classes - found both in two iterations, and is worth keeping for that reason.

**Measured** on `ale2d` with `analytic_hessian=True`, timing `Problem._assemble_hessian_tensor`,
interleaved:

| | Hessian assembly | inner-loop entry text | generated C | `.so` |
|---|---|---|---|---|
| hoisting off | 0.5788 s | 56 967 B (largest entry 5091 B) | 336 526 B | 256 544 B |
| **hoisting on** | **0.5328 s (-7.9%)** | **17 116 B (largest 973 B)** | 350 627 B (+4%) | **248 352 B** |

Correct to 3.0e-13 relative over 655 487 shared entries. The sparsity differs by 387 entries out of
655 744, all of magnitude ~2.8e-17 against `max|H| = 341` - a cancellation landing on exact zero in one
association order and on round-off in the other.

#### The `zero` arm measures more than the entries

The headroom experiment said **-72.8%**: replace every Hessian entry with `0.0` and assembly drops from
0.578 s to 0.157 s. The hoist then removed 70% of the inner-loop text and bought 7.9%. Both numbers are
right; the interpretation of the first was not. **Blanking an entry lets the compiler delete everything
that exists only to feed it** - the subexpression block, the interpolations, the `2ndCOORDDIFF`
pre-pass - so the `zero` arm bounds "the entries *and* their exclusive upstream", which here is an order
of magnitude larger than the entries.

This qualifies §9.4.3 as well, where the same method reported an 80.4% ceiling for the Jacobian. There
the hoist delivered -63% of it, so that ceiling was mostly real - but it was mostly real by luck of the
weak form, not by construction, and a future `zero`-style arm should be read as an upper bound on a
*superset* of what a hoist can reach.

**So the Hessian's cost is mostly not in its entries** - but the first version of this paragraph guessed
where it was instead, and guessed wrong. §9.4.9 measures it: the generated element code is 43% of Hessian
assembly (the entries being about a fifth of that), and the largest single item is not code generation at
all.

### 9.4.9 Where the Hessian time actually goes - and it is mostly not code generation

§9.4.8 left the wrong impression that ~92% of Hessian assembly is upstream of the entries. That was an
inference from one number ("the hoist moved 7.9%"), not a measurement, and it was wrong. Decomposing
`Problem::assemble_hessian_tensor` by skipping one stage at a time gives the whole budget, on `ale2d`
N=12. The switches used for it - `PYOOMPH_HESS_SKIP_ELEMENT`/`_SKIP_SCATTER`/`_SKIP_ACCUM`, each
short-circuiting one stage - were removed once they had produced this table, because they make the
Hessian silently wrong and that is not a thing to leave lying in an assembly routine; three `if`s in
`Problem::assemble_hessian_tensor` bring them back if the question is ever reopened:

| stage | seconds | share | generated code? |
|---|---|---|---|
| `SparseRank3Tensor::accumulate()` inserts | 0.284 | **52%** | no |
| the generated element code | ~0.235 | **43%** | yes |
| the `nvar^3` scatter loop | 0.021 | 4% | no |
| the two `nvar^3` dense buffers per element | ~0.005 | ~1% | no |

Two corrections to what reading the code suggested. The element code is **43%**, not 8% - the entries are
about a fifth of it, the rest being subexpressions, interpolations, the `2ndCOORDDIFF` pre-pass and loop
overhead. And the per-element buffer allocation, which looks alarming (`problem.cpp` allocates
`nvar x nvar^2` and `elements_assembly.cpp:565` allocates a second one purely to throw away the mass
Hessian), is **~1%** - not worth touching.

#### The one-line change that beat the whole hoist

`accumulate()` did two red-black-tree lookups per entry:

```cpp
if (data[i].count(index)) data[i][index] += val;   // lookup, then lookup again
else                      data[i][index] = val;
```

That is ~1.3 million traversals per assembly for 655 744 entries. `std::map<...,double>::operator[]`
value-initialises a missing entry to 0.0, so `data[i][index] += val;` is the same operation with half
the lookups and **bitwise identical** arithmetic - same additions in the same order.

Measured by re-running the same decomposition inside each build, so the comparison does not depend on
the two builds being otherwise comparable (and the `noaccum` arms agree to 0.3%, which is the check):

| | full | noaccum | inserts |
|---|---|---|---|
| double lookup | 0.5454 s | 0.2612 s | 0.2841 s |
| **single lookup** | **0.4813 s** | 0.2605 s | **0.2208 s (-22.3%)** |

**-11.8% of total Hessian assembly**, against -7.9% for the entire hoisting machinery of §9.4.8. Not 50%,
because the second lookup was cache-warm after the first.

### 9.4.10 Replacing the tensor's per-row maps with sorted vectors

The remaining inserts were `std::map` traffic: an ordered red-black tree with a heap allocation per node,
maintaining an ordering that nothing reads until the very end. `SparseRank3Tensor` is written once, in a
burst of ~10^6 `accumulate()` calls, and only afterwards walked in `(j,k)` order by
`finalize_for_vector_product()` and `get_entries()`.

So each row is now a flat `std::vector<Rank3Entry>`: `accumulate()` appends, and a row is **compacted**
- sorted by `(j,k)`, duplicates summed in place - when it has doubled since its last compaction, and
unconditionally by `compact_all()` before either reader runs. The doubling trigger is what keeps the
duplicates from becoming a memory regression: a row holds at most ~2x its compacted size, and the
amortised cost per contribution stays constant.

Two details that are easy to get wrong:

* `finalize_for_vector_product()` starts a new CSR column group on `entry.j > last_col`, which is only
  correct if the row is sorted by `(j,k)` with duplicates already merged - a property the map gave for
  free. Both readers therefore compact first.
* `data` and `compacted_size` are `mutable`, because `get_entries()` is `const` and must compact before
  it can report anything. That keeps the public API - which nanobind binds - unchanged.

Measured on `ale2d` N=12 with `analytic_hessian=True`, the insert figures taken WITHIN each build via
`PYOOMPH_HESS_SKIP_ACCUM` so they do not depend on the builds being otherwise comparable:

| storage | full | noaccum | inserts |
|---|---|---|---|
| map, `count()`-then-assign (original) | 0.5454 s | 0.2612 s | 0.2841 s |
| map, single lookup (§9.4.9) | 0.4813 s | 0.2605 s | 0.2208 s |
| **sorted vector** | **0.3542 s** | 0.2512 s | **0.1030 s** |

**Inserts -63.7%, total Hessian assembly -35.1%** against the original. The `noaccum` control drifted
3.6% between the map and vector builds, which is not accounted for by anything in the change (both
construct an empty container per row), so the cross-build totals are good to a few percent while the
within-build insert numbers are firm.

The tensor is unchanged: identical sparsity (655 617 entries, none unique to either side) and **every
significant entry bitwise identical**, with a largest absolute deviation of 3.0e-14 against
`max|H| = 341` confined to entries at or below the reporting threshold. `std::sort` is not stable, so
contributions to one `(j,k)` may be summed in a different order than they were accumulated - that is the
same last-bit freedom the generated code already has, and here it did not even reach the significant
entries.

Worth keeping in proportion: this is the largest single win in the Hessian path and it is not code
generation. Ranked against each other on the same case, the sparse-tensor storage is worth -35%, the
one-line double-lookup fix inside it -11.8%, and the entire symbolic hoisting machinery of §9.4.8 -7.9%.

### 9.4.11 Decomposing the element half: it is the rank-6 shape fills

§9.4.9 put 43% of Hessian assembly in "the generated element code". That was measured by skipping
`elem_pt->assemble_hessian_tensor()`, which skips the generated function *and* the
`fill_shape_buffer_for_point` callback it makes - so the 43% was never purely generated code. Splitting
it further, by deleting one fill loop at a time from the emitted C and leaving its consumers running
(so the arrays still exist and are still read, they just hold zeros - no dead-code cascade):

| removed | cost | share of Hessian assembly |
|---|---|---|
| the rank-6 Hessian shape fills (C++) | 0.0898 s | **24.4%** |
| the `2ndCOORDDIFF` pre-pass (generated C) | 0.0134 s | 3.6% |
| the first-order `COORDDIFF` pre-pass | 0.0014 s | 0.4% |

Against a 0.3676 s baseline (post-§9.4.10, so the inserts no longer dominate).

**Two arms of that experiment were confounded and are not in the table.** Zeroing the shape buffer
(`noshape`) or the entries (`zero`) makes every Hessian entry evaluate to zero, so
`fabs(hval) > Numerical_zero_for_sparse_assembly` fails and `accumulate()` is never called: those arms
silently drop the insert cost as well, which is why both landed at ~0.155 s and looked like -57%. The
rank-6 row above avoids this by zeroing *only* the rank-6 arrays, leaving the entries non-zero and the
inserts intact in both arms. A third arm, deleting the interpolation accumulation, could not be measured
at all: the harness solves before timing and a zeroed interpolation makes that Newton solve diverge.

#### What this says about §9.4.7 item 8

It splits the item in two, and reverses which half is worth doing.

The **`COORDDIFF` pre-passes are not worth eliminating**: 4.0% between them, which is §8.3-reverted
territory and nowhere near paying for rewriting five `derivative()` sites plus the interface
normal-projector term.

The **rank-6 arrays are**: 24.4%, the largest single item left in Hessian assembly now that the sparse
tensor has been dealt with. And the fix is not the one item 8 proposed. These are filled in C++
(`fill_shape_info_at_s`, the `require_hessian` blocks), so what is wanted is to stop computing and
storing `d2_dx2_shape_dcoord`/`d2_d2x2_shape_dcoord`/`d_d2x_shape_dcoord` at all - which the identities
of §9.4.1 permit, since a second nodal-coordinate derivative of a shape gradient is a sum of products of
first derivatives. That is a runtime change with a code-generation prerequisite, not a code-generation
change.

### 9.4.12 The rank-6 arrays, in closed form

Differentiating §9.4.1's identity a second time gives, on a bulk element,

```
d2( D_i psi_l ) / dX^q_j dX^r_k  =  (D_k psi_l)(D_j Psi_r)(D_i Psi_q) + (D_j psi_l)(D_k Psi_q)(D_i Psi_r)
```

two triple products of FIRST derivatives, symmetric under `(q,j) <-> (r,k)` as it has to be. Verified the
same way as the first-order identities: **5 668 704 comparisons, 0 violations, worst relative deviation
7.3e-12** on a real moving-mesh Hessian problem. (Looser than the ~1e-13 of the first-order checks, as
expected: it is a triple product of quantities that each already carry that error, compared against a
value built through the E-tensor chain.)

`fill_shape_info_at_s` now writes `d2_dx2_shape_dcoord` straight from that expression on bulk elements,
which makes the E-tensor chain **and its per-integration-point `RankSixTensor` heap allocation**
unnecessary - `D2X2_dshape` is only allocated when something still needs it. Interfaces keep the old
path: there the second derivative picks up normal-projector terms this closed form does not carry. That
derivation has since been done and validated - see §9.4.13 - but using it measured slower, so interfaces
still take the E-tensor path deliberately rather than for want of a formula. Under `PYOOMPH_PARANOID_ALE_IDENTITY` both paths are computed, so the
check keeps comparing two independent derivations instead of the closed form against itself.

| | Hessian assembly |
|---|---|
| E-tensor path | 0.3676 s |
| **closed form** | **0.2947 s (-19.8%)** |
| rank-6 fills skipped entirely | 0.2778 s |

The last row is the ceiling, because the array still has to be *written* even by the closed form, and the
closed form reaches 81% of it. What remains is ~0.017 s (4.6% of assembly) of pure write traffic;
removing that means the generated code must stop reading `d2_dx2_shape_dcoord` at all and evaluate the
products inline - a code-generation change for a fifth of the prize this runtime change took. Worth
knowing before anyone starts it.

#### The bug this shipped with first, and why the check did not catch it

The first version gated the whole `if (require_hessian)` block in
`fill_shape_info_at_s_dNodalPos_helper` on `D2X2_dshape` being allocated. That block spans **two
unrelated computations**: the E tensor, and `D_dshape_Dcoords`, from which `int_pt_weights_d2_coords` -
the second derivative of the integration MEASURE - is built. Skipping the second silently stopped
computing a quantity the Hessian needs whether or not the E tensor exists, and every moving-mesh Hessian
came out wrong by 0.3% to 34% against finite differences, with no crash and no diagnostic.

**The 5.7-million-comparison identity check passed throughout.** It had to: under
`PYOOMPH_PARANOID_ALE_IDENTITY` the E-tensor path is deliberately kept alive so the check compares two
independent derivations - which means the flag validates the formula while exercising the code path
production no longer takes. The one configuration that mattered was the one it never ran. The
first-order checks of §9.4.1 do not have this hazard because they only ever *read* arrays that are
filled either way; this change altered what gets filled.

What did catch it, in two minutes, was `tests/test_second_derivatives.py` - a finite-difference
comparison of the assembled Hessian. Validating an intermediate at machine precision is not a substitute
for validating the result.

An artefact worth recording too: the broken build measured -16.1% and the fixed one measures -19.8%,
i.e. doing strictly more work came out faster. The likeliest explanation is that the skipped fill left
`int_pt_weights_d2_coords` holding uninitialised values and the entries then did denormal arithmetic.
A "speedup" that arrives together with wrong numbers deserves the suspicion it got.

### 9.4.13 The interface second derivative: derived, validated, and deliberately not used

§9.4.12 left the codim>0 case open. Carrying the normal-projector through a second differentiation, using
`N·Dpsi = 0` and `P_ij = sum_q X^q_i (D_j Psi_q)`, which gives `dN_ij/dX^r_k = -[N_ik (D_j Psi_r) + N_jk (D_i Psi_r)]`:

```
d2( D_i psi_l ) / dX^q_j dX^r_k  =  (D_k psi_l)(D_j Psi_r)(D_i Psi_q) + (D_j psi_l)(D_k Psi_q)(D_i Psi_r)
    - N_jk (Dpsi_l . DPsi_r)(D_i Psi_q)  - N_ik (D_j psi_l)(DPsi_q . DPsi_r)
    - N_ik (D_j Psi_r)(Dpsi_l . DPsi_q)  - N_jk (D_i Psi_r)(Dpsi_l . DPsi_q)
    - N_ij (D_k psi_l)(DPsi_r . DPsi_q)  - N_ij (D_k Psi_q)(Dpsi_l . DPsi_r)
```

The first line is §9.4.12 exactly; the six N-terms vanish when `el_dim == n_dim`, so this is the general
form and the bulk expression is its specialisation. Validated against the E-tensor chain on an interface
(`el_dim 1 in 2D`, 7 776 comparisons, worst 8.4e-16) and re-confirmed on bulk in 2D and 3D
(`el_dim 3 in 3D`, **114 791 256 comparisons**, 0 violations, worst 3.7e-13).

**Using it on interfaces is a pessimisation, and structurally so.** Measured on `ale2d`:

| | Hessian assembly |
|---|---|
| E-tensor everywhere (pre-§9.4.12) | 0.3676 s |
| closed form on bulk only (shipped) | 0.2947 s |
| closed form everywhere, one general expression | 0.3149 s (+6.9%) |
| closed form everywhere, N-terms branched on `el_dim != n_dim` | 0.3056 s (+3.7%) |

The reason is that the two paths scale in *different* dimensions. The E-tensor's inner contraction costs
`O(el_dim)`, and `el_dim` is precisely what drops on an interface - a line in 2D contracts over one
direction. The closed form costs `O(n_dim)`, which does not drop. So exactly where the closed form takes
on its six extra N-terms, its competitor gets cheaper; on a bulk element the same trade runs the other
way and the closed form wins by 19.8%. `closed_form_d2` therefore stays `(el_dim == n_dim)`, now with a
comment recording the measurement rather than the absence of a derivation.

What the work is worth keeping for is coverage. The interface second-derivative fill ships, and until now
`PYOOMPH_PARANOID_ALE_IDENTITY` had nothing to compare it against - §9.4.12's check covered only the bulk
path. It is now checked against an independently derived closed form, which is the arrangement that
matters given how §9.4.12's bug got through.

#### A refactor tax worth naming

Sharing one `d2_shape_dcoord_closed_form` helper between the fills and the check initially cost **+3.5%
on bulk**, which looked like cross-build drift and was not. The helper took `with_normal_terms` as a
runtime `bool` computed from `el_dim != n_dim`; both are runtime values, so the compiler could not fold
the branch and the bulk path paid for six terms it never used. Since the fills only run when
`closed_form_d2` is already true, passing a literal `false` (and a static zero projector) lets the
inliner delete the N-terms outright: **0.2957 s, +0.3% against the 0.2947 s reference, three passes within
0.001 s.** Sharing the formula costs nothing; sharing it through an unfoldable flag cost 3.5%.

### 9.4.14 Specialising the hanging-node machinery away, per element

With adaptivity on, every entry of every element used to run the hanging-node macros
(`BEGIN_RESIDUAL_CONTINUOUS_SPACE`, `BEGIN_JACOBIAN_HANG`), because the hang/no-hang choice was made at
**generation** time from `ContinuousFiniteElementSpace::can_have_hanging_nodes()`, i.e. from
`with_adaptivity` alone. So a fully non-hanging element - the vast majority even on an adapted mesh -
paid, per entry, a `nummaster` load, two tests, a master loop, and an unconditional evaluation of the
contribution even for a pinned equation. The non-hanging macros test `local_eqn >= 0` *first*.

The runtime half of the fix was already in the tree and dead: `fill_hang_info_with_equations` returns
whether anything in the element really hangs, but the answer was overwritten with `has_hang = true;
// ASSUME ALWAYS HANGING!`, and the `ResidualAndJacobian_NoHang` function-table slot was registered
with the same function as its twin.

**Mechanism.** The same trick as §9.4.6, one level further: the hang macros gained a leading `HANGON`
argument, and the `_impl` a second constant parameter, so *one* emitted body becomes two entry points.

```c
#define BEGIN_RESIDUAL_CONTINUOUS_SPACE(HANGON, EQN, CONTRIB, HANGINFO, LINDEX)          \
  nummaster = ((HANGON) && HANGINFO[LINDEX].nummaster) ? HANGINFO[LINDEX].nummaster : 0u; \
  if (nummaster || (EQN) >= 0)                                                           \
  { const unsigned _nmaster = (nummaster ? nummaster : 1u);                              \
    _res_contrib = CONTRIB;                                                              \
    for (unsigned m = 0; m < _nmaster; m++) { ... } }
```

With `HANGON` folded to 0 the `&&` short-circuits, `HANGINFO` is never loaded, the guard collapses to
`(EQN) >= 0` **before** `CONTRIB`, the loop to one iteration and `hang_weight` to `1.0` - exactly the
shape of `BEGIN_JACOBIAN_NOHANG`, without emitting the body twice. With `HANGON` folded to 1 the
arithmetic is what it always was, plus the pinned-unknown skip, which is why §10's parked note about
that ordering is now only true of the hanging path.

`write_generic_RJM` appends `, const int pyoomph_hang_on` to the `_impl` signature and writes a second
dispatcher `<name>_NoHang` passing 0 where the normal one passes 1; a new
`ResidualAndJacobianSteady_NoHang` table slot mirrors the steady routine, because the steady slot is
what a stationary solve - the common case - actually calls. `elements_assembly.cpp` picks per element
from the fill's return value.

**The one rule that is not an optimisation choice.** For a space belonging to *another* code (a bulk or
opposite-side space read from an interface element) `HANGON` stays the literal 1, unconditionally.
`fill_hang_info_with_equations` reuses those same hang buffers as the local-equation **remap** channel:
it writes `nummaster=1`, `weight=1.0` and the remapped `local_eqn` for every dof, hanging or not
(`src/elements_hanging.cpp:1062-1099`). Folding `HANGON` to 0 there would ignore those entries and
scatter the element into the wrong rows. `FiniteElementSpace::get_hang_on_str` states the three-way
rule once, for exactly this reason.

**Predicate conservatism, deliberate.** `InterfaceElementBase`'s override of the fill returns true
whenever bulk or opposite shapes are required, so an interface element with bulk dependence always
takes the hanging entry point. That is over-conservative - most such elements hang nothing - but
narrowing it needs its own validation pass and buys little (interface elements are a small fraction of
the assembly). Left as an outlook item. The base predicate also covers the
`add_additional_dof_constraint` C1 reductions, which correctly force the hanging path independently of
adaptivity.

**Ceiling and result.** The ceiling was measured first, by compiling with `with_adaptivity=False` on
unrefined meshes - which produces exactly the code the dispatch selects - and it is what decided the
work was worth doing. In-process elemental assembly (`_benchmark_elemental_assembly`), interleaved
arms, one process per arm, min of three rounds:

| unrefined | metric | today (dispatch off) | dispatch on | ceiling | of the ceiling |
|---|---|---|---|---|---|
| `poisson2d` (ndof 14520) | residual | 0.02861 s | 0.02850 s (-0.4%) | 0.02857 s (-0.1%) | - |
| | res+jac | 0.03417 s | 0.03202 s (**-6.3%**) | 0.03196 s (-6.5%) | 97% |
| `ns2d` (ndof 32042) | residual | 0.03809 s | 0.03766 s (-1.1%) | 0.03747 s (-1.6%) | 69% |
| | res+jac | 0.06987 s | 0.05810 s (**-16.8%**) | 0.05716 s (-18.2%) | 93% |
| `ale2d` (ndof 9695) | residual | 0.00662 s | 0.00631 s (-4.7%) | 0.00631 s (-4.8%) | 99% |
| | res+jac | 0.03908 s | 0.03399 s (**-13.0%**) | 0.03381 s (-13.5%) | 96% |

The win is in the Jacobian, not the residual: a residual entry is one multiply-add, so the branch
overhead it saves is a smaller share of it.

On **genuinely refined** meshes (level 1 everywhere, level 2 near one boundary; 71% of the element
assemblies take the specialised entry point) the hanging path does not regress and the mixed population
gains about as much: `poisson2d` res+jac -6.4%, `ns2d` -17.9%, `ale2d` -14.2%.

**What it costs.** Unlike §9.4.6 this doubles only the *adaptive* routines, and nothing at all where
`with_adaptivity` is false (the split is off by construction there), nor in a routine whose spaces all
belong to other codes - the `_NoHang` entry point is only written if the body actually read
`pyoomph_hang_on`, otherwise the table slots alias as before.

| | generated C | `.so` | compile |
|---|---|---|---|
| `solid3d`, adaptive, split off | 378 251 B | 93 752 B | 8.64 s |
| `solid3d`, adaptive, split on | 379 067 B (+0.2%) | 142 952 B (**+52%**) | 10.10 s (+17%) |
| `solid3d`, non-adaptive | 377 194 B | 81 408 B | 8.2-8.5 s (unchanged either way) |
| `ns2d`, adaptive, split off | 138 699 B | 133 320 B | 2.11 s |
| `ns2d`, adaptive, split on | 139 451 B (+0.5%) | 141 560 B (+6.2%) | 2.34 s (+11%) |

That is well under §9.4.6's +114%/+116%, so the available further bound - routing `flag == 2` in the
`_NoHang` dispatcher to the hanging `_impl`, 5 bodies instead of 6 - was measured and deliberately not
applied.

#### Validating it: the same binary, two entry points

`PYOOMPH_DISABLE_NOHANG_DISPATCH` forces every element back through the hanging entry point without
regenerating or recompiling anything, so the two arms of a comparison differ in nothing else.
`PYOOMPH_REPORT_NOHANG_DISPATCH` counts which entry point each element assembly took and says so at
exit - a comparison in which the specialised path never engaged is two identical numbers and a
meaningless pass, the same failure mode the frozen-sparsity work had to guard against.

Seven cases, residual + Jacobian (+ mass matrix where `flag == 2` applies), compared for **exact**
equality, never `allclose`:

| case | NoHang / hang assemblies | result |
|---|---|---|
| unrefined adaptive-capable mesh, unsteady | 144 / 0 | bitwise identical |
| refined mesh, real hanging nodes | 720 / 288 | bitwise identical |
| stationary solve (`Steady_NoHang`) | 180 / 0 | bitwise identical |
| refined mesh, stationary, with mass matrix | 1716 / 264 | see below |
| moving mesh (`ale2d`), refined | 624 / 148 | see below |
| two domains, asymmetric refinement (remap channel) | 1680 / 320 | bitwise identical |
| `add_additional_dof_constraint` on an adaptive mesh | 720 / 288 | bitwise identical |

The two exceptions are **not** semantic. On the two cases whose entries are large sums of products -
the moving-mesh Jacobian and the Navier-Stokes one - the two arms differ by at most 3.4 ulp
(`ale2d` Jacobian 8.7e-18 against a matrix norm of 128; `nsmass` 1.4e-17 in the Jacobian and 4.3e-19 in
the mass matrix, residual bitwise identical). Compiled with `-ffp-contract=off` they are bitwise
identical, entry for entry, including the mass matrix - so the difference is GCC's freedom to contract
`a*b+c` into an FMA differently in two differently-shaped basic blocks, not different arithmetic. The
tcc backend, which contracts nothing, gives bitwise identical results on both cases with contraction
left on. This is the same sensitivity §12 records for gcc-vs-tcc.

Two things had to be fixed in the harness before any of that meant anything, and both are worth
knowing for the next such comparison:

* **The adaptive solves are not bit-reproducible run to run.** The same binary, run twice, produced
  `ale2d` Jacobian entries differing by 1e-32 - and three adapt cycles of Newton amplify that to 1e-18
  in the assembled matrix. Comparing "whatever the solver produced" measures the solver. Both arms now
  snap the dof vector (zero the low mantissa bits) before assembling, so they assemble the same input.
* A **pure-refactor gate** ran first, before any specialisation existed: the rewritten macros with
  `HANGON` printed as the literal 1 everywhere, against the pre-change build. Bitwise identical on five
  of six cases; on `ale2d` the same <=2 ulp FMA signature, pinned down by re-running the *new* build's
  generated C against the *old* macro bodies (installed by hand into `pyoomph/jitbridge/jitbridge.h`),
  which reproduced the old build bit for bit. That isolates "the restructuring is a refactor" from "the
  specialisation is correct", which is why it is worth doing in that order.

Targeted suites: `test_adaptivity.py` + `test_constrained_adaptivity.py` (32 passed),
`test_adaptive_interface_coupling.py` (137 passed), `test_generated_code_expressions.py` +
`test_jacobian_block_flags.py` as the non-adaptive smoke test (27 passed).

### 9.4.15 The Hessian shape flags missed every derived expansion

`shapes_required_Hessian[i]` under-requested: a `HessianVectorProduct` body dereferences
`dx_shapes` for bases the flag walk never marked. Harmless for as long as the runtime fill was
all-or-nothing per space - one flag pulled the whole space in - and immediately fatal once the fill
was narrowed per family (`|HVV|` off by 1.9e-1 relative 15, or `nan` under
`PYOOMPH_POISON_UNREQUIRED`).

The cause is that `__all_Hessian_shapeexps` is the **emission** set, and
`ShapeExpansion::get_spatial_interpolation_name()` ignores `is_derived`: admitting a derived
expansion alongside its undifferentiated twin would declare the same C local twice and the generated
file would not compile. That filter is right and stays; what was wrong is that the same set also fed
`mark_shapes_required`, so every basis read only by the differentiated columns lost its flag. A
second accumulator now collects the expansions unfiltered and is walked only for the flag marking.

A **per-function** reverse scan over 2117 current-ABI corpus files confirms only Hessian functions
were affected (`dx_shapes` in 4, `shape_Pos` in 16 moving-mesh Hessians). `ResJac` is clean - its
walk is over the undifferentiated residual, and differentiation preserves the basis - parameter
derivatives share the `ResJac[i]` struct, and the expression categories never differentiate. A
file-wide union scan finds nothing, which is precisely how this survived: the under-request is
per pass.

#### Outlook

* The Hessian macros (`src/jitbridge.h`) were left alone on purpose: §9.4.9/§9.4.11 show that path is
  dominated by sparse-tensor inserts, not by the emitted code.
* Narrowing the `InterfaceElementBase` predicate to elements that really hang.
* Skipping `fill_hang_info_with_equations` / `interpolate_hang_values` for elements known not to hang
  - *done*, see [assembly_overhead.md](assembly_overhead.md) §2, which also records why most of what
  `interpolate_hang_values` does on a mixed-space mesh is not redundant and cannot be cached away.

## 10. Things deliberately left alone

- **`expanded_additional_field_cache` is written but never read** (`if (false && ...)` in both the
  `field` and `nondimfield` branches). Today it costs memory and buys nothing. Re-enabling it is the
  obvious next lever for models leaning on Python-defined pseudo-fields, but the "for the moment"
  comment suggests it was backed out for a reason that is not recorded.
- **`resolve_corresponding_code` builds an `ostringstream` per placeholder**, and `get_scaling` is a
  virtual dispatched into Python. Both are on the expansion path, but after the memo (§3) they run once
  per distinct subexpression rather than once per visit.
- **The `operator<` implementations for `ShapeExpansion`/`SpatialIntegralSymbol`/...** re-evaluate every
  earlier comparison in each disjunct, so a comparison costs O(n²) field tests instead of O(n). Cheap
  integer comparisons; never showed up in any measurement.
- **The `-O2`/`-O3` choice.** `-O3` costs 3x the compile time of `-O1` on a large element and removes no
  further libm calls. Whether it is worth it for the arithmetic was not measured.
- **Global-parameter caching.** `(*(my_func_table->global_parameters[i]))` is re-read inside the
  innermost loop, and GCC provably cannot hoist it above the `ipt` loop because that loop opens with an
  indirect call through the same non-`restrict` table pointer. Measure it *after* any CSE pass, not
  before.
- **The hanging-node macros evaluate `CONTRIB` before testing `local_unknown >= 0`** (`src/jitbridge.h`)
  — *superseded by §9.4.14.* The ordering is still deliberate, but only where it is needed: a genuinely
  hanging node does need the contribution before the row is known, so the hanging path keeps it, while
  the specialised no-hang bodies now test the equation first and skip the contribution for a pinned
  unknown entirely. Nothing is left here to leave alone.
- **`pow(x,3)`/`pow(x,4)`.** Unlike squaring, `x*(x*x)` and `(x*x)*(x*x)` differ from `pow` in the last
  bits, so they need their own decision rather than riding along with §8.1.

## 11. Environment switches

All change how pyoomph gets there or what it reports, never what it computes.

| variable | effect |
|----------|--------|
| `PYOOMPH_DISABLE_UNIT_PRESCAN` | restore the unconditional `expand().normal()` in `add_residual` (§2) |
| `PYOOMPH_UNIT_FASTCHECK` | ask `collect_base_units` before normalising, for dimensional contributions — off by default (§2.2) |
| `PYOOMPH_PARANOID_UNIT_PRESCAN` | do the skipped normalisation anyway and raise if it disagrees; covers both the prescan and the fast check |
| `PYOOMPH_DISABLE_EXPAND_MEMO` | turn off the placeholder-expansion memo (on by default, §3) |
| `PYOOMPH_ARCHIVE_EXPRESSIONS` | fill `FiniteElementCode::archive` again (§5) |
| `PYOOMPH_POISON_UNREQUIRED` | signalling NaN into every shape buffer the pass did not require; `=all` is the positive control. See [assembly_overhead.md](assembly_overhead.md) §3.1 - it is what found the Hessian flag defect of §9.4.15 |
| `PYOOMPH_DISABLE_SHAPE_FAMILY_SPLIT`, `PYOOMPH_*_HANG_FILL_CACHE`, `PYOOMPH_DISABLE_ASSEMBLY_EXTDATA_SPLIT` | the assembly-overhead levers, [assembly_overhead.md](assembly_overhead.md) §6 |
| `PYOOMPH_TIME_ADD_RESIDUAL` | per-phase timing of `add_residual`/`expand_placeholders` on stderr, with mapper entries, memo hits and distinct-subexpression counts |
| `PYOOMPH_DEBUG_HOIST` | report on stderr why a Jacobian/Hessian entry could not be split for hoisting, with the atoms and their trial-index classes (§9.4.8) |
| `PYOOMPH_DISABLE_RJM_SPLIT` | emit one Residual/Jacobian/Mass function with a runtime flag instead of three specialised bodies behind a dispatcher (§9.4.6) |
| `PYOOMPH_DISABLE_JACOBIAN_HOIST` | emit each Jacobian entry inside the trial loop as before, instead of hoisting its `l_shape`-independent coefficients (§9.4.5) |
| `PYOOMPH_JACOBIAN_HOIST_MIN` | how many expression nodes a coefficient must have before it is worth naming; default 32, higher trades Jacobian time for residual-only time (§9.4.5) |
| `PYOOMPH_DISABLE_HANG_SPLIT` | generation time: emit no hanging-node-free twin of the Residual/Jacobian/Mass body, so every element runs the hanging one as before (§9.4.14) |
| `PYOOMPH_DISABLE_NOHANG_DISPATCH` | run time: send every element to the hanging entry point on an unchanged binary and unchanged generated code. The lever the bitwise validation of §9.4.14 turns |
| `PYOOMPH_REPORT_NOHANG_DISPATCH` | count hanging vs non-hanging entry-point assemblies and print both at exit (§9.4.14). Diagnostic only; a run reporting zero specialised assemblies has proven nothing about them |
| `PYOOMPH_PARANOID_ALE_IDENTITY` | rebuild every moving-mesh shape sensitivity from closed form and compare, reporting counts and worst deviation per identity and per `(el_dim, nodal_dim)` at exit, plus any `J = 0` element (§9.4). Read-only; a run that ends with "NO comparisons" has proven nothing |

## 12. Reproducing

The harnesses live in a scratchpad, not in the repo. Because the run-time changes were all reverted,
these source-rewriting harnesses are the *only* way to reproduce §7–§8: there are no `PYOOMPH_*`
switches left in the tree for any of it. That is deliberate — they measure the idea without committing
the repository to an implementation, which is what made it cheap to reject three of them.

**Generation time.** Build a heavy element, patch `Problem.compile_bulk_element_code` to time itself and
`_pyoomph_core.FiniteElementCode._add_residual` to time ingestion separately; A/B the two expansion
paths inside one binary with the switches above, interleaving the arms, and compare the generated `.c`
byte for byte.

**Run time.** All the harnesses hook `BaseCCompiler.compile` (`pyoomph/generic/ccompiler.py`), the
single entry point both the system compiler and tccbox go through, and rewrite the `.c` between pyoomph
writing it and the compiler reading it. Suppressing code writing instead does not work — it also skips
the multi-return registration and dies with "Cannot identify multi-return function (by unique id)".
Each harness asserts bit-identical residuals against its baseline, which is what distinguishes "this
change does nothing" from "this change broke the rewrite".

One thing to know when comparing compilers: gcc and tcc residuals for the same element differ by 1.2e-11
relative, consistent with gcc contracting multiply-adds into FMA under `-march=native` (where
`-ffp-contract=fast` is the default) while tcc does not. Far above round-off, far below anything a
solver notices — but a tcc-vs-gcc comparison is not a bit-for-bit one.
