# Code generation: the time to produce the C, and the time to run it

Two investigations on the same pipeline, merged here because they keep pointing at each other.

* **Producing the C** (§1–§4) — branch `codegen`. Ingestion, not emission, was the expensive half,
  and two causes account for essentially all of it. Both fixes landed.
* **Running the C** (§5–§9) — whether the emitted code can be made faster. The answer is mostly no:
  pyoomph emits deliberately redundant C and the C compiler is good enough at cleaning it up that
  three implemented, measured changes were all reverted. What survives is the knowledge of which
  apparent inefficiencies are real (few), and one piece of advice that beats every code-generation
  change attempted: use `subexpression()`.

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
| **the three open leads of §9** | unmeasured | unmeasured | **the largest remaining levers** |

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

## 9. Open leads in the subexpression machinery

None of these were attempted. Each is a bigger lever than anything measured above — the first is a
latent correctness bug, the other two are the reason moving-mesh and stability elements generate the
largest code in the project.

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

Any change here changes which branch of `GiNaCSubExpression::derivative` fires, so §9.1 should be
resolved **first**, not discovered afterwards.

### 9.3 Subexpressions in the Hessian

They are stripped outright. `write_generic_Hessian` runs `RemoveSubexpressionsByIndentity` over the
residual to unwrap every marker back to its raw argument before differentiating twice, and
`write_code_subexpressions` throws `"Hessian subexpressions!"` if any survive. So on a problem with
`generate_hessian` — every azimuthal or Cartesian normal-mode stability analysis with an analytic
Hessian — the user's `subexpression()` calls buy nothing at all in the Hessian, which is routinely the
*largest* generated function: in one production element `HessianVectorProduct1` is 455 kB against
`ResidualAndJacobian1`'s 386 kB.

The machinery is half-present rather than absent: `__SE_to_struct_hessian` exists to re-wrap markers
during Hessian generation, `__hessian_subexprs_scanned` tracks what has been folded in, and
`GiNaCSubExpression::derivative` already has an `__in_hessian` branch that builds a nested subexpression
for the inner derivative instead of reading the cache. What is missing is the second-derivative cache
itself — the `d2_subexpr_N_d_f_d_g` equivalent — and the emission of its fill code.

The largest of the three and the least explored. It should not be started before §9.2 is settled,
because on a moving mesh the two share the same escape hatch and the same reason for it.

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
- **The hanging-node macros evaluate `CONTRIB` before testing `local_unknown >= 0`** (`src/jitbridge.h`),
  unlike the non-hanging variants. For a genuinely hanging node the contribution is needed for every
  master, so the ordering is deliberate; the waste is confined to pinned unknowns.
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
| `PYOOMPH_TIME_ADD_RESIDUAL` | per-phase timing of `add_residual`/`expand_placeholders` on stderr, with mapper entries, memo hits and distinct-subexpression counts |

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
