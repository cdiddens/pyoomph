# Where the generated element code loses time

Investigation into whether pyoomph's **emitted C** can be made to run faster, and what the C
compiler already does about it. The sibling document `codegen_speed.md` covers the time spent
*producing* the code; this one covers the time spent *running* it.

The short version: pyoomph emits deliberately redundant C and leaves the cleanup to the C compiler,
and **the C compiler is good enough at it that none of the changes tried here earned their place.**
Three were implemented and measured; all three were reverted. What survives is the knowledge of which
apparent inefficiencies are real (few) and which the optimizer already handles (most), plus one piece
of advice that beats every code-generation change attempted: use `subexpression()`.

All numbers below are in-process timings of `problem._benchmark_elemental_assembly()`, arms built
up front and timed interleaved, min over rounds, on one machine with nothing else running.

## The compiler-flag detour, and why it is a dead end

`--fast-math` (`Problem.parse_cmd_line`, adds `-ffast-math` in `ccompiler.py`) was the starting
point. Four arms, compiled from byte-identical generated C: the `-O3 -march=native` default;
`-fno-math-errno` alone; everything `-ffast-math` implies **except** `-ffinite-math-only`; and full
`-ffast-math`. Elemental residual+Jacobian, relative to the default:

| case | ndof | `-fno-math-errno` | fast-math minus `-ffinite-math-only` | `-ffast-math` |
|---|---|---|---|---|
| poisson3d | 4 624 | +0.1% | −1.0% | −0.6% |
| ns2d | 20 450 | +0.2% | −0.4% | −1.4% |
| coupled | 20 642 | −0.2% | −0.5% | −0.7% |
| ns3d | 7 102 | +0.2% | −4.9% | −4.7% |
| transcendental 2D | 18 430 | **−86.5%** | −89.2% | −89.6% |

On polynomial weak forms the flag is noise. The one case where it is worth 7× is a weak form full of
`exp`/`log`/non-integer `pow`, and there the win is almost entirely `-fno-math-errno`, which changes
no arithmetic at all — residuals came out **bit-identical** to the default, unlike the reassociating
arms (~1e-16 relative).

The reason is visible in the compiled `.so`. Counting libm call sites in the same element:

| | `pow` | `exp` | `log` | total |
|---|---|---|---|---|
| default | 14 | 11 | 2 | 27 |
| `-fno-math-errno` | 3 | 2 | 1 | 6 |
| `subexpression()`, default flags | 3 | 4 | 2 | 9 |
| both | 2 | 2 | 1 | 5 |

Without `-fno-math-errno` a libm call is impure — it may set `errno` — so GCC may neither CSE it nor
hoist it out of the `l_shape` loop. That is the whole mechanism.

And that is the finding that redirected this investigation: **`subexpression()` beats every compiler
flag.** Same element, same clock:

| | elemental res+jac |
|---|---|
| default flags | 0.4397 s |
| `-fno-math-errno` | 0.0594 s (−86.5%) |
| `-ffast-math` | 0.0459 s (−89.6%) |
| **`subexpression()`, default flags** | **0.0392 s (−91.1%)** |
| `subexpression()` + `-ffast-math` | 0.0360 s (−91.8%) |

`subexpression()` does the same CSE at code-generation time, and additionally caches the symbolic
derivatives, which is why its advantage is much larger in the Jacobian (−91.1% vs −86.5%) than in the
residual alone (−46.5% vs −42.9%). Once it is used, the flags are worth ~0.7%.

So the flag is dominated in both quadrants, and the interesting question is what else code generation
could do that the compiler cannot.

## What the generator emits

`FiniteElementCode::write_generic_RJM` (`src/codegen.cpp:5086`) produces, per element:

```
for ipt < n_int_pt {                     // quadrature
   fill_shape_buffer_for_point(...)
   <field interpolation over l_shape>
   <subexpression block>                 // write_code_subexpressions, codegen.cpp:4407
   for l_test < nnode {                  // test function
      BEGIN_RESIDUAL_*(eqn, <one flat expression>)
      if (flag) for l_shape < nnode {    // trial function, innermost
         BEGIN_JACOBIAN_*(eqn, <one flat expression>)   // one per (test field, unknown field)
      }
   }
}
```

Macros in `src/jitbridge.h:627-760`. Each entry is a single flat C expression on one source line,
printed by `print_simplest_form` (`src/codegen.cpp:118`), which by default does `expr.evalf()` and
hands the result to `print_csrc_FEM : public GiNaC::print_csrc_double` (`src/codegen.hpp:361`). Lines
of 15 000 to 350 000 characters are normal in production elements.

## What it already optimizes

Only what the user asks for. `subexpression()` (`pyoomph/expressions/generic.py:1070`) is the sole
CSE mechanism:

- `SubExpressionsToStructs` (`src/codegen.cpp:213-357`) replaces each marker with a
  `GiNaCSubExpression`, de-duplicating by `is_equal` (`240-246`);
- `write_code_subexpressions` emits `double subexpr_N = ...;` (`4497`) and declares
  `d_subexpr_N_d_<field>` derivatives filled once per integration point under `if (flag)`
  (`4509-4582`);
- `GiNaCSubExpression::derivative` (`9128-9245`) applies the chain rule using the *cached* scalar
  rather than re-differentiating the body. This is the part that actually wins.

Beyond that there is no automatic CSE, and the source says so: `src/codegen.cpp:4410` carries
`//Subexpressions // TODO: Check whether it is constant to take it out of the loop`.

## What `-O3` rescues, and what it does not

This section decides which of the obvious-looking items are real. Every one was tested by rewriting
the *emitted C* and recompiling, so no C++ was needed to get the number.

### `pow(x,2.0)` for every square — real, but only for TCC *(implemented, measured, reverted)*

GiNaC's `power::do_print_csrc` only takes its `x*x` fast path when the basis `is_a<symbol>`; pyoomph
bases are `GiNaC::structure` subclasses (`src/codegen.hpp:311,330`), so it never fires and every
square is printed as a `pow` call. The tutorial's torsioned hyperelastic beam
(`docs/source/tutorial/ale/solid/solid_oscillations.py`) emits 210 `pow(`, **all of them with
exponent exactly `2.0`**.

Rewriting all 210 to `(x*x)` in the generated C, recompiling, and re-measuring (residuals
bit-identical, as they must be — `pow(x,2)` and `x*x` agree exactly):

| compiler | `pow@plt` in the `.so` | elemental res+jac before | after | |
|---|---|---|---|---|
| gcc `-O3 -march=native` | 45 → 0 | 0.3095 s | 0.3062 s | **−1.1%** |
| tccbox | 210 → 0 | 3.4888 s | 1.6423 s | **−52.9%** |

GCC folds 165 of the 210 by itself, and the 45 it leaves — all inside `ResidualAndJacobian0` — cost
nothing measurable. TCC has no optimizer, so it pays every one, and removing them makes the element
twice as fast.

Note the other number in that table: **TCC is 11× slower than gcc `-O3`** on this element, and still
5.3× slower after the fix.

**This was implemented, measured, and then reverted.** The implementation replaced GiNaC's
`power::do_print_csrc` for `print_csrc_double` contexts with a printer that emits a product wherever
the basis is a single atomic operand. It worked: 210 `pow(` became 0, `pow@plt` 45 → 0, residual
**and Jacobian** bit-identical to stock printing (max abs difference exactly 0), generated C +0.6%,
and on the real implementation gcc was unchanged while tcc went 3.4565 s → 1.6141 s, **−53.3%**.

It was reverted anyway, for three reasons:

- **It only ever helps tcc, and it cannot currently be scoped to tcc.** Printing is compiler-blind:
  `print_simplest_form` has no idea which backend will consume the text. Making the rewrite
  conditional means threading the target compiler through code generation, which is a larger change
  than the rewrite itself and is not justified by a win that is zero on the compiler almost everyone
  uses.
- **Turning a square into a product duplicates the basis, so anything expensive inside it is
  evaluated twice** — and on tcc, the only compiler this helps, there is no CSE to undo that. The
  implementation forbade it by requiring an atomic operand, but that safety was entirely the
  classifier's doing: for a compiler without CSE, `pow(exp(x),2)` → `exp(x)*exp(x)` is a straight
  loss, and any loosening of the classifier re-opens it.
- **tcc matters much less than it did.** Its selling point is compile speed, and the JIT cache
  (`pyoomph/generic/jit_cache.py`) already removes most repeat compilation. Against gcc `-O3` this
  element runs 11× slower under tcc — still 5.3× with the rewrite — so the honest advice is to use
  the system compiler, not to optimise the slow path.

Two things from that exercise are worth keeping in mind. The GiNaC limitation is real and will be met
again by anyone touching the printer: `power::do_print_csrc` only takes its product path when the
basis `is_a<symbol>`, and pyoomph's bases are `GiNaC::structure` subclasses, so it never fires.

And a near-miss worth recording. The first version of the classifier tested only for the absence of
`(`, on the theory that duplicating anything without a call or a parenthesised group is free. That
let the basis `coordinate_x-0.3` through and emitted

    coordinate_x-0.3*coordinate_x-0.3          instead of   (coordinate_x-0.3)*(coordinate_x-0.3)

— silently wrong by operator precedence. It survived a bit-identical residual *and* Jacobian check on
the solid element, whose bases are all plain variable references, and was caught only by `tests/`:
12 failures in `test_adaptivity.py` and `test_globally_convergent_newton.py`, showing up as an initial
Newton residual of 2.4e+66. GiNaC's own `print_sym_pow` needs no parentheses because it only ever
receives a `symbol`; generalising it to arbitrary printed strings is what creates the obligation.

An `-O` sweep on a larger, unrelated production element (1.94 MB of C, 2917 `pow(`, of which 1501
have integer exponents and 1416 are cube-root-family fractional powers):

| | compile | `.so` | `pow@plt` |
|---|---|---|---|
| `-O0` | 3.1 s | 1 373 048 B | 2491 |
| `-O1` | 9.0 s | 459 576 B | 1454 |
| `-O2` | 24.3 s | 443 192 B | 1454 |
| `-O3` | 29.7 s | 521 016 B | 1454 |

The optimizer is clearly working — everything integer-exponent is folded by `-O1`, and the 1454
survivors are essentially the 1416 genuinely fractional powers. What `-O3` buys over `-O1` here is a
3× longer compile and no fewer libm calls.

### The dead subexpression derivative cache — a correctness smell, not a cost

On a moving mesh, `GiNaCSubExpression::derivative` detects a position symbol and takes an escape
hatch (`src/codegen.cpp:9166-9239`): it differentiates the body on the spot and inlines the result at
every use site, ignoring the cached scalar. The cache is still emitted.

In the tutorial solid element, 225 `d_subexpr_*` are declared and assigned per integration point and
**171 of them are never referenced by any residual or Jacobian entry**; 90 are dead even after
accounting for ones kept alive through other assignments' right-hand sides. In an archived production
element (3.7 MB) the count is 153 declared, 153 assigned, **0 read anywhere in the file**.

Deleting the 90 provably dead assignments and re-measuring: gcc −0.1%, tcc +0.6%. Nothing. They are
computed once per integration point, against `nnode² = 729` Jacobian entries per point — the ratio is
what makes them free.

So this is worth fixing for clarity and for what it reveals, not for speed. What it reveals is worth
acting on: the same escape hatch means that on moving meshes `subexpression()` hoists the *value* but
inlines the *derivative* at every use site, which is exactly the case where the Jacobian statements
grow to hundreds of kilobytes. Related, and worth fixing in the same pass: at
`src/codegen.cpp:9222-9236` an error message is carefully assembled and then not thrown — the
`throw_runtime_error` is commented out.

### Repeated subtrees in the Jacobian entries — this is the real one

Each `BEGIN_JACOBIAN_*` entry is an independent `GiNaC::diff` of the whole residual, printed flat, so
a factor shared between entries is re-expanded into every one of them. Within a single statement the
same subtree also appears repeatedly because GiNaC does not factor `A*X + B*X` when `A` and `B`
differ.

To find out whether this survives `-O3`, the proposed pass was modelled on the emitted C: for each
`l_shape` block, collect the entry expressions, find balanced sub-expressions occurring more than
once, and hoist each to the outermost loop level whose variable it does not mention (`l_shape` →
top of the trial loop, `l_test` only → top of the test loop, neither → above both). Naming a value
cannot change it, so residuals must stay bit-identical — they did, which is what makes the numbers
below trustworthy.

| case | hoists | C size | gcc `-O3` | tccbox |
|---|---|---|---|---|
| hyperelastic solid, 3D | 135 | −12% | **−10.8%** | **−24.9%** |
| Navier-Stokes 2D (control) | 1 | −0% | −3.9% | +1.3% |

Insensitive to the size threshold — at 15, 20, 30, 45 and 80 characters the solid case gives −10.5%,
−11.3%, −10.8%, −11.1%, −11.2%. The win comes from a handful of large repeated groups, not from
aggressive micro-CSE.

**~11% beyond `-O3`, on the case that matters, from a purely mechanical textual model.** The model is
trustworthy: residual **and** Jacobian come out bit-identical to the unmodified build, so it is not
buying speed by corrupting the thing it rewrites. (That check was added after the fact — the first
version of the model only verified residuals, which is exactly the blind spot that would have hidden
a cheaper-but-wrong Jacobian. It passed.)

The Navier-Stokes control confirms the shape of the result: a polynomial weak form has almost nothing
to factor, and the pass correctly finds almost nothing.

### A GiNaC implementation was tried and did not work — reverted

A real pass on the GiNaC tree was written, validated and measured at **−1.9%**, against the −10.6%
the textual model demonstrably achieves on the same element with the same compiler. It never came
close, so it was reverted. This section records what it was and why it failed, so the next attempt
does not start from zero.

The pass sat after each `GiNaC::diff` and before printing — the cheap place, because nothing
differentiates a Jacobian expression again, so a named temporary can be a plain symbol needing no new
GiNaC node type, no derivative rule, and no interaction with the `subexpression()` derivative cache or
the Hessian. It was correct: residuals exactly unchanged, Jacobian moved only by ~3e-14 relative
(`subs` reassociates), generated C deterministic across processes, 245 tests passing with it on.
It simply was not fast.

What is known:

| | distinct temporaries | generated C | elemental res+jac |
|---|---|---|---|
| baseline | — | 320 420 B | 0.3046 s |
| textual model | 45 (each emitted into all 3 sibling `l_shape` loops → 135 definition lines) | 283 208 B | 0.2719 s (**−10.6%**) |
| GiNaC pass, `min_savings=96` (default) | 27 | 266 763 B | 0.2983 s (**−1.9%**) |
| GiNaC pass, `min_savings=48` | 54 | — | 0.3083 s (+1.3%) |
| GiNaC pass, `min_savings=32` | 117 | 294 929 B | 0.3048 s (+0.2%) |
| GiNaC pass, maximal coverage | 504 | 292 853 B | 0.3070 s (+0.9%) |

So the GiNaC pass removes *more* text with *fewer* temporaries and is still five times less
effective. Threshold sweeps across 2, 3, 8, 12, 16, 20, 24, 32, 48, 50, 64, 96, 128 and 512 find no
better regime: 27 temporaries is the optimum and everything more aggressive is neutral or worse -
naming something cheap costs a store and a live register.

Selecting on cost alone rather than on saved work was one real mistake found along the way. It
produced 243 temporaries, which is *slower* than doing nothing, and under tcc slower by 11%, there
being no register allocator to absorb 243 live stack slots. The criterion is now
(occurrences - 1) x cost.

Three hypotheses were tested. Two are dead, one produced a real bug fix, and the gap survives all
three.

**`subs` re-canonicalisation — disproven.** The worry was that GiNaC re-evaluates after substitution
and redistributes the new symbol back into surrounding sums, silently undoing the factoring. Counting
uses of each `_cse_N` in the emitted C: every one is used exactly 16 times, precisely the occurrence
count the pass predicted. The substitution does what it says.

**The cost model — genuinely broken, and fixing it changed nothing.** The cost function summed
interior-node weights and treated leaves as free. But a leaf is a shape expansion or a `subexpr_N`
reference, i.e. a memory load, and it is the loads that dominate. A debug dump of the candidates
showed the consequence starkly: a node with 10 operands was costed at **31**, while a two-operand sum
was costed at **3** — so selection systematically preferred the small one. Charging 1 per leaf (0 for
numeric literals, which are immediates) is the correct model and moved the sweet spot from a
threshold of 32 to 96 — and left the result exactly where it was, at −1.9%.

**Scope — real, quantified, and not sufficient.** The pass runs per
`write_generic_RJM_jacobian_contribution` call, i.e. over the entries of one (test field, space) pair,
while the textual model runs over the whole `l_test` block. Confirmed by the diagnostic: the pass sees
16 occurrences of a subtree that occurs 48 times across the block's three sibling `l_shape` loops. But
each loop can only benefit from its own 16, and widening the count would not change which subtrees
are worth naming *within* a loop — so this explains a factor of three in the counters and nothing in
the arithmetic.

What the evidence actually points at is **coverage**, and it is measured rather than guessed. Bytes of
C inside the Jacobian entry statements, versus bytes moved into temporaries:

| | entry statements | temporaries | total |
|---|---|---|---|
| baseline | 152 049 B | 0 | 152 049 B |
| GiNaC pass, best setting | 97 213 B | 27 707 B | 124 920 B |
| GiNaC pass, maximal coverage (504 temporaries) | 63 262 B | 54 164 B | 117 426 B |
| textual model (45 temporaries) | **18 162 B** | 94 785 B | 112 947 B |

The model collapses the entries by a factor of eight. The GiNaC pass floors at 63 kB *no matter how
low the threshold goes* — 504 temporaries do not get it further. So the two are not selecting the same
things at different thresholds; there is structure the subtree-based pass cannot reach at all.

The leading explanation, untested: GiNaC's `add` and `mul` are **flat n-ary** nodes, so a repeated
*partial* sum or product — a subset of one node's operands — is not a subtree and can never be a CSE
candidate. The textual model, working on printed and explicitly parenthesised output, factors exactly
such groupings. Matching it would mean frequent-subset mining over operand multisets rather than
subtree matching: a materially harder algorithm, and a different piece of work.

### Conclusion: not worth pursuing, because the C compiler already does this

The pass was reverted. Beyond the measured −1.9%, the reason not to return to it is that **this is
work the C compiler is built to do and does well.** Everything in this document points the same way:
`-O3` folds 165 of 210 `pow(x,2)` calls unaided; it collapses 3.7 MB of generated C into a 118 kB
shared library; on the transcendental case it needs only permission (`-fno-math-errno`) to CSE and
hoist libm calls out of the innermost loop by itself. GCC's GCSE/PRE are mature, they see the whole
basic block including everything pyoomph's DAG cannot express, and they are not paying the cost of a
symbolic pass at code-generation time.

The two places where code generation genuinely beats the compiler are both already available and both
much larger effects than anything measured here: `subexpression()`, which wins because it also caches
the symbolic *derivatives* (−91% on a transcendental weak form, better than any compiler flag), and
choosing the system compiler over tcc (11× on a hyperelastic element). Advising users toward those is
worth more than an automatic CSE pass would be.

If someone does return to this, the entry condition should be evidence that `-O3` has actually given
up — e.g. a production element where compile time or `pow@plt` counts show the optimizer bailing out
on 350 kB statements — not a general belief that emitted redundancy must cost something.

## Ranked conclusions

| item | gcc `-O3` | tccbox | verdict |
|---|---|---|---|
| CSE with loop-level placement in the emitter | −11% achievable, **−1.9% achieved** | −25% achievable | tried, **reverted** — see above |
| `pow(x,2)` → `x*x` in the C printer | 0 | −53% | tried, **reverted** — tcc-only, and tcc is fading |
| `-fno-math-errno` in the default flags | −86% on transcendental forms, 0 elsewhere | n/a | tried, **reverted** |
| **`subexpression()` in user code** | **−91% on transcendental forms** | — | **the recommendation**: already available, beats every change tried here |
| drop the `IGNORED RESIDUAL` comment payload | 0 | 0 | not attempted; compile time only, but that is 66% of one 5.8 MB file |
| remove the dead `d_subexpr_*` stores | ~0 | ~0 | not attempted; clarity, not speed |
| hoist trial-function pointers | expected ~0 | untested | not attempted; size only |
| automatic `subexpression()` injection | unknown | unknown | not attempted; narrower than it looks — see below |

A note on the last one. Automatic marker injection before differentiation is the obvious idea, and it
is the *riskier* half: it changes what gets differentiated symbolically, it is blocked from anything
containing a test function (`src/codegen.cpp:250-254` throws), and on `coordinates_as_dofs` codes the
escape hatch above means it can make the Jacobian larger, not smaller. Post-differentiation CSE needs
no new GiNaC machinery at all — nothing differentiates a Jacobian expression again, so a plain symbol
substitution suffices. That is the half that was built, and it did not pay off; injection is the
harder half and there is now no evidence it would do better.

## What was changed in the end: nothing

Every code change explored here was implemented, measured and then reverted. The repository carries
only this document and two documentation fixes (the `--fast-math` help string in
`pyoomph/generic/problem.py` and its counterpart in
`docs/source/tutorial/installation/cmdlineoptions.rst`, both of which now point users at
`subexpression()` instead).

Reverted, with the reasons above:

- **`pow(x,2)` → `x*x` in the printer.** Worked, bit-exact, −53% under tcc, zero under gcc. Cannot
  currently be scoped to tcc; tcc is increasingly irrelevant given the JIT cache.
- **`-fno-math-errno` in the default unix flags.** Worth −86% on a transcendental weak form and
  nothing elsewhere, changing no floating-point value. Note for anyone re-adding it: the compile flags
  are *not* fully covered by the JIT cache key. `SystemCCompiler.get_cache_flag_state()` covers
  `PYOOMPH_DEBUG`, `_optimize_full_speed` and `compile_args`, from all of which the flags are today
  derivable; a hardcoded flag added to the defaults is not, so it needs an explicit epoch in that
  string or every previously cached `.so` is silently reused.
- **The automatic CSE pass.** −1.9% against an achievable −10.6%; see above for why it is unlikely to
  be the right approach at all.

## Things deliberately left alone

- **The `-O2`/`-O3` choice.** `-O3` costs 3× the compile time of `-O1` on a large element and removes
  no further libm calls. Whether it is worth it for the arithmetic was not measured, and switching
  the default on compile-time grounds alone would be guessing.
- **Global-parameter caching.** `(*(my_func_table->global_parameters[i]))` is re-read inside the
  innermost loop, and GCC provably cannot hoist it above the `ipt` loop because that loop opens with
  an indirect call through the same non-`restrict` table pointer. But the CSE pass above subsumes it
  wherever it appears more than once, so it should be measured *after* that pass, not before.
- **The hanging-node macros evaluate `CONTRIB` before testing `local_unknown >= 0`**
  (`src/jitbridge.h:721` vs `734`), unlike the non-hanging variants. For a genuinely hanging node the
  contribution is needed for every master, so the ordering is deliberate; the waste is confined to
  pinned unknowns and was not measured.
- **The `pow(x,3)`/`pow(x,4)` cases.** Unlike squaring, `x*(x*x)` and `(x*x)*(x*x)` differ from `pow`
  in the last bits, so they need their own decision rather than riding along with the exponent-2 fix.

## Reproducing

The harnesses live in the session scratchpad, not in the repo. All of them share one trick: hook
`BaseCCompiler.compile` (`pyoomph/generic/ccompiler.py:107`), which is the single entry point both
the system compiler and tccbox go through and which knows the source filename, and rewrite the `.c`
between pyoomph writing it and the compiler reading it. Suppressing code writing instead does not
work — it also skips the multi-return registration and dies with "Cannot identify multi-return
function (by unique id)" (`src/codegen.cpp:6962`).

- `bench_fastmath.py` — the four compiler-flag arms plus the `subexpression()` cross, all interleaved
  in one process, with the flags asserted via a spy on the compile call so an arm that silently
  failed to get them is an error rather than a clean-looking null result.
- `bench_pow.py` — rewrites `pow(<balanced>,2.0)` to `((x)*(x))`, innermost first.
- `bench_deadderiv.py` — deletes `d_subexpr_*` assignments by fixpoint liveness.
- `bench_cse.py` — the CSE model described above, the one that reaches −10.6%.

Because everything was reverted, these source-rewriting harnesses are also the only way to reproduce
the numbers: there are no `PYOOMPH_*` switches left in the tree for any of it. That is deliberate —
the harnesses measure the *idea* without committing the repository to an implementation, which is
exactly the property that made it cheap to reject three of them.

Each asserts bit-identical residuals against its baseline, which is what distinguishes "this change
does nothing" from "this change broke the rewrite".

One thing to know when comparing compilers: gcc and tcc residuals for the same element differ by
1.2e-11 relative, consistent with gcc contracting multiply-adds into FMA under `-march=native` (where
`-ffp-contract=fast` is the default) while tcc does not. That is far above round-off and far below
anything a solver notices, but it means a tcc-vs-gcc comparison is not a bit-for-bit one.
