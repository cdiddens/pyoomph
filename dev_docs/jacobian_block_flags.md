# Per-block Jacobian properties: proven symmetry and constancy

Status: **implemented and tested** (serial) — the flags are computed for every generated code, exposed
to Python, AND-combined at problem level and printed into `_ccode/_jacobian_structure.txt`. The
consumers they were built for (Schur-complement reuse in [static_condensation.md](static_condensation.md),
PETSc fieldsplit KSP/PC selection) are **not built yet**; nothing reads the flags for decisions today.
A side effect of this work, the print-free global-parameter registration (§4), is active everywhere.

**The idea.** Code generation already decides which (test field, unknown field) blocks of the elemental
Jacobian are structurally nonzero (`contributes_to_jacobian`, see
[structural_assembly.md](structural_assembly.md)). At the moment that decision is made, the *symbolic
block expression* is in hand — and it can prove much stronger properties than "nonzero": that a block
is the (anti)transpose of its mirror block, or that its entries never change between Newton solves.
Those proofs are recorded as per-block bit flags in the JIT func table.

**The contract.** A set bit is a property **proven** from the symbolic expression; an unset bit means
"not proven", never "disproven". Every case the analysis does not fully understand falls through to
unset, so a consumer acting on a set bit is always safe and a consumer ignoring the flags entirely is
never wrong. Blocks without any contribution have all bits 0 — consumers must consult
`contributes_to_jacobian` first (an absent block is identically zero, hence trivially symmetric and
constant).

---

## 1. The bits and where they live

`src/jitbridge.h` (mirrored into `pyoomph/jitbridge/jitbridge.h` at configure time):

| bit | meaning |
|---|---|
| `JACOBIAN_BLOCK_SYMMETRIC` (1) | block (i,j) equals **+transpose** of block (j,i); set on both mirror entries; diagonal blocks: self-transpose |
| `JACOBIAN_BLOCK_ANTISYMMETRIC` (2) | block (i,j) equals **−transpose** of block (j,i) |
| `JACOBIAN_BLOCK_CONSTANT` (4) | entries independent of unknowns, nodal positions (moving mesh), global parameters, time **and** time-stepper weights |
| `JACOBIAN_BLOCK_CONSTANT_FIXED_DT` (8) | as CONSTANT, but may contain BDF/Newmark weights — constant while the step sizes are; implied by CONSTANT |

Two `unsigned char ***` tables, `functable->jacobian_block_flags` and
`functable->mass_matrix_block_flags`, indexed exactly like `contributes_to_jacobian`:
`[res_jac][row contribution class][column contribution class]`. The two constancy bits resolve the
time-derivative question without ambiguity: a `partial_t(u)`-term's Jacobian half carries the BDF
weight, so a transient diffusion block is `SYMMETRIC|CONSTANT_FIXED_DT` while its mass block is fully
`CONSTANT` (the weight lives in the Jacobian, not in M). For the mass table the two constancy bits
coincide.

Python access: `element._get_block_flags()` returns both tables for the current residual set, ordered
like `_get_contribution_names()`; the bit values are module attributes
(`_pyoomph_core.JACOBIAN_BLOCK_*`, kept in sync by hand — see the comment at the definition site in
`src/nanobind/mesh.cpp`).

## 2. How the flags are computed (codegen)

**Recording.** `write_generic_RJM_jacobian_contribution()` (src/codegen.cpp) differentiates the
residual w.r.t. each unknown field and, historically, printed the result and dropped it. Now the
per-(row field, column field) sums of both halves (`diffpart`, `mass_part`) are accumulated into
`FiniteElementCode::jacobian_block_exprs` via `record_jacobian_block_expr()` — but **only** during
the plain `ResidualAndJacobian<r>` pass (`record_jacobian_blocks_for_flags`, set around that one call
in `write_code`): the Steady and dResidual/dParameter variants run through the same differentiation
site and would contaminate the sums. Residual sets derived by expansion mode (azimuthal/Cartesian
normal-mode stability) are skipped entirely — their blocks are complex and transpose-symmetry would
need conjugation, so their flags stay 0, which the contract reads correctly as "nothing proven".

**Aggregation.** At table-emission time the recorded field-pair sums are folded onto *contribution
classes* through `block_contribution_class_name()` — deliberately the same function that names the
`contributes_to_*` table indices, so the two cannot drift apart. Summing per class before testing is
what makes the analysis exact for blocks fed by several weak-form additions or by a bulk/interface
alias pair: per-term ANDing would be sound but incomplete (a sum of symmetric parts is symmetric even
when no single part equals its mirror).

**Constancy** (`scan_block_constancy` / `block_constancy_flags`): one print-free preorder walk that
descends into CSE subexpression wrappers and multi-ret invocation arguments (a plain `.has()` misses
dependencies hidden there). It whitelists what a constant block may contain — derived shape
expansions, test functions, and, on a *fixed* mesh, the geometric symbols (`dx`, element size,
normals) — and clears constancy on anything else: non-derived shape expansions (unknown values),
coordinate-derivative shapes when the mesh moves, global parameters, explicit time, and history
values. Unclassified leaf types default to nonconstant, so new GiNaC structures are conservative
until someone classifies them. BDF-weight dependence (derived shapes with `dt_order>0`) sets only the
dt-dependence, downgrading CONSTANT to CONSTANT_FIXED_DT.

**Callbacks are transparent, not opaque.** A user callback (scalar `CustomMathExpression` or
`CustomMultiReturnExpression`) is *required* to be a deterministic function of its arguments, so it
cannot break constancy by itself — only its arguments can. The scan therefore treats the
`GiNaCCustomMathExpressionWrapper` leaf as what it is, a function identity rather than a value, and
recurses into the argument list of a `GiNaCMultiRetCallback` (`invok.op(1)`). For the scalar case
that recursion is free: `python_cb_function(wrapper, args)` is an ordinary two-operand GiNaC
function, so the walk already visits its arguments; only the multi-return structure is opaque and
needs the explicit descent. A diffusivity `cb(x)` on a fixed mesh is thus provably constant, `cb(u)`
and `cb(t)` are not. The first version of the analysis treated every callback as variable, which
silently cost the whole `cb(x)` class of blocks — note that determinism is a documented contract of
the callback interface, not something the code can enforce.

**Symmetry** (`BlockRoleCanonicalizer`): block (i,j) carries `TestFunction(field_i)` in the row role
and a derived `ShapeExpansion(field_j)` in the column role; its transpose exchanges those roles. The
canonicalizer maps both structures onto fresh registry symbols keyed by role, canonical field
identity (`get_defined_on_domain_equivalent_field()`), basis, derivative attributes, expansion mode
and history-geometry — a **shared** registry between the two mappings, so "same function, same role"
lands on the same symbol. A time-derivative's weight travels as an extra `DTW` symbol with whichever
side carries it, matching the numerics. Then (i,j) is compared against the transpose-rewrite of
(j,i): `is_equal` fast path, else `(A−Bᵀ).expand().is_zero()` for SYMMETRIC and `(A+Bᵀ)` for
ANTISYMMETRIC. Two deliberate bail-outs: a node cap (`node_cap = 20000` — `expand()` is superlinear,
over the cap nothing is proven) and an "unsupported" flag for blocks containing coordinate-derivative
shapes or derived geometric symbols, which carry an implicit *column* index while sitting in the row
role — one canonical symbol cannot represent them and a false positive was possible. This is why
moving-mesh blocks get no symmetry bits, not just no constancy bits.

**Cost.** Measured on an 18-class, heavily nonlinear element (Navier–Stokes plus 14 coupled species
with Arrhenius cross-coupling): 15.20 s codegen+compile with the analysis vs 14.69 s with recording
disabled, ≈3.5 %. The node cap never engaged there.

## 3. Problem level: the AND-union and `_jacobian_structure.txt`

`Problem::assemble_defined_field_list()` (src/problem.cpp) already unions each code's
`contributes_to_*` tables onto the problem-wide `defined_fields` numbering, matched by residual-set
*name*. It now also builds `jacobian_block_flags_union` / `mass_matrix_block_flags_union`
(+ `mass_matrix_contributing_fields`): start from all-bits-set, **AND** in every contributing code's
flag byte, zero the blocks nothing contributes to. A property of the assembled global block holds
only if every contributor proves it.

One correction on top of the plain AND: the pairwise symmetry bits must survive on **both** mirror
entries. A code contributing only to (j,i) breaks the pair relation of the assembled blocks even if
every contributor to (i,j) proves it — and it shows up as an unset bit on the (j,i) side only, so the
post-pass ANDs the pair bits across the two mirror cells (and clears them when only one side has any
contribution at all: a nonzero block cannot be the ±transpose of an absent one).

`get_jacobian_information_string()` renders the result into `_ccode/_jacobian_structure.txt`, one
property matrix per residual set (plus one for the mass matrix where present), two characters per
cell: `S`/`A`/`.` for symmetric pair / antisymmetric pair / contribution without proven symmetry, and
`C`/`c`/blank for constant / constant-while-dt-fixed / not proven. A Taylor–Hood Navier–Stokes cavity
prints:

```
	Jacobian block properties (...)          Mass matrix block properties (...)
	    | 0 1 2 |                            	    | 0 1 2 |
	----|-------|                            	----|-------|
  	  0 | . . AC|                            	  0 | SC    |
  	  1 | . . AC|                            	  1 |   SC  |
  	  2 | ACAC  |                            	  2 |       |
	----|-------|                            	----|-------|
```

— nonsymmetric, nonconstant momentum blocks; the u-p/p-u saddle-point pair antisymmetric (pyoomph's
signs are `−p∇·v` / `+∇·u q`) and constant; a constant symmetric velocity mass. Normal-mode residual
sets print all `.` by design (see §2). The matrices are purely informative here; the union tables on
`Problem` are the intended input for the future consumers.

## 4. Side effect: print-free global-parameter registration

The analysis above must never print a GiNaC expression, because
`GiNaCGlobalParameterWrapper::print` historically allocated the code's local parameter slot on first
encounter — slot order decided which `dResidual<N>dParameter_<i>` routines exist and how
`functable->global_parameters` is laid out, and `print_residual_entry()` even printed *ignored*
residuals into a discarded stream solely for that side effect. That mechanism is now inverted:

- `FiniteElementCode::register_global_parameters_in()` screens an expression print-free (descending
  into subexpressions/multi-ret arguments) and registers parameters in the same first-encounter order.
- `GlobalParameterFunctionScope` (RAII, src/codegen.cpp) wraps each generated-function writer:
  it screens the function's symbolic sources before anything prints, and hoists each parameter into a
  `const double pyoomph_gparam_<i> = *(my_func_table->global_parameters[<i>]);` local at the top of
  the function, which `print` then references by name instead of the double indirection at every use.
  Instrumented writers: ResidualAndJacobian (all variants), HessianVectorProduct, integral/local/
  extremum expressions, tracer advection, Z2 fluxes, initial conditions, Dirichlet conditions.
- Paths without a scope (multi-ret callback bodies, the geometric-Jacobian family) keep working
  through print's fallback: lazy registration plus the self-contained indirect access. That fallback
  is also why the no-print rule for analysis passes **still stands** — a printing pass could still
  register a slot on a fallback path.
- The discarded print of ignored residuals is gone; they emit a bare `0`.

The `dResidualdParameter` selection semantics are unchanged because screening happens at the same
point in the writer sequence where the first print used to happen. Slot numbering *within* a code can
differ from pre-change builds (scan order vs print order) — harmless, the generated
`global_paramindices` table is self-describing, but it means one-time JIT-cache misses.

## 5. Validation

`tests/test_jacobian_block_flags.py` (9 tests, 18 with parametrization) applies a two-sided check:

- **Soundness gate** — for every *set* bit on every problem, the property is verified against
  numerically assembled global class blocks: transpose relation at two random dof states for S/A;
  invariance under randomized dofs, changed parameters, advanced time and changed dt for C; and for c
  without C, that a dt change *does* move the block. The gate itself was negative-controlled during
  development: four hand-injected bogus bits were each caught by a different gate path.
- **Completeness spot checks** — Poisson `S C`, transient diffusion `S c` with mass `S C`,
  convection–diffusion losing S always and C exactly when the wind stops being a literal
  (parameter-driven or an unknown), a callback diffusivity constant for `cb(x)` but not for `cb(u)`
  or `cb(t)`, Stokes u-u `S C` vs Navier–Stokes u-u flagless, moving mesh all-clear, azimuthal sets
  all-zero, and build-twice determinism (two subprocesses — one `Problem` per process, see the
  JIT-loader constraint).

The parameter rework was validated separately: a problem with one parameter in residual, Dirichlet
value, initial condition, integral and local expression shows the hoisted local in every generated
function with zero indirect leftovers, responds exactly to a parameter change without recompilation,
and a Bratu fold-tracking run (arclength + eigenproblem + Hessian + dResidualdParameter) lands on
λ = 3.5138307, the literature value.

## 6. Deliberately not proven, and what is next

- Moving-mesh blocks get neither constancy nor symmetry (§2); hanging-node and interface
  configurations are exercised nowhere in the tests — symmetry survives the `CᵀJC` hanging constraint
  by construction, but nothing asserts it.
- `has_constant_mass_matrix_for_sure` is still inferred from the Hessian pass; the per-block mass
  constancy could strengthen it directly.
- The intended consumers: keep the Schur-complement inverse across solves in static condensation when
  the condensed block is CONSTANT (or CONSTANT_FIXED_DT within a fixed-dt transient), and choose
  fieldsplit KSP/PC per block (CG/Cholesky on proven-symmetric diagonal blocks, full saddle-point
  treatment only where the pair structure demands it). Both read `jacobian_block_flags_union`.
