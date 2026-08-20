# Assembly overhead: hanging bookkeeping, required-shape flags and external-data bloat

Status: **implemented, validated, benchmarked and merged** (serial). End-to-end numbers in §6a. Follow-on to the per-element NoHang
dispatch of [code_generation.md](code_generation.md) §9.4.14, which specialised the *generated* code
but left everything the C++ side does *around* each call untouched. This document covers three such
sources of per-assembly overhead, why each is there, what removing it is worth, and the two places
where removing it turned out to be unsafe.

The short version: the biggest measured win came from an audit nobody had run (the shape-buffer
fills), the most-cited candidate (hanging bookkeeping) is mostly *not* redundant work, and the most
dramatic-looking one (external data inflating `ndof`) is structurally real but costs under a percent.

---

## 1. Why look here at all

`dev_docs/structural_assembly.md` §2 measures elemental evaluation at 56-87% of a full Jacobian
assembly, so the elemental path is the right lever. §9.4.14 took the generated arithmetic; what
remained per element and per assembly was:

* `fill_hang_info_with_equations` + `interpolate_hang_values`, in full, for every element — even ones
  the dispatch had just proven have nothing hanging (the §9.4.14 outlook item),
* a shape-buffer fill that produced whole families of derivatives nothing had asked for,
* external data attached from the OR of *every* contribution, so an output observable widened the
  dense elemental block of every element of the domain.

Measurement first, in all three cases. The levers that produced the baselines are still in the tree
(§6) because each is also the A/B lever for its own change.

### 1.1 Baselines (min of 3, interleaved, in-process `_benchmark_elemental_assembly`)

| case | hang fills, res | hang fills, jac |
|---|---|---|
| bulk_mixed — static C2/C1 Navier-Stokes, 14162 dof | −13.9% | −8.7% |
| bulk_adapt — same, after refinement, 6692 dof | −18.9% | −21.5% |
| iface_normal — interface with normals, 6480 dof | −5.0% | −3.6% |

Per element on `bulk_mixed`: `fill_hang_info_with_equations` 232 ns, `interpolate_hang_values`
1660 ns, against 17.0 µs of Jacobian assembly.

**Read `bulk_adapt`'s number with care**: the measurement lever also forces `has_hang=false`, so
genuinely hanging elements take the NoHang entry as well. It is an upper bound, not a target.

External data, as a fraction of an element's local dofs: interface elements of an
interface+integral-observable problem **66.7%** (6 of 9), bulk elements of a problem whose ODE enters
only an observable **10.1%**. Moving-mesh interfaces sit at 41.3%, which is legitimate and geometric.

The rank-4 `d_dx_shape_dcoord` fill is **40.6%** of elemental Jacobian assembly on a moving mesh —
the largest single item, addressed in §4a.

---

## 2. Workstream A — hanging bookkeeping

### 2.1 What it does, and what is actually redundant

`interpolate_hang_values` has three phases and **no early-out**: hanging positions, hanging values,
and — unconditionally — the *dummy value* interpolation that a mixed-space element (a C1 field
carried on C2 nodes) needs on every assembly. That third phase is genuine, dof-dependent work: it is
not redundant, and no cache can remove it. This is why the `bulk_mixed` ceiling above is not
reachable, and it is the honest reason Stage A's measured gain is small.

What *is* redundant:

* the fill, for an element whose dispatch takes a separately compiled `_NoHang` twin — that code
  provably never reads `hanginfo`;
* the whole call, for an element with no hanging node, no additional dof constraint and an empty
  dummy map;
* the neighbour re-interpolation: `InterfaceElementBase::prepare_shape_buffer_for_integration`
  re-runs `interpolate_hang_values()` on its bulk, bulk-of-bulk, opposite and opposite-bulk elements
  on *every* assembly, so a bulk element adjacent to k interfaces was interpolated k+1 times per pass.

### 2.2 What was built

**The skip is derived from the very function pointer that selects the entry point**, in the same
function, so the two cannot drift apart. It applies only when that pointer is a separately compiled
`_NoHang` twin: `ParameterDerivative` and both Hessian routines have no twin and keep the
unconditional fill. Interfaces that require bulk or opposite shapes always report "hanging" (their
hang buffers double as the equation-remap channel, see §5.2 of code_generation.md), so the remap
channel is never skipped.

The classification (`Nothing` / `Work`) is a per-element member invalidated in `fill_element_info`,
next to `local_dof_contribution_indices_valid`, which is the pattern it copies — that hook fires on
every `assign_eqn_numbers`, covering adapt, remesh, pin/unpin and constraint setup. It is computed
lazily on first use and **not** eagerly inside `fill_element_info`, because interface remap vectors
are rebuilt *after* it.

The dedupe uses a global assembly-pass counter and a per-element stamp. FD perturbation loops call
`get_residuals` once per perturbed dof and each such call must count as its own pass — the perturbed
dofs change the interpolated values — so those paths run under a pass *suspension* that opens a fresh
pass on exit.

Also: the constraint helpers now test the node before doing a `dynamic_cast`, which the fills already
did but `interpolate_hang_values` did not.

### 2.3 Result

Bitwise identical (residual, Jacobian, Hessian-vector) on 12 cases and through a full
solve → adapt → refine → unrefine → solve cycle, with 62007 fills skipped. Engagement, per dump run
(fills skipped/run · interps skipped-by-class/by-stamp/run): `bulk_mixed` 17600/0 · 0/0/17600;
`bulk_adapt` 7128/1584 · 0/0/8712; `iface_normal` 8400/0 · 8400/0/0; `ale_iface` 2880/120 · 0/120/3000.

Benchmark: `iface_normal` res +1.9%, jac +3.6% — the case where the interpolation really was
redundant. On `bulk_mixed` and `bulk_adapt` the gain is ~1%, i.e. essentially the whole
`fill_hang_info_with_equations` cost, because the rest is the dummy-value work described above.

---

## 3. Workstream B — required-shape flags

### 3.1 The audit tooling, built first

`PYOOMPH_POISON_UNREQUIRED` fills every buffer family the *passed* `required` struct does not ask for
with signalling NaN, so generated code reading an unflagged buffer produces NaN instead of a
plausible stale number. It keys on the struct actually handed to the fill, never on
`merged_required_shapes` — the point is to catch a per-pass under-request that the merge would hide.
It never touches `hanginfo` or the remap tables.

Self-test at base: clean on every case, i.e. **no pre-existing under-request in the
residual/Jacobian path**. The negative control matters more than that result: poisoning *before* the
fill proves nothing, because the fill overwrites it. `PYOOMPH_POISON_UNREQUIRED=all` poisons after
each integration point instead, and every case then dies in the Newton solver — which is what
establishes that the readers really do read those buffers.

`dev_docs/scripts/scan_required_shapes.py` scans generated `.c` for flags that are set but never
read. Absence of a read is evidence about the corpus, never proof a field is dead (code_generation.md
§9a). `scan_required_shapes_reverse.py` is the other direction — flags read but never set — and must
be **per function**: a file-wide union is exactly what hid the Hessian defect below.

### 3.2 The normal branch requested the bulk's position shapes unconditionally

`mark_further_required_fields` marks `bulk_shapes->Pos.psi` for any normal, with no
`coordinates_as_dofs` guard — while the structurally identical element-size branch 60 lines below has
one. In 481 of 482 archived generated codes that buffer is never read; 313 of the 593 codes that
request bulk shapes at all request *only* this, and 171 of those are on static meshes where nothing
could consume it.

The cost was not the buffer. A non-NULL `bulk_shapes` triggers a full bulk-element shape fill at every
integration point, the bulk's `interpolate_hang_values`, bulk external-data registration, equation
remapping — and it forces the element onto the hanging entry point, defeating the NoHang dispatch.

Fixed by adding the guard on all three spots (own/bulk/opposite). The ~24 legitimate readers of bulk
`shape_Pos` (contact-line and codim-2 elements, HDG internal facets, FSI) get the flag from the
ShapeExpansion path instead and are unaffected; a static-mesh `var("local_coordinate", domain="..")`
tutorial still emits and reads it.

Measured: `iface_normal` NoHang dispatch 134400/6720 → **141120/0**; neighbour re-interpolation
9.63 ms → **0.12 ms**; `iface_refined` res **−11.7%**, jac **−10.9%**. On a mesh where interface
elements are a few percent of the total the effect disappears into noise — the chain is provably gone
either way (dispatch counts), it just is not what that mesh spends its time on.

### 3.3 `shapes_required_Hessian` under-requested — found by the poison mode

`HessianVectorProduct` bodies dereference `dx_shapes` for bases that the flag walk never marked. It
was harmless only because the fill was all-or-nothing per space: one flag pulled the whole space in.
Narrowing the fill (§3.4) turned it into `|HVV|` off by 1.9e-1 relative 15, and
`PYOOMPH_POISON_UNREQUIRED=1` plus the narrowing turns it into `nan`.

Cause: `__all_Hessian_shapeexps` is the **emission** set, and
`ShapeExpansion::get_spatial_interpolation_name()` ignores `is_derived`. Admitting a derived
expansion next to its undifferentiated twin would declare the same C local twice — the generated code
would not compile. The `is_derived` filter is therefore correct and had to stay; reusing that set to
*also* derive the shape flags is what dropped every basis only the differentiated columns read.

Fixed with a second accumulator collecting the same expansions unfiltered, walked only for the flag
marking. A per-function reverse scan over 2117 current-ABI corpus files confirms **only Hessian
functions** were affected — `dx_shapes` in 4, `shape_Pos` in 16 moving-mesh Hessians. `ResJac` is
clean because its walk is over the *undifferentiated* residual and differentiation preserves the
basis; parameter derivatives share the `ResJac[i]` struct; the expression categories never
differentiate.

### 3.4 Per-family shape fills

The fill produced `shapes`, `dx`, `dS`, `dX` (and second derivatives, gated globally) as soon as a
space was required *at all*. A psi-only space — by far the most common combination, 1046 C2-psi files
against 678 C2-dx_psi — paid for every gradient contraction it never read.

Now each family is filled on its own flag, and `d2x` keys on the per-space flag instead of the global
"any space wants second derivatives". `dS` rides on `dX_psi` (the local-coordinate derivative has no
flag of its own and the code generator marks it as `dX_psi`); a stale comment claiming it rode on
`psi||dx_psi` was corrected.

One structural landmine was fixed on the way: a space required *only* via `dX_psi` and not the
dominant space produced `req == false`, so `dX_shapes` was never written and the generated code would
have read a stale buffer. Zero corpus instances today — but it is now impossible rather than
improbable. The requirement predicate also lived in three hand-synchronised copies and is now one
helper (`RequiredShapeFamilies`) used by the fill, `set_remaining_shapes_appropriately` and the poison.

| case | res | jac |
|---|---|---|
| bulk_mixed | **−11.5%** | **−8.2%** |
| iface_normal | **−11.4%** | **−8.5%** |
| d2x_user (`div(grad(u))` in the residual) | **−14.9%** | **−14.0%** |
| lagr_grad (moving mesh) | −2.2% | −0.7% |

`lagr_grad` is dominated by the rank-4 sensitivities, i.e. §7's item.

Element sizes now accumulate only the requested variant (Cartesian vs. not, plain vs. Hessian)
instead of both whenever either was asked for: ~1% of Jacobian assembly for a code needing one.

---

## 4. Workstream C — external data and the dense elemental block

### 4.1 The structural problem

An element's dense Jacobian block is `ndof × ndof`, and `ndof` counts every value of every attached
external `Data`. The cost is paid whatever the generated code writes: `DenseMatrix::resize` plus
`initialise(0.0)`, then an unconditional `nvar²` scatter loop with a per-entry `eqn_number()` call and
a linear row search. The CRS matrix itself stays clean — the value filter, or the
`field_contribution_index == -2` mask, prunes the zeros — so the damage is entirely elemental-side.

Attachment was driven by `merged_required_shapes`, the OR over **every** contribution: residual,
Jacobian, Hessian *and* integral, local, extremum, Z2-flux and tracer-advection expressions. So an
observable reading the opposite bulk's field attached that bulk's dofs to every interface element for
every Newton assembly, and an ODE variable used only in an observable became a dof on every element
of the domain.

### 4.2 What was built

`assembly_required_shapes` is the same OR restricted to the contributions that are actually assembled
(ResJac + Hessian). **Attachment and `update_equation_remapping` read the same one of the two, always**
— the remap hands out local equation numbers *of* the attached data, so remapping from a wider set
would resolve dofs the element does not carry. Buffer sizing and the evaluators keep the full merge;
the evaluators pass their own per-contribution struct to the fills anyway.

Output-only ODE fields needed no code-generation change: `field_contribution_index == -2` already
means "present but takes part in no contribution", and it discriminates correctly (an ODE variable in
a residual gets a real contribution class; one used only in an observable stays −2). Such links are
not registered as element external data, and `fill_element_info` reads their value straight from the
`Data` with `local_eqn = -1` — the evaluators want values and never equations. The two-pass reindex
keeps a `Data` shared between an observable and a real residual attached, addressed as before.

| case | before | after |
|---|---|---|
| interface element, interface + integral observable of the opposite bulk | `nexternal=6`, `ndof=9` | `nexternal=0`, `ndof=3` |
| bulk element, ODE read only by an observable | `nexternal=1`, `ndof=7` | `nexternal=0`, `ndof=6` |

An 81-entry dense block becomes a 9-entry one. **And it is worth 0.4-0.8% of global assembly.** The
gate for this stage measured dof *inflation*, not its cost; the elemental JIT arithmetic dominates
the `ndof²` memset and scatter by enough that a 9× smaller block on a few percent of the elements
does not show. Recorded here so the next person does not re-derive the expectation from the dof
counts.

### 4.3 What could not be done: the DG attachment gate

An interface element attaches the bulk's DG data unconditionally — a Dirichlet-condition element,
which requires no shape at all, carries half its dofs this way. Gating it on the requirements
**segfaults**: the bulk DG data is addressed *positionally*, as
`external_data_pt(external_offset_bulk + fieldindex)`, and `fill_element_info` marshals every present
DG space unconditionally. Skipping one attachment shifts every later index and reads past the end of
the external data. Narrowing this needs the DG marshalling to learn which spaces were attached, which
is a change to the hot marshalling path and not part of the external-data split. Reverted, with the
reason recorded at the site.

### 4.4 Validation

`ndof` changes, so elemental blocks are incomparable by construction and everything is compared in
**global** numbering. Residual and Jacobian bitwise identical on six cases; the one difference
(`bulk_mixed`, 6.9e-18) reproduces in a same-arm control and that case has no external data at all
(§6.2). Every observable bitwise identical — including the two risk surfaces, the opposite-bulk
gradient and the ODE read through the new direct-`Data` path. SLEPc eigenvalues bitwise identical,
four arclength steps identical to 17 digits, and a state file written in the old layout loads exactly
in the new one.

---

## 5. Invariants a future change must not break

1. **The fill skip and the dispatch must come from one boolean.** If the skip condition is ever
   recomputed independently of the function pointer that selects the entry point, a NoHang body will
   eventually run against a stale `hanginfo` from the previous element.
2. **Attachment and equation remapping read the same requirement set.** Splitting them hands out
   local equation numbers for data the element does not carry.
3. **The equation-remap channel is never skipped and never conditionally flagged.** Interface codes
   pulling bulk spaces keep unconditional hang macros (`get_hang_on_str` returns the literal `1`) and
   report "hanging" from the predicate.
4. **Poison mode keys on the struct passed to the fill**, not on the merge — a per-pass under-request
   is exactly what a merge hides, which is how §3.3 survived for so long.
5. **The reverse scan must be per function**, for the same reason.

## 6. Levers

| lever | effect |
|---|---|
| `PYOOMPH_POISON_UNREQUIRED` | signalling NaN into every unrequested buffer family; `=all` poisons after each integration point (positive control) |
| `PYOOMPH_DISABLE_SHAPE_FAMILY_SPLIT` | restores the all-or-nothing per-space fill. Widens `psi/dx/dX` and restores the **global** `d2x` gate: the second-derivative formulas need geometry quantities that only exist under it, and widening the families alone segfaults |
| `PYOOMPH_DISABLE_HANG_FILL_CACHE` / `PYOOMPH_REPORT_HANG_FILL_CACHE` / `PYOOMPH_PARANOID_HANG_FILL_CACHE` | restore / count / recompute-and-cross-check the hang skip. The paranoid mode runs the full fill on skipped elements, compares, and NaN-poisons their hang buffers |
| `PYOOMPH_DISABLE_ASSEMBLY_EXTDATA_SPLIT` | attaches (and remaps) from the full merge again, covering both the ED0 and the bulk/opposite halves |
| `PYOOMPH_DISABLE_DCOORD_SPLIT` | fills the rank-4 nodal-coordinate sensitivities for every required space again (§4a). Its own lever, not folded into the shape-family one: a wrong flag here drops Jacobian columns silently instead of producing a NaN, so it has to be switched - and FD-compared - on its own |
| `PYOOMPH_MEASURE_SKIP_HANG_FILLS` | **measurement only, unsound in general**: early-returns from both fills. Bitwise-safe only with no hangs, no constraints and no mixed-space dummy map; deliberately does not skip the `eqn_remap` channel, where it would produce garbage equation numbers rather than a stale value |
| `PYOOMPH_REPORT_HANG_FILL_TIME`, `PYOOMPH_REPORT_EXT_DATA` | the baselines of §1.1 |

A comparison whose specialised path never engaged shows two identical numbers and looks like a
perfect result. Every report lever above exists so that cannot happen silently.

## 4a. The rank-4 nodal-coordinate sensitivities, per space

`require_dxdshape` consulted no required-shape flag at all (`flag && moving_nodes &&
!fd_position_jacobian`, with the author's own XXX/TODO next to it), so `d_dx_shape_dcoord` was
filled for **every** required space whether the generated Jacobian differentiated that space's
gradient or not. It is the most expensive family of a moving-mesh fill: 40.6% of elemental Jacobian
assembly on an ALE free surface.

`dx_psi_dcoord` is now a per-space flag (an ABI addition to
`JITFuncSpec_RequiredShapes_For_Space_t`). Getting it *wrong in the safe direction* costs the whole
benefit and getting it wrong in the other direction silently drops nodal-position columns of the
Jacobian - no NaN, no crash - so the marking is deliberately attached to the emission sites rather
than re-derived:

* **Three sites emit a rank-4 read**: the interpolation loop's `required_coorddiffs`, the Jacobian
  expression printer, and the test-function printer. All three mark through one routine-scoped
  member, `current_shapeflag_func_type`, set by the RJM and Hessian writers and cleared by the
  writers of the non-assembled categories. Deriving the flag from only the first of them
  under-requested on exactly one file of the first case tried (a free-surface interface element) -
  the same trap as §9.4.15, caught the same way.
* **The marking must sit inside the branch that reads.** The test-function printer has three
  branches; only the middle one reads the rank-4 array (the first reads no sensitivity, the third
  reads the rank-6 one). Marking before the branch is *correct* but over-requests, and measurably:
  the ALE interface case went from −1.2% to −6.0% once the mark moved inside.

Verification is a per-file scan of every generated `.c` comparing flags against reads, which must
show flags ⊇ reads (and, for the benefit, flags = reads):

| case | required spaces | flagged | actually read | jac |
|---|---|---|---|---|
| ale_iface | C1 C2 | C2 | C2 | **−6.0%** |
| ale_elemsize_both | C2 | C2 | C2 | **−10.7%** |
| lagr_grad | C1 C2 | C1 C2 | C1 C2 | −0.4% |

`lagr_grad` differentiates both of its spaces' gradients, so there is nothing to skip and the honest
result is zero. `ale_elemsize_both`'s win comes from the DL twin of the same gate.

The FD comparator is mandatory here and was run both ways: zero differences at `1e-6` with the split
on and off, while the same check at `1e-14` reports 38 - i.e. it is sensitive enough to have seen a
dropped column. Poison mode covers the rank-4 buffer when unflagged, and residual/Jacobian are
bitwise identical against `PYOOMPH_DISABLE_DCOORD_SPLIT` on every moving-mesh case.

## 6a. What the campaign is worth, end to end

All campaign levers off vs. on, **one binary**, interleaved, min of 4 rounds x 20 repeats,
`_benchmark_elemental_assembly` on a quiet machine:

| case | ndof | residual | Jacobian |
|---|---|---|---|
| bulk_mixed (static C2/C1 Navier-Stokes) | 14162 | **-12.1%** | **-8.6%** |
| bulk_adapt (same, refined, real hanging nodes) | 6692 | **-11.6%** | **-8.3%** |
| iface_normal (interface with normals) | 6480 | **-15.5%** | **-13.0%** |
| iface_refined (asymmetrically refined interface) | 2216 | **-13.1%** | **-10.4%** |
| ale_iface (moving mesh + free surface) | 9695 | **-6.1%** | **-8.0%** |
| lagr_grad (moving mesh, both spaces differentiated) | 4924 | -3.1% | -1.2% |
| d2x_user (second derivatives in the residual) | 1882 | **-13.4%** | **-12.7%** |
| ale_elemsize_both (moving mesh, both element-size variants) | 2945 | -1.9% | **-12.1%** |

**This table understates the campaign**, because only the runtime stages have levers. The
code-generation changes - the normal branch's bulk-position request (§3.2), the Hessian shape flags
(§3.3), the element-size variants and the two small fixes of stage 2 - are in *both* arms of every
row above. §3.2 alone is worth another -11.7%/-10.9% on `iface_refined`, measured base-vs-head.

`lagr_grad` is the honest floor: it differentiates both of its spaces' gradients and carries no
redundant hang work, so there is nothing for any of this to skip.

Engagement, same runs (a comparison whose specialised path never fired proves nothing):

| case | hang fills skipped / run | interps skipped by class / by stamp |
|---|---|---|
| iface_normal | 3360 / 0 | 3360 / 0 |
| ale_iface | 1152 / 48 | 0 / 48 |
| bulk_adapt | 1296 / 288 | 0 / 0 |

`bulk_adapt` keeps 288 real fills - those are the genuinely hanging elements, still taking the
hanging entry point.

## 7. Open
* The DG attachment gate (§4.3), once DG marshalling is attachment-aware.
* Narrowing the `InterfaceElementBase` hang predicate to elements that really hang (it currently
  returns true whenever bulk or opposite shapes are required — conservative and correct).
* Per-value external-data granularity: oomph's `add_external_data` is whole-`Data`, so a bulk node
  contributes all of `u,v,w,p` when one field is read. Needs a wrapper or a pin mask.
* The moving-nodes attachment takes *all* bulk nodes' positions when any interface term needs a
  normal (`elements_interface.cpp`, "Isn't this overkill?"). Narrowing it to face-adjacent nodes is
  correctness-subtle.
* Smaller set-but-never-read candidates the corpus scan reports: own `psi[C2]` 36 files,
  `dx_psi[C2]` 23, `psi[C1]` 11, `dx_psi[C1]` 8, `bulk dx_psi[C2]` 1.

## 8. Issues found that are not this campaign's

1. **Symmetric-solver switching breaks the adaptive interface-coupling tests.** 17 of 137 tests in
   `tests/test_adaptive_interface_coupling.py` fail: one Newton step on a *linear* problem stalls at
   ~1e-10 instead of reaching machine zero, which the test correctly reads as "the analytic Jacobian
   does not match the residual". Setting `exploit_proven_symmetry = False` gives 137/137. The
   attribute's own documentation says leaving it on "cannot produce a wrong result, only different
   pivoting/roundoff" — on these asymmetrically refined, hanging-node interface problems that does
   not hold. It is the same pivoting class of problem as the Crouzeix-Raviart follow-up, which was
   handled by giving the *tests* a different solver.
2. **Two pre-existing nondeterminisms.** The threaded Navier-Stokes cases wobble by ~3-7e-18 in the
   residual (not the Jacobian, not the Hessian-vector product) between two runs of the *same* binary
   at identical settings; and solved states are bistable, so a comparison sweep occasionally flags a
   converged case at |R|≈5e-16. Both reproduce in same-arm controls. Any bitwise comparison here
   needs a deterministic pre-solve state (`dump.py`'s `DUMP_DETERMINISTIC`, which zeroes the low
   mantissa bits) or a same-arm control run before a difference is attributed to a change.
3. **`debug_jacobian_epsilon` is not part of the pre-codegen fingerprint.** Toggling it alone
   produces JIT cache Tier-2 shadow-mode MISMATCH warnings for codes it does not otherwise affect.
4. **`ConstrainFieldsToC1Space` on a refined two-domain interface** throws `Cannot enforce a
   degration to C1 on a C1 vertex node`.
