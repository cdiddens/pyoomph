# Spatial error estimators: co-dimension recovery, per-criterion normalisation, and a dof budget

Written 2026-08-01 on branch `develop`.

Three features on the same pipeline. The first two were planned together, from opposite ends; the
third fell out of writing up §7 and turned a documented constraint into a knob.

1. **Error estimators on boundaries/interfaces** — make the Z2 flux recovery work when the mesh being
   estimated has a co-dimension (a curve in 2D, a surface in 3D). Status: **implemented**; §2.4
   records what building it turned up that §2.1 did not predict.
2. **`Problem.desired_ndof`** — steer adaptation towards a target problem size instead of towards a
   fixed error tolerance. Status: **implemented**; §5.1 records what building it turned up, §5.2 what
   was deliberately left out. §6 (moving the error overrides into C++) remains design only.
3. **Per-criterion error normalisation** (compound-flux groups) — grew out of §7, which was written
   as a constraint to design around. Status: **implemented**; see §7.2.

§7.5 covers the user-facing tutorial section the three features are documented in, including two
framings that were abandoned on evidence. Mesh-construction and plotting findings from building it
live in [mesh_construction.md](mesh_construction.md) §1.

---

Under `mpirun` without `--distribute`, the Z2 patch loop used to be split over the ranks and the
recovered coefficients broadcast back, which did **not** reproduce the same elemental errors on every
rank (third significant digit, from bit-identical state) and so refined a replicated problem to
`nproc` different meshes. Every rank now computes every patch. Why the broadcast merge disagreed at
all is still unexplained — see
[dev_docs/replicated_mpi_correctness.md](replicated_mpi_correctness.md) §3 and §4.

## 1. The pipeline as it stands

Every adaptation path in pyoomph funnels through one function. `Problem::adapt`
([src/problem.hpp:756](src/problem.hpp#L756)) calls `_adapt()`, which is `NB_OVERRIDE`d
([src/nanobind/problem.cpp:102](src/nanobind/problem.cpp#L102)) onto Python's `Problem._adapt`
([pyoomph/generic/problem.py:1964](pyoomph/generic/problem.py#L1964)), which is a one-line forward to
`_adapt_with_interfacial_errors` ([:1721](pyoomph/generic/problem.py#L1721)). So
`solve(spatial_adapt=…)`, `run(…, spatial_adapt=…)`, the initial-adaption loop, the post-remeshing
loop and `arclength_continuation` all go through the same code. Anything added there is added
everywhere.

Its stages, in order:

| # | What | Where |
|---|------|-------|
| 1 | Reset every element's `_elemental_error_max_override` to 0 | [problem.py:1742-1748](pyoomph/generic/problem.py#L1742-L1748) |
| 2 | Raw Z2 errors per bulk mesh, **stored into the override slot** | [problem.py:1749-1751](pyoomph/generic/problem.py#L1749-L1751) |
| 3 | If adapting on an eigenfunction: same again with the eigenvector as dofs, `max`-merged | [problem.py:1753-1797](pyoomph/generic/problem.py#L1753-L1797) |
| 4 | `_apply_refinement_directives()` (C++) + `calculate_error_overrides()` (Python) | [problem.py:1803-1814](pyoomph/generic/problem.py#L1803-L1814) |
| 5 | `InterfaceMesh._override_bulk_errors_where_necessary()`, deepest interface first | [problem.py:1819-1837](pyoomph/generic/problem.py#L1819-L1837) |
| 6 | Collect `errs[meshname]` from the override slots | [problem.py:1840-1844](pyoomph/generic/problem.py#L1840-L1844) |
| 7 | `_adapt_select` → `_harmonise_adapt_selection` → `_adapt_execute` → `_adapt_finalise` | [problem.py:1897-1939](pyoomph/generic/problem.py#L1897-L1939) |

Stage 2 storing the raw error in the *override* slot is the reason stages 4 and 5 can use `max()`
throughout: overrides only ever raise an error, never lower it.

Stage 3 runs under `--distribute` too since the `refine_eigenfunction()` guard came off; what a
distributed run does to the *outcome* of stages 2 and 3 alike — oomph-lib's Z2 recovery neglects the
patches it cannot assemble from locally owned vertex nodes, so elements near the threshold can be
decided differently from serial — is measured in
[mpi_eigenproblems.md](mpi_eigenproblems.md) §9.2.

### The two sentinel values

Stages 4 and 5 do not carry flags. They encode their verdict as a magic error value, chosen relative
to the two thresholds that `select_elements_for_refinement_and_unrefinement`
([src/thirdparty/oomph-lib/include/refineable_mesh.cc:318](src/thirdparty/oomph-lib/include/refineable_mesh.cc#L318))
compares against:

```
must_refine       = 100.0 * max_permitted_error   // far above refine_tol; nothing outvotes it
may_not_unrefine  = 0.5 * (max_permitted_error + min_permitted_error)   // strictly between the two
```

They are written down **twice**, once in C++ ([src/mesh.cpp:586-587](src/mesh.cpp#L586-L587)) and once
in Python ([pyoomph/equations/generic.py:311-312](pyoomph/equations/generic.py#L311-L312),
[pyoomph/meshes/mesh.py:1634-1636](pyoomph/meshes/mesh.py#L1634-L1636)). That duplication is
load-bearing for §7 below and is the single biggest reason to be careful about who may move the
thresholds and when.

---

## 2. Feature 1: why Z2 cannot currently run in co-dimension

The interface side is already almost entirely wired up, and this is easy to miss:

- `InterfaceMesh` derives from `oomph::RefineableMeshBase` ([src/mesh.hpp:317](src/mesh.hpp#L317) via
  [:72](src/mesh.hpp#L72)) and is given its own `Z2ErrorEstimator`
  ([pyoomph/meshes/mesh.py:1545-1548](pyoomph/meshes/mesh.py#L1545-L1548)).
- `SpatialErrorEstimator` attached to an interface domain really does emit `GetZ2Fluxes` for the
  interface element. Verified directly: on `"domain/bottom"` of a `RectangularQuadMesh`,
  `imesh.element_pt(0).num_Z2_flux_terms()` returns 2.
- `_override_bulk_errors_where_necessary` ([pyoomph/meshes/mesh.py:1600](pyoomph/meshes/mesh.py#L1600))
  already converts interface errors into `must_refine`/`may_not_unrefine` on the adjacent bulk
  element, on the *opposite* bulk element when there is one, and then spreads them to elements that
  touch the boundary only at a vertex.

What is missing is only the recovery mathematics. In `get_element_errors`,

```cpp
dim = el_pt->dim();                                  // src/lagr_error_estimator.cpp:1100
```

is the **element** dimension, but the coordinate handed to the recovery basis is the **global**
position:

```cpp
Vector<double> x(el_pt->nodal_dimension());          // :754
el_pt->interpolated_x(s, x);
shape_rec(x, dim, psi_r);                            // :775 — reads x[0 .. dim-1] only
```

and `shape_rec` builds a complete polynomial in exactly those first `dim` global components. The
same pattern repeats at the one other place the recovery basis is evaluated, the nodal averaging.
(Only those two: the elemental error loop compares the recovered flux against the FE flux using the
element's own shape functions, and needs no recovery basis at all.)

So a 1D interface embedded in 2D is fitted with a polynomial in the global **x** coordinate alone.
That is correct if and only if the interface happens to be a graph over the first `dim` global axes.

Measured, not inferred (probe script, `RectangularQuadMesh(N=8)` with a `C2` field and
`SpatialErrorEstimator` on the boundary):

- boundary `"bottom"` (y=0, x varying) → works, errors are sane;
- boundary `"left"` (x=0, **constant**) → the normal matrix is exactly singular and
  `DenseLU::factorise` throws `Singular Matrix`
  ([src/thirdparty/oomph-lib/include/linear_solver.cc:166](src/thirdparty/oomph-lib/include/linear_solver.cc#L166)),
  out of `_override_bulk_errors_where_necessary`, killing the run.

The failure set is not exotic: every vertical wall, every closed droplet interface, every curved free
surface, every 3D boundary plane at `x = const`. The estimator is also not rotation-invariant, which
is the sharpest available test of the fix.

### 2.1 The fix

Do the patch recovery in a **patch-local frame** rather than in global coordinates.

For each patch: origin at the centroid of the patch's nodes; axes from the `dim` dominant directions
of the node-position covariance (a 3×3 symmetric Jacobi eigensolve — small, self-contained, no new
dependency); scale by the patch extent so local coordinates land in roughly `[-1,1]`. `shape_rec`
then receives those local coordinates.

Because a complete polynomial space of order *p* is invariant under affine maps, for codim 0 this is
the *same* recovery, only better conditioned. It is nevertheless **gated to codim > 0**
(`nodal_dimension() != dim`) by default, with an opt-in flag for bulk meshes. Not because it is wrong
for bulk meshes — it strictly improves conditioning for meshes far from the origin — but because
round-off-level changes flip marginal refine/unrefine decisions, and the existing adaptive test
campaign would have to be re-baselined for no gain. Bulk results stay bit-identical.

Three supporting changes:

- **Deterministic frame.** Fix the eigenvector sign (largest-magnitude component positive, index
  tie-break) and the axis order, so a patch assembled on two MPI ranks yields the same frame. Patches
  straddling a process boundary are already approximate for an unrelated reason — see the long note at
  [src/lagr_error_estimator.cpp:532-570](src/lagr_error_estimator.cpp#L532-L570) — but they must at
  least be *consistently* approximate.
- **Non-throwing solve.** Replace `ludecompose`/`lubksub`
  ([:813-819](src/lagr_error_estimator.cpp#L813-L819)) with a truncated pseudo-inverse of the normal
  matrix, using the same Jacobi routine. The matrix is symmetric and at most 10×10. Rank-deficient
  patches — one element, exactly straight, coincident points — then degrade to the best available
  lower-order fit instead of throwing. `Singular Matrix` becomes structurally impossible.
- **Interface meshes inherit their thresholds.** `min/max_permitted_error` are set from the template
  or the Problem only for `MeshFromTemplate*`
  ([pyoomph/meshes/mesh.py:969-979](pyoomph/meshes/mesh.py#L969-L979)). `InterfaceMesh` never gets
  them, so it silently uses oomph-lib's own defaults, `1e-3`/`1e-5`
  ([src/thirdparty/oomph-lib/include/refineable_mesh.h:63-64](src/thirdparty/oomph-lib/include/refineable_mesh.h#L63-L64)) —
  `max` coincides with pyoomph's default, `min` does not (`1e-5` vs `1e-4`). Nobody chose that. Make
  interfaces inherit from their bulk parent and be settable per interface.

### 2.2 Known limitation, documented rather than solved

A patch spanning a sharp geometric corner folds under the tangent projection: both arms project onto
overlapping ranges of the same local coordinate, so the fit is poor and the corner over-refines. That
is usually the desired behaviour anyway. The alternative — parametrising 1D patches by arclength from
the patch vertex — is strictly better for curves but does not generalise to surfaces in 3D, so it was
rejected in favour of one uniform mechanism that degrades gracefully.

### 2.3 Incidental repairs in the same commit

- [src/lagr_error_estimator.cpp:756-763](src/lagr_error_estimator.cpp#L756-L763) prints
  `IPT <n>  x …` to stdout **for every integration point of every patch**, in the `use_Lagrangian`
  branch. Leftover debug output. It has never fired in practice only because both mesh types set
  `use_Lagrangian = False` from Python ([mesh.py:1127](pyoomph/meshes/mesh.py#L1127),
  [:1546](pyoomph/meshes/mesh.py#L1546)) while the C++ constructor defaults it to `true`
  ([src/lagr_error_estimator.hpp:52](src/lagr_error_estimator.hpp#L52)).
- The `shape_rec` doc-comment says "as functions of the global, Eulerian coordinate x". After this
  change that is no longer true for co-dimensional meshes.

### 2.4 What building it turned up

Three things the plan above did not anticipate.

**The nodal averaging cannot average coefficients across frames.** After the per-patch fits, the
recovered flux at a node is obtained by summing the coefficient matrices of every patch the node
belongs to and evaluating the result once. That is only valid while all patches share one basis.
With a frame per patch the coefficients live in different bases and adding them is meaningless: the
evaluation has to happen per patch, in that patch's frame, and the *values* averaged. Algebraically
the two are the same thing when the frames coincide (evaluation is linear in the coefficients), but
not in floating point — the summation order differs. So the global-frame path is kept verbatim
rather than folded into the new one, which is what keeps bulk meshes bit-identical.

**A non-distributed parallel job broadcasts the coefficients between ranks.** Patches are shared out
across processes and each rank broadcasts its results, so a rank ends up evaluating coefficients
that were fitted in *another rank's* frame. Rather than serialise a frame per patch, the receiving
rank rebuilds it: the mesh is not distributed, so it holds every element of the patch, the element
list has just been broadcast in the sender's order, and `build_recovery_frame` is deterministic
given that list. Which is why the eigenvector sign convention of §2.1 is load-bearing and not
cosmetic.

**The mesh's embedding dimension has to be reduced like `dim` is.** The driver reads `dim` from the
first element and `MPI_Allreduce`s it, because a distributed submesh may hold no elements at all on
some rank. `nodal_dimension` now selects the recovery path, so it needs the same treatment - a rank
that guessed differently would take a different branch through a collective.

The LU solve is likewise kept for the global-frame path. The truncated pseudo-inverse would be a
strict improvement there too (it cannot throw), but it is a different rounding, and it is not worth
re-baselining every adaptive test to gain robustness on a path that has not been observed to fail.

Measured after the change, on an 8x8 quad mesh with a `C2` front projected onto a boundary:

| | |
|---|---|
| frame vs. no frame, boundary the old code could do | agrees exactly (0 relative difference) |
| same boundary rotated 30° / 90° | 5e-13 / 2e-14 relative |
| vertical boundary vs. horizontal | agrees exactly |
| bulk mesh, frame forced on vs. off | 3e-12 relative |
| vertical boundary, frame disabled | still throws `Singular Matrix` |

In 3D, two of the three face orientations of a brick (`x=const` and `y=const`) throw without the
frame; only the `z=const` face, the one that is a graph over the first two global axes, ever worked.

### 2.5 Worked example

[dev_docs/examples/sessile_droplet_surfactants.py](dev_docs/examples/sessile_droplet_surfactants.py):
an axisymmetric sessile droplet with insoluble surfactants and a body force near the axis that
stirs a toroidal roll, sweeping surfactant from the apex to the contact line. `SpatialErrorEstimator`
is added **on the free surface**, on both the surfactant concentration and `"normal"` (whose jump is
the interface curvature). Runs 2 s in about 19k dofs.

Two things it is worth reading it for.

**It has to set the interface's refinement thresholds explicitly.** With the Problem-wide defaults,
*every* interface element sits above the refine threshold and the "adaptive" refinement along the
interface is uniform. That is §7.1 in practice: the errors are each mesh's own normalised share, so
a free surface of ~100 elements reports numbers two orders of magnitude larger than a bulk mesh of
tens of thousands. The example sets `0.1 / 0.02` on the interface mesh, via a four-line `Equations`
subclass using the same `after_compilation` hook `RefineToLevel` uses. Note that scaling the
estimator *expression* instead would do nothing at all — the normalisation divides it back out.

**It is honest about not being a demonstration of the bug.** At a 90° contact angle the droplet's
free surface is still a graph over `r`, so the old global-coordinate recovery was merely
ill-conditioned near the contact line, not singular: the two agree to 1e-11 on this geometry. The
cases that genuinely could not run are in the test file, not here. An overhanging contact angle
would be such a case, but reaching one from a hemispherical initial condition needs continuation and
diverges if simply imposed, which is a separate problem from the estimator.

---

## 3. Feature 2: `Problem.desired_ndof`

`desired_ndof: int|None = None` on `Problem`. When set, adaptation aims at a problem *size* rather
than at an error tolerance: refine the elements with the largest errors until `ndof()` is
approximately `desired_ndof`, unrefine the smallest when it overshoots.

Supporting attributes: `desired_ndof_tolerance` (default `0.1`) and a per-step growth cap.

---

## 4. Where the controller has to sit, and why

**Between stage 2/3 and stage 4** of the table in §1 — after the raw estimator errors are gathered,
before `_apply_refinement_directives` and `calculate_error_overrides` run.

This is not a matter of taste. Two placements that look natural are both wrong:

- **Rescaling the error vector at stage 6** (multiply every error by *s* so the top *k* clear the
  fixed threshold) breaks the sentinels of §1. `must_refine = 100·max` survives any `s > 0.01`, but
  `may_not_unrefine = 0.5·(max+min)` only stays above `min_permitted_error` while
  `s > 2·min/(max+min)` — with the default 10:1 ratio that is `s > 0.18`. Below it, every element an
  interface or a `RefineAccordingToElement` callback asked to *protect* is silently unrefined
  instead. The failure is quiet and would look like an unrelated adaptivity bug.
- **Moving the thresholds after stage 5** has the same defect from the other side: the sentinels were
  already computed from the old thresholds and no longer sit where they are meant to sit.

Setting the thresholds *first* means stages 4 and 5 compute their sentinels from the new values, and
the invariant holds by construction. The cost is that interface-driven overrides can push extra
elements over the budget — accepted: those are mandatory refinements anyway, and the controller
re-measures on the next adapt step.

---

## 5. The controller — **implemented**

`Problem.desired_ndof` (default `None`), with `desired_ndof_tolerance` (0.1),
`desired_ndof_damping` (0.7) and `desired_ndof_max_growth` (4.0). Implemented in
`Problem._apply_desired_ndof_controller`, called from `_adapt_with_interfacial_errors` at exactly the
point §4 argues for.

**It is ordinal.** This turned out to be the clarifying property. The controller picks an order
statistic of the error distribution and puts the threshold there; it never uses the *magnitude* of an
error. So on a single mesh it does not care how the estimator is normalised at all. The whole of §7
only bears on it when the ranking is pooled across several meshes — which narrows a worry that looked
foundational down to the multi-domain case, and makes the single-mesh case unconditionally sound.

**The model.** With `N` elements that can still move and current `ndof`, refining `k` of them gives
roughly `ndof·(1 + k(2^d − 1)/N)`; invert for `k`, damp, cap the growth per step. Elements already at
`max_refinement_level` are excluded from `N` when growing and those at `min_refinement_level` when
shrinking — counting either would make the controller aim at a change it has no way to produce.
Halo copies are skipped, or every shared element would be counted once per process holding it.

**The threshold.** Serially a `numpy.partition`. Under MPI the errors live on different ranks, so
rather than gathering them the threshold itself is bisected — 60 rounds of "count locally, MPI-sum,
halve the bracket". Fixed cost, independent of the distribution, and every rank comes out with the
same number by construction, which it must, since it becomes a mesh-wide tolerance. `get_mpi_min` and
`get_mpi_max` were added alongside `get_mpi_sum` for the initial bracket.

**The dead band** flags nothing, so the adaptation reports `nref == nunref == 0` — the signal every
calling loop already breaks on. Without it the controller would hunt about the target forever and
every loop would run to its step limit.

### 5.1 What building it turned up

**Unrefinement is not the mirror of refinement, and the first version stalled on it.** oomph merges a
father only if *all* of its sons agree, and the smallest-error elements do not come in complete
families, so the count below the threshold is only an upper bound on what actually merges — often a
loose one. Measured: shrinking 18790 → target 4000 took 8 steps and *terminated at 4512*, outside the
dead band, having asked for unrefinement and been vetoed. Terminating outside the target with no
explanation is the worst of both worlds.

Fixed by making the controller respond to being ignored: a step that asked for a change and got none
doubles its next request (capped at 32×), and a step that moved resets it. The same 18790 → 4000 now
takes **3 steps** and lands at 4388, inside the dead band. When even the escalated request cannot be
met it says so once rather than looping in silence.

**The escalation had to be taught to ignore the dead band.** First version escalated on any step where
`ndof` did not change — including the dead band, where standing still is the whole point. A run idling
at its target would have arrived at the next genuine adaptation with a 32× request behind it. The
counter is now only armed by steps that actually asked for something.

**A target below the initial mesh crashed.** At the coarsest level no element has a father, so the
unrefinement candidate list is globally empty and the order statistic reduced over a zero-size array.
Both directions now check availability first and report that the target is unreachable, which is a
statement about the mesh rather than an error.

**Measured, on a peaked Poisson source on an 8×8 quad mesh:**

| | |
|---|---|
| target 3000 | 780 → 2766 in 4 steps, terminates |
| target 20000 | 780 → 18790 in 6 steps, terminates |
| 18790 → target 4000 | 4388 in 3 steps |
| target 10 (below the coarse mesh) | stays at 272, reports unreachable |
| via `solve(spatial_adapt=12)`, target 8000 | 7458 |
| target 1200 **plus** `RefineToLevel(3)@top` | 1652, top row all at level 3 |

That last row is §4 in action: the directive wins over the budget, because the thresholds were
decided before the sentinel was computed from them.

**Refinement placement**, which the budget alone says nothing about: with target 8000, mean refinement
level 3.5 within `r < 0.15` of the source peak against 1.2 beyond `r > 0.4`.

### 5.2 Not done: equidistribution at constant ndof

Inside the dead band the controller does nothing. The natural extension is a dof-neutral swap —
refine the worst elements while merging an equal dof cost of the best — which is what a *moving*
feature in a transient run wants, since otherwise a mesh that has reached its target stops following
the solution. It is left out because it has no termination signal: `nref == nunref == 0` is what ends
every adaptation loop, and a controller that keeps swapping never produces it. Doing it properly
needs a separate convergence criterion (swap only while the top-to-bottom error ratio exceeds some
factor, which is self-limiting because refining lowers the top and merging raises the bottom — and
which works far better with `normalize_relative=0`, since relative errors barely move under
refinement).

### 5.3 Decisions taken without being asked

- **The ranking is pooled globally** across all refineable meshes, not per-mesh budgets. It matches
  the fact that a single Problem-level threshold pair has always been broadcast to every mesh, and
  §7.2's per-mesh weighting knob now exists (`weight` on `SpatialErrorEstimator`), so tilting the
  split no longer needs its own mechanism.
- **`desired_ndof` counts `Problem.ndof()`**, i.e. everything, including ODE domains and global
  Lagrange multipliers that no amount of refinement will change. The alternative — counting only
  adaptable dofs — would make the number the user sets differ from the number the solver reports,
  which seemed the worse surprise.

---

## 6. Should the element error overriding move into C++?

There is precedent and a stated reason. `RefineToLevel` and `RefineMaxElementSize` used to be Python
loops and were moved into `Mesh::apply_refinement_directives`
([src/mesh.cpp:577-640](src/mesh.cpp#L577-L640)); the comment at
[pyoomph/equations/generic.py:258-262](pyoomph/equations/generic.py#L258-L262) records why:

> Same values, but evaluated for every element this process holds — **halo copies included** — so a
> halo copy reaches the same verdict as the element it copies, instead of being one more rank-local
> override for the halo exchange to repair afterwards.

What is still in Python: `Equations.calculate_error_overrides()` and its users
([`RefineAccordingToElement`](pyoomph/equations/generic.py#L308-L323),
[NSCH.py:222](pyoomph/equations/NSCH.py#L222), [low_order_NSCH.py:245](pyoomph/equations/low_order_NSCH.py#L245)),
and `InterfaceMesh._override_bulk_errors_where_necessary`
([pyoomph/meshes/mesh.py:1600-1662](pyoomph/meshes/mesh.py#L1600-L1662)).

### 6.1 What moving them would buy

1. **Halo consistency, for free instead of by repair.** Python loops iterate `mesh.elements()`, which
   is rank-local; the C++ pass covers halo copies. Today the divergence is patched afterwards by
   `synchronise_elemental_errors` ([src/mesh.hpp:748-758](src/mesh.hpp#L748-L758)), a MAX-reduce over
   all copies of each shared element. That works, but it is a repair, and the comment there is
   explicit that it exists *because* the overrides are applied rank-locally.
2. **Cost.** One Python-level loop over every element per adapt, with a nanobind property get **and**
   set per element, on top of the loop at stage 2 and again at stage 6. On a multi-million-element
   mesh this is a visible fraction of adapt time and it is pure marshalling.
3. **One definition of the sentinels.** §1 notes they are written down twice. Feature 2 makes the
   thresholds move at runtime, which turns a duplicated constant into a latent bug: if the two copies
   are ever computed at different points in the sequence they disagree, and the symptom is a wrong
   refinement decision with nothing in the log.

### 6.2 Risks — inspect these before starting

1. **`calculate_error_overrides` is a public extension point and cannot be removed.**
   `RefineAccordingToElement` takes an arbitrary Python callback (`level_func: Callable[[Element],int]`).
   No C++ pass can evaluate that. So the move is necessarily *partial*: pure-function-of-element-state
   overrides go to C++, callback-driven ones stay, and `synchronise_elemental_errors` must **stay**
   for the remainder. A partial move that lets someone conclude "overrides are now halo-safe" is worse
   than no move at all — the guarantee would be true of some overrides and false of others, with
   nothing in the code saying which.
2. **Halo elements do not have the same Python-visible state.** A C++ pass evaluating a directive on a
   halo copy is only equivalent if the quantity it reads is itself halo-consistent. `refinement_level()`
   and `size()` are; a field value that a Python callback might read need not be. Any per-element
   quantity added to the C++ pass has to be checked for this individually, not assumed.
3. **The interface→bulk override is a distribution problem, not a language problem.** Moving
   `_override_bulk_errors_where_necessary` to C++ does not fix its known weakness: under MPI,
   `get_opposite_bulk_element()` may return an element this rank does not own, or nothing at all
   ([problem.py:1855](pyoomph/generic/problem.py#L1855) records this). Rewriting it in C++ while that
   is still true buys speed and halo coverage but no correctness, and risks *looking* like it fixed
   something.
4. **The stage ordering becomes a cross-language contract.** §4's whole argument is that the
   thresholds must be set before the sentinels are computed. In Python that ordering is visible in one
   function. Split across languages it becomes an implicit contract that a future refactor can break
   silently. **Mitigation, and it should be treated as mandatory:** pass the two threshold values as
   explicit arguments into the C++ pass rather than letting it read `max_permitted_error()` off the
   mesh. Then a caller that has not decided the thresholds yet cannot call it at all.
5. **Ordering *within* stage 4/5 is load-bearing and undocumented.** Stage 5 runs deepest-interface-first
   ([problem.py:1830-1837](pyoomph/generic/problem.py#L1830-L1837)) and every write is a `max()`. Both
   properties must be preserved exactly; neither is asserted anywhere. Add a test that pins them
   *before* touching the code.
6. **Low test coverage of the override slot itself.** A grep over `tests/` finds exactly one reference
   to this machinery ([test_pyramid_refinement.py:366](tests/test_pyramid_refinement.py#L366)), and it
   is a comment. The behaviour is covered only indirectly, through end-to-end refinement outcomes. A
   move should be preceded by direct tests on `_elemental_error_max_override` after each stage.

**Recommendation.** Do not bundle this with `desired_ndof`. Take risk 4's mitigation (explicit
threshold arguments) even without the move — it is cheap and it is what makes the controller safe.
Treat the move itself as separate work, prerequisites: tests from risk 5 and 6 first, and a decision on
risk 1 about what the halo guarantee is actually being claimed to be.

---

## 7. Relative vs. absolute errors, and per-criterion normalisation

Status: **implemented** (7.2), except the symbolic combination expression of 7.4.

This is the part most likely to be got wrong, because the errors are **not** what they look like.

### 7.1 The errors are already normalised, per mesh

The last stage of `get_element_errors` divides every elemental error by the **mesh-global flux norm**
([src/lagr_error_estimator.cpp:1775-1799](src/lagr_error_estimator.cpp#L1775-L1799)):

```cpp
normalised_compound_flux_error[i] =
    elemental_compound_flux_error(e, i) / (flux_norm[i] + 1.0e-9);   // :1788
```

where `flux_norm[i] = sqrt(∑_elements ∫ |flux_rec|² dx)` over the whole mesh, `MPI_Allreduce`d when
distributed ([:1748-1758](src/lagr_error_estimator.cpp#L1748-L1758)). Three consequences:

1. **The errors are dimensionless.** `max_permitted_error = 1e-3` means "this element carries 0.1% of
   the mesh's total recovered flux norm as error". So a single threshold across meshes carrying
   physically different fields — temperature here, velocity there — is at least *dimensionally*
   meaningful. That is what makes a global ranking defensible at all.
2. **A per-mesh scale factor cancels exactly.** Multiplying every flux on one mesh by a constant
   multiplies both the elemental error and the flux norm by that constant. `SpatialErrorEstimator(u=2)`
   is identical to `SpatialErrorEstimator(u=1)`. Only *ratios between fields on the same mesh*
   (`SpatialErrorEstimator(u=1, v=10)`) do anything — all fluxes land in one compound flux, whose
   error is summed before normalisation
   ([:1587-1596](src/lagr_error_estimator.cpp#L1587-L1596)), and
   `get_combined_error_estimate` then takes the max over compound fluxes
   ([:488-524](src/lagr_error_estimator.cpp#L488-L524)), of which there is normally one. Any future
   "weight this mesh more" knob must therefore live **outside** the estimator; putting it in the flux
   expression does nothing at all, silently.
3. **Normalisation destroys absolute comparability.** Each mesh is normalised by its *own* norm, so a
   perfectly resolved domain and a badly resolved one produce error distributions of similar
   magnitude. Their *shapes* differ, their *scales* do not. A global dof budget ranked on these
   numbers will therefore hand budget to a well-resolved domain roughly in proportion to its element
   count, not to how much it needs.

### 7.2 Feature 3: compound-flux groups — **implemented**

The above was written as a constraint to design around. It became a feature instead.

oomph-lib's Z2 estimator has always supported *compound fluxes*: `ncompound_fluxes()` and
`get_Z2_compound_flux_indices()` partition the flux terms into groups, **each group is normalised by
its own norm**, and `get_combined_error_estimate` combines the groups by taking the maximum
([:488-524](src/lagr_error_estimator.cpp#L488-L524)). pyoomph never overrode either hook, so
everything landed in group 0 and shared one norm. It now uses them:

```python
eqs += SpatialErrorEstimator(u=1)                                        # group ""
eqs += SpatialErrorEstimator(Gamma=1, group="surf", normalize_relative=0, weight=3)
# error[e] = max( err_u[e]/norm_u , 3*err_Gamma[e] )
```

`normalize_relative` and `weight` are properties of the **group**, because the norm being divided out
is a whole-group quantity. Everything lands in the single unnamed group unless a group is named, so
the historical one-joint-norm behaviour is what you get by default.

**Why the maximum, and why it is not negotiable.** The composition rule has to work when independent
parts of a model each contribute a criterion without knowing about each other — the droplet example
in §2.5 already stacks equations from four unrelated places. `max` is the only rule that is
simultaneously order-independent (the result cannot depend on the order equations happened to be
added in) and **monotone**: adding a criterion can only raise an element's error, hence only ever
cause more refinement and less unrefinement. A sum is not monotone in the useful direction — a
criterion could be diluted by adding another, which is exactly the failure the grouping exists to
prevent.

That also settles where vetoes live. "Refine this regardless" and "do not unrefine this" have to
*lower* or pin an element's error, which would break the monotonicity that makes `max` safe. They
stay in the override-sentinel mechanism of §1 (`RefineToLevel`, `RefineMaxElementSize`,
`RefineAccordingToElement`), and criteria stay here. The split is now load-bearing, not incidental.

**Consequence for weighting.** §7.1.2 said a common factor on a mesh's estimator expressions cancels
exactly. That is still true *within a group* — `SpatialErrorEstimator(u=2)` remains identical to
`SpatialErrorEstimator(u=1)`. `weight` is applied **after** the normalisation and therefore does not
cancel; it is the only way to move one criterion up or down against the others. Verified both ways in
`test_factor_inside_a_group_cancels_but_weight_does_not`.

**What this leaves for the dof budget.** §7.2's option (b) — "global ranking with explicit per-mesh
weights" — no longer needs its own mechanism. A per-criterion `weight`, plus `normalize_relative=0`
where cross-mesh comparability is wanted, is that knob. Option (c), per-mesh dof budgets, is still
open and still not obviously needed.

### 7.2.1 How it is built

| Layer | What it carries |
|---|---|
| [codegen.hpp:902](src/codegen.hpp#L902) / `add_Z2_flux` | a group index per flux expression, plus per-group exponent and weight; rejects two criteria that name one group and disagree about it |
| [jitbridge.h](src/jitbridge.h) | `num_Z2_compound_fluxes`, `Z2_flux_group_index`, `Z2_group_normalize_relative`, `Z2_group_weight` (and `_for_eigen` twins) |
| [jit_cache.py](pyoomph/generic/jit_cache.py) | `FORMAT_VERSION` 3 → 4 — the function-table layout changed, so an old cached `.so` must not be reused |
| [elements.cpp](src/elements.cpp) | `ncompound_fluxes()` / `get_Z2_compound_flux_indices()` overrides, plus the two per-group accessors |
| [lagr_error_estimator.cpp](src/lagr_error_estimator.cpp) | reads the per-group settings off element 0 and applies them before the existing max |

Two deliberate choices in there:

- **The default path stays bit-identical.** The generated code emits the group arrays *only* when
  something non-default was asked for; otherwise the pointers stay null, the estimator takes its old
  branch, and the weight multiplication is skipped rather than performed with 1.0.
- **The settings are read off the element, not held on the estimator.** The grouping is declared by
  the equations compiled into the element code, so the element is the only thing that knows which
  criterion is which group. The mesh-level `Z2ErrorEstimator.normalize_relative` still multiplies on
  top, so a whole mesh can be switched to absolute errors from Python without touching the equations.

### 7.2.2 A latent MPI bug this uncovered

`n_compound_flux` is the element count of the `MPI_Allreduce` on `flux_norm` and of the halo error
exchange. It was computed by a rank-local max over non-halo elements, starting from 1 — so before
grouping every rank reached 1 and the disagreement was invisible. With groups it is not: a rank
holding none of a grouped domain would reduce over one entry while its neighbours reduce over
several. Now `MPI_MAX`-reduced before use, alongside the existing reductions for `dim` and
`nodal_dim`. Not reachable before this change, but it was one group away.

### 7.3 Absolute errors, if they are ever wanted

oomph-lib also supports pinning the normalisation outright: if `Reference_flux_norm != 0.0` it
replaces the computed norm for every compound flux
([src/lagr_error_estimator.cpp:1731-1741](src/lagr_error_estimator.cpp#L1731-L1741)). That is a
different thing from `normalize_relative=0`: it keeps a divisor, but a *fixed* one, so the errors stay
comparable across adaptation steps even as the solution evolves (with the computed norm the
denominator moves underneath you, so an unchanged solution on a changing mesh does not give an
unchanged error). Now exposed as `reference_flux_norm` on the `Z2ErrorEstimator` binding.

Also note the `+ 1.0e-9` offset on the denominator (marked `CHANGE C. Diddens` at
[:1788](src/lagr_error_estimator.cpp#L1788)). It is a guard against a vanishing flux norm, a no-op
whenever the norm is much larger than `1e-9`. But it is an *absolute* constant sitting inside an
otherwise scale-free expression: for a nondimensionalisation where the flux norm is itself of order
`1e-9`, it silently halves every error on that mesh. It is also why
`test_relative_error_is_blind_to_the_solution_scale` holds only to `1e-6` and not exactly. Worth
revisiting.

### 7.4 Still open: a symbolic combination expression

Groups give a fixed parametric family: `max_g ( w_g * E_g / N_g^p_g )`. That covers relative,
absolute, blends, weighting and independent criteria — but not a *cross-group* formula (a ratio
between two groups, a `heaviside` gate of one field on another).

The idiomatic pyoomph answer if that is ever wanted is a symbolic combination expression, compiled
like everything else rather than called back into Python:

```python
eqs += ErrorCombination(maximum(z2_error("u")/z2_norm("u"), 3*z2_error("surf"))) @ "domain"
```

with placeholders `z2_error(group)`, `z2_norm(group)` and probably `z2_element_size` /
`z2_mesh_volume` (dividing by element volume gives an error *density*, which is what you want when
elements differ wildly in size). It needs one more generated function in the table
(`GetZ2CombinedError(E[], N[])`, called once per element) and hence another `FORMAT_VERSION` bump.
Several `ErrorCombination`s on one domain would compose the same way groups do — by `max` — so the
rule stays uniform.

Not built. The parametric family covers what has actually been asked for so far.

---

## 7.5 The tutorial section

`docs/source/tutorial/pde/adapt.rst` documents all of this for users, in two examples that were
each built to demonstrate one thing and ended up demonstrating something more useful.

**Moffatt eddies in a wedge** (`adapt/moffatt.rst`). Stokes flow in a sharp corner has an infinite
sequence of counter-rotating eddies, so no tolerance can be chosen -- there is always another eddy
below it. That is the cleanest available motivation for `desired_ndof`. The page's real lesson is
the one that was not planned: the plain energy-norm estimator **ignores the corner entirely**. At
5 000, 20 000 and 80 000 dofs the mesh reaches `r = 9.91e-3` every single time, which is just the
initial spacing at the apex, and the elemental errors are nearly uniform (median 9.7e-7 against max
4.4e-6) because Moffatt eddies carry almost no energy. Handing the estimator
`sym(grad(u))/r**3` instead -- the cascade is self-similar in `log r` -- reaches `6.19e-4` at the
same cost. §2 of this document says the flux is a modelling choice; that page is what it looks like
when the choice is made badly.

**A heated cylinder** (`adapt/cylinder.rst`). Flow past a cylinder with a temperature field and a
trace species a thousand times weaker, entering as a filament away from the cylinder. It is the
worked example for §7.2: under one group the tracer contributes a millionth of the summed error and
never registers; one group per field rescues it. Same budget, 376 vs 1433 elements on the filament.

The cylinder page also carries a negative result worth keeping. It does **not** claim the grouping
is more accurate: the two criteria give Nusselt numbers 6.662 and 7.070, and a larger grouped
computation gives 6.7825 while still moving by four percent, so which is closer to the truth is not
settled by anything done there. Establishing it would need a convergence study aimed at the wall
rather than at the tracer. What is demonstrable, and what the page claims, is that at fixed cost
what you ask the estimator to look at changes what you get.

Two earlier framings of that page were abandoned on evidence, which is worth recording so they are
not re-attempted. Momentum-vs-thermal criteria on the same wake do **not** separate: at Pe=200 and
Pe=4000, `joint`, `grouped` and `temperature`-only agree to four digits, because a joint norm is
dominated by whichever field has the sharper gradients and the two fields want the same region. And
a trace species released *from the cylinder* does not separate either, for the same reason. The
separation needs the weak field to be structured somewhere the strong one is not -- hence the
filament entering away from the body.

Building those pages turned up a set of mesh-construction and plotting findings that have nothing to
do with error estimation; they are written up separately in
[mesh_construction.md](mesh_construction.md) §1. The short version: use a
transfinite O-grid for a wall-bounded problem that will be adapted, because Gmsh's `BoundaryLayer`
field produces a nicer initial mesh but stops converging after the second refinement; and velocity
streamlines cannot be drawn over a deeply refined mesh, because matplotlib's `TriFinder` rejects the
non-conforming triangulation that hanging nodes produce.

---

## 8. Tests

**Feature 1** — `tests/test_boundary_error_estimator.py`, 12 tests, all passing:

- a vertical boundary (`"left"`) no longer throws — the exact case that failed before;
- the same case *still* throws with `use_local_recovery_frame_in_codim = False`, which is what shows
  the frame is doing the work and not something else;
- the frame reproduces the global-coordinate recovery on a boundary the old code could do;
- **rotation invariance** at 30°, 90° and 137°, and agreement between all four edges of a square.
  The global-axis recovery cannot pass this, which makes it the load-bearing test;
- a closed curved interface (`CircularMesh` circumference) gives localised, finite errors;
- a 3D box face at `x = const`, matching the `z = const` face the old code could already do, and
  throwing without the frame;
- a codim-0 test asserting the frame is off by default for bulk meshes *and* that forcing it on
  changes nothing — the evidence that it is a change of basis, not a change of estimator;
- interfaces inherit their refinement thresholds from the bulk mesh.

Ran alongside: `test_adaptivity`, `test_adaptive_interface_coupling`, `test_constrained_adaptivity`,
`test_triangle_refinement`, `test_tet_refinement` (217 passed) and the non-slow part of the 2D/3D
campaigns plus `test_mixed_mesh`, `test_pyramid_refinement`, `test_wedge_refinement` (138 passed).
The `--full` campaigns and the MPI suites have **not** been run.

**Feature 3** — appended to `tests/test_adaptivity.py`, 12 tests: the meaning of the two extremes
(relative is blind to the solution scale, absolute follows it), the geometric blend in between,
absolute error shrinking under refinement, and for the grouping: two groups are exactly the
elementwise max of each alone, adding a criterion can only raise the error, separate groups do not
agree with one joint group, a factor inside a group cancels while `weight` does not, and two criteria
naming one group but disagreeing about it are rejected.

**Feature 2** — `tests/test_desired_ndof.py`, 8 tests, all passing: a peaked-source Poisson reaches
each of two targets within tolerance **and terminates**; the budget is spent where the error is
(refinement level 3.5 at the peak against 1.2 away from it); lowering the target from an
already-refined state shrinks back into tolerance; a target below the initial mesh is left alone; the
thresholds are handed back when `desired_ndof` is unset; the ordinary `solve(spatial_adapt=...)` entry
point works; and — the §4 sentinel hazard, pinned — a `RefineToLevel` directive under a deliberately
too-small budget still gets its mandatory refinement.

Not covered: **MPI**. The threshold bisection is written to be rank-agnostic and the halo skip is in
place, but neither has been run under `mpirun`.

---
