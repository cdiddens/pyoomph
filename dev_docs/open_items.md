# What is still open across the development notes

Status: **audit, started 2026-08-24 at `3abb7c6`, current as of 2026-08-25 at `9259455`.** This merges
the "open", "not done", "still refused" and "what is left" sections of all 47 documents in `dev_docs/`
into one register, and says for each whether the claim is **still true**. It is a map, not a plan: every
item keeps a pointer to the document that owns the reasoning, and none of that reasoning is repeated
here.

**Five commits have landed since the first pass.** Four of them closed or corrected an entry here, and
along the way **two defects turned up that were on no list at all** — the dof corruption of §1a and the
wedge/pyramid refinement abort. Both were found by *acting* on a register entry rather than by reading
one, which is the argument for re-running this exercise instead of trusting the register between passes.
Everything is folded into the entries below rather than appended, so each reads as the current state and
not as a diary; §1b is the summary.

**How each claim was checked.** Every entry below was grepped against the tree — the guard, the class,
the test file, the tutorial page — rather than taken from the document that asserts it. That is the
whole point of the exercise: **eleven claims turned out to be stale**, four of them inside a document
that already contradicts itself elsewhere on the same page. §1 is the list of corrections owed; §2 onwards is
what is genuinely still missing.

Two items are marked `UNTESTED` rather than open or closed: the code path exists and looks right, but
nothing asserts it, so it is unproven rather than absent.

**One correction to this document, made by measuring what it asserted.** The first revision ranked a
distributed-eigen-assembly gap as the most severe open item, on the strength of the *call sites* of
`sync_hanging_values_if_parallel`. Building the case it predicts showed no such effect (§3.1), and the
companion claim about the `get_eigenfunction` bindings was closed code read as open. Inferring from
where a function is called is exactly the mistake `code_generation.md` §9a warns about in another
guise: **the absence of a call is evidence about the call graph, not about the answer.** What the
exercise did turn up is in §1a, and it is worse than either.

---

## 1a. Found and fixed while auditing: a remesh inside a solve corrupts the dof vector

Not previously on any open list as a defect — `mesh_construction.md` §5.1 recorded it as a *hazard*
"read off the source, not as a reproduction", inside a section about a feature that was never built.
It is neither hypothetical nor confined to that feature.

`oomph::Problem::adaptive_unsteady_newton_solve()` snapshots the dofs by flat index before a step and
restores them the same way if the step is rejected on temporal error. `RemeshWhen` fired from
`actions_after_newton_solve()`, which oomph calls from inside `newton_solve()` — between the two. A
remesh there changes `ndof` and the meaning of every index, so the restore writes the old mesh's values
into unrelated dofs, and past the end of `Dof_pt` whenever the remesh shrank the system (`dof(i)` is
`*(Dof_pt[i])`, so that is a write through a garbage pointer).

Reproduced: a hair-trigger `RemeshWhen` on a Laplace-smoothed moving mesh remeshed inside 13 of ~24
steps, four of them shrinking the system. On the run where the coincidence landed, a 619 → 1081 remesh
in a rejected step turned a converged state into `Initial Maximum residuals 2981`, the retry into
`Maximum residuals inf`, and 26 further rejections halved `dt` against the same corrupt snapshot until
the run died with "Max. residual has been exceeded" — **a message that says nothing about remeshing**,
which is why this was never attributed. Four shipped tutorials combine `RemeshWhen` with
`temporal_error`.

Fixed: `actions_after_newton_solve` records the request and `Problem._perform_pending_remesh()` carries
it out once the C++ call has returned, from `_solve_with_adapt_recovery()`, from the two `solve()`
branches that reach oomph directly, and from the preCICE loop. `tests/test_remesh_inside_solve.py`
(4 tests, ~47 s) fails on the pre-fix tree and passes after. `ale/remeshing.py`, which has no temporal
adaptivity and therefore never had the hazard, is unchanged: same 3 remeshes, same 2920 equations.
Details in `mesh_construction.md` §5.1.

---

## 1b. What has moved since the first pass

| what | commit | where it now stands |
|---|---|---|
| A remesh from inside a solve corrupts the flat-index dof snapshot | `dc97952` | **fixed** — §1a, and `mesh_construction.md` §5.1. Was not on any open list as a defect; the doc had it as a hazard "read off the source, not as a reproduction". |
| `RemeshingOptions(on_inverted_element=…)` | `dc97952` | **built**, off by default, and the design's escalation criterion had to be replaced after measurement — `mesh_construction.md` §5.2. Also fixed the inverted-element detector **hanging** under `mpirun`, which had never been run there (§5.4). |
| Refining a wedge or pyramid aborted the run | `bfa6238` | **fixed** — `macro_elements.md` §10a. Not on this list either: the four failing tests read as a curved-boundary problem and were nothing of the kind. It leaves a NEW open item, §3.2. |
| The spurious `-1` Floquet multiplier | `199d5de` | **warned about**, which is all that can be done without changing the collocation — `floquet_multipliers.md` §6. |
| The moving-mesh droplet segfault (`mpi_augmented_systems.md` §6b) | `c33f4b0` | **the evidence was misread**; the experiment turned up an unrelated real defect (arclength continuation across a remesh was broken under `--distribute`), now fixed. See §3.1. |
| "Documentation is the largest single deficit" | `9259455` | **withdrawn** — §2(e). Two of the six documents cited never made the claim and two more were stale. |

Rows 1 and 3 are the ones worth naming: **neither was on this register, or on any other list**, and both
surfaced only because someone went to work on a neighbouring entry. A register of what the documents
*say* is open is a starting point for looking, not a substitute for it — and §2(e) below is the same
lesson from the other end, an entry that was on the list and should not have been.

---

## 1. Corrections owed: claims that are no longer true

These are recorded first because a stale "not done" is worse than no note at all — it sends someone to
build a thing that exists.

| document | what it says | what is actually there |
|---|---|---|
| `jacobian_block_flags.md` **header**, and its README entry | "The consumers they were built for … are **not built yet**; nothing reads the flags for decisions today." | Its own **§7** documents a built consumer. `Problem::get_proven_matrix_symmetry` (`src/problem.cpp:8858`, bound as `_get_proven_matrix_symmetry`) reduces the union tables to a whole-matrix verdict, and both solver base classes consume it through `exploit_proven_symmetry` (`pyoomph/solvers/generic.py:222,847`, `pardiso.py:894`), with `tests/test_symmetric_solver_switch.py` behind it. Only the *originally named* consumers (Schur reuse, fieldsplit) are unbuilt. |
| `code_generation.md` §9.4.5 | table headed "**32 (default)**"; §9.4.7 item 1 "Re-check the hoist limit" carries no DONE marker, unlike items 2, 3, 5, 6, 7 and 8 | The default is **1** (`src/codegen.cpp:2591-2593`), and the doc's own §12 table at line 1612 records the re-run sweep that moved it (1 → −63.7 %, 32 → −62.6 %, 64 → −48.5 %). Item 1 is done; two places in the same file still say 32. |
| `structural_assembly.md` §9 | "**Threaded assembly.** A precomputed scatter index makes a colour- or lock-free parallel scatter feasible. Follow-up." | Built and shipped — `openmp_assembly.md`, `--omp N` / `set_num_threads(N)`, gather-not-scatter, bit-identical, and it *requires* exactly this frozen sparsity (it declines without it). |
| `structural_assembly.md` §9, "Found on the way" | "distributed eigensolving is **broken** in the Python layer. `GenericEigenSolver.get_J_M_n_and_type()` wraps the row-local CSR in a `csr_matrix` of global shape, and scipy rejects it." | Fixed. `pyoomph/solvers/generic.py:1000-1001` builds `shape=(M_nr, n)` with a comment naming this exact failure. |
| `adaptive_refinement.md` §11 | "**Distributed eigensolves.** `--distribute` with SLEPc **segfaults** … Enabling them is: (1) fix that crash; (2) …; (3) …" | (1) is fixed — `mpi_eigenproblems.md` is "implemented and tested", `tests/test_mpi_eigenvalues.py::test_distributed_matches_serial` passes. **(2) and (3) are still open** and are carried forward in §3.1 below. |
| `mpi_eigenproblems.md` §5 | still serial-only: "bifurcation **branch switching**, **left eigenvectors** and **normal forms** (`bifurcation_tools.py`), which build global scipy matrices" | Done at `3abb7c6`. `branch_switching.md` §"Under MPI" records the four accessor substitutions and `tests/test_mpi_branch_switch.py` (18 tests). `_require_non_distributed`'s own docstring (`problem.py:1874`) already says so; §5 did not get the edit. |
| `mpi_augmented_systems.md` header / §7–§10 | the Python custom assembler is "**plan only, nothing implemented**" | **Stage 0 is built.** `Problem.set_custom_assembler` calls `_require_single_rank` (`problem.py:2932,2955`), with a comment citing §10 stage 0 by name. Stages 1–4 are indeed unbuilt. |
| `replicated_condensation_gather.md` §1 | "`citools/test_all_tutorial_scripts.py:290` **skips** `cr_static_condensation.py` on the replicated pass" | The skip is gone (commit `6ba4fc4`); `grep -n condensation citools/test_all_tutorial_scripts.py` is empty. The document's *header* already says the CR case is served by `dof_ordering.md`; §1 kept the old sentence. |
| `viscoelastic_log_conformation.md` §10 | "**A tutorial page for the benchmark.** … no `refs.bib` entries — the bibliography still contains **nothing viscoelastic**." (said twice, in two bullets) | Both are false. `docs/source/tutorial/pde/navier/viscoelastic.rst` (117 lines) is the log-conformation chapter and reproduces the confined-cylinder drag table against Claus & Phillips, with `viscoelastic_cylinder.py` alongside; `refs.bib` carries `Oldroyd1950`, `Giesekus1982`, `FattalKupferman2004`, `Hulsen2005`, `Alves2001`, `Claus2013`. |
| `nonconvergence_diagnostics.md` §1.2 | "`--largest_residuals` … implemented and **documented nowhere**" | Documented, but only as one line in `docs/source/tutorial/installation/cmdlineoptions.rst:160-161`. The section's substantive point survives intact: what is undocumented is the part that matters, the equation-number → `mesh/field` → node mapping the flag prints. |
| `electrohydrodynamics.md` §10.6 | "There is **no tutorial chapter**." | Partly stale. `docs/source/tutorial/mcflow/salts/double_layer.rst` covers the resolved double layer and the gas/liquid pairing. Still missing from §10.6's list: the capacitor, the charged wall in PB and DH, and the EHD drop. |

**One structural gap in the same class.** Nine documents are not linked from `dev_docs/README.md`, so
they are findable only by `ls`: `arclength_inner_product.md`, `assembly_overhead.md`,
`bifurcation_loci.md`, `branch_switching.md`, `critical_wavenumber_tracking.md`,
`distributed_periodic_bc.md`, `openmp_assembly.md`, `quick_continuation.md`, `surfactant_transport.md`.
Six of them are the entire bifurcation/continuation strand.

---

## 2. The cross-cutting themes

Read this section if you want the shortest list of what is actually holding things up. Everything in §3
that recurs, recurs for one of these reasons.

This used to be headed "the five things named by more than one document", and **(e) is why it is not
any more**: it was the one entry here that failed that test, and it failed it because the count was
never checked. It is kept, as a correction rather than a theme, because the way it went wrong is more
useful than the claim ever was.

**(a) History dofs under `--distribute` — mostly closed, and the residue is one call.**
`Problem::get_dofs(t,…)`/`set_dofs(t,…)` were the single blocker behind four separate refusals
(`floquet_multipliers.md` §10.2, `mpi_eigenproblems.md` §5, `mpi_augmented_systems.md` §4,
`bifurcation_loci.md`). They now work (commit `2531e00`). What is still gated on the same mechanism:
`refine_eigenfunction()` (`problem.py:5839`), which is refused for a *second* reason as well — it adapts
the mesh to an eigenfunction and that has never been validated — and `adapt()` / arclength continuation
while a C++ tracker is installed.

**(b) The Python custom assembler is the last MPI-hostile pipeline.** Deflation left it (`deflation.md`),
the Lyapunov coefficient left it (`hopf_normal_form.md` §4), the normal form left it
(`branch_switching.md`), and each departure closed a refusal somewhere else. What is still on it:
`CustomBifurcationTracker` and its family, `DeflationAssemblyHandler`, `CriticalWavenumberTracker`, and
the `HopfTracker` route to the Hopf adjoint. None has ever run under MPI, and `_require_single_rank` now
says so at the door rather than five frames deep in C++.

**(c) The state must be coherent wherever something reaches into it from outside the Newton solve.**
Two instances, and they resolve in opposite directions. `sync_hanging_values_if_parallel` guards the
assembly paths against stale hanging values and is genuinely absent from the eigen path — but measuring
shows nothing depends on it there, because the per-element interpolation covers it (§3.1). The remesh of
§1a is the same shape of question with the opposite answer: the state a remesh leaves behind is coherent,
but the *caller* is holding a flat-index snapshot of the old one, so the incoherence is in oomph's frame
rather than in the mesh. The transferable rule is the one `mesh_construction.md` §5.1 states for
inverted elements — **detect inside, decide inside, act outside** — and it is worth applying to anything
else reached from `actions_after_*`.

**(d) A dense `nbase × nbase` object is the real ceiling on orbits, not the gather.**
`floquet_multipliers.md` §9/§10.3/§10.5 measured this from three directions: the gather is 0.16 s of 13 s,
the monodromy machinery peaks at 1.6 GiB where the gathered Jacobian is 12 MiB, and shift-invert is
12–14× *slower* than forming the monodromy at every size measured because refining the mesh tightens the
multiplier cluster. Every "make it distributed" idea in that strand is optimising 2 % until something
removes the dense object.

**(e) Documentation is a real gap, but a smaller one than the first revision of this document claimed
— and the overstatement was this document's own.** It said "six documents independently note a missing
tutorial chapter for a shipped, tested feature". Checked one by one, that is wrong twice over:

* **`stabilized_navier_stokes.md` and `dof_ordering.md` never make the claim.** The first one's §7 gap
  is `tests/` coverage, not documentation; the second mentions neither a tutorial nor documentation
  anywhere. Two of the six were miscounted outright.
* **Two more were stale and are corrected in §1**: `--largest_residuals` *is* documented (one line in
  `cmdlineoptions.rst`, which undersells it — the dof mapping is what is missing, not the flag), and the
  EHD chapter that §10.6 called absent exists as `mcflow/salts/double_layer.rst` (the capacitor, the
  charged wall in PB and DH, and the EHD drop are what is still missing from it).

What survives is **two** genuinely absent chapters, and they are not independent: a tutorial chapter for
`AdaptiveResolveRecovery` (confirmed — no hit anywhere under `docs/source/`) and the non-convergence
troubleshooting chapter, both owned by `nonconvergence_diagnostics.md`, with
`adaptive_resolve_recovery.md` §8 deferring to it. One strand, not six.

So documentation is **not** the largest single deficit in this tree. That conclusion was an artefact of
counting claims rather than checking them — the same mistake §1's header warns about, made inside the
document that warns about it.

---

## 3. Open by theme

Each entry: the claim, its owning document, and the verification. `CONFIRMED` means the guard, the
absence, or the missing file was checked in this tree today.

### 3.1 MPI

| item | owner | state |
|---|---|---|
| ~~Eigen assembly does not collapse hanging values when distributed~~ | `adaptive_refinement.md` §11 (2) | **NOT A DEFECT — measured.** The call really is absent, but the eigen element loop runs each element's own `interpolate_hang_values()` and the masters are current because every dof write ends in `synchronise_all_dofs()`. On a nonlinear adapted problem (48 hanging nodes, `(1+u²)∇u` + `exp(u)`), forcing the sync before `solve_eigenproblem` moves no eigenvalue at 1 rank, 2 replicated or 2 distributed. |
| ~~`get_eigenfunction` bindings index a distributed vector by global equation number~~ | `adaptive_refinement.md` §11 (3) | **ALREADY FIXED.** All four handlers redistribute onto a globally replicated `LinearAlgebraDistribution` first, so the binding's `Ndof`-long loop is in bounds. |
| **The Python custom-assembler trackers under MPI** (stages 1–4: reduction API, parallel multi-assembly, the B6 hang). | `mpi_augmented_systems.md` §7–§11 | CONFIRMED open; stage 0 done (§1 above) |
| **`blocksolve=True`** and the handlers' block modes; **periodic orbit *tracking*** (distinct from periodic orbits, which work); **`adapt()`/arclength while tracking**; the **no-guess fold/Hopf constructors**. | `mpi_augmented_systems.md` §4 | CONFIRMED |
| ~~A moving-mesh droplet fold-track segfaults at `-n 2 --distribute`, with `ndof` already wrong before tracking~~ | `mpi_augmented_systems.md` §6b | **The evidence was misread** — the "wrong" `ndof` was the base count compared against an augmented one, and the distributed base state matches serial to 8e-12. The experiment turned up a different, real defect (arclength continuation across a remesh was broken under `--distribute`), now fixed. Whether the tracking solve still segfaults is untested by choice: the rule is not to re-run a known MPI segfault. |
| **Distributed periodic BC: refusals for a moving mesh and for adapting after distribution.** `_require_no_distributed_periodic_position_dofs` / `_require_no_distributed_periodic_refinement`. Lifting the second needs a `reconnect_periodic_trees()`. | `distributed_periodic_bc.md` §4 | CONFIRMED — both guards present (`problem.py:1886,4516,4519,2807,5213`) |
| **Periodic BC after a remesh / load balance**: `PeriodicBC.before_finalization` is not re-run, `Mesh::copied_masters` is not carried across. | `distributed_periodic_bc.md` §5 | open, untested |
| **3d remeshers**, distributed or otherwise. Everything is 2d: `Remesher2d` by name, boundary curves are polylines. | `distributed_remeshing.md` §6 | CONFIRMED — `pyoomph/meshes/remesher.py` has `Remesher2d` and `RemesherViaRecreation` only |
| **Read-back of a state across a distributed remesh** (writing works). | `distributed_remeshing.md` §6 | open |
| **Sharded state files** (one per rank). Format decision made, header field reserved, reader refuses anything but `"global"`. Estimated ~1 day. | `distributed_state_files.md` §7 | CONFIRMED — the refusal is at `problem.py:9160` |
| **Eigen/continuation data and tracers in a distributed state**; loading a state whose mesh template differs. | `distributed_state_files.md` §8 | CONFIRMED (they raise) |
| **`MeshPointLocator` under MPI** — designed in full (schedule as a first-class object, one `Alltoallv`, residue pass), not built. Gated on distributed remeshing. | `mesh_point_locator.md` §5 | CONFIRMED open |
| **Interface tracers under MPI are untested**; a particle crossing more than one element into foreign territory in one step is lost rather than migrated. | `tracers.md` §7 | `UNTESTED` — CONFIRMED, `tests/mpi_tracer_worker.py` contains no interface case |
| **Skeleton of an *interface* under `--distribute`** — refused structurally: the cross-rank pairing key is the bulk element's address, and these "bulk" elements are built on the fly. | `internal_facet_fields.md` §7.2 | CONFIRMED (`src/problem.cpp`, `setup_interior_facet_halo_scheme`) |
| **Frozen sparsity does not engage for eigenproblem assembly**, or for augmented systems when distributed. | `mpi_eigenproblems.md` §7, `structural_assembly.md` §9 | open, correct-but-slower |
| **Eigenvectors are replicated on every rank** (`Scatter.toAll`); azimuthal stability distributed is tested only on small problems. | `mpi_eigenproblems.md` §7 | open by choice at today's sizes |
| **Rank-local linear-solver failure under `--distribute`** is declined on purpose; the natural place for the agreement is `throw_solver_failure_as_newton_error`. | `adaptive_resolve_recovery.md` §8 | CONFIRMED |
| **Mesh-data operators + `global_mesh=True`**, and `create_eigendynamics_animation` on a distributed mesh. | `mesh_data_cache.md` §10 | CONFIRMED — `NotImplementedError` at `meshdatacache.py:681` |
| **`Problem.load_balance()` called by hand** dies in `generate_interface_elements` ("bulkmesh was not set") and then segfaults. Pre-existing, reproduces with no macro elements at all. | `macro_elements.md` §8.3, §11 | open, not diagnosed |
| ~~**The deflation perturbation is not partition-independent**~~ DONE (2026-08-29): drawn as a `DeterministicRandomField` per dof type over the node coordinates, so serial, replicated and `--distribute` explore the same search. | `deflation.md` §5 | FIXED |
| **Why the broadcast merge in the Z2 estimator disagreed in the third significant digit.** Unexplained; the fix was to delete the split. Three suspects listed, none excluded. | `replicated_mpi_correctness.md` §6 | CONFIRMED open — the interesting one, because reordered summation cannot produce that difference |

### 3.2 Adaptivity, meshes and geometry

* **Anisotropic / variable split schemes.** `RefinementPattern` was built for them; only the isotropic
  same-type pattern exists. Needs scheme selection, a direction-aware balance rule, and
  `rebuild_from_sons` knowing which pattern made the sons. The hang machinery is already
  scheme-agnostic. — `adaptive_refinement.md` §11.
* **3d mixed across a hex↔tet boundary** (needs the transition cell); wedge/pyramid non-uniform hanging
  still comes from the mesh-level pass. — same.
* **Tet per-element hanging inside `adapt_mesh`**, hence a `refine_selected`/`custom_adapt` hanging gap.
  — same.
* **A son→father local-coordinate map for wedges and pyramids.** New, and the residue of `bfa6238`:
  neither shape implements `get_nodal_s_in_father()`, so their interface topological ids stay incomplete
  and interface refinement coupling across such a mesh falls back to matching by position. The abort
  that used to hide this is gone; the map itself is a separate job, the pyramid especially, whose
  refinement yields six pyramids and four tetrahedra so a son can have a father of a different shape.
  — `macro_elements.md` §10a, §11.
* **Curvilinear macro elements on a moving mesh.** `macro_elements.md` §7.1 was *corrected* in the other
  direction: the route the plan proposed **does** exist, and two wires are missing —
  `Undeformed_macro_elem_pt` is never set, and `RefineableSolidElement::further_build`'s flag never
  reaches a son. Prototyped to 2.2e-16 on the quarter disc. Blocked on a policy decision (§7.2), not on
  mechanism. Q-family only as it stands. — `macro_elements.md` §7.1, §11.
* ~~**Remeshing as the response to an inverted element**~~ — **built**, along with the MPI fix it needed
  (the detection hung under `mpirun` in both modes) and the §5.1 dof-snapshot hazard it sat on top of.
  The design's escalation criterion did not survive measurement and was replaced; see
  `mesh_construction.md` §5.2 for the three counting rules that were tried and why counting inversions
  cannot separate a fold from a transient Newton iterate at all. What is still open there is §5.3's
  second decision (whether the quality trigger's move off the C++ path should have been its own commit —
  moot now, it was) and nothing else.
* **`_internal_facets_` (DG) adaptivity across a coupled interface** remains rejected outright. —
  `interface_refinement_coupling.md` §13.10.
* **3d interface skeletons** (`be->dim() != 1` throws, `src/mesh.cpp:6277`); non-conforming 3d and
  non-uniform 2d-triangle skeleton enumeration; 1d-bulk and 3d skeleton **remeshing**. —
  `internal_facet_fields.md` §7.
* **Closed-loop zeta stays two orders worse than projection** (2.3e-5 vs 8.6e-3 on a circle) and that gap
  is structural. Prefer projection; keep zeta for the cases projection cannot serve. Automatic zeta
  assignment is "of doubtful value now". — `mesh_point_locator.md` §4.4, §12.
* **A symbolic error-combination expression** (`ErrorCombination(...)`, `z2_error(group)` placeholders,
  one more generated function and a `FORMAT_VERSION` bump). CONFIRMED absent — no `ErrorCombination`
  anywhere in the tree. The parametric family covers what has been asked for. —
  `spatial_error_estimators.md` §7.4.
* **`remove_from_boundary` never frees `Boundary_coordinates_pt`** when the map empties, so
  `boundary_coordinates_have_been_set_up()` stays true on a node with none left. A `//FOR PYOOMPH`
  tidy-up. — `boundary_node_membership.md` §8.

### 3.3 Assembly, code generation and solvers

* **Codegen, in the order §9.4.7 asks for.** Item 1 is done (§1 above). Still open: **item 4** in part —
  the third (mass-matrix) body is emitted unconditionally, and both cheap tests that would fold it
  (`mass_part` identically zero; the `ResidualAndJacobianSteady<N>` routine, which has no time
  derivatives by construction) are still worth having; **item 8/9's second half**, dropping the rank-6
  `d2_dx2_shape_dcoord` / `d2_d2x2_shape_dcoord` buffers once the identity substitution lands; and the
  three genuinely dead shape-buffer fields (`nodal_shapes`, `nodal_shape_DL`, `opposite_node_index`),
  removable but layout-changing. The warning attached to that scan is the reusable part: **a field's
  absence from the generated-code corpus is evidence about the corpus, not about the field.**
* **`_jcN`/`_hcN` are recomputed once per hanging master** although they depend only on `l_test`.
  Measure on an adaptively refined mesh; a static mesh shows nothing. — `code_generation.md` §13.6.
* **`expanded_additional_field_cache` is written but never read** (`if (false && …)`), costing memory for
  nothing, and the "for the moment" comment does not record why it was backed out. —
  `code_generation.md` §10.
* **Assembly overhead, five remaining candidates**: the DG attachment gate; narrowing the
  `InterfaceElementBase` hang predicate; per-value external-data granularity (oomph's
  `add_external_data` is whole-`Data`); narrowing the moving-nodes attachment from *all* bulk nodes to
  face-adjacent ones ("Isn't this overkill?" is in the source); and the set-but-never-read shape
  candidates the corpus scan reports. — `assembly_overhead.md` §7.
* **Initialisation: `assign_eqn_numbers` × 3 is 34 % of what remains** (6.4 s of 18.5 s at N=500). Needs
  either a `fill_element_info`-aware skip or a proof the third call is unnecessary. Four smaller items
  behind it, and the profile has **only ever been run on 2d quads** — 1d, 3d, simplices and
  interface-heavy problems are unmeasured. The bigger prize named there is not initialisation at all:
  six sites in `problem.py` call `reapply_boundary_conditions` **twice in a row**, on the eigenvalue,
  tracking and continuation paths. — `initialisation_cost.md` §6, §7.
* **Static condensation**: Jacobian reuse, the line search, arclength continuation and eigenvalue
  problems all stay uncondensed; a component whose rows straddle a rank is refused; bubble *position*
  dofs on a moving mesh are representable but never run; **the distributed case is verified but not
  benchmarked**, and the extra `synchronise_all_dofs()` per Newton step is unconditional where the
  reverse external-data scan could make it conditional. — `static_condensation.md` §9.8, §10.
* **The replicated gather for a selection renumbering cannot make contiguous** (interior-penalty DG).
  Still the answer for that case; no longer needed for Crouzeix–Raviart. —
  `replicated_condensation_gather.md`.
* **Declare the block size from `PETSCSolver`.** §6 of that document establishes the entire measured 15×
  gain comes from `MatSetBlockSize`, and it is a script trick today — CONFIRMED, no `MatSetBlockSize`
  call anywhere in `pyoomph/`. `setNearNullSpace` is untouched. The measurement covers one problem,
  serial, one preconditioner. — `dof_ordering.md` §7.
* **Default `keep_structural_zeros` on**, once the MUMPS-analysis interaction is settled; a values-only
  `J − σM` update for shift-and-invert; the scatter map's memory on large 3d problems, never measured;
  **exposing the sparsity pattern to Python** (`get_jacobian_sparsity()` → `(indptr, indices)`) —
  CONFIRMED absent. — `structural_assembly.md` §9.
* **`try_to_reuse_solver` has no escalation** and can still spin on a matrix that needs one; it defaults
  to `False`. — `linear_solvers.md` §6.6.
* **Schur-complement reuse and fieldsplit KSP/PC selection** from the block flags — the two consumers
  those flags were actually built for, still unbuilt (the symmetry switch of §1 is a third one that
  arrived first). Moving-mesh blocks are proven neither constant nor symmetric; hanging and interface
  configurations are asserted nowhere. — `jacobian_block_flags.md` §6.

### 3.4 Bifurcations, eigenproblems and orbits

* **Branch switching for symmetry-breaking bifurcations** (`m != 0`, `k != 0`). The bifurcating branch
  lives in a *different function space*, so there is no dof vector on this problem to switch onto — the
  mode would have to be reconstructed into a full problem first. Worth naming because
  `AzimuthalSymmetryBreakingHandler` itself works distributed, so a user will expect the switch to
  follow. — `branch_switching.md` §"Not done".
* **The Hopf normal form's factor of two.** Measured against `get_hopf_lyapunov_coefficient`: the ratio
  is 2.000 where the quadratic terms vanish, 0.99984 where they dominate, 2.29 in between — so it is not
  one factor and cannot be fixed by one. The *prediction* is not broken: `perturbation_predictor`
  divides by `2*Re(b)` where the normal form says `Re(b)`, and the two errors cancel exactly on the case
  with a closed-form orbit. Changing either alone breaks it. — `branch_switching.md`.
* **A DAE's algebraic directions land on `(-1)**(number of intervals)`, not 0**, because Gauss–Legendre
  collocation is not stiffly accurate. With an **odd** number of intervals that puts a spurious
  multiplier exactly where a period-doubling bifurcation would be. Radau IIA would put it at 0. Verified
  to 1e-14. **The value still cannot be fixed without changing the collocation, but it is no longer
  silent**: `get_floquet_multipliers()` warns on the signature and names the discriminating experiment
  (re-solve with an even interval count — the artefact moves to `+1`, a period doubling stays at `-1`).
  Keyed on the signature rather than on proving the problem is a DAE, since assembling a mass matrix
  under the orbit handler is refused by name. — `floquet_multipliers.md` §6.
* **Asking for every Floquet multiplier costs `nbase × nT × nbase` complex** — 657 MB at `nbase=1282`,
  unguarded. The clean fix is a lazy reconstruction, but `_last_eigenvectors` is a plain attribute
  several places read directly (CONFIRMED: `bifurcation_tools.py:823`, `periodic_driving_response.py:213,275`),
  so it is a small API change. — `floquet_multipliers.md` §10.5.
* **The native distributed Floquet path** — measured and deliberately not built; see §2(d). It becomes
  right only *together with* removing every `nbase × nbase` object. — §10.3.
* **`refine_eigenfunction()` under MPI** — refused for two independent reasons. — §2(a).
* **`CriticalWavenumberTracker` is serial only**, refuses `k = 0` and `|m| <= 1`, refuses frozen
  sparsity when there is an imaginary contribution, and `_retune_arclength_theta` silently no-ops while
  it is installed. — `critical_wavenumber_tracking.md` §"Traps".
* **Deflation**: `dparameter` is not implemented, so deflation cannot combine with arclength
  continuation (CONFIRMED); `deflated_continuation` cannot connect branch points, which is intrinsic to
  the method. — `deflation.md` §8.
* **`PeriodicOrbitHandler`'s Poincaré-plane constraint is inconsistent**: `-d_plane` is not divided by
  `nelement` while the row is assembled additively per element, so the constraint is effectively
  `n0.x = nelement*d_plane` — a value that becomes rank-dependent under distribution. Worth fixing
  before orbit tracking is distributed. — `mpi_augmented_systems.md` §4.
* **The arclength `"l2"` metric gives zero weight to any dof in no time derivative** (pressure, Lagrange
  multipliers), so a step that only moves those is invisible to the constraint. `"ndof"` is the safe
  choice; a callable weights per field. — `arclength_inner_product.md`.
* **The C++ left eigenvector**, if it is ever wanted: cheaper than it looks, and distributed for free —
  `MyHopfHandler` assembles from the dense element Jacobian, so `Jᵀ` is the same loop with the index
  pair swapped, and the transposed Hessian contraction already exists. — `hopf_normal_form.md`.

### 3.5 Physics modules

* **Positivity: the log formulation for Nernst–Planck.** The largest single piece of designed-not-built
  work in the tree, written out step by step. Note the honest caveat attached to it: it is **not** a free
  improvement, because the log variable makes the equations more nonlinear, so it trades a positivity
  failure for a convergence one and the comparison needs both paths side by side. —
  `electrohydrodynamics.md` §10.1.
* **`FloatingElectrode`, charge regulation, and the Grahame `zeta_model` conversion** — named in the
  design, no code. CONFIRMED absent from the tree. §11.3's `SurfaceChargeConservation(adsorption=…)`
  supplies the *dynamic* version of charge regulation; what is missing is the equilibrium-form class and
  the zeta↔surface-charge conversion. — `electrohydrodynamics.md` §10.4.
* **The Taylor leaky-dielectric drop test** — the one test that would validate the axisymmetric Maxwell
  traction and the whole leaky-dielectric interface at once. Planned, not written; axisymmetry is
  smoke-tested only. Spatial adaptivity and MPI are untested for these equations throughout. —
  `electrohydrodynamics.md` §10.3.
* **Untested EHD surface**, by name: `lippmann_surface_tension` and the two other surface-tension
  expressions (the *sign* of the Lippmann relation should not be trusted to inspection),
  `ElectricFieldProjection`, `with_fixed_amounts`, `SternLayer`. — §10.2.
* **Stabilized Navier–Stokes has no `tests/` coverage at all.** CONFIRMED — `tests/` has
  `test_stabilized_transport.py` (the scalar counterpart) and nothing for `stabilized_ns.py`; validation
  lives in `pyoomph_runs/SUPG/stab/`. The document even names the test to write: the Poiseuille
  consistency check returns roundoff on a 32×8 mesh in seconds. Also there: **`C_I = 4` is 3× the
  classical value and only `C_I = 36` reaches O(h²)** — if any of this is promoted into `pyoomph/`, the
  default should be revisited; `tau` is isotropic where the metric-tensor form is much better; and §2's
  "elementwise zero on C1 velocities" holds on **affine elements only**. — `stabilized_navier_stokes.md` §7.
* **Viscoelastic**, with §1's corrections applied: still open are the **wake experiment of §8.4** (the
  one open question about the numbers, and the right thing to pick up first), the unexplained +0.5–2.0 %
  `du/dy` discrepancy of §8.5, Giesekus against Table 5, **DEVSS-G** (CONFIRMED absent — SUPG exists,
  DEVSS-G does not), 3d, and an inflow-profile helper built on the two existing conformation functions.
* **Salt**: crystallisation is not modelled, and neither is the salt's effect on **viscosity** — a 20 wt %
  brine is ~1.6× thicker than water and both modes treat it as water, which also means the
  Walden-corrected ion diffusivities do not see the thickening. Jones–Dole is the standard correction. —
  `salt_transport.md` §7.
* **AIOMFAC**: dissociation equilibria (bisulfate, ammonia/ammonium) and the `Qcca`/`Rcc` three-ion
  terms; the PEG special case; viscosity and water-diffusivity modules; 13 of the 28 library ions have
  no AIOMFAC parameters and are refused by name. — `aiomfac_electrolytes.md` §7.
* **Coordinate systems**: `RadialSymmetricCoordinateSystem.tensor_divergence` and
  `directional_tensor_derivative` raise (CONFIRMED, `coordsys.py:1830-1834`) — and this one **needs a
  decision before code**: the system models no `theta` coordinate, so the spherical `cot(theta)` rows
  only vanish for a genuinely spherically symmetric tensor, and there is no `define_tensor_field` to
  take as the contract. `BaseDifferentialGeometryCoordinateSystem.vector_gradient` was fixed by
  reasoning and is **unexercised — nothing in the repository subclasses it**; its sibling
  `scalar_gradient` still carries `# TODO: This is likely not right!` (CONFIRMED). —
  `coordinate_system_tensor_ops.md` §1, §7, §8.
* **`time_derivative_of_integral` breaks the telescoping at an adaptation or remesh** — inherent to the
  formulation, and the tests do not adapt. Whether its mass matrix survives the azimuthal expansion is
  pinned by no test, here or for the surfactants. — `electrohydrodynamics.md` §11.5.

### 3.6 Documentation owed

Beyond §2(e): a tutorial chapter for `AdaptiveResolveRecovery` (CONFIRMED absent — no hit anywhere under
`docs/source/`), the non-convergence troubleshooting chapter, and the four diagnostics
`nonconvergence_diagnostics.md` §2 proposes. Of those, **§2.2 is the one with no equivalent elsewhere**:
pyoomph knows the symbolic weak form and the whole equation tree before anything is assembled, so
"Stokes with Dirichlet on the entire boundary and no pressure constraint" is detectable *at setup time*
rather than as a divergence twenty minutes in. §2.3 (counting Lagrange multipliers against constraints)
is the cheaper thing to build first, and the structural sparsity machinery already computes most of it.

---

## 4. Deliberately closed, and should stay closed

Listed so that nobody reopens them from this register. Each has the reason in its own document.

* **Continuous spaces on the facet skeleton** — `internal_facet_fields.md` §2.
* **A distributed dense solve per condensation component** — that is not what condensation is for;
  `static_condensation.md` §9.8.
* **The identity substitution for `COORDDIFF` terms as a runtime win** — measured at 2.2 %; it is worth
  revisiting only for the *generated-code* argument (fewer hoisted coefficients, less compile time), and
  that is a different measurement. `code_generation.md` §9.4.4, §9.4.7 item 8.
* **Simplifying the hoisted coefficients** — measured at 2.3 % by a textual model that overstates by 5×;
  the hoist already partitioned the entry, so the CSE opportunity was consumed by the thing that created
  the coefficients. `code_generation.md` §9.4.7 item 7.
* **Enlarging the Krylov basis instead of shift-inverting** — 3× at best, non-monotonic (`ncv=300` is
  worse than the default), so there is no safe value to raise the default to.
  `floquet_multipliers.md` §10.4.
* **Replacing OpenMP with a `std::thread` pool** — the barrier is not incidental; the chunk sweep shows
  28 % lost once two barriers per chunk stop being amortised. `openmp_assembly.md`.
* **Smoothing the interface normal, and a tangential projector on `(u−w)`** — the first makes mass
  transfer ten times worse, the second is provably a no-op. `surfactant_transport.md`.
* **Arclength patch parametrisation for corner-spanning Z2 patches** — better for curves, does not
  generalise to 3d surfaces; one mechanism that degrades gracefully was preferred.
  `spatial_error_estimators.md` §2.2.
* **A `Data::set_condensed` escape hatch** — a flag on `Data` survives adaptation no better than a
  stored pair does. `static_condensation.md` §10.
* **MPI-IO for state files** — the payload becomes `nproc`-dependent unless sorted globally first, which
  is the gather it was meant to avoid. `distributed_state_files.md` §7.

---

## 5. Index: which document owns which open section

| document | open sections |
|---|---|
| `adaptive_refinement.md` | §11 |
| `adaptive_resolve_recovery.md` | §8 |
| `aiomfac_electrolytes.md` | §7 |
| `arclength_inner_product.md` | "Two things it does not fix" |
| `assembly_overhead.md` | §5 (invariants), §7 |
| `bifurcation_loci.md` | "What an orbit branch refuses", "Other guards" |
| `boundary_node_membership.md` | §8 |
| `branch_switching.md` | "Not done" |
| `code_generation.md` | §9.4.7 (items 4, 8, 9), §10, §13.6 |
| `coordinate_system_tensor_ops.md` | §1 (table), §7, §8 |
| `critical_wavenumber_tracking.md` | "Traps" |
| `deflation.md` | §8 |
| `distributed_periodic_bc.md` | §4, §5 |
| `distributed_remeshing.md` | §6 |
| `distributed_state_files.md` | §7, §8, §12 |
| `dof_ordering.md` | §7 |
| `electrohydrodynamics.md` | §10.1–§10.6, §11.5 |
| `floquet_multipliers.md` | §6, §10.3, §10.4, §10.5, §10.6 |
| `hopf_normal_form.md` | "The C++ left eigenvector" |
| `initialisation_cost.md` | §6, §7 |
| `interface_refinement_coupling.md` | §13.10 |
| `internal_facet_fields.md` | §5.4, §7, §7.2 |
| `jacobian_block_flags.md` | §6 |
| `linear_solvers.md` | §6.6 |
| `macro_elements.md` | §7.1, §8.3, §10a (the residue), §11 |
| `mesh_construction.md` | §5.3 only — §5 is built as of `dc97952` |
| `mesh_data_cache.md` | §10 |
| `mesh_point_locator.md` | §4.4, §5, §12 |
| `mpi_augmented_systems.md` | §4, §6b, §7–§11 |
| `mpi_eigenproblems.md` | §5, §7 |
| `nonconvergence_diagnostics.md` | all of it (planning) |
| `replicated_condensation_gather.md` | all of it (planning) |
| `replicated_mpi_correctness.md` | §6, §8 |
| `salt_transport.md` | §7 |
| `spatial_error_estimators.md` | §2.2, §7.4 |
| `stabilized_navier_stokes.md` | §7 |
| `stabilized_scalar_transport.md` | §10 (two-phase correction reasoned, not measured) |
| `static_condensation.md` | §9.8, §10 |
| `structural_assembly.md` | §9 |
| `tracers.md` | §7 |
| `viscoelastic_log_conformation.md` | §10 |

Four documents carry no open work at all: `openmp_assembly.md`, `precice_setup.md`,
`quick_continuation.md` and `surfactant_transport.md`. Two more are on the list for something weaker
than open work — `arclength_inner_product.md` records limitations of a finished feature, and
`boundary_node_membership.md` §8 records residual *risks* of a shipped repair rather than anything
unbuilt. They are worth reading before touching either subject all the same.
