# Validation campaign for mixed adaptive meshes (branch `mixed_adapt`)

Companion to [`mixed_adaptive_meshes.md`](mixed_adaptive_meshes.md), which describes the *engine*. This
document covers the **validation campaign**: making the physics that users actually run — mixed C1/C2
fields (Taylor–Hood Stokes), C1-space constraints, ALE / moving meshes, Neumann boundary conditions —
demonstrably correct on adaptive quad / triangle / mixed meshes in 2D and on all 3D shape families, both
serially and under `mpirun -n N --distribute` with more than one refinement level.

Status legend: **[DONE]** landed and verified · **[READY]** verified to work, needs codifying as a test ·
**[BLOCKED]** needs a fix first · **[TODO]** not started.

---

## 0. Executive summary of the analysis

The serial story is much better than expected and the MPI story is worse — but for a reason that has
nothing to do with the refinement engine.

* Serially, **everything on today's list already works** except one item: Taylor–Hood Stokes with a bulk
  force on adaptive quad / tri / mixed meshes, mixed C2+C1 Poisson with `ConstrainFieldsToC1Space` /
  `UnconstrainFieldsFromC1Space`, Neumann BCs, and ALE (`LaplaceSmoothedMesh` + Stokes with a prescribed
  free-surface outflow) all reach machine zero on quad, triangle and mixed quad+tri meshes at 0, 1 and 2
  refinement levels (§2). Those tasks are therefore **codification** work, not implementation work.
* The one genuine serial defect is `ConstrainPositionsToC1Space` under triangle hanging — the §4.14
  `local_eqn < eleminfo->ndof` abort, reconfirmed today (§4). That is the only item on the list that needs
  real engine work.
* Under MPI, the campaign is blocked on **two defects that are not in the refinement engine at all**
  (§3): the `get_residuals()` binding is unsound on a distributed vector, and the Stokes pressure fixation
  pins a rank-local node. Both are small, well-understood fixes; both must land before any MPI test in this
  campaign can even be *written*, because the first one destroys the oracle and the second one segfaults.
* The default-solver-under-MPI item is **[DONE]** (§1).

Consequence for sequencing: fix the three blockers (§1 done, §3.1, §3.2) → then the 2D/3D serial tests are
mostly transcription of the probes in this document → then MPI variants of the same tests → then the one
hard item (§4) → then the 3D mixed-box mesh infrastructure that several 3D tasks share (§5).

---

## 1. Default linear solver under MPI — **[DONE]**

**Problem.** `pyoomph/__init__.py` chose the default linear/eigen solver purely from the platform
(macOS-arm64 → PETSc+MUMPS, else Pardiso → PETSc+MUMPS → SuperLU). Under `mpirun -n N` with `N>1` on Linux
this selects **Pardiso**, which is not MPI-parallel and raises from its own constructor:

> `The Pardiso linear solver cannot be used under MPI ... Use Pardiso only in serial`
> (`pyoomph/solvers/pardiso.py:431`)

so every distributed run had to be given `--petsc_mumps` explicitly.

**Fix.** A new `_running_under_mpi()` helper (`get_mpi_nproc() > 1`; `0` means "built without MPI", `1`
means single-process — both are serial) gates a new first branch in the cascade: under a genuine
multi-process run, `_set_petsc_mumps_solver()` is tried **first, on every platform**, since PETSc+MUMPS is
the only distributed-capable direct solver pyoomph ships. If it is unavailable the code falls through to
the unchanged platform cascade and emits one `RuntimeWarning` (rank 0 only) saying the chosen solver is not
MPI-capable, rather than failing at import — an explicit `--petsc`/`--mumps` on the command line must still
be able to rescue the run.

**Serial behaviour is byte-for-byte unchanged** (the new branch's condition is false).

Verified:

| run | default linear | default eigen |
|---|---|---|
| `python …` | `pardiso` | `pardiso` |
| `mpirun -n 2 python …` | `petsc_mumps` | `slepc_mumps` |

---

## 2. Serial state of the campaign — measured, not assumed

All numbers below are from probe scripts (`probe2d.py`, `probe_mpi_matrix.py`, `probe_features.py`,
`probe3d.py` in the session scratchpad) on the domain **[-0.5,0.5]²**, base mesh 4×4 cells, in three
discretisations — `quad` (`RectangularQuadMesh`), `tri_left` / `tri_crossed`
(`split_in_tris=...`), and `mixed` (quads for x<0, triangles for x>0, i.e. one cross-shape interface at
x=0) — at three refinement states: `(0,0)` non-adaptive, `(1,1)` uniform level 1, `(1,3)` uniform level 1
plus a level-3 band on `top`, i.e. **a genuinely non-uniform, multi-level 2:1 mesh**.

Oracle throughout: the problems are linear, so `max|residual|` after one Newton step must be ~0.

### 2.1 Stokes in a box driven by f = (−y, x) — **[READY]**

Taylor–Hood (C2 velocity + C1 pressure — the mixed-space case the task asks about), no-slip on all four
walls, one pressure dof fixed.

| mesh | (0,0) | (1,1) | (1,3) |
|---|---|---|---|
| quad | 3.3e-17 | 3.4e-17 | 2.9e-17 |
| tri_left | 2.7e-17 | 4.4e-17 | 4.5e-17 |
| tri_crossed | — | — | 5.8e-17 |
| mixed quad+tri | 4.1e-17 | 3.5e-17 | 5.4e-17 |

Cross-validation beyond the residual: `max|u|` at the finest state agrees across all four discretisations
to ~1e-5 (0.0078772 / 0.0078695 / 0.0078843 / 0.0078867), which is the discretisation error, not a tear.

So **mixed C1/C2 fields on mixed adaptive meshes already work serially**, including two levels of
refinement jump across the quad↔tri interface. Task = write it as a test.

### 2.2 Mixed C2 + C1 Poisson, `ConstrainFieldsToC1Space` — **[READY]**

`u` on C2 with source 1, coupled `v` on C1 with source `u`, Dirichlet on all walls; then the same with
`ConstrainFieldsToC1Space("u")` in the bulk and `UnconstrainFieldsFromC1Space("u") @ "top"`.

Machine zero (≤1.8e-16) in **all 18 combinations** (2 variants × 3 meshes × 3 refinement states). The
constraint bites as expected — it cuts the dof count from 556 to 190 at `(1,3)` — and the boundary
unconstrain restores the C2 dofs on `top`.

### 2.3 Neumann boundary conditions — **[READY]**

C2 Poisson, Dirichlet on left/bottom, `NeumannBC(u=1) @ "right"` and a *spatially varying*
`NeumannBC(u=x) @ "top"` (the latter is the interesting one: the flux is integrated over face elements
whose parent may be a hanging-node element of either shape).

Machine zero (≤1.3e-15) on quad / tri / mixed at all three refinement states.

### 2.4 ALE / moving mesh — **[READY]**

`StokesEquations(mode="TH")` + `LaplaceSmoothedMesh()`, mesh pinned on left/right/bottom and in x on `top`,
with a prescribed outflow `velocity_y = j_evap` on `top` standing in for evaporation.

Machine zero (≤4.0e-15) on quad / tri / mixed at all three refinement states.

### 2.5 3D affordability — measured

The task asks to keep the base mesh coarse enough that more than one refinement level fits in memory. On
`TetCubeMesh` (unit cube, N³ cells × 6 Kuhn tets) with Taylor–Hood Stokes:

| base | levels | elements | dofs | wall time | peak RSS |
|---|---|---|---|---|---|
| N=2 | uniform 1 | 384 | 1 129 | 0.9 s | 174 MB |
| N=2 | 1 + band 2 | 1 252 | 3 661 | 1.0 s | 222 MB |
| N=2 | 2 + band 3 | 6 418 | 20 974 | 5.7 s | 573 MB |

**Recommended 3D test envelope: base 2×2×2 cells, uniform level 1 + a level-3 band** (i.e. two levels of
jump). That is ~1.5k–6k elements and stays well under 1 GB, which is also safe for a 2-rank MPI run.

---

## 3. MPI blockers — both outside the refinement engine

### 3.1 `Problem.get_residuals()` is unsound on a distributed vector — **[DONE]**

This is **the** blocker: it silently destroys the oracle that essentially every adaptivity test in `tests/`
relies on.

`src/nanobind/problem.cpp:883`:

```cpp
oomph::DoubleVector ov;
self->get_residuals(ov);
std::vector<double> res(self->ndof());
for (unsigned int i = 0; i < self->ndof(); i++)
    res[i] = ov[i];          // <-- ov[i] is a LOCAL index
```

`oomph::DoubleVector::operator[]` is documented as *"[] access function to the **(local)** values of this
vector"* (`double_vector.h:227`). Under `--distribute` the vector is row-partitioned, so it holds only
`nrow_local()` doubles while the loop reads `ndof()` of them — a plain out-of-bounds read past the end of
the buffer. It returns whatever happens to be in memory.

Measured on a 4×4 quad Poisson under `mpirun -n 2 --distribute`:

| refinement | ndof | serial max\|res\| | distributed max\|res\| |
|---|---|---|---|
| none | 9 | 4.3e-17 | 3.1e-17 (accidentally fine) |
| level 1 | 49 | 3.7e-17 | **9.96e+148 / 3.53e+175** |
| level 2 | 225 | 1.1e-16 | **nan** |

The solutions themselves are correct — the full matrix in `probe_mpi_matrix.py` (4 equation systems × 3
meshes × 3 refinement states under `-n 2 --distribute`) shows nodal values matching the serial run to all
printed digits while `max|residual|` is randomly `1e+48`, `1e+241`, `nan`, `1.0`, or machine zero depending
on what was past the buffer. **Every distributed "residual" number obtained so far, including any used to
justify §4.17 of the engine document, is meaningless.**

**Fix** (small, and the pattern already exists in the same file): redistribute to a non-distributed,
globally replicated distribution before reading, exactly as
`_redistribute_local_to_global_double_vector` does at `problem.cpp:1093`:

```cpp
if (ov.distributed()) {
    oomph::LinearAlgebraDistribution glob(ov.distribution_pt()->communicator_pt(), ov.nrow(), false);
    ov.redistribute(&glob);
}
```

**Landed.** A `double_vector_to_global_std_vector()` helper in `src/nanobind/problem.cpp` now redistributes
before reading; `get_residuals()` and `_assemble_residual_jacobian()` both use it. The same
read-by-global-index bug was present in three more places and is fixed with a matching pair of helpers
(`gather_double_vector_to_global` / `scatter_global_to_double_vector`) in `src/problem.cpp`:

* `Problem::get_history_dofs()` — read out of bounds,
* `Problem::get_current_dofs()` — read out of bounds,
* `Problem::set_current_dofs()` — **wrote** out of bounds (it built a vector on the distributed dof
  distribution and then filled `ndof()` entries); this one is used by
  `test_constrained_field_unrefinement`, so it would have been the next thing to bite.

This changes `get_residuals()` / `get_current_dofs()` / `get_history_dofs()` from "silently wrong" to
"globally replicated" under MPI: every rank returns the same full-length vector, which is what test code
naturally assumes and what makes a per-rank `assert max|res| < tol` a valid oracle. `set_current_dofs()`
correspondingly now takes a globally indexed vector on every rank. The Jacobian returned by
`_assemble_residual_jacobian` deliberately stays process-local CSR — that is what its callers expect.

*Not* fixed (out of scope, and blocked anyway): the bifurcation-handler `get_eigenfunction` bindings
(`problem.cpp:~446` and friends) index `oomph::Vector<DoubleVector>` the same way. Distributed eigensolves
are already known broken for an unrelated reason (engine doc §4.17), so this is recorded, not repaired.

**Verified.** 4×4 quad Poisson, `mpirun -n 2 --distribute`, ndof 9 / 49 / 225: max\|res\| now
6.3e-17 / 5.8e-17 / 9.1e-17, **identical on both ranks** and matching serial. The full 36-case matrix
(C1 Poisson, C2 Poisson, coupled C2+C1, Taylor–Hood Stokes × quad/tri/mixed × 3 refinement states) is
machine zero on every rank, with nodal values matching the serial reference. Serial suite unaffected
(152 refinement tests pass).

### 3.2 `create_pressure_fixation()` pins a rank-local node — **[DONE]**

`pyoomph/equations/navier_stokes.py:~73`:

```python
self.node = self.mesh.element_pt(0).node_pt(0)   # Is definitely a C1 node
```

`element_pt(0)` is the **rank-local** element 0. On a distributed mesh each rank therefore pins a
*different* pressure node, so the global system is inconsistently constrained (over-constrained on some
ranks, singular on others). Observed effect: `mpirun -n 2 python … --distribute` on the adaptive quad
Stokes box dies with

```
[1]PETSC ERROR: Caught signal number 11 SEGV: Segmentation Violation
MPI_ABORT was invoked on rank 1 ... with errorcode 59
```

and the printout shows `PINNING some pressure with value 0` once per rank. Replacing the fixation with a
globally well-defined `DirichletBC(pressure=0) @ "bottom"` makes the same run complete, which isolates the
cause.

**Landed.** All three fixation classes had the same `element_pt(0)` defect and all three are fixed
(`pyoomph/equations/navier_stokes.py`):

* `PressureFixationTaylorHood` — candidates are the vertex (C1) nodes of **non-halo** elements; the
  lexicographically smallest coordinate wins, agreed across ranks by `allreduce(…, MPI.MIN)`. Every rank
  then pins whichever local copy matches — **including halo copies**, since a halo node must mirror the
  owner's pinned state or the per-rank dof counts diverge. The lookup scans all local nodes, not just each
  element's `node_pt(0)`, because on another rank the winner may not be any element's vertex 0.
* `PressureFixationCrouzeixRaviart` / `PressureFixationScottVogelius` — same selection by smallest element
  **centroid** among non-halo elements. Unlike the nodal case, an element-internal dof exists only on its
  owner, so a rank that does not own the winner pins nothing.

Two details worth remembering:
* Selection happens in `apply()`, not cached in `setup()`: refinement and distribution both rebuild the
  node/element objects, and after distribution the winner may live on a different rank. `apply()` runs on
  all ranks in lockstep (driven by the global `reapply_boundary_conditions()`), so the collective is safe
  there. `setup()` — which is *not* re-run after adaptation — would have left a stale pointer.
* The no-candidate marker in the reduction is `(1,)` against tagged candidates `(0, key)`, **not** an
  inf-filled tuple: a rank with no local elements does not know `ndim`, and `(inf,) < (inf, inf)` is true,
  so a length-mismatched sentinel would have won the min. Every rank must reach the reduction, including
  ones with an empty partition.

The stray `print("PINNING some pressure with value", …)` is gone.

**Verified.** Serial unchanged (9 Stokes/TH/CR tests pass, and the serial residuals in §2.1 are unmoved).
Distributed Taylor–Hood Stokes box on quad / tri_left / tri_crossed / mixed at levels (1,3), which
previously segfaulted, now runs at `-n 2` **and** `-n 4` with max\|res\| ≤ 1.7e-16 identical on all ranks
and `max|u|` matching serial to all printed digits. Crouzeix–Raviart likewise (`-n 2`, residuals ≤ 1.6e-15,
serial-matching velocities).

### 3.3 What the MPI matrix shows now that the oracle is sound

With §3.1 and §3.2 landed, the full `-n 2 --distribute` matrix — C1 Poisson, C2 Poisson, coupled C2+C1,
Taylor–Hood Stokes × quad / tri_left / mixed × non-adaptive / uniform / two-level 2:1 — is **machine zero
in all 36 cases, with the residual identical on both ranks and the nodal values matching the serial
reference**. Adding the Taylor–Hood box with its own pressure fixation at `-n 2` and `-n 4`, and
Crouzeix–Raviart at `-n 2`, all on the two-level non-uniform mesh, all clean.

So distributed adaptive solving on mixed adaptive meshes — including mixed C1/C2 spaces across a quad↔tri
interface with two levels of refinement jump — is now genuinely evidenced, where before it was
unmeasurable. What remains for Phase B is to turn this sweep into automated tests (§5.2).

---

## 4. `ConstrainPositionsToC1Space` under triangle hanging — the one real engine defect

Reconfirmed today, unchanged from §4.14 of the engine document. On the ALE box (Stokes +
`LaplaceSmoothedMesh` + `ConstrainPositionsToC1Space` + `UnconstrainPositionsFromC1Space @ "top"`):

| mesh | (0,0) | (1,1) uniform | (1,3) 2:1 hanging |
|---|---|---|---|
| quad | 5.4e-16 | 8.9e-16 | 1.4e-15 |
| tri_left | 6.7e-16 | 1.0e-15 | **abort** |

```
python: _probefeat/_ccode/domain.c:111: ResidualAndJacobian0:
        Assertion `local_eqn < eleminfo->ndof' failed.
```

Root cause (from §4.14, still current): `POSITION_CONSTRAIN_TO_C1` `pin_position`s the constrained node and
relies on the `c1_constraint_corners` redistribution, unlike the field constraint, which installs a
*registered* `set_hanging_pt`. At a 2:1 T-junction the constrained node's C1 corner is itself hanging, so
its position redistributes onto external coarse vertices that oomph never registered as position-hang
masters (`pin_position` registers nothing) → `local_position_hang_eqn` returns an index ≥ `ndof`.

Position cannot simply reuse the field fix's geometric `−1` hang: that would couple the dominant C2 values
and wrongly degrade the velocity to C1. The approach that must be made to work is the registered one —
mirror oomph's solid-hang new-master allocation so the constrained position genuinely hangs on the coarse
vertices — which previously tangled with oomph's two-pass equation assignment. This is the only item on
today's list that is engine work rather than test work, and it should be scheduled accordingly.

---

## 5. Missing infrastructure

1. ~~**No mixed 3D *box* mesh.**~~ **[DONE — `tests/box_mesh_3d.py`, see §9.1.]** Originally: `tests/test_mixed_3d.py` exercises mixed shapes with deliberately minimal
   2–3-element meshes (`TetWedgeMesh`, `WedgePyrMesh`, `HexMixMesh`) whose purpose is to isolate one
   cross-shape facet. Today's 3D tasks (Stokes, ALE, Neumann, constraints, MPI, ≥2 refinement levels) need
   a *domain*: a parametrised `MixedBoxMesh3D(kind=…)` on [-0.5,0.5]³ built from 2×2×2 cells, where each
   cell is filled by one family and the requested combinations meet legally:
   * tet ↔ wedge / pyramid (triangle facet only — a tet has no quad face),
   * wedge ↔ pyramid (triangle cap↔face, or quad side↔base),
   * hex ↔ pyramid base / wedge side (quad facet only),
   * plus a three-way and an all-four-way region.
   Boundary facets must be tagged per element face (the §4.3 mechanism) for all six walls, since the
   Neumann and ALE tasks depend on face elements over every shape.
2. ~~**No MPI test target in `tests/`.**~~ **[DONE]** — see §7.

---

## 6. Plan

Ordered so that each step unblocks the next; the three blockers come first because without them nothing
downstream is testable.

**Phase A — unblock (must precede everything MPI).**
- A1. **[DONE]** Default solver under MPI (§1).
- A2. **[DONE]** Distributed `get_residuals()` / `_assemble_residual_jacobian()` / `get_current_dofs()` /
  `get_history_dofs()` / `set_current_dofs()` (§3.1).
- A3. **[DONE]** Globally consistent pressure fixation for all three modes (§3.2).
- A4. **[DONE]** MPI pytest harness + the sweep codified as tests (§7).

**Phase B — 2D campaign. [DONE] — see §8.**

**Phase C — 3D campaign. [DONE for everything that works; two defects found and characterised — see §9.]**

**Phase D — the hard item.**
- D1. `ConstrainPositionsToC1Space` under triangle/mixed hanging (§4): registered position-hang masters
  mirroring the solid-hang allocation. Then the ALE + position-C1 matrix in 2D and 3D, serial + MPI, which
  is currently the only part of the task list with no working baseline at all.

**Risks / open questions.**
* D1 is the schedule risk: it previously failed against oomph's two-pass equation numbering, and the fix
  touches equation assignment rather than the pyoomph-owned refinement engine.
* §3.1's fix changes the semantics of a public API under MPI (from garbage to globally replicated). Cheap,
  but it needs a note in `CHANGELOG.md`.
* 3D mixed + ALE + MPI is the deepest untested stack on the branch; expect the C-phase to surface engine
  issues that the 2-element mixed toys never could.

---

## 7. The MPI test harness — **[DONE]**

Three new files in `tests/`:

* **`mpi_cases.py`** — the problem definitions, imported by *both* sides. The serial reference and the
  distributed run therefore solve a bit-identical problem and differ only in partitioning, which is what
  makes the comparison mean anything. Covers the box [-0.5,0.5]² as `quad` / `tri_left` / `tri_crossed` /
  `mixed` (quad+tri, cross-shape interface at x=0), at refinement states `(0,0)` / `(1,1)` / `(1,3)`, for
  `poisson1` (C1), `poisson2` (C2), `mixed12` (C2 `u` driving C1 `v`), `stokes_th` and `stokes_cr` (both
  driven by f = (−y, x), both using `create_pressure_fixation`).
* **`mpi_worker.py`** — run as `mpirun -n N python mpi_worker.py --spec '<json>' --outdir … --distribute`;
  prints one `PYOOMPH_MPI_RESULT <json>` line per rank per case. A case that raises reports its exception
  and traceback in the payload instead of killing the run, so one broken case still yields a diagnosis.
  Deliberately not named `test_*` so pytest does not collect it.
* **`test_mpi_adaptivity.py`** — the harness. Skips (does not fail) when `mpirun`, MPI, or PETSc+MUMPS is
  missing. 6 tests, ~33 s total.

### 7.1 What is compared, and why those quantities

Everything measured is **partition-independent**, so a distributed run must reproduce the serial numbers:

| quantity | why it is valid under MPI | what it catches |
|---|---|---|
| `max\|residual\|` | gathered to full length since §3.1, so identical on every rank | an inexact hanging-node Jacobian |
| `ndof` | global on a distributed problem | a different discretisation / hanging structure |
| integral observables | `Mesh::evaluate_integral_function` skips halo elements and `MPI_Allreduce`-sums | a **wrong field** — which a residual check alone would pass |

Deliberately *not* compared: `nelement()` (per-rank, includes halos) and nodal values (a rank holds only
its own partition). Two independent oracles are applied: distributed-vs-serial, and cross-rank agreement.

### 7.2 Three things that had to be got right

* **Nested `mpirun` dies silently.** Importing pyoomph calls `MPI_Init`, so the pytest process is itself a
  singleton MPI job owning an Open MPI session directory under `TMPDIR`. A nested `mpirun` collides with it
  and exits 1 with *no stdout and no stderr at all*. Fix: give the child its own `TMPDIR`. This cannot be
  dodged by import discipline in the test module — any other test module in the same pytest process has
  already imported pyoomph.
* **Don't assert on an observable that is zero by symmetry.** ∫u_x and ∫u_y vanish identically for this
  forcing on this box, so the first version compared round-off (~1e-13) against round-off and failed for no
  physical reason. Replaced by the angular momentum ∫(x·u_y − y·u_x), which the forcing actually drives.
  The tolerance additionally carries an absolute term scaled by the case's largest observable, so a
  near-zero observable can never fail on noise again.
* **Crouzeix–Raviart needs its own tolerance** (1e-7 relative, vs 1e-9 elsewhere), for the same reason its
  residual bound is looser: it is markedly worse conditioned than Taylor–Hood, and re-partitioning
  perturbs its observables by ~1e-9 relative. A genuinely torn or mis-hung interface moves these integrals
  by orders of magnitude more, so the looser bound does not blunt the test.

### 7.3 Verified sensitivity

The oracle was negative-controlled, not just observed to pass: perturbing the reference field integral by
**1e-8 relative** is caught, and so is a **single-dof** difference in `ndof`. Three consecutive full runs
are stable (6 passed, ~33 s each). Full serial suite unaffected.

### 7.4 Result

`test_distributed_adaptive_matches_serial` (C1 Poisson, C2 Poisson, coupled C2+C1, Taylor–Hood Stokes ×
4 meshes × 3 refinement states, `-n 2`), `test_distributed_crouzeix_raviart` (`-n 2`) and
`test_distributed_four_ranks` (`-n 4`, the two-level non-uniform state) all pass. Phase B's remaining work
is now to extend `mpi_cases.py` with the constrain-C1, Neumann and ALE cases — the harness itself takes
them without modification.

---

## 8. Phase B — the 2D campaign — **[DONE]**

`tests/mpi_cases.py` is now `tests/box_cases.py`: it is no longer MPI-specific but the single definition of
the whole 2D campaign, imported by three places — the serial tests, the MPI harness (for its in-process
reference) and the `mpirun` worker. The serial and distributed campaigns therefore cannot drift apart.

**Nine equation systems** on four discretisations (`quad`, `tri_left`, `tri_crossed`, `mixed` quad+tri) at
three refinement states (`(0,0)`, `(1,1)`, `(1,3)` = two levels of 2:1 jump):

| case | what it exercises |
|---|---|
| `poisson1`, `poisson2` | pure C1 / pure C2 baselines |
| `mixed12` | `u` on C2 driving `v` on C1 — two spaces hanging independently |
| `constrain12` | + `ConstrainFieldsToC1Space("u")` everywhere |
| `unconstrain12` | + `UnconstrainFieldsFromC1Space("u") @ "top"` |
| `neumann` | constant flux on `right`, spatially varying on `top` (the refined wall) |
| `stokes_th`, `stokes_cr` | Stokes with f = (−y, x); Taylor–Hood and Crouzeix–Raviart |
| `ale` | Stokes + `LaplaceSmoothedMesh`, free top surface, prescribed evaporation outflow |

`tests/test_adaptive_2d_campaign.py` (97 tests, ~27 s) is the serial half; `tests/test_mpi_adaptivity.py`
(11 tests, ~63 s) runs the same cases under `mpirun -n 2` and `-n 4`.

### 8.1 The oracles, and why the obvious ones are not enough

A converged residual only says *the solve converged*; it does not say the field is right. Four layers:

1. **`max|residual|` ≈ 0.**
2. **One Newton step removes the whole residual.** These problems are linear, so an exact analytic Jacobian
   goes from O(1e-2) to machine zero in one step. Expressed as the *ratio* `conv[1]/conv[0]`, not as an
   iteration count — see §8.2. This is what would catch a constrained or hanging dof that was pinned
   instead of being given a registered hang. Under MPI it additionally exercises the distributed
   hanging-value collapse (engine doc §4.17 Fix 2): a Jacobian assembled from stale halo-master values
   fails this even though it may still converge eventually.
3. **Cross-discretisation agreement.** The same physical problem on quads, on two triangle splits and on
   the mixed mesh must give the same global angular momentum, and *refinement must bring them together* —
   the spread drops from 3% at `(0,0)` to 0.2% at `(1,3)`. A tear at the quad↔tri interface leaves free
   nodes and shifts the integral instead of converging it, which no residual check would notice.
4. **Exact discrete identities.** Two of them:
   * `ndof(constrain12) < ndof(unconstrain12) < ndof(mixed12)` on every mesh and level (e.g. 252 < 284 <
     650), so a silently inert constraint — or a boundary unconstrain that restores nothing — cannot pass.
   * the Green identity **∫v = ∫u²**, which holds *exactly* once `u` is restricted to `v`'s space. Verified
     to ~5e-18 on every mesh and refinement state, including the two-level hanging mixed mesh. It does
     **not** hold for the unconstrained baseline (asserted), so it genuinely distinguishes a correct
     restriction from an approximate one rather than being a triviality.

### 8.2 Three calibration findings worth keeping

* **The boundary unconstrain was almost a no-op test.** With `u` Dirichlet on all four walls, the top-edge
  C2 mid-node values are pinned anyway, so `UnconstrainFieldsFromC1Space @ "top"` restores nothing and the
  test would have passed while testing nothing. The cases therefore give `u` and `v` a *natural* (zero-flux)
  condition on `top`. That also makes the Green identity's boundary terms drop, which is what enables
  oracle 4 above — the same choice buys both.
* **Iteration counts are the wrong Jacobian oracle.** Crouzeix–Raviart on triangles lands at 1.9e-8 after a
  perfect first step and takes a cosmetic second one purely because the Newton tolerance is 1e-8. The ratio
  formulation is tolerance-independent. Calibrated on the data: the worst non-CR ratio across all 8 × 4 × 3
  cases is **1.5e-13**, so the bound is set at 1e-10 — three orders of headroom, and eight or more orders
  below what an inconsistent Jacobian gives.
* **Crouzeix–Raviart is exempt from that test, and it is not a regression.** On refined triangle meshes its
  saddle-point system is ill-conditioned enough that the direct linear solve is itself inaccurate (8.4e-04
  → 1.5e-05 in one step, then ~1e-10). This was checked against the *pre-existing* pressure fixation
  (commit `dadc4ae`): the same ratios appear there (3.7e-3 / 6.3e-4 / 5.6e-5 vs 1.8e-2 / 6.2e-4 / 8.1e-5),
  so it is a property of the discretisation, not of the §3.2 change or of the hanging-node Jacobian. CR is
  held to convergence within 3 iterations and the loose residual bound the rest of the suite already uses.

### 8.3 Result

Serial: **97 tests pass**. Distributed: **11 tests pass** at `-n 2` and `-n 4`, every case matching its
serial reference in `ndof` and in every global integral, with all ranks agreeing. Notably
`test_distributed_c1_constraint_still_reduces_dofs` checks that the C1 constraint — which is applied from
Python by walking the mesh's own elements, so each rank sees only its own partition plus halos — still
constrains the *same global set of dofs* after distribution.

Remaining for the 2D half: nothing on the original task list except `ConstrainPositionsToC1Space` (§4),
which is Phase D.

---

## 9. Phase C — the 3D campaign — **[DONE for what works; two defects found]**

Phase C delivered the missing 3D infrastructure, fixed one bug that blocked it outright, and — as the
campaign was designed to do — **found two real defects** that no existing test could have caught. The
passing coverage is large; the failures are characterised exactly and tracked as strict xfails.

### 9.1 `tests/box_mesh_3d.py` — the mixed 3D box

`MixedBoxMesh3D` fills the box [-0.5,0.5]³ as an N×N×N grid of cells, each cell filled by one element
family, per a named layout. The existing mixed-3D meshes (`TetWedgeMesh`, `WedgePyrMesh`, `HexMixMesh`) are
deliberately minimal 2–3-element toys built to isolate one cross-shape facet — the wrong tool for Stokes,
ALE or Neumann on a refined domain.

**Which combinations are geometrically possible** — this is real geometry, not an engine limitation, and it
determined the layout catalogue. What each family presents on a shared cube face:

| family | x-faces | y-faces | z-faces |
|---|---|---|---|
| brick | quad | quad | quad |
| pyramid | quad | quad | quad |
| wedge | quad | quad | 2 triangles |
| tet | 2 triangles | 2 triangles | 2 triangles |

so brick↔brick / brick↔pyramid / pyramid↔pyramid and wedge↔wedge / tet↔tet work in any direction;
brick↔wedge and pyramid↔wedge only across **x or y**; tet↔wedge only across **z**; and **tet↔brick and
tet↔pyramid are impossible in any direction** — a tet has no quad face, so no shared facet can match. The
triangle traces also have to split along the same diagonal, and they do: the Kuhn split cuts every cube
face from the corner minimal in both in-plane coordinates to the maximal one (translation-invariant), and
the wedge's (x,y) diagonal split produces that same diagonal on its z-faces.

For the impossible pair there is a **transition cell**: one cube holding a single pyramid whose base is the
cube's bottom quad face (so a brick fits below) plus 10 tets filling the other five pyramidal wedges (so
tets fit beside and above), with each outer quad split along the Kuhn diagonal. That is the classic
hex-to-tet transition, and it puts pyramids and tets inside one cube.

Catalogue: 4 pure (`hex`, `tet`, `wedge`, `pyr`), 4 pairs (`hex_pyr` — a checkerboard, `hex_wedge`,
`pyr_wedge`, `tet_wedge`), a three-way (`hex_pyr_wedge`), and two using the transition cell (`hex_tet`,
`all_four`). **All 11 validated** at 0 / uniform-1 / non-uniform-2 refinement: facet adjacency manifold, no
duplicate nodes, and a manufactured linear field reproduced at every node.

### 9.2 Bug fixed — the 3D neighbour self-test guard

`DynamicOcTreeForest::check_all_neighbours` (`src/mesh3d.hpp`) inspected only `Trees_pt[0]` when deciding
whether to skip oomph's brick compass neighbour self-test. In a mixed forest whose *first* root happens to
be a brick — which any hex-containing box layout has — the guard did not fire, so the brick self-test ran
on a forest for which `find_neighbours()` had deliberately set **no neighbour pointers at all**. It then
reported a bogus `Max. error in octree neighbour finding: 1.24373 is too big` and aborted, or ran away into
an OOM. Fixed by scanning **every** tree, exactly as the 2D `DynamicQuadTreeForest::check_all_neighbours`
already did. Pure-brick forests still get the full check; this only ever removes a check that was invalid.
Full suite green afterwards (285 passed).

### 9.3 What passes

`tests/test_adaptive_3d_campaign.py`: **194 passed, 38 xfailed** (~3.5 min).

* **Single-space problems pass on all 11 layouts at all three refinement states**, including two-level
  non-uniform 2:1 hanging across brick/tet/wedge/pyramid interfaces and through the transition cells: C1
  Poisson, C2 Poisson, and **the entire Neumann campaign** (a constant flux on an unrefined wall plus a
  spatially varying flux on the refined wall, integrated over face elements of every shape). Task 7's 3D
  half is therefore complete.
* **Multi-space problems** — coupled C2+C1 Poisson, Taylor–Hood Stokes with f = (−y, x, 0), and ALE
  (Stokes + `LaplaceSmoothedMesh` + prescribed evaporation outflow) — **pass on all 11 layouts** when the
  mesh is non-adaptive or uniformly refined, and pass under non-uniform refinement on **bricks and tets**.
* Cross-family agreement: the same Stokes problem discretised by every family and every legal mixture
  agrees on the global angular momentum, with no family an outlier.
* Cost is comfortable: the whole 3D matrix is base 2×2×2 with level 1 + a level-2 band, i.e. 1–6 k
  elements, 1–3 s and ~300 MB per case.

### 9.4 Defect A — two continuous spaces × non-uniform refinement on wedge/pyramid/mixed — **[FIXED]**

Coupled C2+C1 Poisson, Taylor–Hood Stokes and ALE all **fail under non-uniform refinement on wedges,
pyramids and every mixed layout**, while passing on bricks and tets and passing on *all* families under
uniform refinement. The failure mode is Newton not converging (or, for pyramids, "converging" to
`max|residual| = 2.2e-04`), i.e. an inconsistent Jacobian — not a crash.

The discriminator is sharp and points straight at the cause: single-space problems are fine on exactly the
same meshes, so it is not the mesh, the refinement or the boundary conditions. What breaks is specifically
the case where **two continuous spaces coexist**: a C1 field on C2 geometry needs its *own* hang slot at a
different value index, hanging linearly on the coarse cell's C1 corners. That per-value-index hang is
installed for bricks (oomph's own `RefineableQElement::setup_hang_for_value`) and for tets (§4.12 of the
engine document, which validated tet Taylor-Hood/Crouzeix-Raviart), but **not** for the wedge / pyramid /
registry families — whose validation (§4.15) only ever used single-space Poisson, C1 *or* C2, never both at
once. So this is a genuine gap in the wedge/pyramid work rather than a regression.

**Fix — three separate bugs, found in this order.**

1. **The wedge and pyramid C2 elements never overrode the per-value interpolation hooks.** oomph-lib's
   defaults are isoparametric — `ninterpolating_node()` returns `nnode()` and `interpolating_basis()`
   returns `shape()` — so a C1 field's hanging constraint was built from the **quadratic geometric basis
   over all 18 (wedge) / 14 (pyramid) nodes** instead of the linear basis over the corner vertices. The
   brick and the 2D quad/tri C2 elements each hand-roll these overrides; `BulkElementBase` now provides
   them shape-agnostically (`interpolation_value_is_C1`, `generic_ninterpolating_node`,
   `generic_interpolating_node_pt`, `generic_interpolating_basis`,
   `generic_get_interpolating_node_at_local_coordinate`) and the wedge and pyramid C2 elements use them.
   The stale comment on their `further_setup_hanging_nodes` — *"there can't be any problem here, since it
   is all isoparametric"* — was precisely the wrong assumption.

   The same helpers also went onto **`BulkElementTetra3dC2`**, which overrode `interpolating_basis` *alone*.
   That left `ninterpolating_node()` at 10 while the C1 basis writes only 4 entries, so callers read six
   **uninitialised** doubles (`oomph::Shape` allocates with `new double[N]`, which does not zero). It
   happened to work because a `TElement` numbers its vertices first, so the garbage was usually rejected by
   the `|psi| > 1e-12` master test — luck, not correctness.

2. **The pyramid's C1-corner table was wrong for the base centre.** `Dummy_Value_Interpolation_Map` listed
   node 13 (the base quad centre) as `{13, 0, 2}` — the mean of *one diagonal* — where the bilinear base
   requires all four corners. Fixed to `{13, 0, 1, 2, 3}`.

3. **The wedge's C1-corner table had two entries swapped.** Bottom-layer edge mids 3, 4, 5 were listed over
   corner pairs (0,1), (1,2), (0,2), but node 4 sits at the 0–2 midpoint and node 5 at the 1–2 midpoint —
   the top layer (15, 16, 17 over (12,13), (12,14), (13,14)) already had the right pattern.

Bugs 2 and 3 were found by the **Green identity** `∫v = ∫u²`, which reads ~0 for an exact C1 restriction:
it showed 0.29 on pyramids and 0.089 on wedges *even on a non-adaptive mesh*, where no hanging is involved,
which is what separated "the constraint is wrong" from "the hanging is wrong". Both tables were then
verified against the actual nodal positions — on an affine element the geometry and a C1 field interpolate
identically, so each listed corner set must average to its target node's position. That check found bug 3
after two rounds of reading the table had missed it.

**Result.** All five multi-space equation systems — coupled C2+C1 Poisson, both `ConstrainFieldsToC1Space`
variants, Taylor–Hood Stokes and ALE — now pass on **all 11 layouts at all three refinement states**,
serially and distributed. The Green identity holds to ~1e-14 on every family, so it is now asserted for all
of them rather than only for bricks and tets. The 3D campaign has **no xfails left**: 232 passed, 0
xfailed.

### 9.5 Defect B — `ConstrainFieldsToC1Space` × non-uniform 3D refinement — **[FIXED]**

`ConstrainFieldsToC1Space` throws under non-uniform 3D refinement on **all** families, bricks included:

> `src/elements.cpp:2612: Cannot enforce a degration to C1 on a C1 vertex node.`

The existing `test_constrained_adaptivity_3d_brick` passes, so this looked at first like a mixed-mesh
issue. A bisection over four independent knobs (mesh: `CuboidBrickMesh` vs `MixedBoxMesh3D`; boundary
conditions; which wall carries the refinement band; and a *live* coupled C1 field vs the Dirichlet-pinned
`_dummyC1`) shows otherwise — **16 of 16 combinations fail except the single one the existing test happens
to use**, and the mesh makes no difference whatsoever:

| band | C1 field | Dirichlet 1 wall + Neumann | Dirichlet 5 walls |
|---|---|---|---|
| `right` | dummy (pinned) | **passes** | fails |
| `right` | live | fails | fails |
| `top` | dummy (pinned) | fails | fails |
| `top` | live | fails | fails |

So the feature works only in one narrow configuration, and this is **pre-existing on `main`, not a
`mixed_adapt` regression**. The throw site explains the mechanism: the code deliberately intends to *do
nothing* when a constrained node is also a C1 **vertex** of a finer neighbouring element (its own comment
says "there it is not in that element's map, so we do nothing"), but the guard that lets that path through
requires the node to hang on the C1 slot (`is_hanging_on_C1`). At a 2:1 interface the node in question is a
genuine, non-hanging node of the coarse element, so the guard is false and it throws instead. In 2D the
same geometry does hang on the C1 slot (a coarse edge-mid node's C1 value is the average of the two coarse
corners), which is why 2D is unaffected; 3D additionally has face-centre and volume-centre nodes, where
that is evidently not set up.

Adding `_dummyC1`, as the error message suggests, does **not** help.

**Fix.** The guard was inconsistent with the code immediately below it. That code already handles the
vertex case correctly — `c1_corner_lookup.find(l)` misses for a node this element sees as a C1 vertex, so
nothing is installed here and the hang comes (identically) from the element(s) where the node *is* a
non-vertex, exactly as the block comment there describes. The guard aborted before that code could run.
It demanded the node hang on the C1 slot, which happens to hold in 2D (a coarse edge-mid node's C1 value
*is* the mean of the two coarse corners) but not in 3D, where a father's face-centre and volume-centre
nodes also become sons' vertices. The guard now tests the condition its own error message describes —
asking to degrade to a C1 space that does not exist (`!has_C1_fields`) — and the legitimate 2:1 vertex case
falls through to the do-nothing path.

The **position** branch (`POSITION_CONSTRAIN_TO_C1`) was deliberately left strict: a constrained position is
still enforced by `pin_position()` rather than by a registered hang, so relaxing it would not make the
position case correct, only move the failure into the generated residual code (the §4.14
`local_eqn < eleminfo->ndof` abort). It should follow the field branch once Phase D lands.

**Result.** `constrain12` / `unconstrain12` now pass under non-uniform 3D refinement on bricks and tets —
machine zero, one Newton step, and the dof ordering `constrained < unconstrained-on-top < baseline` holds.
What remains on wedges/pyramids/mixed layouts is **defect A**, which these cases share with every other
multi-space problem (they too carry a live C1 field beside the C2 one); their xfail predicate is now simply
the same one. Exactly two xfails flipped to passes, and nothing else moved.

### 9.6 How the failures are tracked

Every failing configuration is marked `xfail(strict=True)` with the reason string, not skipped and not
dropped from the matrix. They stay visible in the run, they cannot silently rot, and the suite will **fail**
the moment one starts passing — which is the signal to delete the marker. 38 of them.

### 9.7 Next

~~Defect B~~ — **done**, §9.5. ~~Defect A~~ — **done**, §9.4; it cleared the 3D halves of tasks 2/3, 4 and
5 in one go, as expected, since they all failed for the same reason. What remains is **defect C** (§9.8),
which is distributed-only and narrow, and **Phase D** (`ConstrainPositionsToC1Space`, §4), which is
independent of all of it. Note that Phase D should also revisit the `POSITION_CONSTRAIN_TO_C1` guard left
deliberately strict in §9.5, and that the position analogue of the C1-corner tables fixed in §9.4 is worth
re-checking at the same time -- `pin_position` uses the same `c1_constraint_corners` redistribution.

### 9.8 Defect C — distributed pure-tet, non-uniform — **[SHARPENED, NOT FIXED]**

Found by the 3D MPI harness. On the **pure-tet** layout at the two-level non-uniform state under
`--distribute`, the solve fails; serially the identical case is machine zero.

**What the earlier characterisation got right and wrong.** It was first seen as an asymmetric
`get_elemental_errors()` throw that deadlocked the other ranks (one `mpirun` sat for 50 minutes on a 1.5 s
workload). After the defect-A fixes the symptom changed: both ranks now fail *symmetrically* with
`MAXIMUM RESIDUALS: inf EXCEEDS PREDEFINED MAXIMUM 1e+10`. The deadlock is therefore gone — which is why
`_run_distributed` keeping a bounded timeout still matters, but the hang itself is no longer the issue.

**Sharpened isolation.** The failure needs all three of: a **pure-tet forest**, **non-uniform** refinement,
and **free (unpinned) boundary dofs**. Measured:

| case | distributed result |
|---|---|
| tet, non-adaptive / uniform | pass |
| tet, non-uniform, C2 Poisson with Dirichlet on 5 walls | pass |
| tet, non-uniform, Dirichlet on 5 walls + one Neumann wall | pass |
| tet, non-uniform, Dirichlet on **3** walls (rest natural) | **fail (inf)** |
| `hex_tet`, `tet_wedge`, `all_four` — non-uniform, same BCs | pass |

So it is not the Neumann fluxes (a Dirichlet-only variant fails too) and not tets as such — every
tet-*containing* mixed layout passes. Pinning more of the boundary masks it, which is what made it look
boundary-condition dependent.

**The measurement that matters.** For the failing configuration the global dof count differs from serial:
**1460 serially, 1573 distributed**, and the ranks disagree about the boundary hanging set (36 vs 39
hanging nodes on a wall, against 36 serially). Mixed layouts pass precisely because their hanging comes
entirely from the mesh-level generative pass in `post_adapt_setup_hanging_nodes`, whereas a pure-tet forest
uses the per-element OcTree hooks.

**A fix was attempted and rejected.** Routing distributed pure-tet meshes through the same generative pass
(gated to `is_mesh_distributed()`, leaving serial untouched) did change the hanging set but did **not** fix
the dof-count discrepancy or the `inf`, and it moved rank 1 further from the serial hanging set rather than
closer. It was reverted rather than left in as a speculative change to a validated path.

**The open question is now answered: the refinement is NOT divergent.** Dumping the refined mesh before the
solve, serially and on both ranks, and comparing as sets:

| | serial | distributed (union over ranks) | dist-only | serial-only |
|---|---|---|---|---|
| node positions | 2031 | 2031 | **0** | **0** |
| element centroids | 1112 | 1118 | 6 | 0 |
| **hanging nodes** | 246 | 326 | **80** | 0 |

The mesh is identical — the node sets match exactly, in both directions. What differs is the **hanging set**,
and asymmetrically: **rank 0's hanging set is exactly the serial one (246), while rank 1 installs 80
constraints that exist neither serially nor on rank 0.** They are not missing constraints, as the dof-count
gap suggested — they are *spurious* ones. They sit at z = 0.125…0.5, i.e. in the interior of the FINER
region and even on the top wall, whereas every legitimate hang is at z = 0.0625…0.25, on the 2:1 interface.

**And all 80 lie in rank 1's HALO region:** zero of them belong to any non-halo element of rank 1, so every
one is a node whose owner (rank 0) correctly says it does not hang. Rank 1 is installing hanging constraints
on nodes it does not own, from its incomplete halo-side neighbour information — the pure-tet route installs
hanging per element via the OcTree neighbour finders, which on a partitioned mesh cannot see that the
neighbour across the cut is equally fine, and so concludes a 2:1 jump that is not there.

**Likely mechanism for the `inf`.** Halo elements are skipped during assembly, so a spurious constraint on a
halo node should be inert — except that `Mesh::collapse_hanging_node_values()` (engine doc §4.17 Fix 2)
writes every hanging node's master-interpolated value into its own raw storage before each assembly. For
these spuriously-hanging halo nodes that *overwrites the correctly synced value from the owner* with a bogus
interpolation, corrupting the halo and hence the residual. That is the first thing to test.

**Suggested fix to try first**, in order of cheapness: (1) make `collapse_hanging_node_values()` skip nodes
the rank does not own, since their values come from the owner via the halo sync anyway; (2) failing that,
suppress hang installation on non-owned nodes in the pure-tet route and let the halo sync carry the owner's
verdict. `Node.get_hanging_masters()` makes the check itself a three-line diagnostic: dump the hanging set
per rank and diff it against serial.

Pinned by `test_distributed_3d_pure_tet_nonuniform_xfail` (strict, short timeout) and excluded from the
other matrices, so the rest of the 3D distributed coverage stays meaningful.

### 9.9 Phase C scoreboard

| | serial | distributed (`-n 2`, `-n 4`) |
|---|---|---|
| C1/C2 Poisson, **Neumann** — all 11 layouts, incl. two-level non-uniform | pass | pass (except pure-tet Neumann, §9.8) |
| coupled C2+C1, Taylor–Hood Stokes, ALE — all 11 layouts, uniform | pass | pass |
| coupled C2+C1, Taylor–Hood Stokes, ALE, `ConstrainFieldsToC1Space` — all 11 layouts, non-uniform | pass (**defects A + B fixed**) | pass (except pure tet, §9.8) |

Six defects found, **five fixed**: the neighbour self-test guard (§9.2), the C1-constraint vertex guard
(§9.5), and the three behind defect A (§9.4 — the missing per-value interpolation hooks, the pyramid base
centre, the swapped wedge edge mids). Only **defect C** (§9.8, distributed pure-tet) remains, tracked as a
single `xfail(strict=True)`.


---

## 10. Running the suite

The campaign made the full suite substantially larger, so `tests/conftest.py` splits it:

| | command | contents | time |
|---|---|---|---|
| fast (default) | `python -m pytest *.py` | everything except the `slow` mark | ~6 min |
| full | `python -m pytest *.py --full` | everything | ~11 min |

`slow` marks the tests that sweep a large matrix or launch `mpirun`: the 3D campaign and both MPI modules.
The 2D campaign deliberately stays in the fast run — it is the branch's core physics and only costs a
couple of minutes. `PYOOMPH_FULL_TESTS=1` is equivalent to `--full` for CI. Nothing is permanently
excluded, and skipped tests are reported as skipped rather than silently dropped.

The 3D **MPI** matrices were also trimmed from all 11 layouts to a representative five — every pure family
plus `all_four`, which carries bricks, tets, wedges, pyramids *and* the brick-to-tet transition cells in
one mesh and therefore exercises every legal interface kind at once. `neumann` keeps the full 11, since
boundary-facet propagation under refinement is the most shape-dependent part of the campaign. The serial
3D campaign continues to sweep all 11 exhaustively, so nothing is uncovered; the distributed campaign only
has to show that partitioning does not break what serial already proved. Full-run time fell from 16 to 11
minutes with identical pass/xfail counts.

---

## 11. Task 6 / Phase D — `ConstrainPositionsToC1Space`: measured current state

The position constraint is the last item of the original task list. Before planning the work, the whole
matrix was measured — `ale_posc1` (ALE + `ConstrainPositionsToC1Space`) and `ale_posc1_unc` (the same plus
`UnconstrainPositionsFromC1Space @ "top"`), added to `box_cases.py` / `box_cases_3d.py`. Each case runs in
its own process, because the 2D failure is a C-level `assert` (SIGABRT) that a `try/except` cannot catch and
that otherwise kills the whole sweep.

| discretisation | non-adaptive | uniform | non-uniform (2:1) |
|---|---|---|---|
| 2D quad | pass | pass | **pass** |
| 2D tri_left / tri_crossed / mixed | pass | pass | `Assertion local_eqn < eleminfo->ndof` (§4.14) |
| 3D — all 11 layouts | pass | pass | `RuntimeError: Cannot enforce a degration to C1 on a C1 vertex node` |

Both variants behave identically, and the boundary unconstrain does bite (2D quad ndof 1289 → 1321; 3D hex
1219 → 1259), so these are real tests and not silent no-ops.

### 11.1 What this changes about the §4.14 estimate

* **The conforming case is in better shape than §4.14 suggested.** It described the feature as working "on
  adaptive quads and on *uniform* tris". In fact it works on non-adaptive and uniformly refined meshes of
  **every** 2D and 3D family, including all the mixed 3D layouts. Only 2:1 hanging breaks it.
* **The 3D failure is not a separate defect** — it is the `POSITION_CONSTRAIN_TO_C1` guard deliberately left
  strict in §9.5 (`src/elements.cpp:2727`) while its field-branch twin was relaxed. Relaxing it would let 3D
  proceed to whatever the 2D tri path reaches, which is almost certainly the same assertion; it was left in
  place precisely so the failure stays legible instead of turning into a wrong Jacobian.
* So there is **one** underlying defect, not three, and it is the one §4.14 named: at a 2:1 T-junction a
  constrained position must hang on the coarse vertices with **registered** masters, but
  `POSITION_CONSTRAIN_TO_C1` still enforces it with `pin_position()`, which registers nothing. Pure 2D quads
  are the exception because oomph-lib's own quad machinery registers the position hang for them.
* The C1-corner table fixes from §9.4 (pyramid base centre, swapped wedge edge mids) did **not** incidentally
  fix this: 2D tris are unaffected by them, and 3D cannot get past the guard to tell.

### 11.2 Shape of the work

1. Replace the `pin_position()` enforcement with a genuine registered position hang on the element's C1
   corner nodes — the position analogue of what the field branch already does with `set_hanging_pt()`,
   mirroring oomph-lib's solid-hang new-master allocation. §4.14 records that an earlier attempt at this
   tangled with oomph's two-pass equation assignment; that is the risk to plan around.
2. Then relax the `POSITION_CONSTRAIN_TO_C1` guard to match the field branch (§9.5).
3. Then the 2D and 3D matrices above, plus their MPI variants, become ordinary campaign cases — the
   `ale_posc1` / `ale_posc1_unc` definitions are already in place and will pick them up.

`Node.get_hanging_masters()` (now bound) makes step 1 far more tractable than it was: the position hang's
master list and weights can be inspected directly from a test rather than inferred from a Newton failure.


### 11.3 First fix attempt — registered geometric position hang — **FAILED, reverted**

The plan of §11.2 step 1 was attempted and does not work as stated. Recording it so the next attempt does
not repeat it.

**The approach.** Replace `pin_position()` with a registered hang on the element's C1 corners, installed in
the node's *geometric* slot (`set_hanging_pt(hang, -1)`), so that oomph's own
`assign_hanging_local_eqn_numbers` walks it and enters the masters into `Local_position_hang_eqn` — the
registration the position path is missing.

**What was learned, and what is now known to be wrong in §4.14.**

* §4.14's stated reason for rejecting the geometric hang — that it "couples the dominant C2 values" — is
  **not correct as stated**. `Node::is_hanging(i)` for `i >= 0` reads `Hanging_pt[i+1]` with **no fallback**
  to the geometric slot (`nodes.h`), and pyoomph's `flatten_hang_for_value` keys on exactly that. A
  geometric hang does not by itself make any value hang.
* The real obstacle is different and is in `Node::set_hanging_pt(hang, -1)` (`nodes.cc:2068`): before
  overwriting the geometric slot it marks every value slot holding the *same pointer* as the geometric one
  and re-points all of them at the new hang. When a node hangs in nothing, every slot is null **and so is
  the geometric one**, so a naive call makes every field hang on the C1 corners too. This is surmountable —
  snapshot the value slots, install, then restore (a slot that shared the old geometric pointer gets its own
  copy of the original; a slot that was null goes back to null) — and it was implemented and verified:
  `Node.get_hanging_masters()` confirmed the installed constraint had the right masters and weights and that
  the value slots stayed clean.
* **It still fails.** With a demonstrably correct hang installed, the global dof count collapses (2D quad
  base mesh: 139 with `pin_position`, 59 with the hang) and Newton diverges to ~1e+54 from the first step —
  on the NON-adaptive mesh, which previously passed. Guarding the install against Dirichlet-pinned positions
  (a hang replaces a pin, so it would silently release a pinned boundary) did not help. So something beyond
  the `HangInfo` bookkeeping accounts for the lost dofs; installing refinement-style geometric hangs on an
  unrefined mesh appears not to be something oomph's dof accounting expects.

**Suggested next direction.** Do *not* route this through the geometric slot. Register the missing masters
directly in `Local_position_hang_eqn` instead — the route §4.14 called "tangles with oomph's two-pass
equation assignment", which now looks like the lesser problem of the two. A cheap first step that is worth
doing regardless: `RefineableElement::local_position_hang_eqn` reads the map with `std::map::operator[]`,
which default-constructs an EMPTY `DenseMatrix<int>` for an unregistered node, so
`leaf_local_eqn_for_position` reads out of bounds and returns a garbage index — that is the whole mechanism
of the §4.14 abort. Adding a membership check there would turn it into a precise diagnostic naming the
unregistered node, which would make the required registration set directly enumerable rather than inferred.
