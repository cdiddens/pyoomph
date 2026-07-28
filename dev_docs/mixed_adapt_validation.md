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

### 9.4 Defect A — two continuous spaces × non-uniform refinement on wedge/pyramid/mixed

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

~~Defect B is the smaller and higher-value fix~~ — **done**, see §9.5. What is left is **defect A**:
installing the per-value-index C1 hang for the wedge/pyramid/registry families, mirroring what the tet
route already does. It is the single remaining blocker for the 3D halves of tasks 2/3, 4 and 5 on
non-uniformly refined wedge/pyramid/mixed meshes — one fix would clear all four at once, since they all
fail for the same reason. **Defect C** (§9.8) is separate and distributed-only. Neither blocks Phase D
(`ConstrainPositionsToC1Space`, §4), which is independent.

### 9.8 Defect C — distributed 3D pure-tet, non-uniform: asymmetric throw → deadlock

Found by the 3D MPI harness (`tests/test_mpi_adaptivity_3d.py`). On the **pure-tet** layout at the
two-level non-uniform state under `--distribute`, `Mesh::get_elemental_errors()` throws an
`OomphException` on *some* ranks and not others during the initial adaption
(`Problem._adapt_with_interfacial_errors` → `mesh.get_elemental_errors()`, `problem.py:1678`).

Because the throw is **asymmetric**, the ranks that did not throw block forever in the next collective, so
the presenting symptom is a **deadlock**, not an error: the first time this was hit, one `mpirun` sat for
50 minutes on a workload that takes 1.5 s serially. That is why `_run_distributed` now takes a bounded
subprocess timeout (900 s, and 240 s for the pinned case) and converts an overrun into an explicit
assertion failure — a distributed deadlock must never be able to stall the suite.

Scope, from the run:
* **not serial** — the identical cases pass in `test_adaptive_3d_campaign.py`;
* **not uniform refinement** — every layout passes distributed at `(1,1)`;
* **not the mixed layouts** — `hex_tet`, `all_four` and `tet_wedge` all pass distributed at `(1,2)`, so
  merely containing tets is not enough; it is the pure-tet forest;
* **equation-dependent** — C1 and C2 Poisson pass on that exact mesh, while Neumann, Taylor–Hood and ALE
  do not, which points at the interfacial-error path rather than at the bulk field.

Pinned by `test_distributed_3d_pure_tet_nonuniform_xfail` (strict, short timeout) and excluded from the
other matrices, so the rest of the 3D distributed coverage stays meaningful.

### 9.9 Phase C scoreboard

| | serial | distributed (`-n 2`, `-n 4`) |
|---|---|---|
| C1/C2 Poisson, **Neumann** — all 11 layouts, incl. two-level non-uniform | pass | pass (except pure-tet Neumann, §9.8) |
| coupled C2+C1, Taylor–Hood Stokes, ALE — all 11 layouts, uniform | pass | pass |
| coupled C2+C1, Taylor–Hood Stokes, ALE — non-uniform, brick/tet | pass | pass (hex; pure-tet §9.8) |
| coupled C2+C1, Taylor–Hood Stokes, ALE — non-uniform, wedge/pyramid/mixed | **defect A** | not reached |
| `ConstrainFieldsToC1Space` — non-uniform, brick/tet | pass (**defect B fixed**) | pass |
| `ConstrainFieldsToC1Space` — non-uniform, wedge/pyramid/mixed | **defect A** | not reached |

Four defects found, **two fixed** (§9.2 the neighbour self-test guard, §9.5 the C1-constraint vertex
guard), two characterised and tracked (§9.4 defect A, §9.8 defect C). All remaining failing configurations
are `xfail(strict=True)`, never skipped or dropped.


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
