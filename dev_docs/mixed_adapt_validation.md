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

### 3.1 `Problem.get_residuals()` is unsound on a distributed vector — **[BLOCKED → fix first]**

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

The same read-by-global-index bug is present in `_assemble_residual_jacobian` (`problem.cpp:~969`,
`res[i] = resi[i]` over `n = J.distribution_pt()->nrow()`); fix both. Audit any other binding that indexes
a `DoubleVector` beyond `nrow_local()`.

Note this changes `get_residuals()` from "silently wrong" to "globally replicated" under MPI, so every rank
returns the same full-length vector — which is what test code naturally assumes and what makes a
per-rank `assert max|res| < tol` a valid oracle.

### 3.2 `create_pressure_fixation()` pins a rank-local node — **[BLOCKED → fix first]**

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

**Fix.** Choose the pinned node deterministically and globally: over **non-halo** nodes only, take the
lexicographically smallest coordinate, `allreduce` to agree on one winner across ranks, and pin it on the
owning rank (and on any rank holding it as a halo, so the pinned state is consistent). Serial reduces to
today's behaviour. The stray `print("PINNING some pressure with value", …)` should also be demoted to the
existing verbosity mechanism.

### 3.3 What the MPI matrix actually shows once the oracle is discounted

Reading only the *solution* columns of the `-n 2 --distribute` matrix (C1 Poisson, C2 Poisson, coupled
C2+C1, Taylor–Hood Stokes × quad/tri/mixed × 3 refinement states): the distributed nodal values reproduce
the serial ones in every case that ran to completion. That is encouraging but **not** evidence — with a
broken oracle and a crashing pressure fixation, nothing here is validated yet. Re-run the whole matrix as
the first thing after §3.1 and §3.2 land.

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

1. **No mixed 3D *box* mesh.** `tests/test_mixed_3d.py` exercises mixed shapes with deliberately minimal
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
2. **No MPI test target in `tests/`.** The pytest suite is serial. Needs a harness that re-launches a
   parametrised worker script under `mpirun -n 2` via `subprocess`, `pytest.skip`s when `mpirun` or
   PETSc+MUMPS is absent, and asserts on the worker's structured stdout. Worker must also compare its
   distributed result against a serial reference of the same problem, since a per-rank residual check alone
   does not catch a globally-consistent-but-wrong field.

---

## 6. Plan

Ordered so that each step unblocks the next; the three blockers come first because without them nothing
downstream is testable.

**Phase A — unblock (must precede everything MPI).**
- A1. **[DONE]** Default solver under MPI (§1).
- A2. Fix `get_residuals()` (and `_assemble_residual_jacobian`) for distributed vectors (§3.1); rebuild via
  `build_for_develop.sh`. Regression: serial `tests/` unchanged; `mpirun -n 2 --distribute` on a Poisson
  ladder must now give machine zero where serial does.
- A3. Fix `create_pressure_fixation()` for distributed meshes (§3.2).
- A4. Add the MPI pytest harness (§5.2) and re-run the full `probe_mpi_matrix` sweep — this is the first
  moment any distributed claim on this branch becomes evidence.

**Phase B — 2D campaign (mostly codification; serial half already verified).**
- B1. Stokes box f=(−y,x), Taylor–Hood, quad / tri_left / tri_crossed / mixed, levels (0,0)/(1,1)/(1,3);
  residual oracle + cross-discretisation agreement of an integral observable (§2.1).
- B2. Same under `mpirun -n 2 --distribute`, with the multi-level case included, plus a serial-reference
  field comparison.
- B3. `ConstrainFieldsToC1Space` / `UnconstrainFieldsFromC1Space` on coupled C2 `u` + C1 `v`, same mesh
  matrix, serial + MPI; assert the dof reduction as well as the residual, so a silently inert constraint
  cannot pass (§2.2).
- B4. Neumann campaign: constant and spatially varying fluxes, on walls that are refined and on walls that
  are not, same mesh matrix, serial + MPI (§2.3).
- B5. ALE: Stokes + `LaplaceSmoothedMesh` + top evaporation outflow, same mesh matrix, serial + MPI (§2.4).

**Phase C — 3D campaign.**
- C1. Build `MixedBoxMesh3D` (§5.1) with the family combinations, and a `refinement_possible` /
  tear-freeness smoke test per combination.
- C2–C5. The 3D analogues of B1/B3/B4/B5 on tet, hex, wedge, pyramid and each mixed combination, at the
  measured-safe envelope of §2.5 (base 2×2×2, level 1 + level-3 band), serial + `-n 2 --distribute`.

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
