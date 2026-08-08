# Bifurcation tracking on a distributed problem

Bifurcation tracking used to refuse `--distribute` outright. It now works for all four augmented
handlers in `src/bifurcation.cpp`: `MyFoldHandler`, `MyHopfHandler`, `MyPitchForkHandler` and
`AzimuthalSymmetryBreakingHandler`. This note records what had to change, what deliberately did not,
and what is still refused.

## 1. What was serial about the handlers

Nothing in the *elemental* assembly. Commit 27a7c23, which made tracking survive a plain (replicated)
`mpirun`, had already established that the augmented Jacobian and residual agree with the serial ones
to 1e-8 under MPI. What was serial was the global bookkeeping around it:

- **Storage.** `Phi`, `Psi`, `Y`, `C`, the azimuthal eigenvector parts and `Count` were plain
  `Vector<double>` of length `Ndof = problem->ndof()`, which stays the *global* row count when
  distributed, and were indexed inside the element loops by *global* equation number.
- **Dof vector.** Every rank pushed all `Ndof` eigenvector pointers plus every scalar unknown, and
  the augmented distribution was built non-distributed.
- **`Count` and `nelement`.** Both came from loops over `mesh_pt()->nelement()` with no halo filter
  and no reduction, so under distribution `Count` over-counted shared equations and the
  normalisation constant `-1/nelement` no longer summed to `-1` across ranks.
- **Direct `GetDofPtr()[global_eqn]`.** Correct on a replicated dof vector, out of bounds on a
  partitioned one.
- **Destructors** rebuilt the base distribution as non-distributed, which is wrong if the base was
  distributed.

## 2. The shared helper

`AugmentedDofDistributionHelper` (`src/bifurcation.hpp`) holds the dof bookkeeping for all four
handlers as a member object rather than a base class — the handlers already derive from
`oomph::AssemblyHandler` plus `AugmentedSparsityProvider`, and a member avoids reordering that.

It has two modes, and the replicated one reproduces the historical behaviour exactly:

| | serial / plain `mpirun` | `--distribute` |
|---|---|---|
| eigenvector dofs pushed | all `Ndof`, every rank | this rank's `nrow_local()` only |
| scalar dofs (param, Omega, Sigma) | every rank | rank 0 only |
| augmented distribution | non-distributed, built in place | `LinearAlgebraDistribution(first_row, n_row_local)`, pointer-swapped in |
| `global_eqn(naive)` | identity | per-rank interleaved translation table |
| `synchronise_scalars` | no-op | `MPI_Bcast` from rank 0 |

The layout is upstream oomph-lib's, from its distributed `PitchForkHandler`
(`src/thirdparty/oomph-lib/include/assembly_handler.cc`, ~2704–2882): rank `d`'s augmented rows are
its base rows, then — in the handler's naive block order — its rows of each eigenvector block, with
each scalar contributing one row at `d == 0`. Every handler describes itself to
`build_augmented_dofs()` as a list of blocks in its own naive order, so the same table construction
serves fold `[u | param | Y]`, Hopf `[u | Phi | Psi | param | Omega]`, pitchfork
`[u | param | Y | Sigma]` and azimuthal `[u | Re | (Im) | param | (Omega)]`.

**Why the distribution is pointer-swapped rather than rebuilt in place.** The dof halo scheme (and
therefore `Problem::global_dof_pt`) keeps a raw pointer to the distribution object it was built
against. Rebuilding that object in place while an augmented distribution is installed would leave the
halo scheme describing a layout that no longer exists. `Problem::SetDofDistributionPt` exists for
this; the base object is kept alive by the helper and restored on destruction.

## 3. Per-handler changes

- Eigenvector *unknowns* became `oomph::DoubleVectorWithHaloEntries` on the base dof distribution,
  read through `.global_value(g)` — which degrades to `[g]` when not distributed, so one code path
  serves all three modes. `Count` likewise.
- The fixed normalisation and symmetry vectors (`Phi` for fold, `C` for Hopf and pitchfork,
  pitchfork's `Psi`, the azimuthal `normalization_vector`) stayed **fully replicated**
  `Vector<double>`. They are read-only after construction and are set from guesses that are already
  identical on every rank, so replicating them removes any synchronisation obligation. The cost is
  `Ndof` doubles per rank per vector, accepted deliberately.
- `Nelement` is now a member holding the `MPI_Allreduce`d count of non-halo elements; `Count` is
  built from halo-skipping loops and then `sum_all_halo_and_haloed_values()`.
- `eqn_number()` still computes the naive number and returns `Dist_helper.global_eqn(...)` of it.
- Each handler overrides `AssemblyHandler::synchronise()` (which oomph already calls at the end of
  `Problem::synchronise_all_dofs`, so **no vendored oomph-lib change was needed** — verify that stays
  true) to refresh eigenvector halos and broadcast the rank-0-owned scalars.
- `get_eigenfunction()` keeps its contract — a globally replicated, non-distributed vector on every
  rank — by gathering when distributed, so the nanobind bindings and every Python consumer are
  unchanged.
- The FD branches perturb base dofs through `Problem::global_dof_pt(elem_pt->eqn_number(n))`. Note
  the argument is the element's *own* equation number, not the handler's: `eqn_number()` now returns
  a translated augmented number, and feeding that to `global_dof_pt` would index nonsense. The same
  applies to the azimuthal forced-zero sets, which hold base equation numbers.
- `realign_C_vector()` reads its members directly (owned rows + `MPI_Allreduce`) instead of the
  replicated dof-pointer table. Fold's long-standing `Y[n] = phin/phisqr` — a division by the sum of
  squares rather than the norm — is preserved; the dof entries it read always pointed into `Y`
  itself, so this always was a rescale of `Y` by its own square sum.

A few pre-existing slips were fixed on the way, all of the same kind — `GetDofPtr()[k*Ndof]` used to
reach the bifurcation parameter, where `Parameter_pt` says it directly. The azimuthal one was reading
`GetDofPtr()[3*Ndof]`, which is out of bounds when `has_imaginary_part` is false.

## 4. Still refused when distributed

- **`blocksolve=True`**, and the handlers' block modes (`solve_block_system`,
  `solve_complex_system`, `Block_augmented_J`, …). They rebuild replicated in-place dof vectors, and
  upstream's own augmented block path throws when distributed too.
- **Periodic orbit tracking.** `PeriodicOrbitHandler` temporarily overwrites arbitrary global dofs
  during assembly; a separate project. Its Poincaré-plane constraint also has a pre-existing
  inconsistency worth fixing first: `-d_plane` is not divided by `nelement` while the row is
  assembled additively per element, so the constraint is effectively `n0.x = nelement*d_plane` — a
  value that would become rank-dependent under distribution.
- **`adapt()` and arclength continuation while tracking.** Blocked by the history-dof refusals in
  `Problem::get_dofs(t,...)` / `set_dofs(t,...)`; independent of the handlers.
- **The frozen-sparsity fast path** declines augmented systems under MPI, so tracking falls back to
  oomph's `parallel_sparse_assemble`. Correct, just slower.
- **The no-guess fold and Hopf constructors**, which derive their eigenvector from a serial linear
  solve on a replicated vector. They throw when distributed; the Python side raises the same message
  earlier, with a better traceback.

## 5. A dense row and column, by construction

`eqn_number()` maps the last local indices of *every* element onto the scalar rows, so the
normalisation row and the parameter column are dense across all ranks. Under a row-distributed matrix
that is one fully dense row assembled from every rank. Correct, and the same as upstream, but a
scaling hot spot — worth revisiting before anyone runs this on a very large problem.

The scalar rows also have a deliberately **empty diagonal** (see the comments at the
`get_sparsity_pattern` implementations): the normalisation equation does not involve the parameter, so
declaring it dense would manufacture a stored zero on the diagonal, which is what invites MUMPS to
plan an elimination onto a null pivot. If a distributed MUMPS solve ever reports a zero pivot on the
bordered rows, that is the place to look, and its ICNTL knobs are reachable through the PETSc options
pyoomph already exposes.

## 6. The `eigenvector_scaling` option

`activate_bifurcation_tracking(..., eigenvector_scaling=...)`, defaulting to `"unit"`, which is
bit-for-bit the historical behaviour: the guess is normalised to unit length and the constraint reads
`c.y = 1`.

The problem with that on a large system is scale. A unit-length vector over `N` dofs has entries of
order `1/sqrt(N)`, so the eigenvector unknowns are tiny, and so are the constraint row's Jacobian
entries `c_i/Count_i`, while the residual they are weighed against is O(1). `"auto"` normalises the
guess by its **largest entry** instead and sets the constraint's right-hand side to the dot product
the rescaled guess actually has (`Normalization_rhs`, which multiplies the `-1/Nelement` constant at
all nine residual sites). The guess therefore satisfies the constraint exactly, and both the unknowns
and the row stay O(1) however large the problem is. `c` is rescaled along with the eigenvector, since
leaving it at unit length would fix only half of the problem.

Measured on the Bratu fold, `tests/mpi_bifurcation_worker.py`:

| mesh | ndof | `unit` max&#124;y&#124; | `auto` max&#124;y&#124; | critical λ |
|---|---|---|---|---|
| N=8  | 451  | 0.1366  | 1.086 | 6.8082638085 |
| N=24 | 4419 | 0.04554 | 1.086 | 6.8081260834 |

`unit` decays with the mesh, `auto` does not, and the located fold is the same to 1e-10. Only the
eigenfunction's amplitude — always arbitrary — differs.

It composes with `set_eigenweight` multiplicatively; the two are independent knobs.

## 7. Testing

`tests/mpi_bifurcation_worker.py` + `tests/test_mpi_bifurcation_tracking.py` (marked `slow`, so
`--full` is needed), following the pattern of the MPI eigen tests: pytest runs serially and launches
the worker under `mpirun`, comparing against an in-process serial run of the same worker. 13 tests:
four cases × {`np=2 --distribute`, `np=3 --distribute`, `np=2` plain}, plus the `auto` scaling.

The plain-`mpirun` cases are there to keep commit 27a7c23 fixed, not for coverage of the new code.

Cases, and what each one is the only cover for:

| case | handler | naive layout | what it pins down |
|---|---|---|---|
| `fold` | `MyFoldHandler` | `[u \| param \| Y]` | one real eigenvector block, one scalar |
| `hopf` | `MyHopfHandler` | `[u \| Phi \| Psi \| param \| Omega]` | two blocks and two scalars — catches a scalar left un-broadcast |
| `pitchfork` | `MyPitchForkHandler` | `[u \| param \| Y \| Sigma]` | a scalar on either side of the block in the naive numbering |
| `azimuthal` | `AzimuthalSymmetryBreakingHandler` | `[u \| Re \| Im \| param \| Omega]` | the axis dofs forced to zero by global equation number |

The critical parameter is a weaker certificate than it looks: it is one number every rank reads off
the same converged augmented state, so it survives a great deal. Two things constrain more:

- **`eigfunc_usqr`**, the mesh integral of the squared tracked eigenfunction. Reaching it runs the
  eigenvector back through `set_current_dofs()`, which scatters by *global* equation number, and
  integrates over non-halo elements with an `MPI_Allreduce`. A broken `eqn_number` translation or a
  missing halo synchronise leaves the critical parameter and the eigenvector norm right and moves
  this. It must be taken **after** deactivating tracking: while tracking is active the dofs are the
  augmented ones, whose length and layout differ between serial and distributed, and padding a
  base-length eigenvector into them measures the padding.
- **cross-rank agreement**, since the parameter and Omega live on rank 0's dof vector alone, and
  `evect_len` catches a rank that returned its own row block instead of gathering.

Two ambiguities the assertions have to allow for, both intrinsic and both visible in a plain
replicated run as readily as a distributed one: the Hopf frequency comes as ±iω and which one the
eigensolver returns is arbitrary (so the magnitude is compared), and the eigenvector's sign or phase
is free (so the eigenfunction integral is taken from the entrywise |v|).

### An unrelated SLEPc failure found while writing this

The first attempt built the azimuthal case on the tutorial's Rayleigh–Bénard problem. SLEPc fails on
it with `error code 73` inside `EPS.solve` — **in serial**, at every shift tried, with or without the
pressure integral constraint, while scipy solves the same eigenproblem fine. The shipped
`axiflow` case of `tests/test_mpi_eigenvalues.py` fails identically. This has nothing to do with
bifurcation tracking (no handler is installed during that eigensolve) and predates this work, but it
does mean the Navier–Stokes azimuthal eigensolve is currently broken with complex PETSc on this
machine and is worth its own look. The azimuthal tracking case is built on a reaction-diffusion
problem instead, which SLEPc handles.

## 8. Verified numbers

Serial versus distributed, `N=8` (`N=20` for Hopf), from the worker's own output:

| case | serial parameter | `np=2 --distribute` | `np=3 --distribute` | eigfunc rel. diff |
|---|---|---|---|---|
| fold | 6.808263808496 | 6.808263808509 | 6.808263808502 | 4e-14 |
| hopf | 1.999999837212 | 1.999999836902 | — | 1.8e-14 |
| pitchfork | 19.739855570997 | 19.739855578031 | — | 2.3e-12 |
| azimuthal | 24.552372532331 | 24.552372532331 | 24.552372532331 | 1e-15 |

The Bratu fold's λ*≈6.808, the Brusselator Hopf's `B = 1+A² = 2` with `ω = ±A = ±1`, and the
reaction-diffusion pitchfork's `λ = 2π² ≈ 19.7392` are all known analytically, so these are not just
self-consistent between the two runs.

## 9. Open: the moving-mesh droplet segfaults

`docs/source/tutorial/advstab/movmesh/hanging_droplet.py` — fold tracking on a moving mesh with an
interface and Lagrange multipliers — **segfaults on rank 0 during the tracking solve** under
`mpirun -n 2 --distribute`. Serially, and under plain `mpirun` (which commit 27a7c23 validated), it
is fine: `Bo_c = 2.946075780049` at `V = 2.0944`.

Not diagnosed, and not re-run since — a segfault under MPI takes PETSc's `MPI_ABORT` with it.

What the one observation does say is that the failure is probably not in the handler. The distributed
run reported `ndof = 7846` where serial has `15693`, i.e. its *base state* was already a different
problem before tracking was activated. The script gets there through `go_to_param` with
`remesh_handler_during_continuation` followed by `force_remesh()`, so it sits directly on distributed
remeshing — `dev_docs/distributed_remeshing.md`, whose stage 5 is deliberately unfinished. The
sensible next step is to check whether a `--distribute` run of that script *without* any bifurcation
tracking already produces the wrong `ndof`, which would settle it without touching the handlers.

Until that is understood, treat moving-mesh problems with remeshing as unvalidated here. The four
handler cases in §7 use fixed meshes.
