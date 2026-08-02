# Eigenvalue problems under MPI, through SLEPc

Status: **implemented and tested.** `tests/test_mpi_eigenvalues.py` (7 tests) covers both MPI
situations, the complex (azimuthal) path, the matrix-manipulator path and the refusal path. File and
line references are to the tree state at the time of writing.

**Goal.** Solve `J v = lambda M v` in parallel via SLEPc whenever pyoomph runs on more than one
process — with `--distribute`, and equally without it, since the eigensolve parallelises either way.
scipy/ARPACK cannot participate: it only ever sees one process' matrices.

---

## 1. What was already there

Almost all of the hard part, as it turned out. `Problem::assemble_eigenproblem_matrices`
(`src/problem.cpp`) builds both matrices on `dof_distribution_pt()`, and oomph-lib's
`get_eigenproblem_matrices` routes a distributed distribution through
`parallel_sparse_assemble` — which pyoomph already overrides. Each rank ends up with a row block in
exactly the layout `MatMPIAIJSetPreallocationCSR` wants: local rows, **global** column indices.

Verified numerically before any code was written, on a 143-dof problem assembled serially and under
`mpirun -n 3 --distribute`. The dof numbering is permuted by `distribute()`, so the matrices cannot
be compared entry by entry; comparing the dense generalized spectrum instead gave

```
sum / trace / Frobenius norm identical to 15 digits
max |eigenvalue difference| over all 143 = 2.8e-11
```

SLEPc needed nothing special either: an EPS with `sinvert` over an MPIAIJ pair converges, and
`-st_pc_type lu` (SLEPc's own default for the spectral transform) is resolved by PETSc to MUMPS or
SuperLU_DIST when the matrix is parallel. A PETSc built without either would fail here; that is a
build requirement, not something to work around.

## 2. What was broken

Everything downstream of the assembly, in Python.

**2.1 The matrices were declared square on every rank.** `SlepcEigenSolver.solve` built
`createAIJ(size=((n,n),(n,n)))` with no communicator, i.e. an `n`-row local block on each of `nproc`
ranks — an `nproc*n`-row matrix. Plain `mpirun -n 2`, with no `--distribute` at all, died with

```
MatMPIAIJSetPreallocationCSR() ... Argument out of range
Row too large: row 255 max 254
```

That nobody had hit this is itself informative: it means eigenproblems had never been run under
`mpirun` at all, with or without `--distribute`.

**2.2 `get_J_M_n_and_type` built `csr_matrix(..., shape=(n,n))`** from an `indptr` of length
`nrow_local+1`. Under `--distribute` that raises inside scipy, before the eigensolver is reached.

**2.3 Two per-rank decisions that would deadlock rather than answer wrongly.** Whether the
eigenproblem is complex was decided from `M_nzz>0`/`J_nzz>0` on the *local* block; a partition
carrying no imaginary entries would take the other branch and then issue different collectives from
its peers. Same for the empty-mass-matrix check. Both are now `MPI_Allreduce`d
(`_mpi_any_rank` in `pyoomph/solvers/generic.py`).

**2.4 Matrix manipulators.** `EigenMatrixSetDofsToZero` — how an `AxisymmetryBC` imposes the axis
conditions at `m != 0` — rewrites whole rows of a square global matrix. No rank has one.

## 3. Design

**Under `mpirun -n>1` the eigenproblem is always solved in parallel, on COMM_WORLD**, whether or not
the mesh was distributed. The two cases differ only in where the row split comes from
(`SlepcEigenSolver._eigen_parallel_layout`):

| | what a rank holds | row split from | sliced? |
|---|---|---|---|
| `mpirun`, no `--distribute` | the whole J and M (oomph assembles in parallel, then replicates) | a contiguous split of `n` imposed here | yes |
| `mpirun --distribute` | its own row block, global column indices | oomph's dof distribution | no |

Solving redundantly on `COMM_SELF` was the first implementation of the replicated case and is not
what anyone wants: `mpirun -n 8` is a request for eight processes to share the work, and doing the
same eigenproblem eight times looks like success while being slower than serial.

What the replicated case does **not** save is matrix memory — every rank still stores the whole J and
M. The win is the shift-and-invert factorisation, whose factors are typically far larger than the
matrix and where both the time and the real memory go. Note also that the *assembly* was already
parallel there: oomph's `get_eigenproblem_matrices` runs `parallel_sparse_assemble` on a temporary
distributed layout and only then redistributes to a replicated one.

The replicated case is in one respect the safer of the two: each row is contributed by exactly one
rank, from matrices that are identical across ranks, so the blocks cannot disagree about a shared row.

**A PETSc that cannot do this is an error, not a fallback** (`_require_parallel_capable`). Two things
are checked when `nproc>1`: that PETSc's own `COMM_WORLD` has the same size pyoomph does (it does not
if petsc4py was built `--with-mpi=0` or against a different MPI than mpi4py), and — only when the
spectral transform will actually factorise, so an explicitly iterative `st_ksp` is not second-guessed
— that MUMPS or SuperLU_DIST is present, since PETSc's own LU is sequential.

**Eigenvectors are gathered back to full global length on every rank**
(`_vector_to_global_array`, via `PETSc.Scatter.toAll`). This is the decision that keeps the change
small: `set_eigenfunction_as_dofs()`, the mesh data cache and the VTK output all index eigenvectors
by global equation number and reach the dofs through `get_current_dofs()`/`set_current_dofs()`, which
already gather and scatter. Nothing on that path had to change. The cost is `neval` vectors, not a
matrix.

`refine_eigenfunction()` is the exception, and it is worth being precise about why, because it looks
like it should work: it too only ever handles a global eigenvector. But `adapt()` carries the
eigenfunction across the adaption in **history levels 3 and 4**, and the history dof accessors refuse
on a distributed problem — an equation number there is global while the vector holds only this rank's
rows, so oomph-lib declares them unsupported and pyoomph throws rather than corrupt the heap
(`Problem::get_dofs(t,...)`, `src/problem.cpp`). The same is true of `adapt()` during arclength
continuation, which uses levels 5 and 6; that is pre-existing and unrelated to eigenproblems.
Distributed history dofs are the prerequisite for both.

**Manipulators move to the backend.** `EigenMatrixManipulatorBase.apply_on_distributed_J_and_M`
takes the eigensolver's own matrices; `EigenMatrixSetDofsToZero` implements it as
`J.zeroRows(rows, diag=1.0)` / `M.zeroRows(rows, diag=0.0)`, restricted to locally owned rows. That
restriction is complete as well as safe: every owned row belongs to a dof of a non-halo local
element, so no rank misses one of its own. The base-class implementation raises, so a manipulator
with no distributed equivalent stops the run instead of silently not applying its constraint.

**`distributed_possible()` is finally consulted.** It had been declared on both solver base classes
for a long time and called from nowhere. `Problem._solve_eigenproblem_helper` now asks it, and
`ScipyEigenSolver` answers no — otherwise scipy would solve each rank's row block as if it were the
whole eigenproblem and return numbers that look entirely reasonable.

## 4. Two bugs found on the way, both outside the eigensolver

**4.1 `set_current_dofs()` never synchronised halo nodes** (`src/problem.cpp`). `set_dofs()` writes
only `Dof_pt`, i.e. this rank's owned dofs; oomph-lib's own routines that update dofs all follow it
with `synchronise_all_dofs()`, but nothing did here because this one is reached from Python. Every
element touching a halo node therefore integrated stale values.

This is **not** eigen-specific — it affected any `set_current_dofs()` under `--distribute`. It
surfaced here because pushing an eigenvector into the dofs and integrating it over the mesh gave a
13% smaller answer than the serial run, while the eigenvalues, the vector length and the vector norm
were all correct. Nothing that looked only at the spectrum would have caught it.

**4.2 `resolve_equations_by_name()` raised on a locally absent submesh**
(`pyoomph/solvers/generic.py`). A corner interface such as `domain/bottom/left` is a single point and
belongs to exactly one partition; on the others the submesh is present but empty and reports no
fields. The old code raised — on *some* ranks, which is not an error but a **split**: those ranks
unwound while the owner walked into the next collective alone. It presented as `MPI_ERR_BUFFER` out
of PETSc, several frames from the name that could not be resolved. A distributed run now resolves
such a name to nothing, trading the mistyped-field-name diagnostic for the ability to run.

## 5. Still serial-only

These assemble the eigenproblem matrices themselves and read the result as a square global matrix,
or sit on augmented-system assembly that throws from C++ ("This likely does not work in distributed
parallel", `sparse_assemble_row_or_column_compressed_base_problem`). They now refuse early, by name,
through `Problem._require_non_distributed`:

- bifurcation tracking and eigenbranch continuation
- periodic orbit tracking / Floquet
- the multi-assembly tensor cache
- Lyapunov exponents
- the periodic driving response
- `refine_eigenfunction()` — for the history-dof reason in §3, not for an assembly reason

Bifurcation tracking in parallel is a separate project: the augmented handlers in
`src/bifurcation.cpp` are serial by construction.

## 6. Testing

`tests/mpi_eigen_worker.py` + `tests/test_mpi_eigenvalues.py`, following the pattern of the
structural-assembly MPI tests: pytest runs serially and launches the worker under `mpirun`, comparing
against an in-process serial run of the same worker.

Eigenvalues alone are a weak certificate — SLEPc computes them collectively, so every rank
necessarily agrees whether or not it was handed the right rows. The assertion that constrains the
result is **`eigfunc_usqr`**, the integral of the squared eigenfunction over the mesh: getting there
runs the eigenvector back through `set_eigenfunction_as_dofs()` → `set_current_dofs()`, which
scatters by global equation number, and then integrates over non-halo elements with an
`MPI_Allreduce`. That is what caught §4.1.

A solve that quietly ran redundantly on every rank is likewise invisible in the answer — same
eigenvalues, same eigenvectors, none of the benefit — so the worker reports the row block the
eigensolver actually used and `_assert_solve_was_split` checks that the blocks tile `[0, ndof)`
exactly: contiguous, non-overlapping, no gaps. Measured without `--distribute` at `np=3`: 255 dofs as
85/85/85.

Three traps worth recording, because each cost a debugging cycle:

- **Everything compared across serial and distributed must be numbering-independent.**
  `distribute()` renumbers the dofs. Eigenvalues and mesh integrals qualify; a dof vector does not.
- **An eigenvector is only determined up to a complex phase.** SLEPc returns `e^{i phi} v` for
  whatever `phi` it lands on, and serial and distributed runs pick different ones. The first version
  of the test used `mode="real"`, which keeps only `cos(phi)` of the eigenfunction, and reported a
  13% discrepancy with nothing wrong underneath — indistinguishable at first glance from the real
  bug in §4.1, which was also in flight. `mode="abs"` is phase-invariant and leaves the shape, which
  is what needs certifying.
- **`azimuthal_m != 0` does not by itself mean a complex eigenproblem.** A SCALAR field expanded as
  `u(r,z) e^{i m phi}` gives a purely real operator — the `m^2/r^2` term is real — so the azimuthal
  scalar test never reached the complex branch it claimed to cover. It takes a vector field, whose
  radial and azimuthal components couple through factors of `i`, and that is
  `test_distributed_axisymmetric_flow` (which also happens to be the only case that installs a
  matrix manipulator). The worker therefore reports `complex_assembly` and `zeromap_size`, and each
  test asserts on them, so "this branch never ran" fails instead of passing quietly.

Measured agreement across serial / `np=2` / `np=3`, including the azimuthal `m=1` complex path and
the axisymmetric-flow manipulator path: eigenvalues to ~6e-9 (shift-invert Krylov-Schur stops at its
own tolerance, and a different factorisation ordering moves the result within it), eigenfunction
integrals to ~1e-15 relative.

## 7. Not done

- **The frozen distributed sparsity does not engage for the eigenproblem assembly**
  (`_get_frozen_sparsity_nnz()` is 0 there), so every eigen assembly rebuilds its pattern. Correct,
  just slower — worth a look now that the path works.
- **Azimuthal stability on a distributed mesh is tested but not exercised at scale.** The test
  problems are small.
