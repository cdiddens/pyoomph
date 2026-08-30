# Eigenvalue problems under MPI, through SLEPc

Status: **implemented and tested.** `tests/test_mpi_eigenvalues.py` (7 tests) covers both MPI
situations, the complex (azimuthal) path, the matrix-manipulator path and the refusal path. File and
line references are to the tree state at the time of writing.

**Goal.** Solve the generalized eigenproblem in parallel via SLEPc whenever pyoomph runs on more than
one process — with `--distribute`, and equally without it, since the eigensolve parallelises either
way. scipy/ARPACK cannot participate: it only ever sees one process' matrices. (It is no longer
*refused* under `--distribute` — the matrices are gathered onto rank 0 and solved there, see
[linear_solvers.md](linear_solvers.md) §9.6 — but that is a fallback, not parallelism.)

---

## 0. The sign convention, once

Worth stating before any code below, because the same two letters mean two different matrices
depending on which side of the nanobind boundary you are on, and every sign bug found in this area so
far has been someone assuming the wrong one.

The C++ core assembles the Jacobian and mass matrix of the residual `R(U, dU/dt) = 0` exactly as they
appear in it:

```
J = dR/dU                M = dR/d(dU/dt)
```

`Problem::assemble_eigenproblem_matrices` (`src/problem.cpp`) and the nanobind binding pass both
through untouched, and the JIT's `ADD_TO_MASS_MATRIX_*` macros (`src/jitbridge.h`) all accumulate
with `+=`, so a term written `weak(partial_t(u), v)` gives a **positive** mass entry. A perturbation
`v exp(lambda t)` of a stationary solution therefore satisfies

```
lambda M v + J v = 0
```

with `Re(lambda) > 0` meaning unstable. This is the convention documented to users
(`docs/source/tutorial/temporal/stability/constraints.rst`) and the one every bifurcation tracker in
`src/bifurcation.cpp` embeds: the fold row is `J Y = 0`, the Hopf rows are `(J + i*Omega*M) V = 0`,
the azimuthal rows are the same with complex `J` and `M`, and eigenbranch tracking adds `+lambda_r M V`.
Note this is the **opposite** of upstream oomph-lib's `HopfHandler`, deliberately: it is what makes
`Omega` the imaginary part of `lambda` rather than its negative.

The negation happens exactly once, in Python, in `GenericEigenSolver.get_J_M_n_and_type`
(`pyoomph/solvers/generic.py`):

```python
matJ = csr_matrix((-J_val, J_ci, J_rs), shape=(J_nr, n))     # note the minus; M is NOT negated
```

so from there on — `SlepcEigenSolver.solve`, the scipy backend, the shift-invert operators, the
matrix manipulators, everything in the rest of this document — the pair on the table is
`(A, M) = (-J, M)` and the problem solved is the ordinary `A v = lambda M v`. `NormalFormCalculator`
and `periodic_driving_response` repeat the same negation locally. `pyoomph/generic/assembly.py` and
`pyoomph/utils/lyapunov.py` deliberately do **not**: they keep the raw `J`, because they are
integrating the residual, not diagonalising it.

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

`refine_eigenfunction()` used to be the exception. It handles a global eigenvector like everything
else here, but `adapt()` carries the eigenfunction across the adaption in **history levels 3 and 4**,
and the history dof accessors were unsupported on a distributed problem — an equation number there is
global while the vector holds only this rank's rows. That is fixed (`Problem::get_dofs(t,...)` /
`set_dofs(t,...)` walk this rank's owned rows and let `synchronise_all_dofs()` carry the values,
commit `2531e00`), and the guard came off once the adaptation itself had been validated — see §9.

**Manipulators move to the backend.** `EigenMatrixManipulatorBase.apply_on_distributed_J_and_M`
takes the eigensolver's own matrices; `EigenMatrixSetDofsToZero` implements it as
`J.zeroRows(rows, diag=1.0)` / `M.zeroRows(rows, diag=0.0)`, restricted to locally owned rows. That
restriction is complete as well as safe: every owned row belongs to a dof of a non-halo local
element, so no rank misses one of its own. The base-class implementation raises, so a manipulator
with no distributed equivalent stops the run instead of silently not applying its constraint.

**`distributed_possible()` is finally consulted.** It had been declared on both solver base classes
for a long time and called from nowhere. `Problem._solve_eigenproblem_helper` now asks it — otherwise a
backend that cannot see a partitioned matrix would solve each rank's row block as if it were the whole
eigenproblem and return numbers that look entirely reasonable.

> `ScipyEigenSolver` used to be the one backend that answered no. It now gathers onto rank 0 instead
> (§9.6 of [linear_solvers.md](linear_solvers.md)), so no in-tree eigensolver trips this check any
> more; it guards backends defined outside pyoomph.

## 4. Three bugs found on the way, all outside the eigensolver

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

**4.3 `rotate_eigenvectors()` worked from a per-rank dof set** (`pyoomph/generic/problem.py`). It
fixes an eigenvector's phase from the dofs named by `dofs_to_real`, resolved through
`dof_strings_to_global_equations()`. Distributed, that resolves to *this rank's* dofs, so a rank
owning none of them got an empty set and died in `numpy.amax` with "zero-size array to reduction
operation maximum" — the rising-bubble benchmark at `np=4 --distribute`, and only there, since with
fewer ranks every rank happened to own some. `dof_strings_to_global_equations()` now `allgather`s the
set when distributed, so the selection is the global union everywhere.

Two things follow from that, and both are easy to get wrong:

- The union is a `set`, whose iteration order is not part of the guarantee that the ranks agree about
  its *contents*, and it feeds a floating-point sum. Sorted before use.
- The phase and the magnitude scale the **whole** eigenvector, so ranks that compute them from
  different data hold different eigenvectors — silently, because the eigenvalue is untouched. Today
  the PETSc eigensolver replicates each eigenvector to full global length
  (`_vector_to_global_array`), so every rank reduces over the identical set and no communication is
  needed; that path is also bit-identical to serial, which is why it is the one taken whenever the
  vector arrives full-length. When only the owned row block arrives, each rank restricts to its own
  rows (otherwise shared rows are counted twice) and each quantity is combined with the operator it
  is *made of*: the magnitude with `MAX` or with a summed numerator **and** a summed count, the phase
  from the summed complex numerator and count. Averaging per-rank averages weights the ranks equally
  however unevenly the dofs are spread over them, and is wrong by percent — `test_rotate_eigenvectors_reduces_correctly`
  fails by 1e-2 under exactly that mutation.

Note the phase itself is ill-conditioned by construction: it is the angle of the *mean* over the
selected dofs, so an antisymmetric mode makes that mean a near-total cancellation whose angle is
meaningless — serially too. The test measures the cancellation and scales its tolerance by it rather
than picking a loose constant: at cancellation 1 the two paths agree **exactly**, at 4e-10 they
differ by 9e-8, which is round-off divided by the cancellation and not a property of the reduction.

## 5. Still serial-only

These assemble the eigenproblem matrices themselves and read the result as a square global matrix,
or sit on augmented-system assembly that throws from C++ ("This likely does not work in distributed
parallel", `sparse_assemble_row_or_column_compressed_base_problem`). They refuse early, by name,
through `Problem._require_non_distributed`:

- the multi-assembly tensor cache
- Lyapunov exponents
  (this builds a replicated system and calls `solve_serial` on every rank — see
  [linear_solvers.md](linear_solvers.md) §9.2 for why it now has to say so first)
The **periodic driving response** has left this list; see §8, and so has
**`refine_eigenfunction()`**, which was on it for the history-dof reason of §3 rather than an
assembly one; see §9.
Branch switching, left eigenvectors and the normal forms are no longer on this list either: they left
the Python custom multi-assembly the way deflation and the Lyapunov coefficient did, and work under a
plain `mpirun` and under `--distribute` — see [branch_switching.md](branch_switching.md)
§"Under MPI" and `tests/test_mpi_branch_switch.py`.

Periodic orbit tracking, Floquet multipliers and `switch_to_hopf_orbit()` are no longer on this list
either — see [floquet_multipliers.md](floquet_multipliers.md) §8 and
[hopf_normal_form.md](hopf_normal_form.md) §4. What is still refused around them: the `HopfTracker`
route to the Hopf adjoint (`use_hopf_tracker_for_adjoint=True`, the Python custom assembler; the
eigensolver route is taken automatically under MPI) and the transient hand-back when leaving a
`with orbit:` block (it writes history dof values, which oomph-lib declares unsupported when
distributed).

Bifurcation *tracking* itself is no longer on this list — see
`dev_docs/mpi_augmented_systems.md`. What is still refused inside it: `blocksolve=True`,
and `adapt()` / arclength continuation while tracking, both for the history-dof reason in §3.

### 5a. Solving an eigenproblem *while* a bifurcation tracker is installed

Also no longer refused, serially or under MPI. It is the **base state's** eigenproblem — what tells
you a codim-2 point is coming along a bifurcation locus — and the assembly needed only its row layout
put back, because oomph's `get_eigenproblem_matrices` installs its own `EigenProblemHandler` anyway.
`Problem::BaseDofDistributionScope`, `dev_docs/mpi_augmented_systems.md` §6c for the mechanism and
`dev_docs/bifurcation_loci.md` §6 for the mode policy.

Three consequences for this document:

- **`get_eigen_row_layout()` now asks `_get_base_dof_distribution_info()`**, not
  `_get_dof_distribution_info()`. The two differ only while a tracker is installed, where the latter
  is the augmented layout — which is still what `tests/mpi_bifurcation_worker.py` wants from it.
  `rotate_eigenvectors` (§4.3) moved to the base layout for the same reason: an eigenvector has base
  length whatever the dof vector is doing.
- **The shift may not be zero while tracking**, and a zero one is refused by name. `petsc.py`'s
  `if self.spectral_transformation and shift` would otherwise leave `ST` at its default and ask MUMPS
  to factorise a matrix the tracker has just made exactly singular.
- **`set_eigenfunction_as_dofs` and `refine_eigenfunction` now refuse while tracking**, by name. The
  first would zero-pad a base-length eigenvector into the augmented dof vector — over the tracker's
  own unknowns — which the `numpy.pad` at its top did silently; the second adapts.

Periodic orbit tracking is now refused *by name* rather than falling through: the old guard tested
`get_bifurcation_tracking_mode() != ""`, which is `""` for orbits, so an eigensolve there would have
assembled on the `nT*Ndof+1` distribution.

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

The suite is marked `slow` and is **skipped without `--full`** — worth knowing, because
`pytest tests/test_mpi_eigenvalues.py -q` reports `11 skipped` in under a second and reads, at a
glance, exactly like a suite that ran.

`test_rotate_eigenvectors_reduces_correctly` (§4.3) is the one test here whose code path nothing in
the normal flow reaches, since the eigensolver replicates its eigenvectors. It manufactures the input
instead: solve normally, slice each eigenvector down to the rank's own rows, feed that back in, and
require the answer to match the full-vector one restricted to the same rows. Both wrong-operator
mutations (`MAX` → mean-of-maxes, mean → average-of-per-rank-averages) fail it by ~1e-2.

### Verified on the rising-bubble benchmark

17925 dofs, leading eigenvalue, all five configurations agreeing to **4.4e-12**:

| config | rank 0's eigen rows | eigenvalue |
|---|---|---|
| serial | 18175 | `-0.106212230504436 + 0.761243489826195i` |
| `np=2` plain | 9088 | `-0.106212230503254 + 0.761243489829197i` |
| `np=2 --distribute` | 12124 | `-0.106212230506938 + 0.761243489829080i` |
| `np=4` plain | 4544 | `-0.106212230500189 + 0.761243489825755i` |
| `np=4 --distribute` | 10920 | `-0.106212230501172 + 0.761243489829077i` |

Without `--distribute` the eigensolver imposes its own even split of the replicated matrix, and 9088
/ 4544 are exactly `ceil(18175/nproc)`. With `--distribute` the split is oomph's own dof distribution
and is uneven, so rank 0's share is not `n/nproc`; the benchmark records only rank 0, so these two
rows say nothing about how the remainder is spread. `_assert_solve_was_split` in the test suite is
what actually checks the blocks tile `[0, ndof)`.

## 7. Not done

- **The frozen distributed sparsity does not engage for the eigenproblem assembly**
  (`_get_frozen_sparsity_nnz()` is 0 there), so every eigen assembly rebuilds its pattern. Correct,
  just slower — worth a look now that the path works.
- **Azimuthal stability on a distributed mesh is tested but not exercised at scale.** The test
  problems are small.
- **Eigenvectors are replicated on every rank** (`_vector_to_global_array` gathers with
  `Scatter.toAll`), which is the right trade at today's sizes and the wrong one for a genuinely
  large problem. See [dev_docs/replicated_mpi_correctness.md](replicated_mpi_correctness.md) §5.

Two bugs in oomph-lib's `CRDoubleMatrix::redistribute`, on the path
`get_eigenproblem_matrices` takes when the problem is *not* distributed, made every eigenproblem with
fewer equations than ranks abort; see
[dev_docs/replicated_mpi_correctness.md](replicated_mpi_correctness.md) §2.

---

## 8. The periodic driving response

`pyoomph/utils/periodic_driving_response.py`. Works under `mpirun`, with and without `--distribute`,
and is solved genuinely in parallel in both cases. `tests/test_linear_response.py` covers the algebra
against a closed form serially; the distributed agreement was measured on the drum tutorial (§8.3).

### 8.1 What it solves

At driving frequency `omega` the response is the bordered real system

```
[ s*J        omega*M    e_d  e_dt ] [xr ]   [0]
[ s*omega*M   -J          .    .  ] [xi ] = [0]
[ e_d^T        .          .    .  ] [l_1]   [1]
[ e_dt^T       .          .    .  ] [l_2]   [0]
```

with `J` the **negated** assembled Jacobian (the `(A,M) = (-J,M)` convention of §0), `s` the sign
orientation, and the two border rows pinning the driving ODE's own two dofs so that the driving is
exactly `cos(omega t)`. Nothing about that changed; what changed is the **ordering of its unknowns**.

### 8.2 Interleave the unknowns, do not permute the blocks

The unknowns used to be ordered by block, `[xr(0..n), xi(0..n), l_1, l_2]`. Once the dofs are
partitioned, rank *r* owning dof rows `[first_row, first_row+nrow_local)` then owns two **disjoint**
row ranges of that system, which `MatCreateAIJ` cannot express — a row-distributed PETSc matrix owns
one contiguous block per rank and nothing else. Interleaving by dof fixes it outright:

```
xr_k -> 2k        xi_k -> 2k+1        l_1 -> 2n        l_2 -> 2n+1
```

Rank *r* then owns `[2*first_row, 2*(first_row+nrow_local))`, contiguous, and the two border rows go
on the last rank. A global column index `j` maps to `2j` / `2j+1` — no owner lookup, no prefix sum, no
communication, and the column indices stay global exactly as `MatMPIAIJSetPreallocationCSR` wants.

Serially the ordering is a permutation of the old one, so the answer is the same up to pivoting.
Measured on the drum tutorial, 1000 frequencies × 10 Bessel modes: **max relative difference 2.8e-12**
against the block-ordered result.

### 8.3 The omega-dependent entry, and why the pattern must not move

The driving ODE contributes `d(EQ_yp)/d(dp) = omega**2`, i.e. `-omega**2` in `matJ`. The old code
overwrote that single entry in the assembled CSR on every iteration — a `SparseEfficiencyWarning` when
the diagonal is structurally absent, and a **global** index write that no rank can perform once the
rows are partitioned. It is now zeroed once at assembly and re-supplied per frequency as two explicit
triplets, on its owning rank only.

That is not just tidiness: it makes the **sparsity pattern of the bordered system independent of
omega**, which is what lets `PETSCSolver.solve_python_built_distributed` keep one Mat and one KSP alive
across the whole scan and MUMPS keep its symbolic analysis. Everything else about a frequency step is
a values-only refresh.

### 8.4 Where the row split comes from

`GenericEigenSolver.get_parallel_row_split()` and `.local_row_block()` — lifted out of
`SlepcEigenSolver` so this module does not have to reach into the PETSc subclass for them, and so a
non-PETSc eigensolver does not break the import. Same policy as §3: under `--distribute` the split is
oomph's own dof distribution and the matrices already hold only this rank's rows; under a plain
`mpirun` the matrices are replicated and a contiguous split of `n` is imposed here, each rank slicing
out its share. Both give one parallel solve rather than `nproc` identical ones.

Verified on the drum (403 dofs → 808 bordered rows) at `np=3`, blocks tiling `[0, 808)` exactly:

| | rank 0 | rank 1 | rank 2 |
|---|---|---|---|
| plain `mpirun` | `[0, 270)` | `[270, 538)` | `[538, 808)` |
| `--distribute` | `[0, 212)` | `[212, 452)` | `[452, 808)` |

### 8.5 The solve itself

`GenericLinearSystemSolver.solve_python_built_distributed(ntot, nrow_local, first_row, mat_local,
b_local)` — a system **pyoomph assembled in Python**, row-distributed, as opposed to one oomph-lib
handed down through `solve_distributed`. Two implementations:

- **PETSc**: an MPIAIJ on `COMM_WORLD` and a `preonly`+`lu` KSP, in slots of their own
  (`_aux_mat`, `_aux_ksp`) — never `petsc_mat`/`ksp`, which hold the Newton solve's factorisation. A
  caller that interleaves the two, which a frequency scan between solves does, would otherwise
  back-substitute against whichever factors were written last. Keeping them apart is what makes
  `_note_external_serial_solve()` *unnecessary* on this path rather than merely survivable. The factor
  package is MUMPS or SuperLU_DIST under MPI and refused by name if neither is present, for the same
  reason as `_require_parallel_capable`: PETSc's own LU is sequential, and quietly solving redundantly
  would look like success while being slower than serial.
- **The base class**: allgather the whole square system onto every rank (`mpi_allgather_square_csr`,
  moved to `generic/mpi.py` from `bifurcation_tools._allgather_square`) and call `solve_serial`. This
  is what the module used to do inline for a system it had built globally in the first place, and it
  keeps pardiso / scipy / accelerate / superlu working under `--distribute` — redundantly, which it
  says once, on rank 0.

The response is replicated to full global length afterwards (`mpi_allgather_vector`), because
`set_eigenfunction_as_dofs()`, the mesh data cache and the VTK output all index an eigenvector by
global equation number. Same trade as §3.

`solve_driving_response()` — the older route, through `solve_eigenproblem` or a Hopf tracker — was
refused as well and needed no change at all: both of those work distributed already. Measured on the
drum at 200 Hz with `use_target=True`, serial and `np=3 --distribute` agree to all 15 digits.

### 8.6 The tutorial script had a second bug behind the guard

`docs/source/tutorial/advstab/response/linear_response_drum.py` projects the response onto Bessel
modes by splining it along the radius. `get_cached_mesh_data("drum", ...)` returns **this rank's
partition** of that radius under `--distribute`, so lifting the guard alone would have produced a run
that finished and wrote plausible-looking nonsense, differently on every rank. It now asks for
`global_mesh=True` inside a `run_with_global_mesh_data(...)` block. Worth remembering for any other
script that post-processes an eigenvector pointwise: making the *solve* distributed does not make the
*analysis* distributed.

### 8.7 Measured agreement

Drum tutorial, 1000 frequencies, 10 Bessel amplitudes each, max relative difference against the serial
run of the same build:

| config | max rel. difference |
|---|---|
| serial, old block ordering | 2.8e-12 |
| `mpirun -n 3` | 3.1e-11 |
| `mpirun -n 3 --distribute` | 1.9e-11 |
| `mpirun -n 3 --distribute --pardiso` (allgather fallback) | 4.3e-12 |
| `mpirun -n 3 --distribute --superlu` (allgather fallback) | 3.1e-11 |

The oscillator tutorial (an ODE, so nothing to partition) agrees to 2.6e-15 at `np=2 --distribute`.

The response amplitudes are the right thing to compare: `distribute()` renumbers the dofs, so nothing
that indexes a dof vector is comparable between a serial and a distributed run.

---

## 9. Adapting the mesh to an eigenfunction

`Problem.refine_eigenfunction()` works under `mpirun`, replicated and `--distribute`. Nothing had to
be changed to make it: the guard was the last user of a blocker that commit `2531e00` had already
removed (§3), and it was kept only because the adaptation itself had never been run distributed. It
now has been, and the suite is `tests/test_mpi_eigen_adapt.py` + `tests/mpi_eigen_adapt_worker.py` —
which is also the *only* coverage this feature has ever had, serial included.

Two cases, both on an adaptive `RectangularQuadMesh` with a diffusivity bump and a source placed
asymmetrically so that the base-state and eigenfunction error fields peak in different places: a
plane real eigenproblem, and an axisymmetric one at `azimuthal_m=1` — the complex path, the second
estimator pass over `Im(v)`, and the forced-zero-dof manipulator (a scalar field is unconstrained at
`r=0` for the base state but must vanish at `|m|=1`).

### 9.1 The oracle that matters is the carry-across, not the mesh

The eigenfunction crosses the adaptation in history levels 3 and 4. **Refining an element leaves the
FE function it interpolates exactly unchanged**, so with one adaptation and no unrefinement the
eigenfunction's integrals must come back out of those levels unchanged — whichever elements were
refined, and on whatever partition. That is the assertion the lifted guard rests on, and it is
partition-independent by construction. Measured:

| | serial | `-n 2 --distribute` | `-n 4 --distribute` |
|---|---|---|---|
| Cartesian | 7.9e-16 | 2.2e-16 | 0 |
| azimuthal `m=1` | 2.1e-7 | 2.1e-7 | 2.1e-7 |

The azimuthal residue is the same in serial, so it belongs to the `m != 0` machinery (axis dofs
forced to zero by a matrix manipulator rather than pinned) and not to MPI. It is not chased here, but
it is the thing to look at if that path ever misbehaves.

### 9.2 The refined mesh is partition-dependent, and that is the estimator

A **replicated** `mpirun` reproduces the serial mesh to the last element (fingerprint, `ndof` and
element count all identical at `-n 2` and `-n 4`, both cases): every rank computes every Z2 patch.

Under `--distribute` it need not. At `-n 2` it happened to match exactly; at `-n 4` the Cartesian case
came out at 1093 dofs against 1129 serially (−3%), with the eigenvalue moving by 2e-6 relative. The
cause is upstream and documented in place: oomph-lib's distributed Z2 recovery **neglects the flux
contributions of patches that can only be assembled from vertex nodes owned by another process** (the
long "NOTE FOR FUTURE REFERENCE" comment in `LagrZ2ErrorEstimator::setup_patches`,
`src/lagr_error_estimator.cpp`). Elements sitting near the refinement threshold are then decided
differently.

This was worth establishing rather than assuming, because a partition-dependent mesh is exactly the
failure class of `adaptive_refinement.md` §8.2. The control that settles it is in the worker as
`--driver base`: the *same* estimator and the *same* number of adaptations, driven by the base state
alone, diverge from serial in the same way (885 → 857 dofs at `-n 4`). So the eigenfunction adds no
partition dependence of its own, and `test_mesh_difference_is_the_estimator_not_the_eigenfunction`
asserts precisely that — a mesh difference that appears *only* when adapting to an eigenfunction
fails the suite.

Everything else is held exactly: all ranks of one run report bit-identical results (the gathered
eigenvector, `ndof`, the mesh fingerprint and every integral), and `PYOOMPH_CHECK_HALO_CONSISTENCY=throw`
passes on the distributed arms of both cases.

### 9.3 Still refused

`refine_eigenfunction()` while a **bifurcation tracker** is installed, distributed or not: adapting
renumbers, which pulls the augmented dof vector out from under the handler. The bifurcation GUI's
`_adapt_to_eigenfunction()` keeps that refusal and has dropped its distributed one.
