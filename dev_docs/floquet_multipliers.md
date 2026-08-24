# Floquet multipliers of a periodic orbit

Two formulations live side by side in `Problem.get_floquet_multipliers()`
(`pyoomph/generic/problem.py`), selected by `method=`:

* **`"condensed"`** (default since this work) — `pyoomph/generic/floquet.py`. Condenses the block
  bidiagonal orbit Jacobian into the monodromy matrix and takes its eigenvalues.
* **`"periodic_schur"`** — opt-in, same file. Periodic QR: the same condensation, but the transfer
  matrices are never multiplied. §7.
* **`"eigenproblem"`** — the original code, kept verbatim as
  `Problem._get_floquet_multipliers_eigenproblem`. One large singular pencil over all time points.

§1–§4 are the condensation and why it replaced the other; §5 records what was measured; §6 the
things that are *not* fixed and the traps that remain.

---

## 1. What the orbit Jacobian looks like

`PeriodicOrbitHandler` numbers the augmented unknowns time-major,
`global_eqn = Ndof*tindex + base_eqn` (`src/bifurcation.cpp`, `eqn_number()`), giving
`nT = n_tsteps()` blocks of `nbase` unknowns plus the scalar period `T`.

`floquet_mode` is on for `bspline_order == 0` (`mode="floquet"`) **and** for `bspline_order < -2`
(`mode="collocation"`, any order) — the constructor sets both. So the default collocation orbits
already have the explicit end-of-period block and always did; only `central`, `BDF2` and `bspline`
do not, and `is_floquet_mode()` refuses those.

Per element `ie` of the time discretization (Lagrange order `m`; `m = 1` for the plain midpoint
`mode="floquet"`), `get_jacobian_collocation_mode()` loops `inode < el->nnode()-1` for the
equations but over all `el->nnode()` for the unknowns. So the element

* writes **row blocks** `ie*m .. ie*m+m-1`,
* into **column blocks** `ie*m .. ie*m+m`.

Row blocks `0 .. nT-2` are therefore each written by exactly one element, and row block `nT-1` is the
wrap-around identity `v_{nT-1} - v_0 = 0`. That last block row is what both formulations key off.

`PeriodicOrbitHandler::get_time_element_node_indices()` hands this structure to Python; the
collocation branch reads it off the time mesh's `TimeNode::get_index()`, and the midpoint branch
states the same thing as the pairs `(ti, ti+1)`.

## 2. The condensation

Slice the element's sub-block as `[E0 | L]`, `E0` being the first `nbase` columns:

```
L @ [v_{ie*m+1}; ... ; v_{ie*m+m}]  =  -E0 @ v_{ie*m}
C_ie := last nbase rows of L^-1 (-E0)         # nbase x nbase transfer
Mono  = C_{Nelem-1} @ ... @ C_1 @ C_0
multipliers = eig(Mono)
```

The rows and columns of an element are contiguous ranges of the CSR matrix (the time mesh gives
element `ie` the nodes `ie*order + in`), so the sub-block is a plain two-sided slice.
`_TimeElement.__init__` asserts the consecutiveness rather than assuming it, and
`check_orbit_jacobian_structure()` verifies that no equation row actually reaches outside its own
element's columns — the period column being the one legitimate exception, since `dR/dT` is dense down
the whole matrix and is dropped from the Floquet problem exactly as the eigenproblem formulation
drops it. Getting the structure wrong yields plausible multipliers rather than an error, which is why
both checks are on by default.

**The mass matrix is arbitrary, and may be singular.** `M` never appears on its own: it is already
inside `E0` and `L`, weighted by `dpsi/ds` (collocation) or `±invds` (midpoint). Nothing inverts it,
which is what a shooting formulation would have to do. `L` stays invertible where `M` has zero rows
because it carries the `0.5*J` terms.

## 3. Why the old formulation was replaced

`method="eigenproblem"` builds the pencil `J v = mu M v` over the whole `nT*nbase` space, with `M`
the identity on the last block alone, so the wrap-around row becomes `v_{nT-1} - v_0 = mu v_{nT-1}`
and `gamma = 1/(1-mu)`. Correct, and the same Fairgrieve–Jepson idea — but:

* The mass matrix has rank `nbase` in an `nT*nbase`-dimensional space, so generically only `nbase`
  eigenvalues are finite. The rest are infinite and are removed by `valid_threshold=10000`.
* An infinite `mu` maps to `gamma = 0`, so that threshold cannot distinguish a spurious eigenvalue
  from a genuinely small multiplier. Small multipliers are discarded wholesale.
* How many survive depends on the eigensolver, so the **number of returned multipliers varied** —
  including between a serial run and an `mpirun` of the same script. The Langford tutorial carried
  an explicit workaround saying so; it has been removed.
* A shift had to be supplied by hand (`shift=3` in the tutorial).
* Cost and memory scale with `nT*nbase`, not `nbase`.

The condensation has none of these: no shift, no threshold, exactly `nbase` multipliers.
`shift` and `valid_threshold` are still accepted for signature compatibility but warn when passed.

## 4. Dense or matrix-free

Forming `Mono` costs one solve per time element with `nbase` right-hand sides. Applying it
matrix-free costs one solve per time element **per Arnoldi iteration**, so it only pays off while few
multipliers are wanted — the switch is on `n/nbase`, not on `nbase` alone (`dense_threshold=2000`,
plus a rule that keeps the dense route when `4n > nbase` up to `_DENSE_MONODROMY_CAP = 4000`).

The first version switched on `nbase` alone and, at `nbase=802` with `n=None`, ran ARPACK with
`k=800` out of 802 — the worst of both, and it is also the case `eigs` cannot do at all
(`k < nbase` is required). Hence the `n`-aware rule.

## 5. Measurements

Langford ODE (`nbase=3`, `NT=50`, `order=3`), against the analytical non-trivial multiplier: the
condensed and the eigenproblem values agree to every printed digit at every continuation step, and
both track the analytical curve to the accuracy of the time discretization. The condensed method
returns 3 multipliers at every step; the trivial one sits at `|lambda-1| ~ 1e-15` on a well converged
orbit. Repeated for `mode="floquet"` and `mode="collocation"` with `order` 1, 2 and 3 — same
agreement in each, with the expected loss of accuracy at low order.

1D Brusselator PDE orbit (`tests/benchmarks/bench_periodic_orbit_1d.py`), `NT=30`, `order=3`:

| `N` | `nbase` | orbit ndof | dense (all multipliers) | matrix-free (8) | eigenproblem (8) |
|---|---|---|---|---|---|
| 40 | 162 | 5023 | 0.57 s | 0.38 s | 0.41 s |
| 200 | 802 | 24863 | 65 s (802 values) | 277 s (8 values) | 223 s (8 values) |

Dense and matrix-free agree to 3e-13 on the dominant six, the eigenproblem method to 3e-11. Note the
second row: **the matrix-free route was slower than forming the whole monodromy**, because the
Brusselator's multipliers are clustered around 0.992 and Arnoldi converges badly on that. It is kept
for the case where `nbase**2` simply does not fit, not because it is generally faster.

Reproducibility: the multipliers are *not* bit-identical between a serial run and a replicated
`mpirun`, because the Jacobian assembly itself is not (oomph dispatches to `parallel_sparse_assemble`
for any `nproc>1`, which sums element contributions in a different order). Measured on an orbit whose
period agrees to the last bit, the multipliers differ by ~3e-11 relative. What *is* now stable is the
count.

## 6. What is not fixed

* **The plain product loses the smallest multipliers**, though far less than expected — see §7 for
  the measurement and for `method="periodic_schur"`, which fixes it.
* **A DAE's algebraic directions do not give zero multipliers.** Gauss–Legendre collocation is not
  stiffly accurate: `|R(inf)| = 1`. The perturbation of an algebraic direction is the degree-`order`
  polynomial vanishing at the `order` Gauss points of the element, and since those points are
  symmetric about the midpoint its value at the end of the element is `(-1)**order` times its value
  at the start. Over the orbit that accumulates to exactly `(-1)**(number of time intervals)` —
  verified to 1e-14 for `order` 1, 2 and 3 at both parities in
  `tests/test_floquet_multipliers.py::test_dae_algebraic_multiplier_sign`.
  **With an odd number of intervals this puts a spurious multiplier on `-1`, where a period-doubling
  bifurcation would be.** It is a property of the discretization, not of the condensation — the
  eigenproblem method finds the same value, just to six digits fewer — and an even number of
  intervals moves it next to the trivial `+1` instead. A Radau IIA collocation would put it at 0.
* **`--distribute` works**, for the orbit and for the multipliers — §8. Two things around it do not:
  `switch_to_hopf_orbit()` and the transient hand-back when a `with orbit:` block exits.


## 7. `method="periodic_schur"`

### What it does

Subspace iteration run through the chain instead of multiplying it. Starting from an orthonormal
`Q`, sweeping `V, R_ie = qr(C_ie @ V)` along the orbit gives, exactly and at every sweep,

```
Mono @ Q_old = Q_new @ S,      S = R_{p-1} ... R_0   (upper triangular)
```

so with `W = Q_old^H Q_new` the multipliers are `eig(W @ S)`. Subspace iteration drives `W` upper
triangular at rate `|lambda_{k+1}/lambda_k|` per sweep, and where it is, `W @ S` is upper triangular
too: `lambda_k = W[k,k] * exp(sum_ie log R_ie[k,k])`. **That sum is where the overflow goes away** —
the product of the diagonals is accumulated in logs and never formed.

The QR sign freedom is fixed so `R` has a positive real diagonal (`_positive_diagonal_qr`), without
which the diagonal is not a magnitude and the sweep-to-sweep comparison compares phases.

Indices sharing a modulus — a complex conjugate pair, or the `+1`/`-1` a DAE produces — can never be
separated: the rate is exactly 1 there. They are left as a diagonal block of `W` and diagonalized
directly, each factor first divided by the geometric mean of its own diagonal so the block product
stays O(1) while its scale is carried in logs alongside.

Eigenvectors are not produced in the original basis by this route, so they are recovered afterwards
by inverse iteration on the dense monodromy — well conditioned precisely because the eigenvalue
handed to it is accurate.

### Stagnation

The first version ran all 200 sweeps on the DAE cases: `+1` and `-1` have the same modulus, so the
off-triangular part of `W` stops shrinking and no further sweep can help. Breaking after three
sweeps without at least a 10% reduction takes those from 200 sweeps to 6 and 7, same answers.

### What it is worth — measured

Against a **120-digit product of the same transfer matrices**, so the comparison isolates the
product's roundoff from the time discretization. Stuart–Landau plus an upper-triangular stiff chain
`z_k' = -a_k z_k + c z_{k+1}`, `a = [2,4,8,16,32]`, `NT=48`, `order=3` — spectrum spanning 26 orders
of magnitude:

| exact (120 digits) | plain product, `c=1` | plain product, `c=1e6` | periodic Schur, `c=1e6` |
|---|---|---|---|
| 1.0 | 1.1e-16 | 1.1e-16 | 8.9e-16 |
| 3.4876e-06 | 4.7e-11 | 4.7e-11 | 3.6e-16 |
| 3.4872e-06 | 1.2e-16 | 4.9e-16 | 1.1e-15 |
| 1.2112e-11 | 1.3e-16 | 0 | 5.3e-16 |
| 8.2843e-14 | 1.5e-16 | 1.1e-15 | 6.4e-15 |
| 7.2907e-23 | 3.2e-16 | **9.4e-06** | 2.4e-13 |
| 3.5807e-26 | 1.6e-16 | **1.4e-02** | 2.6e-13 |

(relative errors). Two things to take from this:

* **The plain product is far better than its reputation.** With ordinary coupling it is at machine
  precision across all 26 decades — LAPACK's balancing inside `numpy.linalg.eig` does the work
  periodic Schur would. It only breaks down for the bottom two multipliers, and only once the
  monodromy is made strongly non-normal.
* **Where it does break down, periodic Schur is eleven orders of magnitude better**, and it is also
  better on the near-degenerate pair at 3.487e-6 (3.6e-16 against 4.7e-11) in every case.

Worth knowing before reaching for it: for a mode that stiff the *discrete* multiplier has little to
do with `exp(-a*T)` anyway, because Gauss collocation's stability function tends to `±1` rather than
0 (§6). So the regime where the accuracy gain matters is narrow. It is opt-in for that reason.

### Cost, and the clustered case

`sweeps * Nelem` matrix products of size `nbase`, i.e. dense-only; refused above
`_DENSE_MONODROMY_CAP`. On the 1D Brusselator (`N=40`, `nbase=162`) the spectrum is clustered around
0.992, so it stagnates after 4 sweeps with a single unseparated block of 159 and falls back to what
is essentially the plain product — 1.4 s against 0.57 s for the default, same answers to 2e-15. That
degradation is graceful and correct, but it is also the common case for a PDE orbit: **periodic Schur
buys accuracy on graded spectra, and nothing at all on clustered ones.**


## 8. `--distribute`

`PeriodicOrbitHandler` used to be entirely replicated -- `Tadd`, `x0`, `n0`, `du0ds` and `Count` were
global-`Ndof` `std::vector`s indexed by global equation number, `eqn_number()` returned the naive
number untranslated, and the dof vector was built non-distributed. It is now written against
`AugmentedDofDistributionHelper`, the same way the four bifurcation-tracking handlers were
(`mpi_augmented_systems.md` Part I):

* `Dist_helper.initialise()` first in the constructor, `restore_base_distribution()` in the destructor.
* `Tadd`, `x0`, `n0`, `du0ds` and `Count` are `DoubleVectorWithHaloEntries` on the base distribution.
  The element loops read them by BASE equation number through `global_value()`, which degrades to
  plain `[]` when not distributed -- so one code path serves all three modes.
* `Count` and the global element count come from `setup_count_and_nelement()` (halo-skipping plus a
  reduction), so the `1/Count` weights still telescope to 1 across ranks.
* The naive layout `[u_0 | u_1 | ... | u_{nT-1} | T]` goes to `build_augmented_dofs()` -- it is exactly
  the historical `Ndof*tindex+eqn` numbering, so `eqn_number()` only has to run its result through
  `Dist_helper.global_eqn()`.
* Every `*(GetDofPtr()[glob_eqn])` in the six residual/Jacobian routines became
  `Problem::global_dof_pt(elem_pt->eqn_number(i))` -- the element's OWN base equation number, not the
  handler's translated one.
* `backup_dofs`, `restore_dofs`, `set_dofs_to_interpolated_values` and
  `update_phase_constraint_information` write the base dofs wholesale rather than through an element,
  so unlike the assembly loops they cannot use `global_value()` (a row this rank neither owns nor
  halos has no entry to reach). They loop over owned rows and then push to the halos.
* A `synchronise()` override refreshes the `Tadd` halos and broadcasts the rank-0-owned period; oomph
  already calls it from `Problem::synchronise_all_dofs`, so no vendored change was needed.
* `dof_distribution_helper()` now answers non-NULL, which also makes `BaseDofDistributionScope` work
  during orbit tracking. The Python-side refusal to eigensolve while an orbit handler is installed
  stays deliberately -- Floquet multipliers are the right tool there.

Then the Floquet side needed one thing more. **Under `--distribute` the augmented rows are
interleaved per rank** -- rank 0's base rows, then rank 0's rows of each time block, then rank 1's --
so a gathered orbit Jacobian is NOT in the time-major order the condensation slices along, and every
block it cut would be the wrong one. `PeriodicOrbitHandler::get_naive_equation_order()` hands out the
naive -> augmented translation and `_to_time_major()` applies it to both axes. It returns an empty
list when not distributed, where the two orders already agree, and the permutation is then skipped.

That the *structure check* caught this rather than the condensation answering wrongly is the reason
it is on by default; see §2.

### Validation

`tests/test_mpi_floquet.py`, Stuart-Landau kinetics pointwise on a 1D mesh plus diffusion. Its
spatially uniform state is exactly `u=cos(t)`, `v=sin(t)` with `T=2*pi` (diffusion annihilates a
uniform field), so the guess handed to the handler is the answer, on a mesh with enough elements to
distribute -- and no Hopf tracker is needed to get there, which matters because that route is still
serial (below). Serial against `mpirun -n 4 --distribute`, `N=40`, `NT=24`, `nbase=162`, orbit
`ndof=4051`:

| | serial | `-n 4 --distribute` |
|---|---|---|
| period | 6.2831995733762831 | 6.2831995733762831 (bit-identical) |
| Floquet multipliers | | agree to 2e-15 .. 4e-14 |
| 8 sampled orbit states | | bit-identical to 14 digits |

The period comes out bit-identical across serial and 2, 3 and 4 ranks. The multipliers differ only by
the assembly's summation order, which the partitioning changes.

### What is still refused around it

* **`switch_to_hopf_orbit()`** needs the first Lyapunov coefficient, computed by the Python custom
  assembler in `bifurcation_tools.py`, which is still serial (`mpi_augmented_systems.md` Part II). It
  now says exactly that instead of letting the generic refusal claim orbit tracking is unsupported.
  Build the orbit guess yourself and call `activate_periodic_orbit_handler`, or pass `dparam` and
  `orbit_amplitude` to skip the Lyapunov coefficient.
* **Leaving a `with orbit:` block** seeds three history levels so that a plain `run()` continues the
  orbit transiently. Writing history dof values is not implemented when distributed -- oomph-lib
  declares the `t>0` dof accessors unsupported there -- so `PeriodicOrbit.__exit__` warns and drops
  the orbit instead. Everything inside the block is unaffected.

### The bugs found on the way

Three, and the second and third are the interesting ones.

1. **`T_global_eqn` was a block short.** Set to `Ndof*Tadd.size()` instead of `Ndof*n_tsteps()`,
   which broke even the serial Newton solve. Anything else indexing the naive layout deserves the
   same scrutiny.
2. **`Problem::set_history_dofs` overran the heap** (`src/problem.cpp`). `Problem::set_dofs(t,...)`
   already refuses when distributed, with a comment explaining that the loops there use global
   equation numbers against a local vector -- but `set_history_dofs` gets there only *after* its own
   fill loop, which builds `dofs` on the dof distribution (`nrow_local` entries) and then writes
   `ndof()` -- the GLOBAL count -- of them into it. At `ndof=162` on 4 ranks that is ~120 doubles
   past the end of the buffer, three times over. This is pre-existing and was simply unreachable:
   its only callers are `PeriodicOrbit.__exit__` and `refine_eigenfunction()`, and both used to be
   refused on the Python side first. The refusal is now hoisted above the fill loop.

   Worth knowing for the next time this shape of bug appears: the corruption surfaced as a glibc
   "corrupted double-linked list" inside an unrelated `malloc` much later, and PETSc's signal handler
   then called `MPI_Abort`, which itself allocates -- so the job **hung** on the already-held malloc
   lock instead of dying. A run that hangs with every rank at 0.2% CPU, with `PetscSignalHandler` and
   `PMPI_Abort` above `__libc_malloc` in the backtrace, is this and not a missing collective.
3. **`get_current_dofs()[0][:nbase]` read the wrong entries.** `PeriodicOrbit.__exit__` and
   `change_sampling()` sliced the first `nbase` entries off the gathered *augmented* vector to get
   the base state, which is only the base block when the two orderings agree. `_current_base_dofs()`
   now goes through the same naive ordering the Floquet permutation uses.


## 9. Where the time actually goes, and what parallelizes

Profiled on the distributed 1D orbit of §8 scaled up to `nbase=1282` (`N=320`, `NT=24`, orbit
`ndof=32051` -- the largest these MPI constraints allow), **all ranks pinned to one BLAS thread**.

That pinning is not a detail. The first profile was taken without it, and every rank of a 4-rank run
had OpenBLAS spawning threads across all cores: the transfer solves read 13.9 s where the same work
takes 1.34 s on one rank. Every wall-clock number below is with `OMP_NUM_THREADS=1` exported through
`mpirun -x`, and any future measurement here has to do the same or it is measuring oversubscription.

### The breakdown, one rank

| step | time | note |
|---|---|---|
| gather the orbit Jacobian | 0.16 s | 12 MiB |
| permute to time-major | 0.00 s | |
| element LUs (`Nelem=8`) | 0.22 s | |
| transfer solves | 1.34 s | 8 elements x 1282 right-hand sides |
| matmul chain | 0.52 s | |
| dense eig | 0.68 s | |
| eigenfunction reconstruction | **~172 s** | before batching; see below |

### What was fixed

* **The eigenfunction reconstruction was 93% of a full call.** It pushed each eigenvector through the
  chain on its own, so asking for all 1282 multipliers meant 1282 separate passes of sparse solves:
  185 s, against 13 s for everything else. `orbit_eigenfunctions()` now pushes them all through
  together. Same arithmetic per column, 14x on the whole call, and it is a *serial* win as much as a
  parallel one. `test_eigenfunction_closes_the_orbit` pins the result through the invariant
  `v(s=1) = lambda*v(s=0)`.
* **The transfer solves are shared out by column** (`transfer_matrices()`), which is the one piece
  that genuinely parallelizes: 1.34 s -> 0.72 s -> 0.48 s on 1, 2 and 4 ranks. One `Allgatherv` for
  all elements together.
* **The product and the eigendecomposition are done on rank 0 and broadcast.** Every rank used to
  repeat both. That is not merely wasted work: the redundant `numpy.linalg.eig` went from 0.68 s on
  one rank to 1.76 s on four as the copies contended for memory bandwidth, which made a 4-rank
  Floquet solve *slower overall* than a serial one despite its solves being 2.8x faster.

Do **not** replace the per-element transfer matrices with propagating the identity through the whole
chain, which does the same solves holding one block instead of `Nelem` of them. It looks like a
strict improvement and is not: the accumulated block enters the next element as the right-hand side
`E0 @ X` of an ill-conditioned solve, so its rounding is amplified at every stage. On the stiff
non-normal chain of §7 the two smallest multipliers went from ~1e-15 relative error to 5e+2 and
4e+4. `test_periodic_schur_beats_the_product_at_the_bottom` catches it.

### End to end, and why it stops there

| | 1 rank | 2 ranks | 4 ranks |
|---|---|---|---|
| all 1282 multipliers | 13.1 s | 13.4 s | 16.1 s |
| the 8 dominant ones | 3.7 s | 3.6 s | 4.7 s |

**Floquet multipliers do not parallelize usefully across MPI ranks at these sizes.** What is left
after the three fixes is dense linear algebra on `nbase x nbase` objects, and ranks on one node
contend for the same memory bandwidth rather than adding to it. What MPI buys is the orbit *solve*
being distributed (§8) and the problem fitting at all -- not a faster multiplier calculation.

### Why the native distributed path was not built

The plan's stage 4 was to stop gathering the orbit Jacobian: extract each time element's blocks as
distributed PETSc Mats and run SLEPc on a shell matrix with one KSP per element. Two measurements
rule it out:

* **The gather is 0.16 s of 13 s**, so there is no time in it.
* **The gathered Jacobian is 12 MiB; the dense monodromy machinery is 1.6 GiB** (peak RSS at
  `nbase=1282`, dominated by the `nbase x nT*nbase` complex eigenfunction array and the eig
  workspace). A problem too large to gather the Jacobian is, by two orders of magnitude, already too
  large to form the monodromy -- so the memory case the native path was meant to serve does not
  arise before the method itself does.

The route that *would* serve very large `nbase` is the existing matrix-free operator, which needs no
`nbase x nbase` object at all. Its limit is Arnoldi convergence on clustered spectra (§5), not the
gather.
