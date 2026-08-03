# Hanging-node values: a fixed MPI staleness bug, and an open nondeterminism

Status: **one bug fixed and tested; a second, larger one found, characterised and OPEN.** Found while
benchmarking the distributed eigensolver (`mpi_eigenproblems.md`), which is how a chain of unrelated-
looking symptoms got followed back to its source. File references are to the tree at the time of
writing.

---

## 1. The symptom chain

The rising-bubble azimuthal stability tutorial (`docs/source/tutorial/advstab/azimuthal/`) was being
timed under `mpirun -n 2` with no `--distribute`. It died in the **linear solver**:

```
petsc.py:687 solve_distributed -> MatMPIAIJSetPreallocationCSR
Argument out of range / Column too large: col 9837 max 9786
```

That looks like a PETSc problem and is not one. Working backwards:

| what was measured | result |
|---|---|
| the two ranks' global `ndof` at that solve | **9714 vs 9787** — the ranks disagreed |
| the meshes after one `adapt()` | 428 elements and 1891 nodes on both, but 24 nodes each rank did not share |
| the pinning on the 1867 shared nodes | identical — pinning was innocent |
| the solution entering `adapt()` | bitwise identical |
| the elemental errors out of the estimator | **212 of 212 differed, by 60-92%** |

So: identical solution in, wildly different error estimates out, therefore different refinement,
therefore different meshes, therefore a linear solver handed inconsistent sizes. The linear solver was
the messenger. Rank 0 refined the bottom of the bubble, rank 1 the top.

## 2. The bug that was fixed

A hanging node's raw value/position storage is a **cache of its masters**, refreshed only as a side
effect of an assembly or output pass over the elements that contain it. oomph-lib splits the assembly
**by element** across ranks (`First_el_for_assembly` / `Last_el_plus_one_for_assembly`) whenever
`nproc > 1` — distributed or not. So each rank only ever refreshed the hanging nodes inside its own
element range, and on a replicated mesh the rest kept whatever they held before, which after
adaptation is **zero**.

`sync_hanging_values_if_distributed` existed for exactly this staleness but was gated on
`is_mesh_distributed()`. The trigger is not distribution, it is element-partitioned assembly. Renamed
to `sync_hanging_values_if_parallel` and gated on `nproc > 1` (`src/problem.cpp`):

- **distributed** → `collapse_hanging_node_values()`, unchanged;
- **replicated** → `interpolate_hanging_values()`, which also restores hanging **positions** — equally
  stale on a moving mesh, and not covered by the node-only routine.

Measured effect:

```
hanging-node mismatches   120 of 1007  ->  22 (1 ulp, previously zeros)
elemental error max|diff|   2.289e-03  ->  2.596e-16
mesh after adapt #1   [(9366,428,1891),(9362,428,1891)]   -> [(9224,422,1867),(9224,422,1867)] AGREE
mesh after adapt #2   [(18450,881,3739),(12174,554,2431)] -> [(18744,899,3817),(18744,899,3817)] AGREE
bubble under plain mpirun   crashed  ->  runs; eigenvalue matches serial to 4e-12
regressions                          ->  112 passed
```

## 3. The bug that is still open

The 22 surviving 1-ulp differences are **not** an MPI problem and not floating-point summation order.

> `Mesh::interpolate_hanging_values()` returns a different answer each time it is run on the same
> input, **in a single-process serial run**.

```
serial, no mpirun, two runs of the same binary:
  positions before the call : SAME   (bit-identical input mesh)
  positions after the call  : DIFFER (~13 of 2014, by one ulp)
```

MPI never caused it; it only supplied a second observer. **Results involving hanging nodes are
therefore not reproducible run to run, today, in serial.** Usually invisible — but a 1-ulp change in
an elemental error can flip a refinement decision at a threshold, and from there the answer diverges
macroscopically. That is exactly the 60-92% divergence of §1.

### 3.1 Minimal reproducer

No MPI, no solve, no adaptation, no eigensolver:

1. take the rising-bubble problem to `initialise()`;
2. dump every nodal position;
3. call `mesh._interpolate_hanging_values()`;
4. dump again;
5. run the whole thing twice and compare the second dumps.

### 3.2 What has been ruled out, each by experiment

| candidate | how it was tested | verdict |
|---|---|---|
| stale/partial coverage | dumped every hanging entry the pass writes | all 34 differing entries **are** written |
| the assembly writing afterwards | sync alone, no assembly | sync alone already produces the difference |
| flattening not leaf-only | code: recursion ends at real dofs/pinned data | pure; masters non-hanging by construction |
| element-dependent flattening | `node_is_c1_constrained_for_*` read only node state, never `this` | element-independent |
| ordering / non-idempotence | ran the sync 1x, 2x, 3x | idempotent — fixed point after one pass |
| the dofs | bitwise compare | 0 of 4807 differ |
| the leaves | classified every differing entry | 0 non-hanging, 0 pinned differ |
| hang weights | dumped them | exact binary fractions (-0.125, 0.375, 0.75) |
| mesh geometry at setup | compared after `initialise()` | 0 of 2014 differ |
| uninitialised / invalid memory | `valgrind memcheck --track-origins` | **0 errors from 0 contexts** |
| pointer/heap-address container order | `setarch -R` (ASLR off) | still differs |
| Python iteration order → build order | `PYTHONHASHSEED=0` | still differs |

A pure function over bit-identical inputs cannot vary. One of those premises is false and it has not
been found.

### 3.3 Untested suspects

- **Threading** in the element loop, or in a library it calls (BLAS/MKL) — a nondeterministic
  reduction order would fit every observation above.
- **The JIT code cache** — whether the generated element code is byte-identical between runs, and
  whether a cache hit and a fresh compile produce the same arithmetic (FMA contraction, vectorisation).

Both are cheap to test and neither has been.

## 4. Also fixed on the way

`OomphCommunicator::broadcast(const int&, DenseMatrix<double>&)` sent the matrix dimensions with
`MPI_UNSIGNED_LONG` while the locals are `unsigned` — an eight-byte write into a four-byte object, on
every receiving rank. Genuine undefined behaviour on any MPI run that broadcasts a `DenseMatrix`; it
survived because on little-endian x86 the extra bytes are the zero half. Fixed and recorded in
`src/thirdparty/INFO_oomph-lib`. It moves the computed errors by ~5e-7 relative, and it is **not** the
cause of §3.

## 5. Wrong turns worth not repeating

Recorded because each cost a build-and-run cycle and the reasoning looked sound at the time:

- **"The linear solver mis-declares replicated matrices."** It does not; oomph hands it a genuine row
  split even without `--distribute` (71+72 on a 143-dof problem). The linear solver was reporting a
  divergence, not causing one.
- **"The per-node flux container is a pointer-ordered `std::set`."** True in
  `LagrZ2ErrorEstimator::get_element_errors`, and a real latent hazard — but changing it to an
  insertion-ordered vector left the result *bit-identical*, because patch matrices are allocated
  sequentially so set order already equalled insertion order. Reverted rather than kept.
- **"The patch broadcast is incomplete."** It is not: both ranks end with all 286 patches and identical
  per-node contribution counts.
- **"A second, post-assembly sync will make the last write uniform."** It did not converge (15 → 11 →
  17 across variants, i.e. noise). Reverted.
- **"The divergence is seeded in the refined geometry."** No — the mesh is bit-identical after
  `initialise()`.

The common thread: everything was searched for as a difference *between ranks*, and correctly found
that nothing differed, because the defect is not between ranks at all.
