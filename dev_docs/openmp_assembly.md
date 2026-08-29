# OpenMP element assembly

Opt-in threading of pyoomph's element loop: `--omp N` on the command line, or
`problem.set_num_threads(N)`. Off by default (`N = 1`), in which case none of what follows is
reachable and the serial loops run exactly as they always did.

## What this rests on

Commits `25b24ea` and `5b30fa9` retired the process-wide mutable state that made a parallel element
loop impossible: the generated `.so`'s file-scope `my_func_table`, `_currently_assembled_element`,
the single `Default_shape_info_buffer` (replaced by `ShapeBufferPool`, one buffer chain per
assembling thread), the hang-interpolation pass id (now `thread_local` behind an atomic id
generator), `__CurrentJITCode`, and the codegen ambient flags. `release_thread_shape_buffer()` was
written then for a worker that did not yet exist; it exists now.

## The design: gather, not scatter

Evaluating an element's dense block is 56-87 % of a Jacobian assembly
(`structural_assembly.md` section 2) and is embarrassingly parallel. Scattering those blocks into the
CSR value array is not: two elements sharing a dof write the same slot.

Serialising those writes with atomics would work, but it makes the summation order - and with it the
last bits of every matrix entry - depend on the thread schedule. Instead the scatter is **inverted**:

* the elements are processed in **chunks**, and the chunks run **sequentially**;
* **phase 1** computes every element's block in parallel into its own slice of a scratch buffer;
* **phase 2** gives each thread a disjoint range of **target slots** and has it sum that slot's
  contributions.

The gather index is stably sorted by slot, so each slot's contributions arrive in element order -
the order the serial loop adds them in - and the chunks are themselves in element order. The result
is **bit-identical** to the serial assembly. That is the property the tests assert (`==`, not
`allclose`), and it is what lets `--omp` be used without re-validating anything.

The price is the gather index (two ints per scatter entry, cached per equation numbering next to the
frozen sparsity) and the phase-1 scratch (bounded by the chunk length, 1 MB per matrix by default;
override with `PYOOMPH_ASSEMBLY_CHUNK_DOUBLES`, which counts doubles per matrix).

## Where it applies

| Loop | Covers |
|---|---|
| `Problem::assemble_with_frozen_sparsity` | Newton residual+Jacobian, and the eigenvalue/mass-matrix assembly (oomph's `get_eigenproblem_matrices` routes back through `sparse_assemble_row_or_column_compressed`) |
| `..._base_problem`, frozen fast path | augmented systems (fold/Hopf/pitchfork/azimuthal, arclength) and `assemble_multiassembly` |
| `assemble_distributed_with_frozen_sparsity` | hybrid MPI+OpenMP Jacobian |
| `assemble_distributed_residuals_only` | hybrid MPI+OpenMP residual |
| `Problem::get_residuals_by_elemental_assembly` | the serial residual-only sweep, which oomph-lib does in `problem.cc` |

Under MPI the workers make no MPI calls - all exchange happens outside the parallel region - so
`MPI_THREAD_FUNNELED` is all that is required of the MPI build.

### Hybrid MPI + OpenMP: `mpirun --bind-to none` or nothing

**mpirun pins each rank to one core by default, and that silently turns `--omp` into a no-op.** The
threads are created, the loop runs threaded, every number is right - they simply take turns on the
one core the rank is allowed to use. `Problem::parallel_assembly_possible()` now compares
`sched_getaffinity` against the thread count and says so once.

Assembly only (no solve), 2D Q2/Q1 cavity, 32 042 dofs, `--distribute`, on a 4-core machine.
Speed-up of the SLOWEST rank against one rank on one thread:

| ranks x threads | default binding | `--bind-to none` |
|---|---|---|
| 1 x 2 | 0.99x | 1.41x |
| 1 x 4 | 0.99x | 1.69x |
| 2 x 2 | 1.36x | 1.73x |
| 4 x 1 | 1.60x | 1.57x |

(residual-only assembly, same runs: 1 x 4 goes 0.93x -> 1.99x, 2 x 2 goes 1.57x -> 2.36x, and 4 x 1
is 3.35x either way, since pure MPI needs no threads to spread out.)

With the binding fixed, ranks and threads are close to interchangeable for the Jacobian - 4x1, 2x2
and 1x4 all land at 1.6-1.7x. For the RESIDUAL, ranks win clearly (3.4x at 4x1 against 2.0x at 1x4):
its threaded share is small, so the serial hanging pre-pass and the CSR/exchange dominate, while the
MPI split cuts all of them. So the useful rule is: **use ranks first, and threads for what ranks
cannot buy** - memory (a distributed mesh holds one share per rank) or a rank count the mesh
partitioner cannot sensibly go beyond.

## Measured

2D lid-driven cavity, Q2/Q1 Navier-Stokes, 100x100 elements, 89 402 dofs, on a 4-core machine with
8 MB of L3 (so 8 "threads" is SMT, not 8 cores). In-process timings, arms interleaved:

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| element loop (phase 1) | 143 ms | 73 ms | 38 ms | 41 ms |
| whole `assemble_jacobian` | 193 ms | 132 ms | 97 ms | 98 ms |

The element loop itself scales 3.7x on 4 cores. The whole call gains 2.0x, and the two numbers that
separate them are worth knowing:

* ~50 ms of the call is the CSR hand-out and the frozen-fill statistics, which are serial and
  untouched by any of this;
* ~16 ms is the serial hanging pre-pass. It is skipped entirely on a mesh with neither hanging nodes
  nor dummy values (`ParallelAssemblyPlan::needs_hang_prepass`); this cavity has C1 pressure on
  C2-only nodes, so it does not qualify. This is the largest remaining serial fraction of a threaded
  assembly and the obvious next thing to attack.

### Load balancing across unequal elements

Nothing about phase 1 is split by element COUNT. The elements of a chunk are handed out by
`schedule(dynamic, gran)`, i.e. by work-stealing, so a mesh whose elements cost wildly different
amounts - two domains with different physics, a cheap scalar field next to a 3D Navier-Stokes block,
bulk elements next to interface elements - balances itself without anyone having to estimate a cost
per element. Chunk boundaries do not respect domain boundaries either, and do not need to.

What *was* fixed by count, and did break, is the dispatch **granularity**. It used to be a constant
8. The chunk length comes from a budget on `nvar*nvar`, so a domain with large elements gets a SHORT
chunk - 3D Q2/Q1 Navier-Stokes has `nvar = 3*27 + 8 = 89`, which admits a few dozen elements per
chunk - and a grain of 8 then handed the whole chunk to one or two threads while the rest idled. The
grain is now derived per chunk, aiming for about eight dispatches per thread:

    gran = clamp(chunk_length / (8 * nthreads), 1, 8)

Measured, 3D Q2/Q1 cavity, 8x8x8, 10 845 dofs, `assemble_jacobian` at 4 threads: **90 ms with the
fixed grain of 8, 65 ms with the derived one**. The homogeneous 2D case is unaffected (its chunks are
long enough that the formula returns 8 anyway) and still runs at 2.0x.

With that in place the measured imbalance - busiest thread against the mean of their phase-1 BUSY
times, which `PYOOMPH_REPORT_OMP_ASSEMBLY` prints - is 1.01-1.05 on all three shapes tested: the
homogeneous 2D cavity, the large-element 3D block, and two domains sharing a problem where one is a
scalar Poisson and the other Navier-Stokes. Note that the imbalance figure excludes the wait at the
barrier on purpose; without that, a thread that finished early and a thread that never got work look
identical.

The one thing still split by count is phase 2, and there it is the right measure: every gather entry
is one multiply-free add, so equal counts really are equal work.

### The chunk length

Phase 2 is 4-7 ms, i.e. ~6 %, and it is the one part that cares where the chunk sits relative to the
cache: it reads the block buffer in TARGET-SLOT order, essentially at random within the chunk, where
the serial loop reads each block sequentially out of L1.

The default is 1 MB of doubles per matrix, and it was swept rather than reasoned - from 8 K to 4 M
doubles at four threads on an 8 MB L3, on both an expensive element kernel (the cavity above) and a
cheap one (Q2 Poisson, 250 k dofs):

| doubles/matrix | 8 K | 16 K | 64 K | **128 K** | 512 K | 1 M | 4 M |
|---|---|---|---|---|---|---|---|
| cavity, `assemble_jacobian` | 147 ms | 115 | 116 | **115** | 111 | 109 | 111 |
| Poisson, `assemble_jacobian` | - | 252 | 238 | **236** | 243 | 240 | 240 |

Everything from 16 K up is within ~5 %, and the two ends fail for different reasons - only one of
which is a cliff:

* **Small.** The chunk holds a few dozen elements and its two barriers stop being amortised. 8 K is
  28 % worse. This is what `min_chunk` in the plan builder guards against.
* **Large.** Phase 2 roughly doubles once the chunk leaves the cache (3.8 ms at 32 K against 6.8 ms
  at 1 M on the cavity), but at ~6 % of the loop that barely moves the total, and phase 1 gets a
  little back from having fewer barriers. There is no cliff here.

So the value is not critical and there is deliberately no cache-size detection. Two earlier
measurements that suggested otherwise - a 32 MB chunk reading 1.6x SLOWER than serial - were
confounded by a plan rebuild firing on every call, which is why the plan is now deliberately
independent of the thread count. `PYOOMPH_REPORT_OMP_ASSEMBLY` prints the three parts separately;
use it before believing any chunk-size conclusion, including this one.

Serial no-regression, same problem at 57 122 dofs, measured across separate processes before and
after the change: element loop 0.98x, `assemble_jacobian` 1.01x, residual 1.01x - all inside the
+-5 % spread between repeated runs of the *same* build, so there is no measurable serial cost.

## The races that had to be closed first

1. **`functable->current_res_jac` is written per element during assembly.** The pitchfork and
   azimuthal handlers switch a code's residual form in the middle of their own `get_jacobian()`, on a
   table shared by every element of that code. The field cannot become `thread_local` (it lives in a
   C struct that generated code includes, and `jitbridge.h` is part of the JIT cache key), so the
   writes are diverted instead: `get_current_res_jac()` / `set_current_res_jac()` in
   `thread_state.hpp`, backed by a per-thread map that is only consulted inside a parallel region.
   Off that path both are the plain field access they replace.
2. **`interpolate_hang_values()` writes into shared `oomph::Node` storage.** Handled by a serial
   pre-pass that does the interpolation once, stamped with a fresh pass id which the workers then
   adopt - so `HangInterpGate::skip()` matches for every element and no worker writes a node at all.
   That is work removed, not merely a lock avoided.
3. **Python callbacks hold no GIL in a worker.** `pyoomph_core` links neither Python nor nanobind, so
   the nanobind layer installs a pair of GIL hooks (`install_gil_hooks`, the same pattern as
   `functable->invoke_callback`). The assembly hands the GIL over around the parallel region and the
   `CustomMathExpression` / `CustomMultiReturnExpression` trampolines take it in `_call`. Callbacks
   therefore serialise - a callback-heavy problem gains little - but the answer is right.

## When it declines

It reports the reason once and runs the serial loop:

* a `fd_jacobian` or `fd_position_jacobian` domain - the FD loops perturb nodal data shared with
  every neighbouring element, which no scheme can parallelise;
* any assembly-time diagnostic env var (`PYOOMPH_REPORT_EXT_DATA`, `PYOOMPH_REPORT_HANG_FILL_CACHE`,
  `PYOOMPH_FROZEN_FILL_BREAKDOWN`, ...) - their counters are process-wide, and when they are on their
  whole point is to be believed;
* no frozen sparsity (`keep_structural_zeros` off, a value-dependent pattern, compressed-column
  output): the map-based fallbacks are not threaded;
* a build whose CMake configuration found no OpenMP.

`problem._get_parallel_assemblies_done()` counts the loops that actually ran threaded. Check it
before believing any measurement: a fast path that silently fell back looks exactly like a working
one, because both give the right answer.

## Interaction with the linear solver and the parameter scanner

`set_num_threads(N)` sets both halves - the element loop and the solver (Pardiso via
`mkl_set_num_threads`). Note the asymmetry: the threaded ASSEMBLY is bit-identical, but a threaded
direct SOLVER is not bit-reproducible, so a converged state can move in its last digits. That was
already true of `set_num_threads` before any of this, and `problem._set_num_assembly_threads(n)`
sets the assembly count alone when the two effects have to be told apart (which is what the tests
do). `--omp N` is additionally read straight from `sys.argv` in
`pyoomph/__init__.py`, before anything imports MKL or PETSc, because those latch their thread counts
from the environment at import time. Without `--omp`, `OMP_NUM_THREADS` defaults to 1 so that a
third-party OpenMP runtime cannot open a second pool of threads next to ours.

`ParallelParameterScan` runs `max_procs` simulations at once, so its children stay single-threaded:
`single_threaded_childs` (the default) pins the solver's threading through the environment and
pyoomph's own through an appended `--omp 1`.

## Packaging: which wheels actually have it

The build option is `PYOOMPH_USE_OPENMP` (`AUTO`/`ON`/`OFF`, default `AUTO`, see
`docs/source/tutorial/installation/cmakeoptions.rst`). `AUTO` links OpenMP if the toolchain has it -
which is why the capability of a wheel would otherwise be decided, silently, by whatever the build
machine happened to have. `pyoomph._pyoomph_core.has_openmp` says which it was; the wheel jobs pass
`ON` so that a runner which loses OpenMP fails the build instead of shipping a serial wheel, and
`tests/test_openmp_assembly.py` skips on it rather than failing everywhere it is absent.

* **Linux.** GCC always brings `libgomp`, and auditwheel vendors it into the wheel. A second
  `libgomp` next to numpy's or torch's is merely wasteful, not fatal.
* **Windows (MinGW/UCRT64).** `-fopenmp` pulls in `libgomp-1.dll`, which needs `libwinpthread-1.dll`
  and `libgcc_s_seh-1.dll` of its own - `-static-libgcc -static-libstdc++` covers the extension, not
  libgomp. delvewheel bundles all three, so the wheel stays self-contained. Exceptions must keep
  being caught before they leave the parallel region (they are): the extension's static libgcc and
  libgomp's dynamic one are two unwinders in one process.
* **macOS.** AppleClang has no OpenMP at all and Homebrew's `libomp` cannot be used: its bottle
  requires the runner's macOS release, and delocate refuses to bundle that into a 10.13 wheel.
  `citools/build_static_libomp.sh` builds LLVM's runtime from source at the wheel's own deployment
  target and links it statically, so nothing lands in `.dylibs`.
  `.github/workflows/test_mac_openmp_wheel.yml` builds one wheel and checks all of it: nothing
  bundled, `kmpc_fork_call` present in the extension, `minos` still 11.0, `has_openmp` true, and the
  bit-identity suite green.

  Static linking does not, in principle, avoid libomp's per-process registration - a second copy in
  the process is what produces `OMP: Error #15` - so a `KMP_DUPLICATE_LIB_OK` default was tried and
  then removed. Measured on a macos-14 runner against scikit-learn (which does bring its own
  runtime): the threaded assembly survives in both import orders, and the variable is already set to
  `True` before pyoomph is imported - `threadpoolctl`, which scikit-learn depends on, sets it at
  import time for exactly this reason. The package that brings the second runtime brings the setting
  with it. One earlier run with PyTorch 2.13 DID
  abort in the first threaded assembly, with no `OMP:` diagnostic of any kind - unexplained, not
  reproduced with scikit-learn, and no reason on its own to set a process-global variable that every
  other OpenMP library in the process would see.

Rejected: replacing OpenMP with a `std::thread` pool, which would remove the packaging problem on
all three platforms. The OpenMP surface here is small enough to port (one parallel region, one
dynamic `for`, two barriers), but the barrier is not incidental - the chunk sweep above shows a 28 %
loss once the two barriers per chunk stop being amortised, measured against libgomp's tuned
spin-then-sleep. Hand-rolling that, plus a pool whose lifetime survives fork/MPI/several `Problem`s,
buys one platform that `build_static_libomp.sh` already buys without touching the hot path.
