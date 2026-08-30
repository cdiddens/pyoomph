# pyoomph — parallelization for AI coding assistants

Companion to [`AGENTS.md`](../AGENTS.md). How to run a pyoomph script on more than one
core, and — more importantly — what a *script* has to do differently so that it is
still correct when it does. The distilled version of `docs/source/tutorial/parallel/`.

**The headline: almost nothing changes in the script.** Parallelization is a launch-time
decision, not a modelling one. The same `Problem` subclass, the same `Equations`, the
same weak forms run serially, threaded, and under `mpirun`. Do not build a "parallel
version" of a script — build the script, then launch it differently. The rules below are
the handful of exceptions.

## The three modes

| Launch | Assembly | Linear solve | Mesh storage |
|---|---|---|---|
| `python3 script.py` | serial (solver may still thread internally) | serial/threaded | one process holds everything |
| `python3 script.py --omp N` | threaded over elements | threaded | one process holds everything |
| `mpirun -n P python3 script.py` | split over ranks | distributed (rows) | **every** rank holds the whole mesh |
| `mpirun -n P python3 script.py --distribute` | split over ranks | distributed | partitioned (METIS) + halos |

They compose: `mpirun -n 2 python3 script.py --omp 2` is two ranks of two threads.

Recommendation for a user who just wants it faster: **`--omp N` first** — it is one flag,
needs no extra install, and is bit-identical to the serial run. Reach for MPI when the
problem is too big for one machine, or on a cluster.

## OpenMP: `--omp N`

```bash
python3 my_simulation.py --omp 4
```
or in the script, `problem.set_num_threads(4)`. Both also set the linear solver's thread
count.

- **Bit-identical to the serial assembly.** Deliberate, not luck: pyoomph inverts the
  scatter into a gather so every matrix entry receives its contributions in element
  order. `--omp` can be toggled without re-validating any result. (A threaded direct
  *solver* is a separate matter and is not bit-reproducible.)
- Needs a build with OpenMP (`PYOOMPH_USE_OPENMP`, default `AUTO`). Check with
  `from pyoomph import _pyoomph_core; print(_pyoomph_core.has_openmp)`.
- Pyoomph sometimes **declines** to thread and says so once, then runs serially — most
  often for a domain using a finite-difference Jacobian, or a build without OpenMP. The
  answer is identical either way, so before believing a speed measurement check
  `problem._get_parallel_assemblies_done()`.
- A `CustomMathExpression` callback evaluated during assembly takes the Python GIL in
  turn, so a callback-heavy problem gains little.
- Expect ~3.7x on the element loop and ~2.0x on the whole Jacobian assembly at 4 threads
  (the CSR structure handout and hanging-node pre-pass are inherently serial).

## MPI: `mpirun`

```bash
mpirun -n 4 python3 my_simulation.py               # replicated mesh
mpirun -n 4 python3 my_simulation.py --distribute  # partitioned mesh
```

**MPI is a compile-time option and is OFF in the PyPI wheels.** A `pip install pyoomph`
cannot run in parallel. Building with it:

```bash
python -m pip install --no-build-isolation -e . --config-settings=cmake.define.PYOOMPH_USE_MPI=ON
```

Check an installation with `python -m pyoomph check mpi`, or in a script with
`from pyoomph.generic.mpi import has_mpi; has_mpi()`. `mpi4py` must be importable;
`--distribute` additionally needs `pymetis` (`pip install pymetis`). The rank helpers are
`get_mpi_rank()` and `get_mpi_nproc()` from the same module.

The number of processes is fixed by the launcher — a script can neither choose nor change
it.

**Without `--distribute`** every rank holds the whole mesh and the whole unknown vector;
only the element loop and the linear solve are parallel. It saves time, not memory.
**With `--distribute`** METIS partitions the mesh and each rank keeps its own part plus
halos. The initial mesh is still built on every rank, partitioned on rank 0, then pruned.

### The one rule that changes how you write the script

> **Every rank must agree on the state of the simulation.**

Anything non-deterministic — an unseeded random number, an iteration order that depends
on object identity, a decision taken from a locally-computed float that differs in its
last bits — makes the ranks diverge. The usual symptom is not a wrong answer but a
**deadlock**, with all ranks stuck in different collective calls.

- Need a random field? Use `DeterministicRandomField` (`pyoomph.expressions.utils`),
  which broadcasts its point cloud from rank 0. Pass `seed=` to make it reproducible
  across runs too.
- Need a Python-side value computed once? Compute it on rank 0 and broadcast:
  `value = get_mpi_bcast(compute() if get_mpi_rank()==0 else None)`.
- Never guard `add_mesh`/`add_equations` by rank: problem setup is collective and every
  rank must build the same problem.
- Do not iterate over a `set` of objects when the order affects the result — `id()`-based
  ordering differs per rank, and a fixed `PYTHONHASHSEED` does *not* screen for it.

### Linear solvers under MPI

With more than one process pyoomph defaults to `--petsc_mumps` when available. The serial
backends (`pardiso`, `superlu`, `umfpack`, Apple `accelerate`) are not refused: the system
is gathered onto rank 0, solved there, and scattered back, with a warning. That is still a
win for an assembly-dominated problem, but rank 0 needs the whole matrix.

- `--petsc_mumps` — distributed sparse LU. Robust default; memory-limited like any direct
  solver, especially in 3D.
- `--petsc` — PETSc Krylov solvers. What eventually scales, but needs a preconditioner
  suited to the equations; a poor choice simply will not converge.

Both require PETSc built with MUMPS (`--download-mumps=yes`). Settable in the script via
`problem.set_linear_solver(...)`.

Do **not** force a uniform solver across scripts to make a comparison tidy — it changes
what is computed. `petsc_mumps` collapses arclength continuation in some setups, and plain
iterative `--petsc` fails outright on augmented (bifurcation-tracking) systems.

### Console output

`--mpi-output` selects `condensed` (default, rank 0 only), `all` (every rank, tagged) or
`off` (unfiltered).

## Pitfalls

- **`mpirun` pins each rank to one core by default**, which makes `--omp N` a no-op that
  still looks like it works — the threads exist, the loop is threaded, the numbers are
  right, they just take turns on one core. Launch with `mpirun --bind-to none`, or a
  binding giving each rank as many cores as it has threads. Pyoomph prints a one-off note
  when it notices.
- Keep ranks × threads at or below the physical core count.
- For a fixed core budget, ranks beat threads for the *residual* (its threaded share is
  smaller); they are roughly interchangeable for the *Jacobian*. Take ranks first, and add
  `--omp` for what more ranks cannot give: memory headroom, or a core count beyond what
  the partitioner divides sensibly.
- Eigenvalue/stability work under MPI needs a **complex** PETSc/SLEPc build on
  `PYTHONPATH`; without it pyoomph quietly falls back to the scipy eigensolver.

## See also

- `docs/source/tutorial/parallel/` — the full human chapter (`openmp.rst`, `mpi.rst`).
- [`advanced.md`](advanced.md) — DG facet unknowns under `--distribute`.
