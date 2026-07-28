# Test cases

These are the test cases for pyoomph. Run them **in this directory** with

> python -m pytest *.py

You must have installed `pytest` via

> python -m pip install pytest

## Fast run vs. full run

The command above is the **fast** run (~6 min): it skips the tests marked `slow`, which are the ones that
sweep a large matrix or launch `mpirun` — the 3D adaptive-mesh campaign and both MPI modules. It still
covers everything else end to end, including the 2D adaptive-mesh campaign.

Before merging a branch, run **everything** (~11 min):

> python -m pytest *.py --full

or, where passing a flag is awkward (CI), set `PYOOMPH_FULL_TESTS=1`. Nothing is permanently excluded —
`--full` runs the entire suite, and the skipped tests are reported as skipped rather than silently dropped.

### What the wheel builds run

A second, independent axis: the `campaign` marker covers the four mixed-adaptive-mesh validation modules
(`test_adaptive_2d_campaign.py`, `test_adaptive_3d_campaign.py`, `test_mpi_adaptivity.py`,
`test_mpi_adaptivity_3d.py`). The wheel-building workflow deselects them:

> python -m pytest tests -m "not campaign"     # 177 tests, ~4 min

set as `test-command` in `pyproject.toml`. cibuildwheel repeats that command once per Python version per
platform, so a suite that takes minutes locally costs a multiple of that there. What CI keeps is the test
set that predates the campaign, which is what a wheel actually needs to certify: that the built extension
imports, compiles elements and solves. Proving the refinement engine correct across every discretisation
is a branch-merge job, not a per-wheel one.

The two axes are orthogonal. `slow` is about what *you* wait for while working; `campaign` is about what
CI pays per wheel. A default local run still executes the 2D campaign, and `--full` still runs everything:

| invocation | tests | who |
|---|---|---|
| `pytest *.py` | 581 collected, `slow` skipped | working locally |
| `pytest *.py --full` | 581 | before merging a branch |
| `pytest tests -m "not campaign"` | 177 | the wheel builds |

## The adaptive-mesh campaign

`test_adaptive_2d_campaign.py` and `test_adaptive_3d_campaign.py` exercise the physics that adaptive
(hanging-node) meshes have to support — mixed C1/C2 spaces (Taylor-Hood Stokes), the C1-space field
constraints, Neumann fluxes, and ALE moving meshes — on quad / triangle / mixed meshes in 2D and on every
geometrically possible combination of bricks, tetrahedra, wedges and pyramids in 3D.

The problem definitions live in `box_cases.py` (2D) and `box_cases_3d.py` (3D), with `box_mesh_3d.py`
building the 3D domains; these three are helper modules, not test modules. Sharing them with the MPI
harness is what keeps the serial and distributed campaigns from drifting apart.

A number of 3D configurations are marked `xfail(strict=True)` with a reason: they are known defects, not
gaps in coverage. See `dev_docs/mixed_adapt_validation.md` §9. Being strict, they will fail the suite as
soon as they start passing, which is the signal to remove the marker.

## MPI tests

`test_mpi_adaptivity.py` (2D) and `test_mpi_adaptivity_3d.py` (3D) check distributed adaptive refinement.
pytest itself stays serial: each test launches `mpi_worker.py` under `mpirun -n N ... --distribute` in a
subprocess and compares the per-rank results against a serial reference computed in-process from the same
case module. Only partition-independent quantities are compared (the gathered residual, the global dof
count, and MPI-reduced integral observables), and they are checked both against serial and across ranks.

These tests **skip** rather than fail if `mpirun`, MPI support, or PETSc with MUMPS is unavailable, so a
serial-only installation still runs the full suite. Do not run pytest itself under `mpirun`.

### Debugging a distributed adaptive run

Set `PYOOMPH_CHECK_HALO_CONSISTENCY` to have every adapt cross-check that the processes still agree about
the elements they share — their positions, refinement levels, pending refinement flags, and the error
estimates about to decide their fate:

| value | effect |
|---|---|
| unset, `0`, `off` | off (the default; no cost) |
| `1`, `warn`, `report` | report mismatches to stdout and carry on |
| `2`, `throw` | raise on the first mismatch |

```bash
PYOOMPH_CHECK_HALO_CONSISTENCY=throw mpirun -n 2 python my_script.py --distribute
```

Reach for this whenever a distributed run disagrees with the serial one, produces a partition-dependent
answer, or diverges where serial converges. Divergent meshes are silent at the point they happen and only
surface much later as a wrong `ndof`, an `inf` residual or a deadlock — this names the offending elements
by position at the adapt that created them. The verdict is agreed across processes, so throwing mode fails
the whole job rather than one rank while the others block.

`Mesh.check_halo_consistency()` runs the same check on demand from Python; it is collective, so every rank
must call it. `test_halo_consistency_check_stays_clean[_3d]` keeps the campaign clean under `throw`.
