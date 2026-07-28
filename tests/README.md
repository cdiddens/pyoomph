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
