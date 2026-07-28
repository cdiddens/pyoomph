# Test cases

These are the test cases for pyoomph. Run them **in this directory** with

> python -m pytest *.py

You must have installed `pytest` via

> python -m pip install pytest

## MPI tests

`test_mpi_adaptivity.py` checks distributed (MPI) adaptive refinement. pytest itself stays serial: each
test launches `mpi_worker.py` under `mpirun -n N ... --distribute` in a subprocess and compares the
per-rank results against a serial reference computed in-process from the same definitions
(`mpi_cases.py`). The two helper modules are not test modules and contain no tests of their own.

These tests **skip** rather than fail if `mpirun`, MPI support, or PETSc with MUMPS is unavailable, so a
serial-only installation still runs the full suite. Do not run pytest itself under `mpirun`.
