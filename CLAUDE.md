# Working on pyoomph

pyoomph is a multi-physics finite-element framework: a C++ core (`src/`, exposed to Python through
nanobind in `src/nanobind/`) that symbolically differentiates weak forms with GiNaC and JIT-compiles
the resulting residual/Jacobian code, on top of a vendored copy of oomph-lib.

## Git and testing: ask, do not act

**Committing, pushing, and running large pytest suites are only done when the user asks for them.**
Reminding the user that something is worth committing or testing is welcome; doing it unprompted is
not, because these runs take far too long to spend without being asked.

That applies to the full `tests/` suite, the MPI suites, and the tutorial harness
(`citools/test_all_tutorial_scripts.py`, 127 scripts). A full tutorial pass takes over an hour; two
passes for an A/B comparison take most of an afternoon. Batch changes up and run once, rather than
after each fix. Targeted single scripts and small benchmarks are fine without asking.

The main branch should be always quite stable. Development happens on the branch "develop", which is merged into main regularly.
Urgent direct fixes can be implemented directly in "main" and have to be included into "develop".

## Building

After **any** C++ change (`src/`), run

    ./build_for_develop.sh

not `ninja` on its own. It does an editable `pip install`, which is what makes Python import the new
extension; building without installing leaves the stale module in place and you will debug a binary
that does not contain your change.

The `.pyi` stub (`pyoomph/_pyoomph_core.pyi`) is generated from the nanobind docstrings by
`nanobind.stubgen` during the build and mirrored into the source tree. To fix what the stub says, fix
the docstring in `src/nanobind/` and rebuild - do not edit the stub.

Never rebuild while a long benchmark or test pass is running: the install swaps the module underneath
it and the results become a mixture of two builds.

## Running things

Run scratch scripts, benchmarks and diagnostics from the session scratchpad, **never** from the
user's own folders under `pyoomph_runs/` - every run creates an output directory next to the script.

You can work in the Scratchpad subfolder!

Fast tests of existing tutorials can be achieved by adding --quick-test to the command line. It will stop after the first Newton solve and provides a single output.

Give each run its own fresh working directory. Many tutorials write dump/restart files and take a
shortcut when they find them, so a second run in the same directory measures something entirely
different from the first.

MPI should be run with maximum 8 cores (and then smaller problems, less than 40000 dofs)
Maximum number of dofs should be 200000.

### Eigenvalue and stability problems need a complex PETSc

`PYTHONPATH` is unset in a non-login shell, so petsc4py is not importable at all there and pyoomph
falls back to the scipy eigensolver without saying much about it - a tutorial that should exercise
SLEPc then quietly tests something else. Anything using eigensolvers, azimuthal stability or Floquet
analysis needs the complex build.

**The path differs per machine, so look it up rather than pasting one from here** - and check it, because
a nonexistent entry in `PYTHONPATH` is not an error, it just leaves you without PETSc.
On the development computers, `PETSC_DIR`, `PETSC_ARCH_REAL` and `PETSC_ARCH_COMPLEX` are defined.
Include `$PETSC_DIR/$PETSC_ARCH_COMPLEX/lib` into `PYTHONPATH` to get complex PETSc and test it with

    find ~/code -maxdepth 5 -name petsc4py -type d      # the arch dir is its parent's parent
    PYTHONPATH=<candidate> python3 -c "from petsc4py import PETSc; import numpy; \
        assert PETSc.ScalarType is numpy.complex128; import slepc4py; print('complex PETSc ok')"

Likewise for real PETSc.


The MPI and 3D-adaptivity suites are marked `slow` and are **skipped** without `--full`:

    python3 -m pytest tests/test_mpi_eigenvalues.py -q --full

Do not force a linear solver on the tutorials to make a comparison uniform. It changes what is being
computed: `petsc_mumps` collapses `hopf_switch`'s arclength continuation, and plain `petsc` (iterative
KSP) fails outright on the augmented systems. Leave each script's own choice alone and say so.

## Benchmarking

Measure **in-process** (time the assembly or solve calls), not the wall time of the whole script. For
short scripts most of the wall time is interpreter and PETSc import, so the flag under test cannot
move it more than a few percent and the measurement is pure noise.

Machine load corrupts wall-clock A/B comparisons badly - a 6x inflation and a sign flip have both
been observed on this machine. Interleave the two arms (A, B, A, B) rather than running all of A then
all of B, and do not run anything else, including subagents, while measuring.

Before believing any frozen-sparsity comparison, check that the path actually engaged
(`_get_frozen_sparsity_nnz() > 0`). A comparison where the fast path silently fell back shows two
identical numbers and looks like a perfect result.

## Conventions

- **Vendored oomph-lib.** Changes inside `src/thirdparty/oomph-lib/` are marked with a `//FOR PYOOMPH`
  comment *and* described in `src/thirdparty/INFO_oomph-lib`. Both, every time.
- **Comments explain why.** The codebase documents reasoning and rejected alternatives, not
  restatements of the code. Match that: when a piece of code exists because of a specific failure,
  say what the failure was. However, keep it brief.
- **Correct the record.** Several long-standing comments in this codebase turned out to describe
  behaviour that had since changed (a default flipped, a bug fixed, a code path `#define`d out). If
  you find one, fix it in the same commit rather than working around it.
