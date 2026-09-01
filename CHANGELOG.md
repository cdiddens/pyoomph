# Changelog

## [0.2.1]

Released as urgent patch of 0.2.0. 

### Fixed

- **A two-sided interface could exhaust the machine's memory during the first assembly.** The pruned sparsity pattern (`prune_structural_zeros_by_field_coupling`, on by
  default since 0.2.0) let the two sides of a coupled interface ask each other for an answer each was
  still computing, which recursed without bound. 

## [0.2.0] - 2026-08-30

About six weeks and 940+ commits since 0.1.9. Four themes: h-adaptivity generalised from quads and
bricks to every element shape, MPI support extended to nearly everything that was serial-only,
substantially faster assembly and code generation, and several new physics modules. The compiled core
also moved from pybind11 to nanobind.

### Added

- **Adaptive refinement for every element shape**: triangles, tetrahedra, wedges, pyramids and mixed
  forests of them (2d and 3d), with hanging nodes, 2:1 balancing, unrefinement and mixed-order
  (Taylor-Hood, Crouzeix-Raviart, MINI) spaces. Shared nodes are identified topologically, never by
  position. `dev_docs/adaptive_refinement.md`.
- **Curved boundaries for every element type**, through one shape-generic `GenericMacroElement`, in 2d
  and 3d, from templates and from gmsh, including under `--distribute`. `dev_docs/macro_elements.md`.
- **Adaptivity across coupled domain interfaces**: two domains sharing an interface are kept
  conforming automatically, serially and distributed. `dev_docs/interface_refinement_coupling.md`.
- **Precomputed CSR sparsity, value-only re-assembly and solver symbolic reuse** (on by default),
  covering the mass matrix, the multi-assembly, every bifurcation tracker and the distributed exchange
  plan. `dev_docs/structural_assembly.md`.
- **Threaded element assembly, `--omp N`** (off by default), bit-identical to the serial loop, and
  combinable with MPI. macOS uses a GCD backend. `dev_docs/openmp_assembly.md`.
- **Static condensation of element-local dofs** (`StaticCondensation`, `Problem.condense_dofs`);
  51-65 % faster factorisation of a Crouzeix-Raviart system. Serial and distributed, experimental.
  `dev_docs/static_condensation.md`.
- **Unknowns on the interior-facet skeleton** (`at_internal_facets=True`), surviving adaptation,
  remeshing, `--distribute` and state files; HDG example in the tutorial.
  `dev_docs/internal_facet_fields.md`.
- **Eigenvalue problems under MPI through SLEPc**, with gather-to-root for the serial solvers and
  scipy/ARPACK. `dev_docs/mpi_eigenproblems.md`.
- **Remeshing, state files and periodic boundary conditions under `--distribute`.**
- **Periodic orbits, Floquet multipliers, bifurcation tracking, branch switching and deflation under
  MPI**, replicated and distributed.
- **Floquet multipliers by structured condensation** (the default), returning exactly `ndof`
  multipliers with no shift or magnitude threshold, plus a shift-inverted matrix-free route for large
  problems. `dev_docs/floquet_multipliers.md`.
- **An interactive bifurcation GUI**, with parameter switching, two-parameter loci, field plots,
  deflation and periodic orbits. `dev_docs/bifurcation_loci.md`.
- **Axisymmetric pinch-off and coalescence**: `AxisymmetricReconnection(rmin=..., distmin=...)` lets a
  free surface change its topology, nearly volume conserving, under MPI too. Replaces
  `AxisymmetricPinchoffAndCoalescence`. `dev_docs/axisymmetric_topological_changes.md`.
- **Viscoelastic flow in the log-conformation representation** (Oldroyd-B, Giesekus, PTT, FENE-CR,
  FENE-P). `dev_docs/viscoelastic_log_conformation.md`.
- **Residual-based stabilization** of Navier-Stokes and of scalar transport, replacing
  `pyoomph.equations.SUPG`. `dev_docs/stabilized_navier_stokes.md`.
- **Electrostatics, electrolytes and electrohydrodynamics**: Poisson-Boltzmann, Ohmic conduction,
  Poisson-Nernst-Planck, surface charge, Maxwell stress, electroosmotic slip.
  `dev_docs/electrohydrodynamics.md`.
- **Dissolved salts**: an ion and salt library, salt transport with the ambipolar diffusivity, salt
  retention under evaporation, salt-induced Marangoni flow, and AIOMFAC electrolyte activity
  coefficients. `dev_docs/salt_transport.md`, `dev_docs/aiomfac_electrolytes.md`.
- **Liquids may be given by concentration** in `Mixture(...)`, e.g.
  `water + 1*milli*molar*get_pure_liquid("surfactant")`.
- **Surfactant transport in one class**, conservative by default. `dev_docs/surfactant_transport.md`.
- **Tracer particles, rewritten**: bulk and interface, moving meshes, trails, payloads, remeshing and
  `--distribute`. `dev_docs/tracers.md`.
- **A mesh point locator** replacing `MeshAsGeomObject`, with closest-point-projection transfer for
  interfaces without a usable zeta and periodic zeta on closed loops. `dev_docs/mesh_point_locator.md`.
- **A content-addressed JIT cache** over deterministic code generation.
- **A Spectra eigensolver**, so targeting an eigenvalue no longer needs PETSc/SLEPc.
- **TQMesh as a second 2d meshing backend**, and OCC geometry support in `GmshTemplate`.
- **Second-order spatial derivatives**, with complete Hessians.
- **Vector fields for `DirichletBC` and `InitialCondition` as a whole**, e.g.
  `DirichletBC(velocity=vector(1,0))`.
- **`RemeshWhen(on_inverted_element=True)`**: remeshing as the response to a folded mesh, where a
  smaller time step cannot help.
- **Spatial error estimators on interfaces**, per-criterion normalisation, and adaptation towards a
  dof budget. `dev_docs/spatial_error_estimators.md`.
- **Per-block Jacobian symmetry and constancy are proven** and used to select the symmetric solver and
  eigensolver paths. `dev_docs/jacobian_block_flags.md`.
- New tutorial chapters: parallelization (OpenMP and MPI), spatial adaptivity, viscoelastic flow, the
  electric double layer, salts and ions, tracers, facet fields and HDG, static condensation, branch
  switching, the bifurcation GUI, coordinate systems, and the agent guide (`AGENTS.md`).

### Changed / Improved

- **The core extension is built with nanobind**, `src/pybind/` is now `src/nanobind/`, and CPython
  3.12+ gets a single Stable-ABI wheel. Python 3.9 is no longer supported.
- **The reference cycles that kept every `Problem` and `Mesh` alive are gone**, along with a family of
  leaks the migration exposed.
- **The whole Python package type-checks clean** under pyright and mypy, which fixed a number of
  genuine defects along the way.
- **Elemental assembly and generated code got faster**: non-hanging elements are dispatched to a
  hang-free path, unused shape families and hanging bookkeeping are skipped, loop-invariant buffer
  reads are bound to locals (11-15 % of an elemental Jacobian), Jacobian and Hessian entries are
  hoisted, `subexpression()` now reaches the analytic Hessian, and `-fno-math-errno` is a default
  flag. `dev_docs/assembly_overhead.md`.
- **An adaptation that refines and unrefines nothing no longer touches the problem**, so the equation
  numbering and the sparsity pattern survive it.
- **`Problem.initialise()` is ~27 % cheaper per dof.**
- **Tensor index conventions are now self-consistent, and two of them changed**: `contract(A,b)` is
  `A_ij*b_j` (was `A_ji*b_j`), and `div(T)[i]` is `d_j T_ij` (was `d_j T_ji`), which makes `div` the
  adjoint of `grad`. Symmetric tensors are unaffected.
- **The coordinate-system keyword is `coordsys` everywhere**; the five old spellings warn.
- **The normal-mode coordinate systems** gained tensor divergence and directional derivative, so
  `GCL=True` and the viscoelastic module can be combined with normal-mode stability analysis.
- **Framework-only methods on the equation and mesh-template classes are underscore-prefixed**, and a
  number of dead methods and write-only attributes were removed.
- **Arclength continuation got a mesh-independent inner product**, and `quick_mode` continues without
  an eigensolve per step.
- **Anything the MPI ranks must agree on is independent of the hash seed and of heap addresses.**
- **Removed**: the PFEM prototype, the standalone MUMPS and PaStiX solvers, `pyoomph.equations.SUPG`,
  the `wrong_strain` option, and the `MeshAsGeomObject` backend.

### Fixed

- **The 3x3 Gauss-Legendre knot table for 2d quads had two transposed digits**, making the rule
  asymmetric: results on quad meshes were wrong by ~1e-9 however fine the mesh.
- **Refining across a quad/triangle interface could tear the mesh** (and segfault), because the quad
  neighbour lookup could match a triangle node by accident.
- **Distributed adaptivity could refine different elements on different ranks**, silently corrupting
  the mesh: pyoomph's per-element error overrides ran rank-locally. Errors are now synchronised
  owner-to-halo, the 3d 2:1 balancing pass is globally consistent, and there is an opt-in halo
  consistency check.
- **`ConstrainPositionsToC1Space`, `ConstrainFieldsToC1Space` and mixed-order spaces** aborted or
  produced an inconsistent Jacobian on non-uniformly refined 3d and simplex meshes.
- **A `RemeshWhen` firing inside a solve corrupted the dof vector** under temporal adaptivity; the
  remesh is now deferred until the C++ call returns.
- **Inverted-element detection hung under MPI**; the verdict is now reduced across ranks.
- **`activate_bifurcation_tracking(blocksolve=True)` segfaulted** rather than being refused.
- **MPI dof and residual accessors read a distributed vector by global equation number**, returning
  garbage or overrunning the heap; `create_pressure_fixation()` pinned a different dof on every rank.
- **Several bifurcation defects**: the first Lyapunov coefficient's diagnostics and guards, four in
  `NormalModeBifurcationTracker`, the sign of the azimuthal tracker's `M_imag` term, and the tracked
  eigenvalue, which flipped the reported stability at every bifurcation.
- **Pardiso could return a wrong solve silently**; a symmetric factorization is now a pivoted one.
  macOS Accelerate trapped on singular matrices and never applied the deflation rescale.
- **The Windows JIT produced DLLs with no libm calls**: `tcc` exits 0 on an undeclared symbol, and
  `strcpy` was never declared for it.
- **`expr += number` produced complex constants**, which broke code generation far from the cause.
- Several coordinate-system errors: the azimuthal row of the axisymmetric tensor divergence, its
  directional tensor derivative in all three branches, divergences summing over coordinates the mesh
  does not have, and a transposed `vector_gradient` in the differential-geometry base class.

### Packaging & CI

- Stable-ABI (abi3) wheels for CPython 3.12+; the macOS wheels ship their own OpenMP runtime.
- New workflows: wheel tests across Python 3.11-3.14 and all platforms, tutorial scripts run against a
  fresh wheel, prebuilt PETSc/SLEPc artifacts, and a nightly runner for `develop` that also builds the
  documentation.
- The test suite is split into a fast default run and a `--full` run.


## [0.1.9] 

Roughly five months and 250+ commits since the 0.1.8 release. The two biggest
themes are a new mesh-element family (pyramids, wedges, and their bubble-enriched
tetrahedral relatives, all with proper 3D facet support) and a substantially
reworked build/release pipeline (CMake + scikit-build-core and source distributions). 

Alongside that:
several new solver backends, and a long tail of correctness fixes in the FEM core.

### Added

- **New element types**: pyramids and wedges, including C1/C2 variants and the
  bubble-enriched `TetraC1TB`/`Tetra3dC1TB`/`Tetra3dC2TB` tetrahedra, with
  proper facet-based boundary/interface detection (replacing boundary-node-only
  identification) and 3D Gmsh facet support.
- **New solver backends**: a macOS Accelerate-framework linear solver and
  eigensolver. PETSc gained automatic field-split   index sets and general solver improvements.
- **`pyoomph check`** (`python -m pyoomph check solver|eigen|compiler|all`):
  reworked solver selection, checking, and reporting, including install hints
  for missing optional dependencies (MKL/Pardiso, PETSc/SLEPc).
- **Parallel/MPI groundwork**: basic METIS-based mesh partitioning, basic load
  balancing, Dirichlet-by-matrix-manipulation as an alternative to the classical
  implementation, and distributed Dirichlet index spreading over MPI.
- **New physics/numerics**:  latent heat support for `PrescribedMassTransfer`, 
  time derivatives of integrals, matrix-valued `IntegralExpression`s, 
  an `InvertSymmetricMatrix`  multi-return expression, additional local dof constraints 
  (C1 confinement /  ALE constraining), an adaptive bifurcation tracker, 
  and `RemesherViaRecreation`.
- **Source distribution (sdist)** generation, wired into CI and verified with a
  full fresh-environment install-from-source test.
- New GCL and Rayleigh-Plateau-instability tutorials; an inverse-problem
  tutorial; `AGENTS.md`/agent-facing docs for AI-assisted development.
- numerical-data-file loading as numpy array with column and parameter information

### Changed / Improved

- Extensive internal refactoring of hanging-dof handling (new space-information
  structures, restructured hang buffers, streamlined `fill_hang_info_with_equations`)
  and of DG field handling.
- The compiled core extension moved from a top-level `_pyoomph` module to
  `pyoomph._pyoomph_core`.
- Removed unused oomph-lib thirdparty code (FSI, multi-domain, spectral
  elements, DG elements, spines, triangle meshes, the LAPACK QZ eigensolver) —
  a meaningful source-tree size reduction.
- Solid mechanics performance improved; 1D axisymmetry coordinate (polar) range
  reworked to `2x2` matrices in the vector gradients instead of `3x3` with a zero row/column.
- All of `src/` (excluding thirdparty code) commented and documented, with
  pybind11 binding docstrings added throughout.
- Large tutorial-documentation pass: numerous code blocks converted from
  downloadable scripts to `literalinclude`, several documentation gaps filled,
  full spellcheck.

### Fixed

- Interface-dof bugs breaking adaptive multi-physics interfaces, C1TB
  interfaces, and edge cases on interfaces with opposite orientation.
- Hele-Shaw factors corrected
- `CSplineInterpolator` bug; Jacobian sanity checking added to catch a class of
  silent bug where a misnamed override (e.g. `define_residual` instead of the
  correct `define_residuals`) would otherwise just never get called.
- residual/Jacobian checking (e.g. "has residual but no Jacobian row/col"); 
  DG element sorting bug in Jacobian assembly (wrong comparator);
  an accidentally-commented line that left Jacobian codegen empty in some
  cases.
- Higher-codimension (codim-3) code paths; vector gradients on higher
  codimensions.
- hanging dofs for 2D facets on 3D meshes; 
  finite differences and 2D hanging interface dofs on 3D meshes; 
- a load_state issue on adaptive meshes fixed
- an unsymmetric-mass-matrix case now warns instead of
  silently producing wrong results on scipy/ARPACK-based eigensolvers.
- A segfault in `MPI_Init` when extra CLI arguments are passed.

### Windows support

- Fixed `WinError 32` ("file in use") crashes in `pyoomph check` and any
  script that tears down a `Problem` while its temp/output directory is being
  deleted: the log file and persistent output files (`ODEFileOutput`,
  `IntegralObservableOutput`) are now closed proactively in `Problem.release()`
  instead of waiting for eventual garbage collection, matching the existing
  proactive DLL-unload behavior.
- Fixed a related `ValueError` ("path is on mount 'C:', start on mount 'D:'")
  when the code/output directory and working directory are on different
  drives, by falling back to an absolute compiler source path.
- Windows wheels now build via MSYS2/MinGW + CMake instead of the old
  `setup.py`-based flow; added an on-demand CI workflow.

### Packaging & CI

- Migrated the build backend from `setup.py` to CMake + scikit-build-core.
- Wheels are now built via `cibuildwheel` across Linux (manylinux), macOS
  (x86_64 and arm64), and Windows, looping over Python 3.10-3.15 (3.15 via
  `cpython-prerelease`) in a single job per platform.
- Added a dedicated workflow to prebuild static CLN/GiNaC as reusable
  artifacts, with auto-detection of current CLN/GiNaC versions from
  ginac.de (falling back to known-good pinned versions if that lookup fails).
- Fixed a `.gitignore` bug (`*.txt` was silently excluding the tracked root
  `CMakeLists.txt`, among others, from the sdist) that had made the sdist
  fundamentally unbuildable.
