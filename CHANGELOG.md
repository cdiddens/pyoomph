# Changelog

## [0.1.9] 

Roughly five months and 250+ commits since the 0.1.8 release. The two biggest
themes are a new mesh-element family (pyramids, wedges, and their bubble-enriched
tetrahedral relatives, all with proper 3D facet support) and a substantially
reworked build/release pipeline (CMake + scikit-build-core and source distributions). 

Alongside that:
several new solver backends, and a long tail of correctness fixes in the FEM core.

### Added

- **Adaptivity across coupled domain interfaces.** Two domains that share an interface (tied by
  `ConnectFieldsAtInterface`, `ConnectMeshAtInterface` or any other opposite-interface connection) are
  adapted individually by oomph-lib, so a refinement criterion stated for one of them left the other with
  no reason to follow -- and the opposite-element matcher, which pairs interface elements by exact
  vertex-position sets, then died with "Cannot locate opposite element". The only way to have an adaptive
  coupled interface was not to: `RefineToLevel()` on *both* sides, so that they could not disagree
  (see `docs/source/tutorial/multidom/simple_fsi.py`). pyoomph now keeps the two sides carrying identical
  boundary facets automatically, refining the coarser one where they differ, interleaved with each mesh's
  own 2:1 balancing, to a joint fixed point. Works under `mpirun --distribute`, where the two domains are
  partitioned independently. Diagnosable with `Problem.check_interface_conformity()`, which reports
  facets with no counterpart separately from facets whose counterpart is not on the process holding them
  -- under MPI those are different defects needing different fixes. The two sides are reconciled on their
  pending refine/unrefine *decisions*, after both meshes have decided and before either acts, rather than
  on the errors that produced them: oomph merges a father only if all of its sons agree, so an
  unrefinement vetoed by a son that does not touch the interface is invisible to any comparison made at
  the interface. That is what keeps a coarsening interface from being merged away and refined straight
  back, which would be correct but would re-interpolate the patch from the merged father and lose its
  fine-scale solution. Coupled domains with different `max_refinement_level` are allowed: the shallower
  cap governs the shared interface, while each domain still refines to its own cap away from it. Where
  that cannot work -- `RefineToLevel` refines uniformly and does not respect `max_refinement_level`, so
  one side can be driven past a cap the other cannot follow -- the run now stops with the offending
  facets and the reason, instead of the opposite-element matcher's bare "Cannot locate opposite node".
  Elements that touch a coupled interface at a single *vertex* are kept graded with it as well: they
  carry no boundary facet, so nothing that enforces conformity can see them, and a domain whose whole
  refinement is forced from the other side used to leave them arbitrarily coarser -- a refined band one
  element thick with a four-level drop beside it, while every conformity check reported success. The
  2:1 rule is now closed at the interface vertices too, in the same fixed point.
  See `dev_docs/interface_refinement_coupling.md`.
- **Vector fields can be given to `DirichletBC` and `InitialCondition` as a whole**, i.e.
  `DirichletBC(velocity=vector(1,0))` and `InitialCondition(velocity=vector(u,v))` instead of naming
  every component. The value is split onto the field's components positionally -- the same
  correspondence `var("velocity")` itself is built with -- so it is correct in any coordinate system,
  including an axisymmetric one where the components are not x/y. The padding `vector()` adds up to
  `GiNaC_vector_dim()` is ignored; a non-zero component the field has no slot for is an error rather
  than silently dropped, as is a vector given for a scalar field.
- **`RefineToLevel` and `RefineMaxElementSize` are now evaluated in the C++ core** instead of by a Python
  loop over the elements on each adapt. Same criteria, same values, unchanged API -- but they now cover
  every element a process holds, halo copies included, so a distributed run needs no repair pass to make
  the halo layer agree with its owner about them.
- **New element types**: pyramids and wedges, including C1/C2 variants and the
  bubble-enriched `TetraC1TB`/`Tetra3dC1TB`/`Tetra3dC2TB` tetrahedra, with
  proper facet-based boundary/interface detection (replacing boundary-node-only
  identification) and 3D Gmsh facet support.
- **New solver backends**: a macOS Accelerate-framework linear solver and
  eigensolver. PETSc gained automatic field-split   index sets and general solver improvements.
- **Halo consistency check for distributed runs** (`PYOOMPH_CHECK_HALO_CONSISTENCY=1|throw`, off by
  default): every adapt cross-checks that all processes agree about the elements they share -- positions,
  refinement levels, pending refinement flags and the error estimates about to decide their fate --
  reporting or raising if they do not. Divergent meshes are silent where they happen and only surface much
  later as a wrong `ndof`, an `inf` residual or a deadlock; this names the offending elements by position
  at the adapt that created them. The verdict is agreed across processes, so raising mode fails the whole
  job rather than one rank while the others block. Also callable from Python as
  `Mesh.check_halo_consistency()`.
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

- **Tensor index conventions are now self-consistent -- two of them changed.** `grad` was already the
  Jacobian, `grad(u)[i,j] = d(u_i)/d(x_j)`, and `matproduct`/`double_dot` were already standard, but
  `contract` and the divergence of a rank-2 tensor were not, in ways that cancelled for the idiomatic
  terms and so went unnoticed. Both now follow the standard adjacent-index convention:
  - **`contract(A,b)` (and `A @ b`) is now `A_ij*b_j`, was `A_ji*b_j`.** Symmetrically,
    `contract(a,B)` is now `a_j*B_ji`. The two orders have therefore swapped meaning: the advection
    term `u.grad(u)` is now `contract(grad(u),u)`, and `contract(u,grad(u))` is `grad(|u|^2/2)`. If
    you have equations written the old way round, swap the arguments -- or better, spell it
    `matproduct(grad(u),u)`, which never changed. Only the mixed vector/matrix case moved;
    vector-vector (dot product), matrix-matrix (`A:B`) and anything involving a scalar are untouched,
    so a `contract(S,S)` stays exactly as it was. `weak()` never went through `contract` and is
    unaffected.
  - **`div(T)[i]` of a rank-2 tensor is now `d_j T_ij`, was `d_j T_ji`.** This makes `div` the adjoint
    of `grad`, so `div(grad(u))` is the vector Laplacian (it used to be `grad(div(u))`), and makes
    `div(T)` the integration-by-parts partner of `weak(T,grad(v))` with the traction
    `matproduct(T,n)`. For symmetric tensors -- every usual stress -- nothing changes at all. A flux
    tensor has to be assembled accordingly: the momentum flux carrying `u` along `rho*q` is
    `dyadic(u,rho*q)`. The conservative (`GCL=True`) Navier-Stokes momentum flux was updated to
    match; no other equation in pyoomph took the divergence of a tensor. Each coordinate system's
    `tensor_divergence` implements the second-index convention directly.
  - `dot()` now also accepts one matrix and one vector, with the same standard convention, where it
    previously raised "dot is only allowed between vectors". `dot` and `contract` agree in every
    shape they share; two matrices still raise, since `matproduct` and `double_dot` are both
    plausible readings. The index order of every operator is now stated in its docstring.
- **`expr += number` and `expr -= number` no longer produce complex constants.** `__iadd__` and
  `__isub__` were bound only against `std::complex<double>`, and nanobind converts an int or a float to
  that without complaint, so the in-place forms quietly yielded e.g. the complex numeric `-1.0+0.0i`.
  Nothing looked wrong symbolically -- it prints as `-1` and compares equal to `-1` -- but the C printer
  then emitted `std::complex<double>(-1.0,0.0)` into a real-valued residual and the compiler rejected
  the element, so the failure surfaced as `command '/usr/bin/cc' failed` far from its cause. `*=` and
  `/=` were never affected; they always had int/double overloads. In practice this made
  `RadialSymmetricCoordinateSystem(Rcenter=...)` unusable for any non-zero `Rcenter` given as a plain
  number, since its gradients and divergence shift the radius with `coords[0] -= self.Rcenter`;
  wrapping the value in `Expression()` was the accidental workaround.
- Fixed three divergences that reached for a derivative with respect to a coordinate their mesh does
  not have. Each died with "Cannot expand the field 'coordinate_z'" (or `_y`) instead of dropping a
  term that is zero anyway, because the summation range came from the operand's padded slot count
  rather than from the mesh dimension:
  - `div(f*identity_matrix())` on a two-dimensional Cartesian mesh -- i.e. the pressure part of any
    stress divergence written out by hand -- since `CartesianCoordinateSystem.tensor_divergence`
    summed over all three coordinates whatever the dimension.
  - `div(vector(a,b,c))` with a nonzero third component on a two-dimensional Cartesian mesh, likewise.
  - `div(vector(u_r,u_phi,0))` on a one-dimensional radial axisymmetric mesh, where
    `AxisymmetricCoordinateSystem.vector_divergence` gated its axial term on `nops()`, which is three
    for every padded vector.
- Fixed the azimuthal row of `AxisymmetricCoordinateSystem.tensor_divergence`, whose connection term
  should be `(T_rphi + T_phir)/r`. On a two-dimensional axisymmetric mesh it read
  `(T_phir - T_rphi)/r`, which is zero for a symmetric tensor and the wrong sign otherwise; on a
  one-dimensional radial mesh it read `(2*T_phir - T_rphi)/r`, wrong even for a symmetric tensor. Only
  reachable from a hand-assembled tensor such as `dyadic(vector(h,0,0),vector(0,0,k))`, since
  `define_tensor_field` puts the azimuthal component on the diagonal only and `vector_gradient` is
  swirl-free -- plain axisymmetry has no azimuthal velocity component -- which is why it survived. The
  azimuthal-symmetry-breaking system, which does carry swirl, already had it right.
- Fixed `BaseDifferentialGeometryCoordinateSystem.vector_gradient`, which returned the transpose of
  what every other coordinate system returns (`d(u_j)/d(x_i)` instead of `d(u_i)/d(x_j)`).
- Corrected the `upper_convected_derivative` docstring, which stated the stretching terms with the two
  transposes interchanged. The code was and is right for pyoomph's gradient convention.
- The spatial-derivative operators (`partial_t` with ALE, `material_derivative`,
  `directional_derivative`, `convected_derivative`, `upper_convected_derivative`) now all document
  that they use the *surface* gradient on a domain with a co-dimension, and that `var("u",domain="..")`
  is how to get the bulk one -- as `grad` and `div` already did. `directional_derivative` and
  `convected_derivative` had no docstring at all.
- Extensive internal refactoring of hanging-dof handling (new space-information
  structures, restructured hang buffers, streamlined `fill_hang_info_with_equations`)
  and of DG field handling.
- The compiled core extension moved from a top-level `_pyoomph` module to
  `pyoomph._pyoomph_core`.
- **`Problem.get_current_dofs()` returns two numpy arrays**, not one array and a Python
  `list[bool]`. The second half is parallel to the first and to every other dof-length vector the
  class hands out, so it should be usable the same way -- `dofs[~is_positional]` now works. Its
  docstring also said it flagged *pinned* dofs; it has always flagged nodal *position* dofs (a
  pinned value is not a dof and does not appear in this vector at all), and now says so.
  `Problem.get_all_values_at_current_time()` passes the array through unchanged.
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

- **Adaptivity, mixed quad+tri meshes**: refining across a quad↔triangle interface could tear the mesh.
  oomph-lib's quad neighbour lookup maps a new node's position into the edge neighbour with the quad box
  map and then asks that neighbour whether it holds a node there; across a mixed interface the neighbour is
  a triangle, whose local coordinates live in `[0,1]²` against the quad's `[-1,1]²`, so the lookup could
  match a triangle node *by accident* and hand back a node belonging to a completely different edge. The
  quad son adopted it, which both duplicated a node and dragged a real node of the coarse triangle onto the
  quad's edge (via the hanging-node position interpolation), folding every triangle that owned it. Seen on
  gmsh meshes that put a quad boundary layer (`Quads=1`) inside a triangular mesh, where one `adapt()` moved
  coarse-mesh nodes by a sizeable fraction of an element and produced visibly torn output; uniform
  refinement of such a mesh could also segfault. Cross-shape node sharing is done topologically instead.
  The 3D brick lookup carries the same guard; it is unreachable there today (a mixed 3D forest uses no
  octree neighbour pointers) and was added so the planned cross-shape 3D neighbour finder cannot inherit
  the problem.
- **MPI**: per-element error overrides are now agreed between processes by a MAX-reduction over all
  copies of an element, rather than by letting the owner's value win. An override is not always computed
  on the rank that owns the element it applies to: an interface element pushes its error onto the bulk
  element behind it, including — at a coupled interface — the bulk element of the *opposite* domain, and
  since two coupled domains share no nodes the partitioner cuts them independently, so the rank holding
  the interface element routinely holds only a halo copy of that opposite element. Owner-wins discarded
  such an override silently, on every rank.
- **MPI**: under `mpirun -n N` with `N>1`, PETSc+MUMPS is now selected as the default
  linear/eigen solver on every platform (previously the platform cascade picked the
  serial-only Pardiso on Linux, which then raised from its own constructor). Serial
  solver selection is unchanged.
- **MPI**: `Problem.get_residuals()`, `get_current_dofs()`, `get_history_dofs()` and
  `_assemble_residual_jacobian()` read an `oomph::DoubleVector` by *global* equation
  number, but that vector is row-partitioned on a distributed problem — they returned
  out-of-bounds garbage (`nan`, `1e+148`, …) as soon as `ndof > nrow_local`.
  `set_current_dofs()` wrote out of bounds in the same way. All now gather/scatter
  properly, so under MPI every rank sees the same full-length vector. (The Jacobian
  returned by `_assemble_residual_jacobian` remains process-local CSR, as before.)
- **`ConstrainPositionsToC1Space` on non-uniformly refined meshes**: aborted at any
  2:1 T-junction on triangle, mixed and all 3D meshes. A constrained node's position
  redistributes onto its C1 corners, but those corners are recorded by whichever
  element sees the node as a non-vertex -- which may be a *neighbour*, so they can be
  vertices of another element that oomph-lib never registered as position-hang masters
  here. Reading their local equation number then returned garbage. Those masters are
  now registered explicitly. Works on every element family in 2D and 3D.
- **Mixed-order spaces on 3D wedges/pyramids/tets**: a C1 field living on a C2-geometry
  element was interpolated with the *geometric* (quadratic, all-nodes) basis instead of
  the linear basis over the corner vertices, because the wedge and pyramid C2 elements
  never overrode oomph-lib's isoparametric `ninterpolating_node`/`interpolating_basis`
  defaults. Any Taylor-Hood, coupled C2+C1 or ALE problem on a non-uniformly refined
  wedge/pyramid/mixed 3D mesh therefore got an inconsistent Jacobian and Newton failed
  to converge. `BulkElementBase` now provides these hooks shape-agnostically, and the
  wedge, pyramid and tetrahedral C2 elements use them. The tet previously overrode
  `interpolating_basis` alone, which left callers reading uninitialised entries of the
  `Shape` array.
- **C1-space constraints on 3D wedges/pyramids**: two wrong entries in the C1-corner
  tables meant `ConstrainFieldsToC1Space` degraded a field to something that was not
  the element's C1 space at all, even without adaptivity. The pyramid's base-quad
  centre was expanded over one diagonal instead of all four base corners, and two of
  the wedge's bottom-layer edge midpoints were tied to the wrong corner pairs.
- **3D adaptivity**: `ConstrainFieldsToC1Space` aborted with "Cannot enforce a
  degration to C1 on a C1 vertex node" on any 3D mesh carrying a 2:1 (non-uniformly
  refined) interface, plain bricks included. A constrained node is legitimately a C1
  *vertex* of the finer neighbouring elements — and of the sons created at a father's
  face/volume centre — and the code already handled that by leaving the hang to the
  elements where the node is a non-vertex; the guard aborted before reaching it. It
  demanded the node hang on the C1 slot, which holds in 2D but not in 3D. The guard now
  tests what its message describes: degrading to a C1 space that is not present.
- **3D adaptivity**: `DynamicOcTreeForest::check_all_neighbours` inspected only the
  *first* tree when deciding whether to skip oomph-lib's brick compass neighbour
  self-test. A mixed 3D forest whose first root happened to be a brick therefore ran
  that self-test on a forest for which no neighbour pointers are set at all, aborting
  with a bogus "Max. error in octree neighbour finding" (or running away into an
  out-of-memory). It now scans every tree, as the 2D equivalent already did.
- **MPI**: adaptive refinement could refine different elements on different processes, silently
  corrupting the mesh. oomph-lib's error estimator synchronises the errors it computes from haloed to
  halo elements -- but pyoomph applies its own per-element error *overrides* afterwards
  (`RefineToLevel`, `RefineMaxElementSize`, `RefineAccordingToElement`, interface-driven bulk overrides
  and any user `calculate_error_overrides`), and those ran rank-locally, so an element could be marked
  "must refine" on the process that owns it and "do not refine" on a process holding a halo copy. The
  ranks then built different meshes: stale coarse elements survived in the halo layer, the tree-based
  hanging-node search installed 2:1 constraints that globally do not exist, and the global equation
  numbering diverged, so the first Newton step produced `inf` (earlier, an asymmetric throw that
  deadlocked the remaining ranks). Because the halo/haloed element lists are built by walking the leaves
  of the same trees, a single divergence also misaligned every later halo exchange. The final error
  vector is now synchronised owner-to-halo before adaptation. Note oomph-lib has a check for exactly
  this, but only under `PARANOID`, which pyoomph does not enable by default.
  What decided whether a run was affected was not *which* criterion was used but *where it was stated*:
  a criterion on the bulk mesh reads only the element's own geometry, so a halo copy agreed with its
  owner anyway, whereas one restricted to a boundary/interface (`... @ "domain/top"`) reaches the bulk
  through the interface elements -- which a rank holding only halo copies of those bulk elements does
  not have. All of `RefineToLevel`, `RefineMaxElementSize` and `RefineAccordingToElement` were affected
  in that position.
- **MPI**: the 3D 2:1 refinement-balancing pass (`TemplatedMeshBase3d::enforce_refinement_balance`) was
  rank-local: each process selected its own elements (halo copies included) and decided when to stop from
  its own selection count, around a collective `refine_selected_elements()`. The selection is now unioned
  across processes and the loop terminates on the global set being empty.
- **MPI**: `create_pressure_fixation()` (Taylor-Hood, Crouzeix-Raviart and
  Scott-Vogelius) pinned the pressure dof of the *rank-local* element 0, so each rank
  constrained a different dof and distributed Stokes solves crashed. The pinned
  node/element is now selected deterministically and agreed across ranks.
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


