# Development notes

Long-form records of what was built, what was measured, and what was rejected — the reasoning behind
code that the source comments only have room to point at. Each document says at the top what state its
subject is in. Many are cited by file name from `src/`, `pyoomph/` and `tests/`, so renaming one means
chasing those citations.

**Conventions.** Everything here is measured unless it says otherwise; a claim that was reasoned rather
than executed says so. Rejected alternatives are kept with the reason, because the reason is usually
the useful part. Code examples in `examples/` are runnable companions, not snippets.

## Adaptivity and meshes

| | |
|---|---|
| [adaptive_refinement.md](adaptive_refinement.md) | Hanging nodes, the topological tree route for every element shape, mixed meshes, distributed adaptivity. The reference for how a hanging node is represented at all. |
| [macro_elements.md](macro_elements.md) | Curved boundaries for every element type: the shape-generic transfinite blend, opaque parametric coordinates, and why moving meshes were left alone. |
| [interface_refinement_coupling.md](interface_refinement_coupling.md) | Keeping two coupled domains conforming across a shared interface through an adapt. |
| [boundary_node_membership.md](boundary_node_membership.md) | A node marked as being on a boundary it is not on, and the post-adapt repair. |
| [adaptive_resolve_recovery.md](adaptive_resolve_recovery.md) | Recovering from a Newton failure in the re-solve that follows an adaptation. |
| [spatial_error_estimators.md](spatial_error_estimators.md) | Z2 in co-dimension, `desired_ndof`, and per-criterion error normalisation. |
| [mesh_construction.md](mesh_construction.md) | Boundary-layer meshes that survive refinement, Gmsh element winding, and inverted elements. |
| [mesh_point_locator.md](mesh_point_locator.md) | Point location and mesh-to-mesh transfer: zeta, closest-point projection, and the L2 projection solve. |
| [mesh_data_cache.md](mesh_data_cache.md) | Typed cache keys, and merging a distributed mesh's data into one global view. |

## MPI

| | |
|---|---|
| [replicated_mpi_correctness.md](replicated_mpi_correctness.md) | `mpirun` without `--distribute`: the two mistakes every defect there was an instance of. Read first if a replicated run misbehaves. |
| [distributed_remeshing.md](distributed_remeshing.md) | Remeshing under `--distribute`, both remesher paths. |
| [distributed_state_files.md](distributed_state_files.md) | `save_state`/`load_state` on a distributed problem. |
| [mpi_eigenproblems.md](mpi_eigenproblems.md) | Distributed eigenvalue problems through SLEPc. |
| [mpi_augmented_systems.md](mpi_augmented_systems.md) | Bifurcation tracking under `--distribute` (done), and the plan for the Python custom assembler (not). |

## Assembly, code generation and solvers

| | |
|---|---|
| [code_generation.md](code_generation.md) | Where code-generation time goes, and whether the emitted C can be made faster. (Mostly it cannot — the C compiler is already good at it.) |
| [structural_assembly.md](structural_assembly.md) | Precomputed CSR sparsity, value-only re-assembly, and the distributed exchange. |
| [linear_solvers.md](linear_solvers.md) | Backend reuse contracts, Pardiso static pivoting, MUMPS' value-dependent analysis, and reporting a solve failure without ending the run. |
| [nonconvergence_diagnostics.md](nonconvergence_diagnostics.md) | Planning only: what exists for "my Newton solve does not converge", and what is worth building. |

## Physics and equations

| | |
|---|---|
| [viscoelastic_log_conformation.md](viscoelastic_log_conformation.md) | The log-conformation representation, and the confined-cylinder benchmark it was validated against. |
| [tracers.md](tracers.md) | Passive tracer particles: formulation, adaptation, remeshing, MPI. |
| [coordinate_system_tensor_ops.md](coordinate_system_tensor_ops.md) | Which tensor operators each coordinate system implements, and — §6 — how to test one at all. |

## Environment

| | |
|---|---|
| [precice_setup.md](precice_setup.md) | The preCICE `.deb` must match the distribution release. |
