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
| [internal_facet_fields.md](internal_facet_fields.md) | Unknowns on the interior-facet skeleton (HDG traces, mortar multipliers): per-facet storage, the pinned opposite dummy, 3d enumeration, and how the fields survive adaptation, remeshing, `--distribute` and a state file. |
| [mesh_data_cache.md](mesh_data_cache.md) | Typed cache keys, and merging a distributed mesh's data into one global view. |

## MPI

| | |
|---|---|
| [replicated_mpi_correctness.md](replicated_mpi_correctness.md) | `mpirun` without `--distribute`: the two mistakes every defect there was an instance of. Read first if a replicated run misbehaves. |
| [distributed_remeshing.md](distributed_remeshing.md) | Remeshing under `--distribute`, both remesher paths. |
| [distributed_state_files.md](distributed_state_files.md) | `save_state`/`load_state` on a distributed problem. |
| [mpi_eigenproblems.md](mpi_eigenproblems.md) | Distributed eigenvalue problems through SLEPc. |
| [mpi_augmented_systems.md](mpi_augmented_systems.md) | Bifurcation tracking under `--distribute` (done), and the plan for the Python custom assembler (not). |
| [floquet_multipliers.md](floquet_multipliers.md) | Condensing the block bidiagonal orbit Jacobian into the monodromy matrix instead of solving one large singular pencil, the opt-in periodic Schur, why a DAE's algebraic directions land on `±1` rather than 0, and periodic orbits under `--distribute`. |

## Assembly, code generation and solvers

| | |
|---|---|
| [code_generation.md](code_generation.md) | Where code-generation time goes, and whether the emitted C can be made faster. (Mostly it cannot — the C compiler is already good at it.) |
| [initialisation_cost.md](initialisation_cost.md) | Why `initialise()` took 25 s for a million dofs before anything was solved, what removed a quarter of it, and why skipping the elemental equation numbering cannot work (`assign_local_eqn_numbers` also rebuilds `eleminfo`). Also: how to profile any of this when `perf` is unavailable and `cProfile` cannot see nanobind. |
| [structural_assembly.md](structural_assembly.md) | Precomputed CSR sparsity, value-only re-assembly, and the distributed exchange. |
| [jacobian_block_flags.md](jacobian_block_flags.md) | Per-block proven symmetry/constancy bits from the symbolic block expressions, their problem-level AND-union in `_jacobian_structure.txt`, and the print-free global-parameter registration that came out of it. Consumers not built yet. |
| [static_condensation.md](static_condensation.md) | Eliminating element-local dofs (CR bubbles, DL/D0/DG fields) from the assembled system and reconstructing them after the Newton update. Halves the factorisation time; serial, distributed and replicated MPI; experimental. |
| [replicated_condensation_gather.md](replicated_condensation_gather.md) | Planning only, and no longer needed for Crouzeix-Raviart, which [dof_ordering.md](dof_ordering.md) serves instead: gathering a condensed block whose rows straddle the uniform row split of a replicated (`mpirun` without `--distribute`) run. Still the answer for a selection renumbering cannot make contiguous, e.g. interior-penalty DG. |
| [dof_ordering.md](dof_ordering.md) | Choosing the global dof numbering from Python: nodal blocks for a block preconditioner, element blocks for static condensation. The permutation hook inside `assign_eqn_numbers`, and aligning the replicated MPI row split with the blocks. |
| [linear_solvers.md](linear_solvers.md) | Backend reuse contracts, Pardiso static pivoting, MUMPS' value-dependent analysis, reporting a solve failure without ending the run, and running a serial backend under `mpirun` by gathering onto rank 0. |
| [nonconvergence_diagnostics.md](nonconvergence_diagnostics.md) | Planning only: what exists for "my Newton solve does not converge", and what is worth building. |

## Physics and equations

| | |
|---|---|
| [viscoelastic_log_conformation.md](viscoelastic_log_conformation.md) | The log-conformation representation, and the confined-cylinder benchmark it was validated against. |
| [stabilized_navier_stokes.md](stabilized_navier_stokes.md) | SUPG/PSPG/LSIC/GLS/ASGS/VMS for the momentum equations, the equal-order pairs they enable, and what a bulk stabilization leaves behind on a Neumann boundary. |
| [stabilized_scalar_transport.md](stabilized_scalar_transport.md) | The same for advection-diffusion, mixture composition and temperature: shared `tau` machinery, why `div(grad(c))` had to be written as `trace(grad(grad(c)))`, and the measurement that no stabilization perturbs the interface physics. |
| [tracers.md](tracers.md) | Passive tracer particles: formulation, adaptation, remeshing, MPI. |
| [aiomfac_electrolytes.md](aiomfac_electrolytes.md) | Activity coefficients of a salt solution: the AIOMFAC middle- and long-range parts in all three back-ends, the parameter audit against the current AIOMFAC source (two ions had been attached to the wrong species), and the agreement with AIOMFAC itself. |
| [salt_transport.md](salt_transport.md) | Salts without any electrostatics: one field per salt with the ambipolar diffusivity, the interface condition that keeps a salt in an evaporating liquid (three ALE forms, three different terms), salt-induced Marangoni and its direction, and the measured agreement with Poisson-Nernst-Planck. |
| [electrohydrodynamics.md](electrohydrodynamics.md) | Electrostatics, electrolytes (PNP / Poisson-Boltzmann / Debye-Hückel / leaky dielectric) and the coupling into the flow: why the potential formulation, the shared permittivity scale that makes two-domain coupling work, the surface-charge sign, and the three EHD routes — one of which is silently wrong on a free surface. §10 is the open-issues list, including the log formulation for Nernst-Planck and an audit of what no test covers; §11 is the conservative (GCL) transport of the surface charge and the ions on a moving mesh, why the old surface-charge form did not converge in the time step at all under evaporation, ad-/desorption rates, and two sign conventions that were documented backwards. |
| [coordinate_system_tensor_ops.md](coordinate_system_tensor_ops.md) | Which tensor operators each coordinate system implements, and — §6 — how to test one at all. |

## Environment

| | |
|---|---|
| [precice_setup.md](precice_setup.md) | The preCICE `.deb` must match the distribution release. |
