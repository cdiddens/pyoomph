# Tracer particles

Status: **done and in use.** `TracerParticles` (`pyoomph/equations/tracers.py`) advects passive
particles through a bulk domain or confined to an interface, on static and moving meshes, in 1/2/3
dimensions, with path-integrated payloads, a rolling position history for trails, handover between
domains, and `--distribute`. Covered by `tests/test_tracers.py` (31 cases) and
`tests/test_mpi_tracers.py` (10 cases, `--full`).

This replaces the original implementation wholesale. That one was dead code - no test, no example,
no prose documentation, and the only `.rst` was an empty autodoc stub - and it was wrong in ways
that would not have announced themselves. §5 lists what was wrong, because several of those defects
are the reason particular tests are written the way they are.

---

## 1. The formulation

A particle sits at `x_p(t) = X(s(t), t)`, where `X` is the mesh map and `J = dX/ds`. In general

```
ds/dt = J⁺ (v − dX/dt|_s)                      J⁺ = Moore-Penrose pseudo-inverse
```

and the whole design follows from what that becomes in the two cases:

**Bulk (codimension 0).** `J` is square, so `J J⁺ = I` and the equation collapses to

```
dx_p/dt = v
```

The mesh velocity **cancels analytically**. So the code integrates the *physical position* and uses
the local coordinate `s` purely as a chart for evaluating the field. Nothing ever computes a mesh
velocity, `eval_flag("moving_mesh")` is never consulted, and a particle in a moving mesh with `v = 0`
does not move because every Runge-Kutta stage derivative is *identically* zero - not because two
computed velocities cancel to within rounding. That is why `test_moving_mesh_zero_advection_does_not_move_tracers`
can assert `< 1e-13` rather than a tolerance.

**Interface (codimension 1).** `J` is `(d−1) × d`, `P = J⁺ J` is the orthogonal projector onto the
tangent space, and

```
dx_p/dt = P v + (I − P) dX/dt|_s
```

- the tangential part of the advection field, plus the normal part of the interface's own motion.
That is "advected tangentially, co-moving normally" exactly, with no explicit normal or tangent
algebra anywhere in the code. After each accepted sub-step the position is re-anchored onto the
interface by the same least-squares inversion, which pins the normal offset at machine zero instead
of letting it random-walk over thousands of steps.

The two cases share one code path. `TracerCollection::place_at` solves `X(s,τ) = y` by Newton when
the system is square and by Gauss-Newton on the normal equations `(J Jᵀ) ds = J r` when it is not;
in the second case the normal part of the residual is deliberately left in, because that residual
*is* the offset from the surface.

## 2. Time consistency within a step

The solver only ever defines `X^{n+1}`, `X^n`, `X^{n-1}`. The configuration *within* a step is
defined here as a Lagrange interpolation in time of the **nodal positions** (`TracerTimeConfig`).
Because the shape functions do not depend on time,

```
J(τ) = Σ_k w_k(τ) J^k        dX/dτ = Σ_k w'_k(τ) Σ_l ψ_l x_l^k
```

are *exact* rather than approximated, and `dX/dτ` is the exact derivative of the very interpolant
`J` is taken from. That last point is what makes a mesh moving exactly with the flow produce an
identically zero derivative.

The order is quadratic (three levels) where the stored history allows it and linear otherwise; it
demotes itself on an impulsive start, where `t(1) == t(2)` makes the quadratic basis singular.
`test_time_interpolation_order_is_honoured` measures the difference, and getting a case where it
*shows* took two attempts, both worth recording:

* in the **bulk** the stage derivative with `v = 0` is identically zero at any interpolation order,
  so a bulk particle stays put regardless and the test passes vacuously;
* a **spatially uniform** interface motion telescopes: `∫ dX/dτ dτ = X(1) − X(0)` for any
  interpolant that hits both endpoints.

So the test deforms the interface non-uniformly *and* moves the particle along it.

## 3. Integrator and element exit

Bogacki-Shampine 3(2), FSAL, with a PI-style controller on the embedded error. Not a 5(4) method:
the advection field is only `C⁰` across element faces, so a higher-order method cannot realise its
order on any sub-step that straddles one, while costing six stages times up to three history levels
of generated-code calls each.

Element exit needs no exit-time prediction at all. The test is "the `s` that inverts the new
position falls outside the reference domain", after which the particle is looked for in the
elements sharing a node with the current one. If it is in none of them, the sub-step was too long
and is rejected and halved - which is the right response anyway, and is also what keeps the
integration from ever needing to locate a point in the time-interpolated configuration, for which no
locator exists.

This deleted `factor_when_local_coordinate_becomes_invalid` (base + six overrides + two shared
helpers, ~250 lines) and with it the 2d-only barrier: it threw for **every** 3d element, so tracers
in 3d were impossible. Containment now goes through `reference_domain_kind` /
`inside_reference_domain` / `clamp_to_reference_domain` (`src/refdomain.{hpp,cpp}`), lifted out of
`MeshPointLocator` so the two cannot drift apart, and covering wedges and pyramids for free.

## 4. Bookkeeping, identity, MPI

**The physical position is the authoritative state.** `elem` and `s` are derived and are dropped
whenever `Mesh::bump_topology_generation()` announces that cached element pointers may be stale.
That counter replaced comparing against the address of a cached kd-tree: the invalidation `delete`d
it, so the allocator handing the same address back made the change invisible, and the stored pointer
was dangling either way.

Every particle carries a persistent, never-recycled `TracerId`.

**Distribution.** Ownership is "the process on which the particle's element is not a halo". A
particle is deliberately allowed to advect *through* the halo layer - whose nodal positions and dof
values are synchronised copies of the owner's - and ownership is reconsidered only at the end of the
step, where the level-0 locator is exactly valid. Migrating mid-step would instead require the
receiving process to place the particle in the time-interpolated configuration it did not integrate.
`exchange_migrants()` is then one `Alltoall` plus two `Alltoallv`s of fixed-stride records, and the
round loop is bounded and throws rather than spinning.

Every tracer collective is entered by every process, including ones holding no particles;
`test_a_process_with_no_particles_does_not_hang` seeds everything in one corner precisely to keep
that true.

State files hold the whole particle set, gathered and sorted by identity, so a file written at one
process count reads at another. Seeding goes through `add_tracers_collective`, where every process
proposes the same candidates and the lowest-numbered one holding a candidate in a non-halo element
keeps it - so the particle set and its identities do not depend on the partitioning at all.
`test_distributed_result_matches_the_serial_one` asserts exactly that, at 1e-11.

## 5. What the old implementation got wrong

Recorded because most of it was silent, and because the tests are shaped around it.

* **The Jacobian was taken at history level 0 for every sub-step.** `fill_shape_info_at_s` was
  called with the default `history_index = 0` regardless of the time fraction, so the geometry was
  frozen at the end-of-step configuration while the velocity was blended between levels.
* **The ALE term was not blended at all**, and carried a `# TODO: ALE term in past?`.
* **The ALE term was not emitted unless the mesh had position dofs.** `eval_flag("moving_mesh")`
  resolves to a compile-time 0/1 from `coordinates_as_dofs`, so a mesh moved by macro elements or by
  direct node manipulation dragged its tracers along with it, silently. This is now not a fixed bug
  but a removed one: in the bulk the term does not exist. `test_moving_mesh_...` includes that exact
  configuration.
* **2d only**, in two independent places: the element-exit predicate threw for every 3d element, and
  the seeding raised `RuntimeError("Implement here tracer grid generation for other dimensions")`.
* **`prepare_shape_buffer_for_integration` was never called**, so a `partial_t()` inside an advection
  expression read whatever timestepper weights the previous assembly had left behind.
* **The generated code emitted `dx = shapeinfo->int_pt_weight[0]`** at a point that is not an
  integration point, making a stale value silently available.
* **Advection was hooked to `after_newton_solve`**, so it fired on stationary solves, on every
  arclength continuation step, and once per spatial-adaptation level. It now runs from a new
  `Problem.actions_after_transient_solve()`, once per accepted timestep, with a last-advected-time
  guard behind it.
* **`_update_mesh` replaced the collection on every call**, discarding every particle and all
  transfer-interface wiring - which is why `TracerTransferAtInterface` had never worked.
* `get_tags` existed in C++ but was never bound, so the `tag` argument was write-only.

Three defects found *outside* the tracer code while building this, all pre-existing and all fixed
here because a tracer test is what exposed them:

* **`expand_all_and_ensure_nondimensional` divided by zero** whenever the *first* component of a
  vector expression was identically zero: it took component 0 as the reference unit unconditionally,
  and a zero component's factor is 0. So `vector(0, 1)` was not expressible in **any** expression
  reaching that helper - integral observables included, not just tracers.
* **`set_tracer_advection_velocity` could not accept an identically zero field**, for the same
  reason one step earlier. "Advect by nothing while the mesh moves" is the sharpest ALE test there
  is, so it had to become expressible.
* **`add_plot` never inferred `mode="tracers"`.** A tracer name was recognised as a plottable field
  but the mode inference still made it a `tricontourf` and then failed asking for a colorbar, so
  tracers could only be plotted by passing the mode by hand.
* **`Time::ndt()` counts stored timestep *sizes***, while `time(t)` walks back over them, so level
  `t` is valid for `t <= ndt()`. Capping the interpolation order at `ndt()` silently forced linear
  interpolation on every problem - caught only because the order test then showed bit-identical
  results for both orders.

## 5a. Spatial adaptation, and a bug it uncovered

Refinement replaces a leaf element by its sons; unrefinement **deletes** them. Anything holding an
element pointer across either is holding a pointer the mesh has invalidated, so tracers have to be
told. They are: `Problem.actions_after_adapt` bumps every mesh's topology generation, and the
collection re-locates every particle from its stored physical position before the next advection.

That bump had to be added to the **Python** `actions_after_adapt`, and this is worth knowing beyond
the tracers: `pyoomph::Problem::actions_after_adapt` in `src/problem.cpp` is **dead code for every
Python-defined problem**, which is every problem. The Python override replaces it wholesale and does
not call `super()`, so the C++ body - generation bump, `ensure_dummy_values_to_be_dummy()`,
`setup_pinning()` - never runs. (`setup_pinning()` happens to be repeated in the Python version;
`ensure_dummy_values_to_be_dummy()` is not, and has therefore never run after an adapt. That is
pre-existing and untouched here, but somebody should look at it.)

The failure this caused is a good example of why the test asserts a mechanism rather than a number.
Without the bump the tracers kept dereferencing invalidated elements for twelve adaptations and
still produced the analytic answer to 2e-16, because:

* oomph keeps a **refined** element's parent object alive, with its nodes and their current values,
  so evaluating in a stale parent gives a coarser interpolation of the same solution - plausible,
  and for a field that depends only on geometry, identical;
* a pointer into an **unrefined** element's deleted son is undefined behaviour that need not crash,
  and did not.

So `test_tracers_survive_refinement_and_unrefinement` asserts that on every step where the element
count changed, *every* particle was re-located - `get_relocations_last_step() == nlocal()`. That
assertion fails without the fix; the value assertions beside it do not. The other two adaptation
tests (an interface mesh, which is torn down and rebuilt entirely, and a curved boundary, where
refinement genuinely moves the domain by snapping new nodes onto the arc) are behavioural guards
against particle loss and pass either way.

## 5b. Remeshing

Different from adaptation in kind, not degree: remeshing builds an entirely new mesh object that
shares no element and no node with the old one and discretises the domain differently. Nothing about
a particle's element pointer survives, so the collections are carried over to the replacement and
every particle is re-located from its stored physical position.

Bulk collections were already carried over (`Problem.remesh_handler_during_solve` re-points them at
the new mesh). **Interface collections were not**, and the effect was total: an interface mesh is
rebuilt as a new object rather than swapped into the problem's mesh dict, so the collection and
every particle in it simply disappeared. They now travel along the `previous_mesh` chain in
`InterfaceMesh.__init__`, which covers arbitrarily nested interfaces for free.

Two smaller things fell out of that:

* `TracerParticles._mesh` was left pointing at the mesh that had been thrown away. It happened to
  work, because the bulk carry-over shares the `_tracers` dict by reference rather than copying it,
  so the stale mesh still resolved to the right collection. `after_remeshing` now re-binds to the
  equation tree's current mesh, at a point where it has its elements.
* The duplicate-name guard in `_bind_mesh` compared mesh objects, which is exactly wrong here: a
  remesh legitimately presents a different object for the same domain. It compares domain names now.

`place_globally` also snaps an interface particle onto the located point. During a run that is a
no-op, since the sub-step re-anchor already keeps the offset at zero, but after a remesh the new
interface discretises the same boundary slightly differently and the particle would otherwise sit
fractionally off it until the next sub-step. Anything reading positions in between would see the
invariant broken.

`test_interface_tracers_survive_a_remeshing_event` and
`..._stay_on_the_interface_after_a_remeshing_event` both fail without the carry-over; the bulk one
passes either way and is there as a guard.

## 6. Where the accuracy stops

Worth stating, because the tolerance knob does not reach it.

* **The in-step interpolation of the mesh configuration caps everything at its own order** (2 or 3),
  however tight `rtol` is set. The solver simply does not know where the mesh was in the middle of a
  step.
* **A `C⁰` velocity field caps the integrator.** Every sub-step that straddles an element face has a
  non-smooth right-hand side and BS3 degrades locally to order 2 there. The controller shrinks the
  step, which is correct but costs sub-steps.
* **Blending an analytic function of the coordinates over history levels is only approximate.** The
  advection field is blended over the nodal time-history levels, which is *exactly* right for a
  solved FE field - it is then the field with time-interpolated nodal values - but for an analytic
  `f(x)`, `Σ w_k f(x_k) ≠ f(Σ w_k x_k)` unless `f` is affine. The error is quadratic in the per-step
  mesh displacement, i.e. the same order as the configuration interpolation itself.
  `test_nonuniform_advection_on_moving_mesh_converges_to_the_static_answer` measures that order
  rather than asserting an exactness that does not hold.
* **What "machine zero" means here.** The exact cases are exact because the test data makes the
  stage derivative identically zero or constant along the path. On a real Navier-Stokes field on a
  curved deforming mesh a tracer is exactly as accurate as the field it is handed, and no more.

## 7. Still open

* **Interface tracers under MPI are untested.** The machinery does not special-case them and there
  is no reason to expect it to fail, but nothing asserts it, so treat it as unproven. Extending
  `tests/mpi_tracer_worker.py` with an interface case is the missing piece.
* **A particle crossing more than one element into foreign territory within a single step is lost**
  rather than migrated. The halo layer is one element deep, so this is a CFL-like limit of roughly
  one element per step near a partition boundary; the general fix is a global residue pass
  (`Allgatherv` + `Allreduce(MIN, rank)`), which the point locator's design already anticipates.
* **Cost has not been measured.** Physical-space integration adds one Newton inversion per stage
  against the old s-space form, and the sub-step count is now set by an error controller rather than
  by the old fixed "~10 sub-steps per element". The honest prior is "similar, with an accuracy
  guarantee"; anything stronger needs an in-process measurement (see CLAUDE.md).
* **`InterfaceElementBase::get_normal_at_s`** takes the orientation of past-level normals from the
  current normal. Fine for one step of mesh motion, thin ice for an advection expression using
  `var("normal")` at level 2 on a violently moving interface.
