# Tracer particles

Status: **done and in use.** `TracerParticles` (`pyoomph/equations/tracers.py`) advects passive
particles through a bulk domain or confined to an interface, on static and moving meshes, in 1/2/3
dimensions, with path-integrated payloads, a rolling position history for trails, handover between
domains, periodic re-injection, and `--distribute`. Covered by `tests/test_tracers.py` (47 cases) and
`tests/test_mpi_tracers.py` (13 cases, `--full`).

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

## 5c. Three defects the wavy-channel tutorial exposed

`docs/source/tutorial/ale/tracers.py` was the first case with a genuinely moving mesh, a domain
several units wide and a state file written every output. It found three separate things, and the
first two both presented as "particles disappear", which is why they are easy to conflate.

* **The placement Newton could not reach its convergence threshold.** `place_at`'s inner iteration
  declared success only when the step in reference coordinates fell below a fixed `1e-14`. What that
  iteration can actually reach is the rounding noise of the residual, `eps*|x|`, divided by the
  element scale `|dX/ds|`. On a mesh of 0.1-sized elements at `x ~ 4` that floor *is* about `1e-14`,
  so the threshold was unreachable: the iteration spent all 25 rounds bouncing on it, the placement
  was reported as failed, and the particle was dropped as having left the mesh - while sitting well
  inside its own element. It is distance-dependent, which is what made it look arbitrary: in the
  tutorial every loss was at `x >= 1.4` and none below, and 189 particles became 64 within two time
  units. The threshold is now scaled by that floor, using the *smallest row norm* of `J` so that a
  flat or stretched element relaxes it rather than tightening it.
  `test_no_particle_is_lost_in_the_interior_of_a_moving_mesh` is parametrised on the distance from
  the origin for exactly this reason: the `at_origin` case passes either way.
* **The cached point locator did not notice that the mesh had moved.** A `MeshPointLocator` freezes
  the nodal positions it was built from - kd-tree, element boxes, affine fits - and the collection
  cached one per time level, keyed on the mesh's *topology* generation. But mesh motion changes the
  geometry without changing the topology, so a locator built at one step was used to answer "is this
  point inside the mesh" at the next. It showed up loading a state file: the mesh was restored to its
  stored configuration and the particles near the moved boundary were then placed against a locator
  describing the initial one, and were dropped as outside. There is now a `geometry_stale` flag, set
  wherever the configuration may have moved (`advect_all`, `relocate_all`, `set_mesh`, `_load_state`).
  The element walk asks `get_adjacency_locator()` instead, because node-element incidence does not
  move - otherwise every step of a moving mesh would pay for a full locator rebuild.
* **The rolling position history was not in the state file.** Only positions, payloads and identities
  were, so a restored state came back with every particle in the right place and no trail at all, and
  the trails then grew back from scratch rather than continuing. State format 0.1.3 adds it: the
  per-particle sample count goes into the id/tag array, which becomes three entries per particle, and
  the samples are appended after the fixed-size blocks so neither array needs a worst-case stride.
  While fixing it, the ring buffer's reallocation on a capacity change turned out to reassign the
  buffer while leaving `hist_n` and `hist_head` pointing into it as if the old samples were still
  there; it re-rings properly now, which is also what lets a restored history survive being bound to
  an equation asking for a different capacity.

One more, in the same area: `after_transient_solve` guards against advecting twice at the same time,
and a state file may put the clock *back* - a rollback, a restart from an earlier dump. The guard
then refused to advect until the run had caught up with the time it last saw, leaving the particles
frozen in place while the flow moved on. `after_remeshing`, which a state load goes through, now sets
the guard to the restored time.

## 5d. Periodic re-injection

`TracerPeriodicBoundaryCondition` is the counterpart of `TracerTransferAtInterface` for a domain
that is periodic in itself: a particle that has run out of the mesh is offered its position plus
each registered shift, and is taken back in at the first image that lands inside, finishing the rest
of its timestep from there. Three things about the shape of it:

* **The shifts are registered on the collection, not on a boundary.** A shifted position that lands
  inside the mesh *is* the periodic image - a domain in which two different shifts both land inside
  would have to be larger than its own period - so there is nothing to detect. Which end of a pair
  the equation is attached to therefore does not matter, attaching it to both is a no-op the second
  time, and a particle leaving through a corner where two periodic directions meet falls out for
  free. Same argument as the one that removed the boundary-set intersection from `try_transfer`.
* **The wrap happens inside the advection, after the transfer interfaces have had their chance.**
  So a neighbouring domain still wins over a wrap, and the particle finishes its step from the image
  rather than stopping at the boundary - a wrap costs no accuracy. The re-placement is at the level-0
  configuration while the particle is at `timefrac < 1`, which is the same approximation `adopt()`
  already makes for a cross-domain handover: the walk in `place_at` corrects the element.
* **MPI needs a collective round of its own.** `exchange_migrants()` routes a particle to the owner
  of the HALO element it ended in, and the far end of a periodic domain is not a halo of the near
  end - under a partitioning that knows nothing about the periodicity it is usually not in the
  sending process's mesh at all. So a particle whose image is not local is parked, and
  `exchange_reinjections()` offers every process's parked particles to all of them with the same
  claim protocol as seeding (lowest-numbered process holding it wins), inside the existing migration
  round loop so the re-injected particle gets another round to finish its step in.
  `test_periodic_reinjection_crosses_processes` asserts `get_reinjections_last_step() >= 1` rather
  than only the positions: with both ends of the domain on one process the local path would answer
  identically and the test would be checking nothing.

The trail is dropped on a wrap. A trail is a path through the plotted coordinates and a wrapped path
is not continuous there, so keeping the samples from before the jump would draw one line straight
back across the whole domain on the next output.

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
