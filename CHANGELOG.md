# Changelog

## [0.1.9] 

Roughly five months and 250+ commits since the 0.1.8 release. The two biggest
themes are a new mesh-element family (pyramids, wedges, and their bubble-enriched
tetrahedral relatives, all with proper 3D facet support) and a substantially
reworked build/release pipeline (CMake + scikit-build-core and source distributions). 

Alongside that:
several new solver backends, and a long tail of correctness fixes in the FEM core.

### Added

- **Any liquid can be given by concentration.**
  `Mixture(water + 1*milli*molar*get_pure_liquid("my_surfactant"), temperature=20*celsius)` -- and
  `5*gram/litre*...` for a mass concentration -- instead of a mass fraction worked out by hand. The
  literature values for a soluble surfactant, and every isotherm in
  `pyoomph.materials.surfactant_isotherms`, are written against `molarconc_`, so a mass fraction was
  the one form nobody has. Unlike a dissolved salt, such a species is *always* a real component: it
  counts towards the mass fractions, gets a mole fraction and a share of the density, and needs the
  registered `MixtureLiquidProperties` for the full set of names. The remaining fractions describe the
  base mixture and are scaled by `1-sum w` to make room, so `water + 20*percent*glycerol + 1*mM*surf`
  keeps a 20 % glycerol *base*.
  - **Two conventions, because there are two volumes**, selected by `concentration_basis`.
    `"base_mixture"` (the default) is how a solution is made -- mix the base, measure *its* volume, add
    the solute by that volume -- so it is a mass balance and claims nothing about the volume the solute
    occupies; it is `_set_salt_initial_conditions` with the apparent molar volume set to zero.
    `"solution"` is moles per volume of the finished solution, i.e. what the solver reports: it solves
    `w = c*M/rho(w)` against the mixture's own density by fixed-point iteration, so `molarconc_<name>`
    at t=0 is exactly the value that was typed. The two differ by the solute's own mass fraction, i.e.
    not at all in the dilute limit.
  - A `temperature=` is required, since essentially every density correlation has one.
  - A dimensionless factor still means a fraction. A factor carrying any other unit is now refused at
    the multiplication with a message naming the three accepted forms, rather than surviving as a
    "fraction" of 1 mol/m^3 until `float(total)` inside `Mixture` failed on it.

- **Salt as a real composition field.** `CompositionFlowEquations(..., salt_treatment="component")`
  promotes every dissolved salt from a dilute solute to an ordinary component: a mass fraction that
  sums to unity with the solvents, a mole fraction, and a share of the volume. The composition
  equations then transport it, and the evaporation condition it needs is the `j_i = 0` case of the
  term they already write for any non-volatile component -- none of the three ALE branches the
  dilute treatment needed arises, and the salt is conserved to 1e-14 by machinery that knows nothing
  about salts. `"dilute"` remains the default.
  - The density becomes volume-additive, `1/rho = w_solv/rho_solv + sum w_s V_phi,s/M_s`, with the
    solvent correlation evaluated at the *renormalised* salt-free composition. `V_phi` is stored per
    ion and combined by stoichiometry, the same way the ambipolar diffusivity is -- the additivity
    holds to 0.1 cm^3/mol against measured salt values -- and reproduces brine density to 0.1% at
    5 wt%, 0.4% at 10 and 1.5% at 20. A salt whose ions have no tabulated volume is refused: zero is
    not a harmless default for a volume.
  - The ion concentration fields survive as substitutions of the mass fraction, so a surface tension
    law, an activity coefficient or an observable written against `c_Na_p` works in every mode.
  - **The two modes agree on more than expected**: with a prescribed evaporation rate they thin a
    film identically at any concentration, because the salt's volume is conserved and the film
    therefore loses volume at `j/rho_solvent` whatever is dissolved in it. What differs is the liquid
    -- at 3 molal the dilute treatment still reports the density of water and a water mass fraction
    of one -- which is what feeds anything that computes an evaporation rate rather than being handed
    one.
  - The AIOMFAC conversion changes rather than disappearing: `molefrac_*` now counts a salt as one
    particle where AIOMFAC counts its `nu` ions, so `gamma_pyoomph = gamma_AIOMFAC (1+f)/(1+nu f)`.
    A test asks both modes for the same physical state and requires the activities to agree while
    the coefficients and mole fractions differ.
  See `dev_docs/salt_transport.md` section 6.

- **AIOMFAC with ions: activity coefficients of a salt solution.** A dissolved salt changed the
  surface tension but not the *activity* of the solvent, so a drying brine evaporated like pure
  water. AIOMFAC is the only model in the library with an electrolyte generalisation and only its
  short-range half was present; the middle-range ion interactions and the long-range
  Pitzer-Debye-Hueckel term are now there too, for the solvents as well as the ions. Water activity
  and molality-based ionic activity coefficients agree with AIOMFAC 3.13 to six digits, for aqueous
  NaCl and for water + glycerol + NaCl; gamma_pm of NaCl reproduces the literature minimum near 1
  molal, and the dilute limit follows the Debye-Hueckel limiting law. `set_activity_coefficients_by_unifac`
  picks the salts up by itself, so the vapour pressure over a brine now drops as it should, and
  `get_ion_activity_coefficient` and `get_mean_ionic_activity_coefficient` expose the ionic ones. A
  pure solvent with a salt in it counts: that case had no activity machinery at all before.
  - **The maths is written once and rendered three ways** -- symbolic GiNaC, numpy, and generated C
    -- against the expression-generator interface the short-range part already had, rather than
    written out three times as that part historically was.
  - **The parameters were regenerated from the AIOMFAC source** by `citools/generate_aiomfac_parameters.py`,
    which is kept in-tree so the next AIOMFAC release is a rerun. The audit found two ions attached
    to the wrong species (subgroup 246 was called CH3COO- and is IO3-, 247 was called SCN- and is
    OH-), ion molar masses in g/mol among neutrals in kg/mol *and discarded on the way in*, seven
    ions with placeholder parameters AIOMFAC does not have, 155 changed and 528 missing interaction
    entries, and 246 pairs AIOMFAC marks as never determined -- which now raise rather than being
    silently treated as ideal. Salt-free mixtures are unchanged to 1e-8; the residual difference is
    that the old table had been round-tripped through single precision.
  - AIOMFAC's `BRR`/`CRR` are deliberately **not** imported: they belong to a different temperature
    parameterisation that AIOMFAC only uses for selected fit datasets, and pyoomph's B and C mean
    something else again, so importing them would be wrong rather than merely different.
  See `dev_docs/aiomfac_electrolytes.md`.

- **Salt transport, evaporation and salt-induced Marangoni flow, with no electrostatics involved.**
  A salted material handed to `CompositionFlowEquations` used to generate exactly the same system as
  an unsalted one — the composition equations build their field list from the mixture *components*,
  and the ion table is a different dictionary that nothing read. It is now picked up automatically
  (`salts="auto"`), and `pyoomph/equations/salt_transport.py` solves one field per salt with the
  ambipolar diffusivity `D = (z+-z-)D+D-/(z+D+-z-D-)`, derived from the ion table rather than
  tabulated: NaCl 1.610, KCl 1.994, CaCl2 1.335, Na2SO4 1.230, HCl 3.336 against measured 1.610,
  1.990, 1.335, 1.230, 3.340 (1e-9 m²/s). One field per *salt* rather than two per pair is what makes
  electroneutrality structural — the ion concentrations are stoichiometric substitutions, so
  `get_charge_density()` is literal zero — and it is the only option without a potential to hold the
  two ions together.
  - **A non-volatile solute stays behind when the solvent evaporates**, which is not the natural
    boundary condition of the weak form: that one is zero *diffusive* flux, and a receding surface
    then sweeps the salt out with the vapour. `MultiComponentNavierStokesInterface` supplies the
    condition next to the one it already supplies for the volatile components, and it is a different
    term for each of the three ALE forms. With `GCL=True` — the conservative form, whose natural
    condition already is zero flux through a moving boundary — the dissolved amount is conserved to
    machine precision (3e-15) at any step size; the other two converge at second order in `dt`.
  - **Salt raises the surface tension** (`SaltProperties.surface_tension_increment`, tabulated for 29
    salts: NaCl +1.64, CaCl2 +3.66 mN/m per mol/L, and the strong acids negative), so Marangoni flow
    runs *towards* the enriched region — the opposite of a surfactant, and worth knowing when reading
    an evaporating drop. The law is written against the *ion* concentrations, which exist under both
    electrolyte models, so the same interface drives the same Marangoni stress whether the ions come
    from the electroneutral model or from `PoissonNernstPlanck`.
  - **The two models agree where they overlap**: an electroneutral gradient relaxing in a 1 µm box
    decays to 0.610553 of its amplitude under the electroneutral model and 0.610554 under full
    Poisson-Nernst-Planck, against the analytic 0.610498. They may not share a domain, since a
    substituted ion concentration shadowed by a solved one would run and be wrong.
  - `add_salt` now also takes a salt directly, `water.add_salt("NaCl", 1*milli*molar)`, and two salts
    sharing an ion add up on it instead of overwriting.
  See `dev_docs/salt_transport.md`.

- **Salts, and mixtures that carry them.** `pyoomph.materials.ions` also registers the common salts
  and strong acids, and `get_salt("NaCl")` fetches one the way `get_ion` and `get_pure_liquid` fetch
  theirs. A salt names two ions and pulls them out of the ion library when it is constructed, so it
  cannot name an ion that does not exist; the stoichiometry is *derived* from the two charge numbers
  rather than parsed out of the name, which is why Na2SO4 comes out 2:1 -- sulfate is divalent, not
  because the name contains a 2. Multiplying a salt by a concentration dissolves it, and `Mixture`
  now takes that alongside the solvent fractions:

      mix = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

  The dissolved species stay out of the fraction bookkeeping: fractions must sum to unity and a
  concentration is not one of them, and at 1 mM a salt is 6e-5 of the solution by mass, so pretending
  it displaces some of the water would be a bigger error than ignoring it
  (`DissolvedSpeciesComponent.mass_fraction_in` is there when that needs checking). A single ion
  works the same way -- `c*ion` dissolves it while `fraction*ion` keeps the mixture-component meaning
  an ion inherits from `PureLiquidProperties`, with the units telling the two apart. `molar`
  (mol/litre) joined the units. Because the Walden correction reads the *mixture's* viscosity, 20 wt%
  glycerol slows Na+ from 1.33e-9 to 8.0e-10 m^2/s and drops the conductivity of 1 mM NaCl from 12.6
  to 7.6 mS/m without any of that being wired up separately. Glycerol also gained its measured
  relative permittivity (42.5); a mixture still does not average that automatically, since linear
  mixing is a poor rule for it, but the error now names
  `set_by_weighted_average('relative_permittivity')` as the way to ask.

- **A library of ionic species, `pyoomph.materials.ions`.** The 28 common ions (H+ through HPO4 2-)
  are registered with the same `@MaterialProperties.register()` decorator every other material uses,
  and `get_ion("Na+")` fetches one exactly as `get_pure_liquid` and `get_surfactant` do -- a fresh
  instance per call, so dissolving it in one liquid cannot reach another. `add_ion` and `add_salt`
  take a name and go through the same lookup, so an electrolyte is
  `water.add_salt("Na+", "Cl-", 1*milli*mol/liter)`; a name that is not registered still needs an
  explicit `charge_number` and `diffusivity`, which `new_ion` then puts into the same table. The
  datum stored per ion is the limiting molar conductivity at 25 °C and the diffusivity follows from
  it by Nernst-Einstein: one number per ion, so the conductivity closure and the transport model
  cannot end up describing different ions. Both the temperature dependence and the solvent dependence
  come from the fractional Walden rule `λ⁰μⁿ = const`, applied by the solvent
  (`BaseLiquidProperties.get_ion_diffusivity`) since an ion does not know what it is dissolved in.
  That puts D(Na+) at 0 °C within 0.03% of the measured value where a constant `λ⁰` is 47% out, makes
  an electrolyte's conductivity rise at the ~2 %/K a conductivity meter compensates for instead of
  being exactly temperature independent, and estimates Na+ in glycerol as 737x slower than in water
  rather than identical to it. H+ carries a fitted exponent of 0.63 against everyone else's ~0.94,
  which is Grotthuss proton transfer being far less sensitive to the solvent viscosity than Stokes
  drag.

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

- **The generated element code binds its loop-invariant buffer reads to locals.** Every
  `shapeinfo->`/`eleminfo->` access whose leading indices do not depend on the test/trial node -- the
  shape buffers, the nodal data, the hanging-node rows, the local-equation table, the loop bounds, the
  timestepper weights, and the Jacobian/mass row strides inside the scatter macros -- is now read once
  at the top of the integration-point body and indexed from there. It had to be re-read at every use:
  the `jacobian[...] +=` store in the innermost trial loop may alias `shapeinfo` as far as the C
  compiler knows, and qualifying the output pointers with `restrict` does not help (it was tried, and
  produces a byte-identical object file). Worth 20% of the emitted instructions and 11-15% of an
  elemental Jacobian across the tutorial elements, with the residual-only path -- which has no such
  store -- unchanged, and every residual, Jacobian and mass matrix bitwise identical. `-O3` cannot do
  this itself: it is not allowed to. Revertible with `PYOOMPH_DISABLE_BUFFER_ALIASES`.

- **`-fno-math-errno` is now a default compile flag** for the system compiler. It changes no
  arithmetic -- residuals are bit-identical -- but it lets the compiler treat libm calls as pure and so
  hoist them out of the trial loop. On polynomial weak forms it is noise; on pyoomph's own hyperelastic
  mesh smoothers, where the user cannot reach in and wrap anything in `subexpression()`, it halves
  `HyperelasticSmoothedMesh` (41.9 -> 20.7 ms elemental residual+Jacobian) and cuts `YeohSmoothedMesh`
  by a factor of 5.7 (127.2 -> 22.3 ms). `SystemCCompiler.get_cache_flag_state()` now hashes the
  computed flag list itself, so a future flag change invalidates the JIT cache on its own and needs no
  epoch.

- **`use_subexpressions` removed from `HyperelasticSmoothedMesh` and `YeohSmoothedMesh`.** Both
  activate the coordinates as dofs by construction, so the subexpression derivative cache always took
  the position-symbol escape hatch: the body was differentiated on the spot and inlined at every use
  site, and the cached scalar was written and never read. Passing `True` was measured at **+87%**
  elemental residual+Jacobian, 16% more generated C and 2.4x the `pow()` calls, i.e. the option could
  not pay off on any element these classes can be attached to. Nothing in the repository passed it.

- **An adaptation that refines and unrefines nothing no longer touches the problem.** Deciding to do
  nothing is the normal end state rather than an edge case: oomph-lib only leaves its own adaption loop
  once an `adapt()` has reported 0/0, so with `spatial_adapt>0` the last adaptation of every solve is a
  no-op by construction, and a mesh sitting at `max_refinement_level` with errors still above the
  refinement tolerance never reports anything else. That no-op was not free -- every interface mesh was
  torn down and rebuilt, the global mesh reassembled, and the equations reassigned, which invalidates
  the Jacobian sparsity pattern unconditionally, so the frozen sparsity was thrown away and rebuilt for
  a numbering that had not changed. The refine/unrefine decision is now taken *before* anything is torn
  down (which the coupled-interface reconciliation above already needed) and the whole block is skipped
  when it comes out empty. On the FSI tutorial the equation numbering, and with it the sparsity pattern,
  now survives every step in which the mesh does not move.
- **`subexpression()` now also works in the analytic Hessian.** Since 2024 the Hessian generator
  unwrapped every `subexpression()` marker before differentiating the residual twice, so on a problem
  with `setup_for_stability_analysis(analytic_hessian=True)` -- every bifurcation tracker, azimuthal and
  Cartesian normal-mode stability analysis -- the wrapping bought nothing at all in what is routinely
  the *largest* generated function. It is now kept: the outer Hessian index wraps `d(body)/d(field)` in
  a nested subexpression whose own cached derivative is the second derivative, so the values and both
  derivative levels are computed once per integration point instead of being inlined into every entry
  of the `nnode^2` double loop. On a three-species element with a shared transcendental activity law,
  `HessianVectorProduct0` went from 809 kB to 72 kB and assembling the Hessian tensor from 0.196 s to
  0.0054 s. Nothing changes for problems that do not call `subexpression()`, and the computed Hessian
  is unchanged to round-off -- checked against the same residual written without the wrapper, on a
  plain nonlinearity, with `partial_t` inside the wrapper (the mass-matrix Hessian), on an axisymmetric
  `m=1` azimuthal tracker, and on a moving mesh. On moving meshes the coordinate index still inlines,
  as it does in the Jacobian. Note this is the same trade `subexpression()` already made in the
  Jacobian, and it can go the other way: wrapping something cheap now makes the Hessian function
  slightly *larger*, because hoisting costs a declaration and a fill regardless of how little inlined
  text it saves. Wrap the expensive shared terms, as the docs already advise.
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
- **The normal-mode coordinate systems can now take the divergence and the directional derivative of a
  rank-2 tensor.** `CartesianCoordinateSystemWithAdditionalNormalMode.tensor_divergence` and
  `.directional_tensor_derivative` used to raise, and so did
  `AxisymmetryBreakingCoordinateSystem.directional_tensor_derivative`. All three are implemented, for
  `ndim == edim`; the surface and point cases still raise, now with a message that says so. The
  practical consequence is that `NavierStokesEquations(GCL=True)` and the viscoelastic module can be
  combined with `setup_for_stability_analysis(additional_cartesian_mode=True)` and
  `azimuthal_stability=True`, which was previously refused outright. Expanding by `exp(I*k*z)` rather
  than `exp(I*m*phi)` needs no connection terms at all -- a Cartesian frame does not rotate and there
  is no `1/r` -- so the whole content is three first-order derivative operators, which are the same
  ones the class's existing `scalar_gradient`, `vector_gradient` and `vector_divergence` already spell
  out. `docs/source/tutorial/advstab/cartesiannormal/rivulet.py` runs with `GCL=True` and reproduces
  its eigenvalue curve; `rising_bubble.py` does too, with the two formulations converging together
  under refinement as their `int rho*u_i*div(u)*v_i` difference requires.
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
- Fixed the azimuthal frame rotation of `AxisymmetricCoordinateSystem.directional_tensor_derivative`,
  which was wrong in all three of its branches -- each in a different single slot, because each spelled
  the rotation out separately. `C_rz` read `+T_phiz` instead of `-T_phiz` in the `(r,z,phi)` layout;
  the `use_x_as_symmetry_axis` layout read the transposed `T_rz` for `C_zphi`; and the radial-mesh
  branch had no azimuthal term on the diagonals at all, and dropped the radial derivative on the
  off-diagonals. All four call sites now share one derivation. As with the `tensor_divergence` fix
  above, every wrong entry multiplies an azimuthal off-diagonal, which plain axisymmetry cannot build,
  so nothing reachable was affected and the tests use hand-assembled basis dyads. A cheap check that
  catches all three: the rotation must satisfy `C(T^transpose) == C(T)^transpose`.
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

- **`EquationTree.__add__` on an already-placed tree.** `+` hands the children of *both* operands to
  the new node it returns, rewriting their `_parent`. Doing that to a tree that had already been
  placed with `@ "domain"` (or added to the `Problem`) left the placed children pointing at a new root
  nobody keeps, and the damage surfaced arbitrarily far away as "Mesh is None" in
  `pin_redundant_lagrange_multipliers`. Adding to, or with, a placed tree is now refused up front and
  names the path it was placed at; assemble a domain's equations first and place the result
  afterwards. The merge itself moved to an internal `_merge_with`, which the recursion and the
  boundary-pattern expansion still use, since there the operands are nobody else's.

- **Quadrature, 2D quadrilateral elements**: the 3x3 Gauss-Legendre knot table had two digits
  transposed. Five of the nine entries of `Gauss<2,3>::Knot` read `0.774596662941483` for
  `0.774596669241483`, and only on the *positive* knot, so the rule kept the correct total weight but
  stopped being symmetric — which is why it survived: an asymmetric quadrature is invisible to any
  test whose reference value is computed on the same mesh. The effect was a fixed defect in the
  assembly rather than an error that refinement could remove: the integral of a mid-side shape
  function derivative over an element came out as 7e-9 instead of identically zero, so a field lying
  exactly in the C2 space no longer produced a zero residual, and results on 2D quadrilateral meshes
  were wrong by about 1e-9 *however fine the mesh*. Measured on a Poisson problem whose exact
  solution is linear, the deviation at the mid-side nodes drops from 8.5e-10 to 4.0e-15 (2e-16 was
  already reached by `C1`, and by triangles at any order, which use different rules). The knot is now
  written once and negated, so the rule is symmetric to the last bit. An audit of every `Gauss<D,N>`
  knot and weight table against exact Legendre roots found no other defect above 4e-15;
  `tests/test_quadrature.py` now tests the rules directly, across lines, quads, triangles and bricks.

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


