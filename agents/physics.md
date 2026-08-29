# pyoomph — the built-in physics equation libraries

Companion to [`AGENTS.md`](../AGENTS.md). Every ready-made `Equations` class under
`pyoomph/equations/`, with the constructor keywords that matter. **Prefer these over a
hand-rolled weak form whenever the physics matches.** When a one-liner here is not enough,
the class itself is short and readable — grep it before guessing a keyword.

## The catalogue

- **`poisson.py`**: `PoissonEquation(name="u", space="C2", source=None, coefficient=1)` —
  `-div(coeff*grad(u))=f`, supports continuous and DG spaces. In a **dimensional** problem
  `coefficient` carries `length²` and `source` carries the units of `u` (see the
  dimensionless-residual rule above), e.g.
  `PoissonEquation(name="u", source=1*kelvin, coefficient=1*meter**2)`; `coefficient=1`
  raises "residual contribution is not dimensionless". `DiffusionEquation`
  adds a time derivative to get `∂t u - div(D grad(u)) = f`. Neumann conditions are the
  generic `NeumannBC` (`NeumannBC(u=-g)` imposes `coeff*grad(u).n = g`);
  `PoissonFarFieldMonopoleCondition` handles unbounded domains.
- **`advection_diffusion.py`**: `AdvectionDiffusionEquations(fieldnames="advdiffu", diffusivity=1, wind=var("velocity"), source=...)`
  for scalar transport; `AdvectionDiffusionFluxInterface`, `AdvectionDiffusionInfinity`.
- **`navier_stokes.py`**: `StokesEquations(...)` and `NavierStokesEquations(...)` (adds
  inertia) — the main flow-solver classes. Shared keywords:
  `dynamic_viscosity=`, `mass_density=`, `mode="TH"` (Taylor-Hood, the default and
  inf-sup stable) or `"CR"` (Crouzeix-Raviart), `bulkforce=` (any body-force vector added
  to the momentum residual), `gravity=` (as a vector), `boussinesq=True` (constant density
  in the continuity equation), `fluid_props=` (a material object instead of raw numbers).
  So a Boussinesq buoyancy is `bulkforce=Ra*var("temperature")*vector([0,1])` — **do not
  hand-roll the momentum equation just to add a body force.** Fields declared: the vector
  `"velocity"` (components `velocity_x`/`velocity_y`/`velocity_z`) and the scalar
  `"pressure"`; both names are also the `set_scaling` keys.
  Interface equations: `NavierStokesFreeSurface(surface_tension=1, ...)` (free
  surface with surface tension/curvature), `NavierStokesContactAngle(contact_angle=90*degree)`,
  `NoSlipBC` (`DirichletBC` subclass), `NavierStokesSlipLength`,
  `NavierStokesPrescribedNormalVelocity`, `ConnectVelocityAtInterface`.
- **`ALE.py`**: mesh-motion equations for free-boundary/moving-mesh problems —
  `PseudoElasticMesh(E=..., nu=...)`, `LaplaceSmoothedMesh(...)`,
  `HyperelasticSmoothedMesh(...)`, `YeohSmoothedMesh(...)`; helpers
  `ConnectMeshAtInterface`, `PrescribedMovingMesh(umesh=...)`,
  `EnforceVolumeByPressure(volume=...)` (fix an enclosed volume via internal pressure).
- **`solid.py`**: `DeformableSolidEquations(constitutive_law=..., coordinate_space="C2",
  mass_density=..., bulkforce=..., isotropic_growth_factor=...)` — finite-strain solid;
  the nodal positions become the unknowns (field `"mesh"`, reference `var("lagrangian")`),
  so the VTU output already shows the deformed shape. Constitutive laws:
  `GeneralizedHookeanSolidConstitutiveLaw(E=, nu=)`,
  `IncompressibleHookeanSolidConstitutiveLaw(E=)`.
  `LinearElasticitySolidEquations(E, nu, mass_density=0, bulkforce=0, ...)` is the
  small-strain version (E and nu are positional). Loads:
  `SolidTraction(T)` (a traction vector) and `SolidNormalTraction(P)` (a pressure), both
  positional; an unequipped boundary is traction-free. `FSIConnection(velocity_offset=0)`
  couples a solid domain to a `StokesEquations`/`NavierStokesEquations` fluid domain. Two
  rules that are easy to get wrong: it is attached to the **fluid** side of the shared
  interface (`fluid_eqs += FSIConnection() @ "interface"`), and the solid equations must
  be built with **`scale_for_FSI=True`** or the stresses are balanced wrongly (it refuses
  otherwise). The fluid needs a moving mesh (`LaplaceSmoothedMesh()` or similar), and
  `FSIConnection` supplies the mesh coupling, the kinematic condition and the traction
  transfer by itself, so write no velocity or mesh BC of your own on that interface and
  add **neither** `ConnectMeshAtInterface` **nor** `ConnectVelocityAtInterface` there.

  **A solid needs three scales, even when the problem is stationary.** Its test scale is
  `temporal**2/(mass_density*spatial)`, so leaving `mass_density`/`temporal` at their
  default 1 makes the residual non-dimensionless and the setup is rejected. The idiom is
  to pick the elastic wave time:

  ```python
  self.set_scaling(spatial=L, mass_density=rho, temporal=L*square_root(rho/E))
  eqs = DeformableSolidEquations(constitutive_law=GeneralizedHookeanSolidConstitutiveLaw(E=E, nu=nu),
                                 coordinate_space="C2", mass_density=rho)
  eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "clamped_end"   # True = pin where it is
  eqs += SolidTraction(vector([0, -F/h])) @ "loaded_end"
  ```
  For a static problem `rho` is arbitrary (it cancels); any positive value works.
- **`cahn_hilliard.py`**: `CahnHilliardEquation(sigma=, epsilon=, mobility=)` on its own,
  and `SimpleNSCH(fluid_plus, fluid_minus, sigma=, epsilon=, mobility=)` — the
  batteries-included phase-field two-phase flow (Cahn-Hilliard + Navier-Stokes), with
  `SimpleNSCHWettingInterface`/`CahnHilliardWettingInterface` for contact-angle wetting.
  (`SimpleNSCH` lives here, *not* in `NSCH.py`.)
- **`NSCH.py`**: `CompositionNSCHPhaseField(epsilon, mobility, sigma_nsch, ...)`, the
  materials-driven phase-field model that couples to `multi_component.py`; plus
  `RefinePhaseFieldGradients` and `DisjunctDomainMarkerNSCH`.
- **`low_order_NSCH.py`**: `MaterialBasedLowOrderNSCH(fluidA, fluidB, epsilon, mobility)` and
  `LowOrderNSCH`, a cheaper low-order variant; `LowOrderNSCHWetting` for the contact angle.
- **`multi_component.py`**: multi-species/multi-phase transport built on material
  property objects (see Materials below). `CompositionFlowEquations(fluid_props, ...)`
  is the main "batteries-included" assembler (Navier-Stokes + species transport +
  optional temperature). `MultiComponentNavierStokesInterface(interface_props, ...)`
  is the main free-interface class with mass transfer/Marangoni/surfactants. It carries its own velocity
  coupling and traction transfer, so do **not** add `ConnectVelocityAtInterface` — but
  between two moving-mesh domains it does **not** connect the meshes, so
  `ConnectMeshAtInterface` is still required alongside it.
  `TemperatureConductionEquation`/`TemperatureAdvectionConductionEquation` for heat.
- **`contact_angle.py`**: dynamic contact-line models plugged into
  `DynamicContactLineEquations(model=..., wall_normal=...)`, e.g.
  `PinnedContactLine()`, `UnpinnedContactLine(theta_eq=..., cl_speed_exponent=1)`
  (Cox-Voinov for exponent 3), `YoungDupreContactLine(...)`, `WenzelContactLine(...)`,
  `CassieBaxterContactLine(...)`. Both `DynamicContactLineEquations` and
  `NavierStokesContactAngle` optionally take `cox_voinov=True` (plus `U_wall` and
  `cox_voinov_microscopic_length`), which imposes the angle bent by Cox-Voinov up to the
  size of the attached free surface element instead of the microscopic one.
- **`lubrication.py`**: `LubricationEquations(mu=, sigma=, disjoining_pressure=...)`
  for thin-film/lubrication-theory flows (film height + pressure).
- **`darcy.py`**: `DarcyEquation(fluid_props, permeability=, porosity=)` for porous-media flow.
- **`helmholtz.py`**: `HelmholtzEquation(k=, complex=False)` — `Δu+k²u=0`, e.g. for
  acoustics/wave problems in frequency domain.
- **`kuramoto_sivashinsky.py`**: `KuramotoSivashinskyEquations(...)` for thin-film
  interfacial pattern formation.
- **`stokes_stream_func.py`**: `StreamFunctionFromVelocity(...)` — post-processing
  stream function from a computed velocity field (2D/axisymmetric).
- **`harmonic_oscillator.py`**: `HarmonicOscillator(omega=, damping=, driving=)`, an
  `ODEEquations` example/utility for a damped/driven oscillator.
- **`ode.py`**: `DynamicODEEquations(**eqs)` — declare an ODE system by its *residuals*,
  one per variable, without writing an `ODEEquations` subclass: each keyword names a
  variable and its value is the expression that must vanish, e.g.
  `DynamicODEEquations(x=partial_t(var("x"))-var("y"), y=partial_t(var("y"))+var("x"))`.
  The test function is multiplied in for you.
- **`viscoelastic.py`**: `ViscoelasticEquations(model=..., relaxation_time=, polymer_viscosity=,
  formulation="log-conf"|"conformation", space="C2", add_polymer_stress_to_momentum=True,
  stabilization=None)` with pluggable models `OldroydB()`, `Giesekus(alpha=)`,
  `PTT(epsilon=, kind=)`, `FENE_CR(L=)`, `FENE_P(L=)`; `ViscoelasticInflowBC`.
  **It is added *alongside* a flow equation, not instead of one** — it evolves the
  conformation tensor and adds the polymer stress to an existing momentum equation, whose
  `dynamic_viscosity` is then the *solvent* viscosity:

  ```python
  eqs  = NavierStokesEquations(dynamic_viscosity=eta_s, mass_density=rho)   # or StokesEquations
  eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=lam, polymer_viscosity=eta_p)
  ```
  The unknown is the **log-conformation** `Psi = log(C)` by default, in the fields
  `log_conformation_xx/_xy/_yy` (`conformation_*` under `formulation="conformation"`). To
  get a stress, exponentiate and apply the model's relation — for Oldroyd-B
  `tau = (eta_p/lam)*(C - I)` with `C = scipy.linalg.expm(Psi)`.
  `add_polymer_stress_to_momentum=False` leaves the momentum equation alone, which is how
  you check a constitutive model against an imposed velocity field.

  **A coupled solve started from `Psi = 0` at a finite shear rate diverges.** Make the
  driving rate a global parameter, solve at zero, and ramp it with `go_to_param` — that
  reproduces the analytic Oldroyd-B simple-shear stresses (`tau_xy = eta_p*gdot`,
  `N1 = 2*eta_p*lam*gdot**2`) to 3e-7.
- **`potential_flow.py`**: `PotentialFlow(potential_name="phi", ...)` for inviscid
  irrotational flow, with `PotentialFlowFreeInterface(surface_tension=...)`,
  `PotentialFlowNormalVelocity`, `PotentialFlowFarField`, `PotentialFlowInterfaceEnd`.
  (`PotentialFlowFreeInterface1/2/3` are deprecated aliases that warn.)
- **`stabilized_ns.py`**: `StabilizedNavierStokes(space="C2C1", viscous_form=, stabilization=,
  tau_formula="shakib"|"codina"|"tezduyar", ...)` — residual-based SUPG/PSPG/LSIC
  stabilization, which is what lets equal-order velocity/pressure spaces work; plus
  `ImposedTraction`, `BackflowStabilization`, `StabilizationBoundaryFlux`.
- **`stabilization.py`**: the shared machinery behind the `stabilization=` keyword of
  `AdvectionDiffusionEquations`/`CompositionAdvectionDiffusionEquations`/the temperature
  equations. Pass `stabilization="SUPG"` (or an iterable of `"SUPG"`, `"GLSDIFF"`,
  `"ASGSDIFF"`, `"DC"`, or a `ScalarTransportStabilization` instance) when advection
  dominates diffusion and the solution oscillates. Off by default everywhere.
- **`surfactants.py`**: `SurfactantTransportEquations(surfactants, diffusivity=, ...)` —
  interfacial surfactant transport, usable standalone on any free surface and driven
  automatically by `MultiComponentNavierStokesInterface`. Also `SurfactantEndFlux`,
  `SurfactantsAtSolidInterface`. Defaults to the conservative (GCL) form; see
  [`materials.md`](materials.md) for the isotherms that feed it.
- **`salt_transport.py`**: `SaltTransportEquations(salts, fluid_props=, ...)` — dissolved
  salts as ion pairs (rather than independent species), with `FrozenSaltConcentrations`
  and `SaltConcentrationsFromMassFractions`.
- **`electrostatics.py`**: `ElectricPotentialEquations(permittivity=|relative_permittivity=,
  charge_density=, conductivity=)`; `PoissonBoltzmannEquations(ions=|bulk_concentration=, valence=1, temperature=,
  linearized=False)` and `DebyeHuckelEquations(debye_length=...)` for electric double
  layers (the latter is the linearized case, and takes the screening length directly —
  `phys_consts.debye_length(...)` computes it); `NernstPlanckEquations(ions, ...)` for ion
  transport; `OhmicConductionEquations`. The potential field is named **`"phi"`** by
  default (`potential_name=` changes it) and all these classes forward extra keywords such
  as `relative_permittivity=`/`permittivity=` and `temperature=` to
  `ElectricPotentialEquations`. Boundary/interface classes:
  `ElectrodeBC(voltage, potential_name="phi")`,
  `SurfaceChargeBC`, `SurfaceChargeConservation`, `ElectricFarFieldCondition`,
  `ElectricPotentialConnection`, `ThinDielectricLayer`, `SternLayer`, `IonFluxBC`.
- **`electrohydrodynamics.py`**: couples the above to flow —
  `MaxwellStressEquations`/`ElectricBodyForceEquations` (bulk force, two equivalent
  formulations), `MaxwellStressInterface` (the jump at a dielectric interface),
  `ElectroosmoticSlip(zeta_potential=...)`.
- **`tracers.py`**: `TracerParticles(advection=var("velocity"), seed=...)` — massless
  tracer particles advected with the flow, for visualization or residence-time studies.
  Seeds: `TracerSeedPoints(positions)`, `TracerSeedGrid`, `TracerSeedRandom`,
  `TracerSeedElement`, `TracerSeedCallable` — **seed coordinates are plain
  nondimensional floats** (in units of the `spatial` scale), not dimensional
  expressions. `TracerTransferAtInterface`/`TracerTransferToInterface` move tracers
  between domains, `TracerPeriodicBoundaryCondition` wraps them around.
  Read the positions back with
  `problem.get_mesh("domain").get_tracers(name="tracers").get_positions()`, an
  `(N, dim)` array of nondimensional coordinates (also `get_ids`, `get_tags`,
  `get_payloads`, and the `gather_*` variants which collect across MPI ranks).
  Tracer positions **are** carried through `save_state`/`load_state`.
- **`topological_changes.py`**: automatic topology changes of a moving mesh —
  `AxisymmetricReconnection(rmin=, distmin=, volume_conservation=True)` pinches off or
  merges an axisymmetric interface when it gets too thin. Needs a
  `TopologicalChangesGmshTemplate`/`TopologicalChangesTQMeshTemplate` mesh (which can
  rebuild the domain after the change) rather than a plain `GmshTemplate`; also
  `DisjunctDomainMarker` to label the resulting separate pieces.

Many of these physics modules are heavily parametrized — when writing a script, prefer
grepping the actual class in the corresponding file for the full constructor signature
and docstring rather than relying purely on the one-liners above.

