# pyoomph — reference for AI coding assistants

This file is a condensed, code-verified reference for AI assistants helping a user
write **simulation scripts** with pyoomph (i.e. using the installed `pyoomph` package
as a library — this is *not* about hacking on the pyoomph framework's own C++/Python
source). Point an AI assistant at this file before asking it to write a pyoomph
problem script.

pyoomph is a Python front-end to the C++ library `oomph-lib`. You describe a PDE (or
ODE) system as a **weak form built from symbolic expressions**; pyoomph
differentiates it symbolically (Jacobian, parameter derivatives, Hessians — including
w.r.t. moving-mesh coordinates), generates C code, compiles it on the fly, and links
it back into the running script. Users normally never touch C++ directly —
everything is expressed through the Python API below. 
All equations are solved together in a fully implicit, monolithic manner.


## Mental model

A typical pyoomph script has exactly three kinds of classes, plus a boilerplate entry point:

1. **`Equations`** (or `ODEEquations`) subclass(es) — declare unknown fields and the
   weak-form residual(s) of the physics. Reusable, mesh/domain-agnostic.
2. **`Problem`** subclass — builds the mesh(es), instantiates equations, attaches
   boundary/initial conditions and outputs, and binds equations to named
   domains/boundaries via the `@` operator.
3. **`GmshTemplate`** (or `MeshTemplate`) subclass(es) — define the geometry of (multiple connected) domains.
4. A `if __name__=="__main__":` block that instantiates the `Problem` as a context
   manager and calls `.solve()` (stationary) or `.run(endtime, ...)` (transient).

```python
from pyoomph import *
from pyoomph.expressions import *  # var, var_and_test, weak, grad, div, partial_t, dot, ...

class MyEquation(Equations):                 # or ODEEquations for 0D systems
    def __init__(self, *, name="u", space="C2", source=0):
        super().__init__()
        self.name, self.space, self.source = name, space, source

    def define_fields(self):
        self.define_scalar_field(self.name, self.space)   # or define_vector_field/define_tensor_field

    def define_residuals(self):
        u, v = var_and_test(self.name)
        self.add_residual(weak(grad(u), grad(v)) - weak(self.source, v))

class MyProblem(Problem):
    def define_problem(self):
        self.add_mesh(LineMesh(minimum=-1, size=2, N=100))   # domain "domain", boundaries "left"/"right"
        eqs = MyEquation(source=1)
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        eqs += TextFileOutput()
        self.add_equations(eqs @ "domain")

if __name__ == "__main__":
    with MyProblem() as problem:
        problem.solve()      # stationary; use problem.run(endtime, outstep=...) for transient
        problem.output()
```

Every custom `Equations`/`ODEEquations` subclass overrides two hook methods:
- `define_fields(self)` — declare unknowns.
- `define_residuals(self)` — build the weak-form residual(s) and call `add_residual`.

Equations objects **compose with `+`** (e.g. physics `+` boundary conditions `+`
initial conditions `+` outputs), and are **bound to a mesh domain or boundary name
with `@`** (e.g. `eqs @ "domain"`, `DirichletBC(u=0) @ "left"`). The combined tree is
passed to `Problem.add_equations(...)`.

## `Equations` / `ODEEquations` API (`pyoomph/generic/codegen.py`)

Class hierarchy: `BaseEquations` → `Equations` (has a spatial domain) and
`ODEEquations` (0D, no mesh — internally backed by an `ODEStorageMesh`). `InterfaceEquations`
(subclass of `Equations`) lives on a boundary/interface of a bulk domain and can
declare `required_parent_type` / `required_opposite_parent_type` to assert which bulk
equations must be present. `eqA + eqB` produces a `CombinedEquations`. `eqs @ "name"`
produces an `EquationTree`, the thing passed to `Problem.add_equations`.
The domain and boundary names of the `EquationTree` must match the hierachical topological structure of the bulk/boundary names of the `GmshTemplate` or `MeshTemplate`.

Hooks to override for `Equations` (all optional, default `pass`):
- `define_fields(self)` — call `define_scalar_field`/`define_vector_field` or `define_ode_variable` (on `ODEEquations`).
- `define_residuals(self)` — call `add_residual`/`add_weak` any number of times.

Field declaration (on `Equations`):
- `define_scalar_field(name, space, scale=None, testscale=None)`
- `define_vector_field(name, space, dim=None, scale=None, testscale=None)`
- `space` is one of `"C1"`, `"C1TB"`, `"C2"`, `"C2TB"`, `"D1"`, `"D1TB"`, `"D2"`,
  `"D2TB"`, `"DL"`, `"D0"` (`C*` = continuous Lagrange, `D*` = discontinuous/DG,
  `*TB` = bubble-enriched on triangles/tetrahedrals, `DL` = discontinuous affine linear, `D0` = elementwise constant).
- On `ODEEquations`: `define_ode_variable(*names, scale=None, testscale=None)`.

Residual assembly (on `BaseEquations`):
- `add_residual(expr)` — the fundamental method: add an already-built weak-form
  expression (must include the test function, e.g. via `weak(...)` or
  `expr*testfunction(name)` for ODEs) to the residual vector.
- `add_weak(a, b, *, lagrangian=False, coordinate_system=None)` — shorthand for
  `add_residual(weak(a, b))`.
- Programmatic (rarely used directly — normally use the `InitialCondition`/
  `DirichletBC` equation classes instead): `set_initial_condition(field, expr)`,
  `set_Dirichlet_condition(field, expr)`.

## Symbolic expression API (`from pyoomph.expressions import *`)

| Function | Purpose |
|---|---|
| `var(name)` | Bind a field or special variable: `"time"`, `"coordinate"`, `"coordinate_x"/"_y"/"_z"`, `"mesh"`, `"lagrangian"`, `"normal"`, `"velocity"`, or any user-defined field name. |
| `var_and_test(name)` | `(var(name), testfunction(name))` in one call — the standard way to start `define_residuals`. |
| `testfunction(name)` | The FE test function of a field. |
| `weak(a, b)` | `∫_Ω a·b dΩ` — the fundamental weak-form bilinear pairing; `b` is (a derivative of) a test function. |
| `grad(f)` / `div(f)` | Gradient / divergence; automatically becomes the *surface* gradient/divergence on interface (codimension>0) domains. |
| `partial_t(f, order=1, ALE="auto")` | Time derivative `∂ₜⁿf`; `ALE=True/"auto"` corrects for mesh motion on moving meshes. |
| `dot(a, b)` | Vector dot product. |
| `contract(a, b)` | Generalized dot/Frobenius/matrix-vector product depending on tensor rank. |
| `vector([...])` | Build a vector-valued expression, e.g. `vector([-var("coordinate_y"), var("coordinate_x")])`. |
| `nondim(name)` | Nondimensional version of `var(name)`. |
| `evaluate_in_past(expr)` | Value of `expr` at the previous timestep (e.g. for error estimators). |
| `time_scheme("BDF1"/"BDF2"/"TPZ"/"MPT", expr)` | Wrap a residual term to use a specific time-stepping scheme for that term. |
| Elementary math | `sin`, `cos`, `exp`, `log`, `sqrt`, `pi`, `minimum`/`maximum`, etc. — behave like GiNaC symbolic functions, usable directly on `var(...)` expressions. |

**Units** (`from pyoomph.expressions.units import *` or included in `from pyoomph import *`):
base units `meter`, `second`, `kilogram`, `kelvin`, `mol`, `ampere`; SI prefixes
(`milli`, `micro`, `nano`, `kilo`, ...); derived units like `mm`, `minute`, `hour`,
`pascal`, `mmHg`. Multiply numeric literals by these to get dimensional
`Expression`s, e.g. `5*milli*meter` or `0.1*mm/second`. Use `problem.set_scaling(...)`
to define the nondimensionalization (e.g. `set_scaling(spatial=1*mm, temporal=1*second)`).

**Coordinate systems** (`pyoomph/expressions/coordsys.py`): `cartesian` (default),
`axisymmetric`, `axisymmetric_flipped`, `radialsymmetric`, or a custom
`BaseCoordinateSystem`. Set via `Problem.set_coordinate_system(...)` or pass
`coordsys=` to a moving-mesh equation.

## `Problem` API (`pyoomph/generic/problem.py`)

Override `define_problem(self)` to build the problem (mesh + equations). Key methods:

| Method | Purpose |
|---|---|
| `add_mesh(mesh_template)` | Register a `MeshTemplate` (or ready-made mesh, see below); only valid inside `define_problem`. |
| `add_equations(eqs_at_domain)` | Attach an `EquationTree` (`equations @ "domain_name"`); only valid inside `define_problem`. |
| `get_equations(path)` / `get_mesh(name)` | Retrieve previously added equations/mesh by name. |
| `solve(*, spatial_adapt=0, timestep=None, temporal_error=None, ...)` | Stationary Newton solve if `timestep=None`; otherwise a single transient step. |
| `run(endtime, timestep=None, *, outstep=None, numouts=None, spatial_adapt=0, temporal_error=None, maxstep=None, startstep=None)` | Time-march to `endtime`, calling `output()` periodically. `outstep=True`/a float sets fixed output interval; `numouts=N` splits `[0,endtime]` into N outputs; `temporal_error=<tol>` enables adaptive time-stepping; `spatial_adapt=1` enables mesh adaptivity each step. |
| `output(stage="")` | Invoke all attached output equations (writes files for the current state). |
| `set_initial_condition(...)` | Applies `InitialCondition` equations (called automatically on first `solve`/`run`). |
| `define_global_parameter(**params)` | Named continuation/ramp parameters usable inside expressions, e.g. `self.Re = self.define_global_parameter(Re=1.0)`. |
| `set_scaling(**kwargs)` | Nondimensionalization: `temporal=`, `spatial=`, or `fieldname=<scale>`. |
| `set_coordinate_system(csys)` | `"axisymmetric"`, `"radialsymmetric"`, etc. |
| `go_to_param(**kwargs)` | Pseudo-arclength continuation until a named global parameter reaches a target, e.g. `go_to_param(Re=100)`. |
| `activate_bifurcation_tracking`, `find_bifurcation_via_eigenvalues`, `solve_eigenproblem` | Stability/bifurcation analysis on the same residuals. |
| `set_output_directory(name)` | Override the default output directory (defaults to the script's filename minus `.py`). |

`Problem` is a context manager — always wrap usage as `with MyProblem() as problem:`
so compiled code/mesh resources are released on exit. `problem += x` is shorthand for
adding a mesh/equations/plotter, mirroring `+=` composition of equations.

## Meshes (`pyoomph/meshes/simplemeshes.py`, `pyoomph/meshes/gmsh.py`)

Ready-made templates (pass to `Problem.add_mesh(...)`):

| Class | Key kwargs | Domain/boundary names |
|---|---|---|
| `LineMesh(N=10, size=1.0, minimum=0.0)` | 1D interval | domain `"domain"`, boundaries `"left"`/`"right"` |
| `RectangularQuadMesh(size=1.0, N=10, lower_left=[0,0], split_in_tris=False)` | 2D rectangle | boundaries `"left"/"right"/"top"/"bottom"` |
| `CircularMesh(...)` | 2D disk | |
| `CuboidBrickMesh(...)` | 3D brick | |
| `CylinderMesh(...)`, `SphericalOctantMesh(...)` | 3D | |
| `PointMesh(...)` | 0D single point (host for ODE-like equations that still want a spatial "location") | |

For unstructured/complex 2D/3D geometry, subclass `GmshTemplate` and override
`define_geometry()` using pygmsh-style primitives (`point`, `line`, `spline`,
`circle_arc`, `plane_surface`, ...).

Custom meshes: subclass `MeshTemplate`, override `define_geometry()`, use
`new_domain(name)`, `add_node_unique(...)`, `add_facet_to_boundary(name, [...])`.

Domain/boundary names defined by the mesh are exactly the strings used with `@`.
Interfaces between two domains (or `eqs @ "boundary_name"`) become their own
`InterfaceMesh` automatically.

## Generic building-block equations

In `pyoomph/meshes/bcs.py` (boundary conditions — use with `@"boundary_name"`):

| Class | Purpose |
|---|---|
| `DirichletBC(**fields)` | Strong Dirichlet condition, e.g. `DirichletBC(u=0, v=1)@"left"`. |
| `NeumannBC(**fluxes)` | Natural/flux BC matching the bulk equation's integration-by-parts choice. |
| `EnforcedBC(**constraints)` | Arbitrary constraint enforced via a Lagrange multiplier, e.g. `EnforcedBC(u=var("u")-var("v"))` adjusting `u` to match `u=v` |
| `EnforcedDirichlet(**fields)` | Dirichlet enforced weakly (Lagrange multiplier) instead of strong pinning. |
| `PeriodicBC(...)` | Periodic boundary matching. |
| `PythonDirichletBC`, `PinWhere`, `UnpinDofs` | Programmatic/conditional dof pinning. |

In `pyoomph/equations/generic.py`:

| Class | Purpose |
|---|---|
| `InitialCondition(**fields)` | Set initial values per field, e.g. `InitialCondition(u=bump_expr)`. |
| `SpatialErrorEstimator(*fluxes, **fields)` | Drives h-adaptivity from jumps of `grad(field)` (or custom flux expressions) across elements. |
| `RefineToLevel`/`RefineToMaxLevel`/`RefineMaxElementSize`/`RefineAccordingToElement` | Mesh-refinement controls. |
| `RemeshWhen(...)` | Trigger automatic 2D remeshing on mesh distortion. |
| `ProjectExpression(**projs)` | L2-project an arbitrary expression onto a field, for output/diagnostics. |
| `TemporalErrorEstimator(**fieldfactors)` | Drives adaptive time-stepping (used with `run(..., temporal_error=...)`). |
| `IntegralObservables(**exprs)` / `ExtremumObservables(...)` | Track domain integrals / min-max of fields over time, written to output. |
| `IntegralConstraint`/`AverageConstraint` | Enforce an integral/average constraint via a Lagrange multiplier (e.g. fixed total mass). |
| `ConnectFieldsAtInterface(fields)` | Couple fields of two domains meeting at an interface. |
| `LocalExpressions(**exprs)` | Named auxiliary expressions available to sibling/child equations. |

Outputs (`pyoomph/output/*.py`, add via `+=`, written on `problem.output()`):
`TextFileOutput()` (plain text dump of nodal values), `MeshFileOutput()` (VTU/mesh
file for ParaView), `ODEFileOutput()` (for `ODEEquations` domains), 
`IntegralObservableOutput()`. `pyoomph.output.plotting.MatplotlibPlotter` can be
attached to `problem.plotter` for built-in matplotlib-based 2D field plots.

## Built-in physics equation libraries (`pyoomph/equations/*.py`)

These are ready-to-use `Equations`/`ODEEquations` classes for common physics — prefer
them over hand-rolled weak forms when the physics matches. All live under
`pyoomph.equations.*` and are imported explicitly, e.g.
`from pyoomph.equations.navier_stokes import NavierStokesEquations`.

- **`poisson.py`**: `PoissonEquation(name="u", source=None, coefficient=1)` —
  `-div(coeff*grad(u))=f`, supports continuous and DG spaces. `DiffusionEquation`
  adds a time derivative to get `∂t u - div(D grad(u)) = f`. `PoissonFlux` (Neumann),
  `PoissonFarFieldMonopoleCondition` (unbounded domains).
- **`advection_diffusion.py`**: `AdvectionDiffusionEquations(fieldnames="advdiffu", diffusivity=1, wind=var("velocity"), source=...)`
  for scalar transport; `AdvectionDiffusionFluxInterface`, `AdvectionDiffusionInfinity`.
- **`navier_stokes.py`**: `StokesEquations(dynamic_viscosity=1, mode="TH"|"CR"|"SV"|"mini"|...)`
  and `NavierStokesEquations(...)` (adds inertia) — the main flow-solver classes.
  Interface equations: `NavierStokesFreeSurface(surface_tension=1, ...)` (free
  surface with surface tension/curvature), `NavierStokesContactAngle(contact_angle=90*degree)`,
  `NoSlipBC` (`DirichletBC` subclass), `NavierStokesSlipLength`,
  `NavierStokesPrescribedNormalVelocity`, `ConnectVelocityAtInterface`.
- **`ALE.py`**: mesh-motion equations for free-boundary/moving-mesh problems —
  `PseudoElasticMesh(E=..., nu=...)`, `LaplaceSmoothedMesh(...)`,
  `HyperelasticSmoothedMesh(...)`, `YeohSmoothedMesh(...)`; helpers
  `ConnectMeshAtInterface`, `PrescribedMovingMesh(umesh=...)`,
  `EnforceVolumeByPressure(volume=...)` (fix an enclosed volume via internal pressure).
- **`solid.py`**: `DeformableSolidEquations(constitutive_law=...)` with pluggable
  `GeneralizedHookeanSolidConstitutiveLaw(E=, nu=)` or
  `IncompressibleHookeanSolidConstitutiveLaw(E=)`; `LinearElasticitySolidEquations`
  for small-strain linear elasticity; `SolidTraction`/`SolidNormalTraction`;
  `FSIConnection` couples a solid domain to a `StokesEquations`/`NavierStokesEquations`
  fluid domain (fluid-structure interaction).
- **`cahn_hilliard.py`** / **`NSCH.py`** / **`low_order_NSCH.py`**: phase-field
  two-phase-flow models (Cahn-Hilliard + Navier-Stokes). `SimpleNSCH(fluid_plus, fluid_minus, sigma, epsilon, mobility)`
  and `MaterialBasedLowOrderNSCH(fluidA, fluidB, epsilon, mobility)` are the
  "batteries-included" entry points; `CahnHilliardWettingInterface`/
  `LowOrderNSCHWetting` add contact-angle wetting BCs.
- **`multi_component.py`**: multi-species/multi-phase transport built on material
  property objects (see Materials below). `CompositionFlowEquations(fluid_props, ...)`
  is the main "batteries-included" assembler (Navier-Stokes + species transport +
  optional temperature). `MultiComponentNavierStokesInterface(interface_props, ...)`
  is the main free-interface class with mass transfer/Marangoni/surfactants.
  `TemperatureConductionEquation`/`TemperatureAdvectionConductionEquation` for heat.
- **`contact_angle.py`**: dynamic contact-line models plugged into
  `DynamicContactLineEquations(model=..., wall_normal=...)`, e.g.
  `PinnedContactLine()`, `UnpinnedContactLine(theta_eq=..., cl_speed_exponent=1)`
  (Cox-Voinov for exponent 3), `YoungDupreContactLine(...)`, `WenzelContactLine(...)`,
  `CassieBaxterContactLine(...)`.
- **`lubrication.py`**: `LubricationEquations(mu=, sigma=, disjoining_pressure=...)`
  for thin-film/lubrication-theory flows (film height + pressure).
- **`darcy.py`**: `DarcyEquation(fluid_props, permeability=, porosity=)` for porous-media flow.
- **`helmholtz.py`**: `HelmholtzEquation(k=, complex=False)` — `Δu+k²u=0`, e.g. for
  acoustics/wave problems in frequency domain.
- **`kuramoto_sivashinsky.py`**: `KuramotoSivashinskyEquations(...)` for thin-film
  interfacial pattern formation.
- **`potential_flow.py`**: `PotentialFlow(potential_name="phi", ...)` for irrotational
  flow (`Δφ=0`, `u=∇φ`), with `PotentialFlowFreeInterface(...)` for free-surface
  potential flow (e.g. capillary waves, bubble oscillation).
- **`stokes_stream_func.py`**: `StreamFunctionFromVelocity(...)` — post-processing
  stream function from a computed velocity field (2D/axisymmetric).
- **`harmonic_oscillator.py`**: `HarmonicOscillator(omega=, damping=, driving=)`, an
  `ODEEquations` example/utility for a damped/driven oscillator.

Many of these physics modules are heavily parametrized — when writing a script, prefer
grepping the actual class in the corresponding file for the full constructor signature
and docstring rather than relying purely on the one-liners above.

## Materials (`pyoomph/materials/`)

Multi-component/multi-phase equations (`multi_component.py`, `NSCH.py`, `darcy.py`,
`contact_angle.py`) take fluid/interface property objects (`AnyFluidProperties`,
`AnyFluidFluidInterface`, ...) rather than raw numbers. See
`pyoomph.materials.default_materials` for predefined materials and
`pyoomph.materials.generic` for the base classes to define custom ones. Surfactant
adsorption isotherms live in `pyoomph.materials.surfactant_isotherms`; UNIFAC/AIOMFAC
activity-coefficient models in `pyoomph.materials.UNIFAC.*`/`activity.py`.

## Conventions observed across the official tutorial examples

- Always `from pyoomph import *` then `from pyoomph.expressions import *` at the top.
- Custom equation classes take their physical parameters as constructor kwargs and
  store them as `self.xxx`, so they stay reusable across problems.
- `Problem.__init__` sets default parameter values as `self.xxx = ...`;
  `define_problem` does the actual mesh/equation assembly and may reference `self.xxx`.
- In `GmshTemplate.define_geometry`, the parameters of the `Problem` can be accessed using the `get_problem()` method, 
  usually with a `cast` to the actually expected `Problem` subclas.
- Output directory defaults to the script's filename without `.py`; override with
  `problem.set_output_directory(...)`.
- Some tutorial variants (e.g. "adaptive", "axisymmetric" versions) are not
  standalone — they `from base_script import *` and only override a couple of
  settings. Don't assume every example file runs in isolation; check its imports.
- End every script with:
  ```python
  if __name__ == "__main__":
      with MyProblem() as problem:
          problem.solve(); problem.output()          # stationary
          # or: problem.run(endtime, outstep=..., numouts=..., temporal_error=..., spatial_adapt=...)
  ```

## Where to look for more (in this repo)

- `docs/source/tutorial/` — the full human tutorial (rst + literalinclude'd `.py`
  example scripts); `temporal/` starts with ODEs (simplest), `spatial/` covers
  stationary PDEs, `pde/` covers spatio-temporal PDEs, `ale/` covers moving
  meshes/free surfaces, `multidom/` covers multiple coupled domains, `mcflow/`
  covers the multi-component/materials equations, `dg/` covers Discontinuous
  Galerkin, `advstab/` covers stability/bifurcation analysis, `math.rst`/`math/`
  lists all built-in math functions and keyword variables (`var("...")` names).
- `pyoomph/generic/codegen.py`, `pyoomph/generic/problem.py` — the core `Equations`/
  `Problem` machinery (source of truth for exact method signatures).
- `pyoomph/equations/*.py` — all built-in physics; read the target file directly for
  full constructor signatures/docstrings before writing code against it.
- Rendered API docs: https://pyoomph.readthedocs.io/en/latest/tutorial.html
