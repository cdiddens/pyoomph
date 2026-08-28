# pyoomph — reference for AI coding assistants

This file is a condensed, code-verified reference for AI assistants helping a user
write **simulation scripts** with pyoomph (i.e. using the installed `pyoomph` package
as a library — this is *not* about hacking on the pyoomph framework's own C++/Python
source). Point an AI assistant at this file before asking it to write a pyoomph
problem script.

**Reading this file alone should be enough to write a complete, running pyoomph script**
for a custom problem: custom equations, custom mesh and geometry, dimensional units,
boundary and initial conditions, output, adaptivity, remeshing and parallel execution.
Deeper material lives in the [`agents/`](agents/) subdirectory (see
[Companion reference files](#companion-reference-files) at the end) — go there when the
task actually touches its subject, not before.

For working on pyoomph's *own* source, see `CLAUDE.md` and `dev_docs/` instead.

pyoomph is a Python front-end to the C++ library `oomph-lib`. You describe a PDE (or
ODE) system as a **weak form built from symbolic expressions**; pyoomph
differentiates it symbolically (Jacobian, parameter derivatives, Hessians — including
w.r.t. moving-mesh coordinates), generates C code, compiles it on the fly, and links
it back into the running script. Users normally never touch C++ directly —
everything is expressed through the Python API below. 
All equations are solved together in a fully implicit, monolithic manner.

## When helping users with pyoomph:

1. Prefer existing equation classes over implementing weak forms manually.
2. Only derive weak forms manually when no built-in equation exists.
3. Keep custom Equations reusable and mesh-independent.
4. Use symbolic expressions exclusively.
5. Use dimensional quantities unless the user explicitly asks for nondimensional equations.
6. Follow the conventions used throughout the tutorial.
7. Do not guess constructor arguments — read the class in `pyoomph/equations/*.py`.
8. Write one script that runs serially and in parallel; never a separate "MPI version".

Avoid reimplementing:

- Navier-Stokes
- Stokes
- Poisson
- Advection-diffusion
- ALE
- Lubrication

unless the user specifically requests a custom weak formulation.

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
from pyoomph.expressions.units import * # units for dimensions

class MyEquation(Equations):                 # or ODEEquations for 0D systems
    def __init__(self, *, name="T", space="C2", source=0, alpha=1*(milli*meter)**2/second):
        super().__init__()
        self.name, self.space, self.source = name, space, source
        self.alpha = alpha

    def define_fields(self):
        self.define_scalar_field(self.name, self.space, testscale=scale_factor("temporal")/scale_factor("T"))   # or define_vector_field/define_tensor_field

    def define_residuals(self):
        T, Ttest = var_and_test(self.name)
        self.add_residual(weak(partial_t(T),Ttest)+weak(self.alpha*grad(T), grad(Ttest)) - weak(self.source, Ttest))

class MyProblem(Problem):
    def __init__(self):
        super().__init__()
        self.alpha=1*(milli*meter)**2/second
        self.L=1*milli*meter
        self.T0=20*celsius
        
    def define_problem(self):
        self.set_scaling(spatial=self.L,temporal=self.L**2/self.alpha,T=self.T0)
        self.add_mesh(LineMesh(minimum=-self.L/2, size=self.L, N=100, name="domain"))   # boundaries "left"/"right"
        eqs = MyEquation(source=1*kelvin/second*(var("coordinate_x")/self.L)**2)
        eqs += InitialCondition(T=self.T0)
        eqs += DirichletBC(T=self.T0) @ ["left","right"]        
        eqs += TextFileOutput()
        self.add_equations(eqs @ "domain")

if __name__ == "__main__":
    with MyProblem() as problem:
        problem.run(10*second,outstep=1*second)      # use problem.solve() for stationary
        problem.output()
```

Every custom `Equations`/`ODEEquations` subclass overrides two hook methods:
- `define_fields(self)` — declare unknowns.
- `define_residuals(self)` — build the weak-form residual(s) and call `add_residual`.

Equations objects **compose with `+`** (e.g. physics `+` boundary conditions `+`
initial conditions `+` outputs), and are **bound to a mesh domain or boundary name
with `@`** (e.g. `eqs @ "domain"`, `DirichletBC(u=0) @ "left"`). The combined tree is
passed to `Problem.add_equations(...)`.

```python
	def define_problem(self):
		eqs = NavierStokesEquations(...)
		eqs += LaplaceSmoothedMesh(...)
		eqs += InitialCondition(...)
		eqs += DirichletBC(...)
		eqs += MeshFileOutput()

		self.add_equations(eqs @ "fluid")
```

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
- `add_weak(a, b, *, lagrangian=False, coordsys=None)` — shorthand for
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
| `subexpression(expr)` | Compute a shared sub-expression once in the generated C and reuse it (also caches its derivatives). Wrap any moderately expensive repeated term. |
| Elementary math | `sin`, `cos`, `exp`, `log`, `square_root` (**not** `sqrt`), `absolute`, `signum`, `heaviside`, `pi`, `minimum`/`maximum`, `rational_num(n,d)`. |
| Tensor helpers | `identity_matrix`, `transpose`, `trace`, `determinant`, `inverse_matrix`, `matproduct`. |

## Units and nondimensionalization

`from pyoomph.expressions.units import *` — a **separate import**: units are not in
`from pyoomph import *` nor in `from pyoomph.expressions import *`. Base units `meter`,
`second`, `kilogram`, `kelvin`, `mol`, `ampere`; SI prefixes (`milli`, `micro`, `nano`,
`kilo`, ...); derived units `newton`, `pascal`, `joule`, `watt`, `volt`, `coulomb`,
`farad`, `gram`, `liter`, `minute`, `hour`, `atm`, `mmHg`, `celsius`, `degree`,
`percent`. Multiply numeric literals by these to get dimensional `Expression`s, e.g.
`5*milli*meter` or `0.1*milli*meter/second`. There is **no `mm`/`cm`/`km` shorthand** —
compose them from a prefix and a base unit.

**Write the physics dimensionally and let pyoomph nondimensionalize it.** This is the
default and strongly preferred way — only go nondimensional when the user asks for it.
`problem.set_scaling(...)` declares the scale of each quantity:

```python
self.set_scaling(spatial=1*milli*meter, temporal=1*second, T=1*kelvin,
                 velocity=1*milli*meter/second)
```

Every field then has a scale (`scale_factor("T")`) and a test-function scale
(`test_scale_factor("T")`). Inside a reusable `Equations` class, express `scale`/`testscale`
through `scale_factor(...)` rather than hard-coded numbers, so the class stays usable at
any set of scales — the idiom is
`testscale=scale_factor("temporal")/scale_factor("T")`, which makes the residual
dimensionless. `nondim(name)` gives the nondimensional counterpart of `var(name)`.

### The one rule that governs every residual

**`weak(a, b)` integrates over the *nondimensionalized* domain.** The measure `dΩ` carries
no units by default (`weak(..., dimensional_dx=True)` opts into the physical `m^d`), so
every residual contribution must satisfy

> `a * b` is dimensionless — nothing else.

`b` is a test function, whose scale you choose with `testscale=`. So the recipe is: pick
`testscale` as whatever makes `a*b` dimensionless. For a heat equation whose leading term
is `weak(partial_t(T), Ttest)`, that is `testscale=scale_factor("temporal")/scale_factor("T")`
— exactly the skeleton at the top of this file.

Two consequences that catch people out, because they make units of a *coefficient* differ
from the textbook strong form:

- A gradient pairing `weak(coeff*grad(u), grad(utest))` with `testscale=1/scale_factor(u)`
  contributes `coeff / L²`. **So `coeff` must carry `length²`**, not the diffusivity's usual
  units. This is why `PoissonEquation(coefficient=1*meter**2)` is right and
  `coefficient=1` is a hard error in a dimensional problem.
- A source pairing `weak(f, utest)` with the same testscale contributes `f / u`. **So the
  source carries the units of `u` itself**, not of `u/length²`.

Both terms have to be dimensionless *individually*, so with that testscale the two are
pinned: the equation actually solved is `-coefficient·Δu = source` with `coefficient` an
**area** and `source` in the units of `u`. A problem posed as `-D·Δu = f` with a physical
diffusivity (say `D` in m²/s and `f` in K/s) is the same equation multiplied through by a
constant — multiply both sides by whatever turns `D` into an area, here one second, and
pass `coefficient=D*second`, `source=f*second`.

A stationary problem needs no `temporal=` scale: only terms that actually appear in the
residual are checked, and `scale_factor("temporal")` is referenced only by equations that
have a time derivative.

If you get it wrong, pyoomph refuses at setup with

```
The added residual contribution is not dimensionless.
It still carries the base unit: meter
All terms agree on the unit meter^(-2), i.e. it is consistent with itself but not dimensionless.
```

and then prints every scale in play plus the offending term, expanded. Read that list: it
names the field, the scale and the test scale it used, so the fix is usually one factor.
A residual that is *self-consistent but not dimensionless*, as above, means a coefficient
is short exactly that power of length. Note that a wrong-but-consistent choice can also
slip through when a scale absorbs it — a unit that comes out as `kelvin/meter**2` where you
expected `kelvin` means the source or coefficient units were off even though it ran.

**Pitfalls that bite hard and surface far from the cause:**

- **Non-integer exponents must be exact rationals, never Python floats.** Write
  `x**rational_num(19,20)`, not `x**0.95`. GiNaC's unit handling gives up on a float
  exponent applied to a quantity that still carries units, and the error shows up
  somewhere else entirely. Put the decimal in a trailing comment. The same applies to
  `**rational_num(1,2)` for a square root of a dimensional quantity.
- **A missing `set_scaling` entry silently means a scale of 1** in the corresponding SI
  base unit, which for e.g. a micrometre-scale problem gives a badly conditioned system
  rather than an error. Set a scale for every field whose natural magnitude is far from 1.
- **Don't mix dimensional and nondimensional expressions.** Any quantity fed into a
  substituted field or a material property must be consistently one or the other; a bare
  float where an `Expression` with units is expected is a dimension error waiting to
  happen.
- `problem.get_scaling(...)` and multiplying/dividing by a unit are how you convert a
  computed nondimensional number back to a physical one for output.

## Coordinate systems

`Problem.set_coordinate_system(...)` takes either the **string** `"axisymmetric"`,
`"axisymmetric_flipped"`, `"radialsymmetric"`, `"cartesian"`, or the ready-made
**instance** of the same name. The instances live in `pyoomph.expressions` (so
`from pyoomph.expressions import *` has them) — the *classes* they are built from are in
`pyoomph/expressions/coordsys.py`, which is where a custom `BaseCoordinateSystem`
subclass goes. `coordsys=` on an individual equation or on `weak(...)` overrides it
locally.

**Axis convention for `axisymmetric`: `coordinate_x` is the radius r and `coordinate_y`
is the axial coordinate z.** So on a `RectangularQuadMesh` the symmetry axis r=0 is the
`"left"` boundary and the outer radius is `"right"`. `axisymmetric_flipped` swaps them
(x becomes the symmetry axis). The r-weight of the metric makes the symmetry axis a
natural boundary for a scalar field — **do not pin anything there**. `AxisymmetryBC` is
for vector fields (and for the m-dependent conditions of azimuthal stability), not for a
plain scalar.

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
| `define_global_parameter(**params)` | Named continuation/ramp parameters usable inside expressions, e.g. `self.Re = self.define_global_parameter(Re=1.0)`. Their value is always a **plain dimensionless number** (`param.value = 2.5`, `go_to_param(Re=100)`); to give one a dimension, multiply by a unit where it is used, e.g. `DirichletBC(velocity_x=self.U_lid*(milli*meter/second))`. That works inside residuals and Dirichlet values and stays differentiable for continuation. |
| `set_scaling(**kwargs)` | Nondimensionalization: `temporal=`, `spatial=`, or `fieldname=<scale>`. |
| `set_coordinate_system(coordsys)` | `"axisymmetric"`, `"radialsymmetric"`, etc. |
| `go_to_param(**kwargs)` | Pseudo-arclength continuation until a named global parameter reaches a target, e.g. `go_to_param(Re=100)`. |
| `activate_bifurcation_tracking`, `find_bifurcation_via_eigenvalues`, `solve_eigenproblem` | Stability/bifurcation analysis on the same residuals. |
| `set_output_directory(name)` | Override the default output directory (defaults to the script's filename minus `.py`). |
| `set_linear_solver(name)` | `"pardiso"`, `"superlu"`, `"umfpack"`, `"petsc"`, `"petsc_mumps"`, `"accelerate"`. Leave it alone unless there is a reason — see [`agents/parallel.md`](agents/parallel.md). |
| `set_num_threads(n)` | OpenMP threads for assembly and linear solver; same as `--omp n`. |
| `save_state(fn)` / `load_state(fn)` | Dump/restore the full simulation state for resuming. |

`Problem` is a context manager — always wrap usage as `with MyProblem() as problem:`
so compiled code/mesh resources are released on exit. `problem += x` is shorthand for
adding a mesh/equations/plotter, mirroring `+=` composition of equations.

## Meshes (`pyoomph/meshes/simplemeshes.py`, `pyoomph/meshes/gmsh.py`)

Ready-made templates (pass to `Problem.add_mesh(...)`). `size`, `N` and `lower_left`
accept either a scalar (same in every direction) or a per-axis list, and every template
takes `name=` to set its domain name (default `"domain"`), e.g.
`RectangularQuadMesh(size=[R, H], N=[20, 40], lower_left=[0, 0], name="domain")`.

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

In `pyoomph/equations/generic.py` (boundary conditions — use with `@"boundary_name"`; they used to
live in `pyoomph/meshes/bcs.py`, which no longer exists):

| Class | Purpose |
|---|---|
| `DirichletBC(**fields)` | Strong Dirichlet condition, e.g. `DirichletBC(u=0, v=1)@"left"`. |
| `NeumannBC(**fluxes)` | Natural/flux BC matching the bulk equation's integration-by-parts choice. |
| `EnforcedBC(**constraints)` | Arbitrary constraint enforced via a Lagrange multiplier, e.g. `EnforcedBC(u=var("u")-var("v"))` adjusting `u` to match `u=v` |
| `EnforcedDirichlet(**fields)` | Dirichlet enforced weakly (Lagrange multiplier) instead of strong pinning. |
| `PeriodicBC(...)` | Periodic boundary matching. |
| `PythonDirichletBC`, `PinWhere`, `UnpinDofs` | Programmatic/conditional dof pinning. |
| `AxisymmetryBC` | The r=0 conditions, including the m-dependent ones for azimuthal stability. |

**Where two Dirichlet conditions meet, the one added last wins.** A corner node belongs
to both adjacent boundaries, so in a lid-driven cavity

```python
eqs += DirichletBC(velocity_x=U_lid, velocity_y=0) @ "top"
eqs += NoSlipBC() @ ["left", "right", "bottom"]     # added last -> corners are u=0
```
gives the *non-leaky* lid (corner velocity 0), while swapping the two lines gives the
*leaky* lid (corner velocity `U_lid`). Both are well posed and they are different
problems, so make the order deliberate rather than incidental.

`InactiveDirichletBC`, `AxisymmetryBCForScalarD0Field`, `PinMeshAtDistanceToInterface` and
`InteriorBoundaryOrientation` are the rarer ones and live in `pyoomph/equations/additional.py`,
which is not pulled in by `from pyoomph import *` - import it explicitly.

In `pyoomph/equations/generic.py`:

| Class | Purpose |
|---|---|
| `InitialCondition(**fields)` | Set initial values per field, e.g. `InitialCondition(u=bump_expr)`. |
| `SpatialErrorEstimator(*fluxes, **fields)` | Drives h-adaptivity from jumps of `grad(field)` (or custom flux expressions) across elements. |
| `RefineToLevel` | Mesh-refinement control (`RefineMaxElementSize`/`RefineAccordingToElement` are in `pyoomph/equations/additional.py`). |
| `RemeshWhen(...)` | Trigger automatic 2D remeshing on mesh distortion. |
| `ProjectExpression(**projs)` | L2-project an arbitrary expression onto a field, for output/diagnostics. |
| `TemporalErrorEstimator(**fieldfactors)` | Drives adaptive time-stepping (used with `run(..., temporal_error=...)`). |
| `IntegralObservables(**exprs)` / `ExtremumObservables(...)` | Track domain integrals / min-max of fields over time, written to output. **Integrates over the physical domain** — see below. |
| `IntegralConstraint`/`AverageConstraint` | Enforce an integral/average constraint via a Lagrange multiplier (e.g. fixed total mass). |
| `ConnectFieldsAtInterface(fields)` | Couple fields of two domains meeting at an interface. |
| `LocalExpressions(**exprs)` | Named auxiliary expressions available to sibling/child equations. |

## Output and plotting

Output equations (`pyoomph/output/*.py`) are added to a domain with `+=` like any other
equation and are written whenever `problem.output()` runs:

| Class | Writes |
|---|---|
| `MeshFileOutput()` | VTU/mesh files for ParaView — the default choice for 2D/3D fields. |
| `TextFileOutput()` | Plain-text dump of the nodal values. Good for 1D and for plotting with anything. |
| `TextFileOutputAlongLine(start=, end=, N=)` | Values sampled along a line/curve through the mesh, independent of the nodes. |
| `GridFileOutput(lower, upper, N=/dx=)` | Interpolation onto a regular Cartesian grid. |
| `ODEFileOutput(first_column=...)` | For `ODEEquations` domains; `first_column=` writes e.g. a parameter instead of time. |
| `IntegralObservableOutput()` | The `IntegralObservables`/`ExtremumObservables` time series. |

For a scan or summary file not tied to the output steps, use
`problem.create_text_file_output(fname, header=[...])` and `.add_row(...)`.

**Observables** are read back with `problem.get_mesh(domain).evaluate_observable(name)`.
Unlike `weak(...)`, `IntegralObservables` integrates over the **physical, dimensional**
domain: `Area=1` evaluates to the real area *with units* (1e-4 m² for a 1 cm square), and
`evaluate_observable` returns a dimensional `Expression`, not a float. So

```python
eqs += IntegralObservables(Area=1, Usqr=dot(var("velocity"), var("velocity")),
                           Urms=lambda Usqr, Area: square_root(Usqr/Area))
...
Urms = float(problem.get_mesh("cavity").evaluate_observable("Urms")/(milli*meter/second))
```
Entries may reference earlier ones by name through a `lambda`, as `Urms` does here. Divide
by a unit (or by `problem.get_scaling(...)`) *before* calling `float()` — `float()` on a
value that still carries a unit raises a `RuntimeError` that names the leftover unit, and
`create_text_file_output(...).add_row(...)` calls `float()` on everything you hand it.

**Plotting during the run.** Subclass a plotter, override `define_plot(self)`, and assign
an instance to `problem.plotter`; it then renders at every `problem.output()`.

- `pyoomph.output.plotting.MatplotlibPlotter` — 2D (and axisymmetric) field plots.
  Inside `define_plot`: `self.set_view(xmin, ymin, xmax, ymax)`, `self.add_colorbar(...)`,
  then `self.add_plot("domain/field", colorbar=cb)` for a colour map, or with
  `mode="arrows"`/`"streamlines"` for a vector field. `"domain/boundary"` as the first
  argument draws the interface line (the interface needs at least one equation on it — an
  empty `InterfaceEquations()` is enough). `transform=["mirror_x", None]` plots mirrored
  and unmirrored copies side by side, the usual way to show an axisymmetric result.
  Also `add_arrow_key`, `add_text`, `add_time_label`, `add_scale_bar`.
- `pyoomph.output.plotting1d.MatplotlibPlotter1D` — 1D domains as ordinary x-y graphs
  (including the (x,y) curve of a 1D mesh embedded in 2D/3D).
- `pyoomph.output.plotting3d.PyVistaPlotter` — 3D rendering via PyVista.

Plots can be regenerated from written output without re-solving: `--runmode p`.

## Built-in physics equation libraries (`pyoomph/equations/*.py`)

These are ready-to-use `Equations`/`ODEEquations` classes for common physics — prefer
them over hand-rolled weak forms when the physics matches. All live under
`pyoomph.equations.*` and are imported explicitly, e.g.
`from pyoomph.equations.navier_stokes import NavierStokesEquations`.

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
  is the main free-interface class with mass transfer/Marangoni/surfactants.
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
  formulation="log-conf"|"conformation")` with pluggable constitutive models `OldroydB()`,
  `Giesekus(alpha=)`, `PTT(epsilon=, kind=)`, `FENE_CR(L=)`, `FENE_P(L=)`;
  `ViscoelasticInflowBC`. The log-conformation formulation is the default and is what keeps
  high Weissenberg numbers stable.
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
  [`agents/materials.md`](agents/materials.md) for the isotherms that feed it.
- **`salt_transport.py`**: `SaltTransportEquations(salts, fluid_props=, ...)` — dissolved
  salts as ion pairs (rather than independent species), with `FrozenSaltConcentrations`
  and `SaltConcentrationsFromMassFractions`.
- **`electrostatics.py`**: `ElectricPotentialEquations(permittivity=|relative_permittivity=,
  charge_density=, conductivity=)`; `PoissonBoltzmannEquations`/`DebyeHuckelEquations` for
  electric double layers; `NernstPlanckEquations(ions, ...)` for ion transport;
  `OhmicConductionEquations`. Boundary/interface classes: `ElectrodeBC(voltage)`,
  `SurfaceChargeBC`, `SurfaceChargeConservation`, `ElectricFarFieldCondition`,
  `ElectricPotentialConnection`, `ThinDielectricLayer`, `SternLayer`, `IonFluxBC`.
- **`electrohydrodynamics.py`**: couples the above to flow —
  `MaxwellStressEquations`/`ElectricBodyForceEquations` (bulk force, two equivalent
  formulations), `MaxwellStressInterface` (the jump at a dielectric interface),
  `ElectroosmoticSlip(zeta_potential=...)`.
- **`tracers.py`**: `TracerParticles(advection=var("velocity"), seed=...)` — massless
  tracer particles advected with the flow, for visualization or residence-time studies.
  Seeds: `TracerSeedPoints`, `TracerSeedGrid`, `TracerSeedRandom`, `TracerSeedElement`,
  `TracerSeedCallable`; `TracerTransferAtInterface`/`TracerTransferToInterface` move them
  between domains, `TracerPeriodicBoundaryCondition` wraps them around.
- **`topological_changes.py`**: automatic topology changes of a moving mesh —
  `AxisymmetricReconnection(rmin=, distmin=, volume_conservation=True)` pinches off or
  merges an axisymmetric interface when it gets too thin. Needs a
  `TopologicalChangesGmshTemplate`/`TopologicalChangesTQMeshTemplate` mesh (which can
  rebuild the domain after the change) rather than a plain `GmshTemplate`; also
  `DisjunctDomainMarker` to label the resulting separate pieces.

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
activity-coefficient models in `pyoomph.materials.UNIFAC.*`/`activity.py`. Electrolytes
are their own thing: `import pyoomph.materials.ions`, then
`water.add_salt("NaCl", 1*milli*molar)` — see [`agents/materials.md`](agents/materials.md).

## Moving meshes, free surfaces and remeshing

A moving (ALE) mesh is not a different kind of problem — it is one more equation on the
same domain. Add a mesh-motion equation from `pyoomph/equations/ALE.py` to the bulk, and
the nodal positions become unknowns like any other field:

```python
eqs  = NavierStokesEquations(dynamic_viscosity=mu, mass_density=rho)
eqs += LaplaceSmoothedMesh()                      # or PseudoElasticMesh(), HyperelasticSmoothedMesh()
eqs += NavierStokesFreeSurface(surface_tension=sigma) @ "top"   # the free surface itself
eqs += DirichletBC(mesh_y=0) @ "bottom"           # pin the mesh where it must not move
self.add_equations(eqs @ "liquid")
```

- The mesh position is the field `"mesh"` (`var("mesh")`, components `mesh_x`/`mesh_y`/
  `mesh_z`); `var("lagrangian")` is the undeformed reference position. Pin mesh components
  with an ordinary `DirichletBC(mesh_x=...)`; the value `True` means "pin at the current
  value", e.g. `DirichletBC(mesh_x=0, mesh_y=True) @ "left"` lets the left boundary slide
  vertically but not horizontally. A prescribed motion is just an expression:
  `DirichletBC(mesh_x=1+0.5*var("lagrangian")[1]*var("time")) @ "right"`.
- `partial_t(f)` defaults to `ALE="auto"`, i.e. it already subtracts the mesh velocity on a
  moving mesh. Do not hand-roll the convective correction.
- The free surface itself is an `InterfaceEquations` on a boundary
  (`NavierStokesFreeSurface`, `MultiComponentNavierStokesInterface`); contact lines are a
  further `@"boundary/corner"` interface (`NavierStokesContactAngle`,
  `DynamicContactLineEquations`).
- `ConnectMeshAtInterface` ties the meshes of two domains that share an interface;
  `EnforceVolumeByPressure(volume=...)` fixes an enclosed volume via an internal pressure.

**Remeshing (2D only).** When the mesh eventually distorts too far, pyoomph can rebuild it
from the current deformed boundary and interpolate the solution onto the new mesh. Two
things are needed, and forgetting the first is the usual failure:

```python
from pyoomph.meshes.remesher import Remesher2d   # RemeshWhen/RemeshingOptions/RemeshMeshSize
                                                 # come from pyoomph.equations.generic
mesh = MyGmshMeshTemplate()
mesh.remesher = Remesher2d(mesh)         # <- on the TEMPLATE, without this RemeshWhen does nothing
self.add_mesh(mesh)
...
eqs += RemeshWhen(RemeshingOptions(max_expansion=2, min_expansion=0.3, min_quality_decrease=0.2))
eqs += RemeshMeshSize(size=0.02) @ "right/top"   # target element size at that corner
```

`RemeshWhen` also takes its thresholds directly as kwargs, plus
`on_invalid_triangulation=True`/`on_inverted_element=True` to remesh as an emergency
measure. Remeshing is *not* the same as h-adaptivity: for adaptivity use
`SpatialErrorEstimator(...)` plus `solve(spatial_adapt=n)`/`run(..., spatial_adapt=n)`, and
`RefineToLevel`/`RefineMaxElementSize` to constrain it. The two combine.

See [`agents/advanced.md`](agents/advanced.md) §4 for how `activate_coordinates_as_dofs`
works underneath and for the interpolation guarantees across a remesh, and
[`agents/examples.md`](agents/examples.md) recipe 3 for a complete free-surface script.

## Parallelization

**Do not write a "parallel version" of a script.** Parallelization is chosen at launch
time; the same `Problem`, `Equations` and weak forms run serially, threaded and under
`mpirun`.

```bash
python3 script.py --omp 4                          # OpenMP threads, bit-identical to serial
mpirun -n 4 python3 script.py                      # MPI, every rank holds the whole mesh
mpirun -n 4 --bind-to none python3 script.py --distribute --omp 2   # partitioned mesh + threads
```

The two rules that do affect the script:

1. **Under MPI every rank must agree on the state of the simulation.** Anything
   non-deterministic (an unseeded random number, an `id()`-ordered iteration, a rank-local
   decision) makes the ranks diverge, and the symptom is a *deadlock*, not a wrong number.
   Use `DeterministicRandomField` for random fields, and
   `get_mpi_bcast(x if get_mpi_rank()==0 else None)` for anything computed in Python.
   Never guard `add_mesh`/`add_equations` by rank — setup is collective.
2. **MPI is a compile-time option and is OFF in the PyPI wheels.** Check with
   `python -m pyoomph check mpi`.

`--omp` needs nothing extra and is the first thing to reach for. Note that `mpirun` pins
each rank to one core by default, which silently turns `--omp` into a no-op — use
`--bind-to none`. Full details, solver choice under MPI and the pitfall list:
[`agents/parallel.md`](agents/parallel.md).

## Command-line flags every script gets for free

`Problem` installs its own argument parser, so any script accepts these without the author
doing anything:

| Flag | Purpose |
|---|---|
| `--outdir DIR` | Output directory (default: script filename without `.py`). |
| `--runmode d\|o\|c\|p` | Delete-and-run / overwrite / continue from the dumped state / re-plot only. |
| `--quick-test` | Stop after the first successful Newton solve and write one output. Ideal for checking that a script runs at all. |
| `--omp N`, `--distribute`, `--mpi-output` | Parallelization (above). |
| `--pardiso`, `--superlu`, `--umfpack`, `--petsc`, `--petsc_mumps`, `--accelerate` | Linear solver. |
| `--slepc`, `--slepc_mumps`, `--arpack` | Eigensolver. |
| `--tcc`, `--distutils` | JIT C compiler (internal TCC vs. system compiler). |
| `-P name=value ...` | Override problem parameters from the command line. |
| `--verbose`, `--largest_residuals N` | Diagnostics; the latter reports which dofs carry the largest residual, which is the fastest way to find a non-converging equation. |
| `--no-cache`, `--suppress_compilation`, `--suppress_code_writing` | JIT-code debugging. |

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

End every script with:
```python
if __name__ == "__main__":
    with MyProblem() as problem:
        problem.solve(); problem.output()          # stationary
        # or: problem.run(endtime, outstep=..., numouts=..., temporal_error=..., spatial_adapt=...)
```

## Companion reference files

The detailed references live in the [`agents/`](agents/) subdirectory. This file is
self-contained for an ordinary problem; read the companion file when the task actually
touches its subject.

| File | Read it when the task involves |
|---|---|
| [`agents/examples.md`](agents/examples.md) | You want a working skeleton to start from: 2D BVP with observables and a parameter scan, two domains coupled at a shared interface, a free-surface/ALE flow, parameter continuation, custom `GmshTemplate` geometry, combined spatial+temporal adaptivity, save/resume. |
| [`agents/materials.md`](agents/materials.md) | Real fluids/solids by name, mixtures, multi-component transport, interfaces between phases, surfactants and isotherms, evaporation/mass transfer, UNIFAC/AIOMFAC activity coefficients. Anything taking a `fluid_props`/`interface_props` object. |
| [`agents/parallel.md`](agents/parallel.md) | OpenMP threads, MPI, `--distribute`, determinism requirements across ranks, choosing a linear solver under `mpirun`. |
| [`agents/advanced.md`](agents/advanced.md) | Eigenvalues, linear stability, bifurcation tracking and classification, branch switching, periodic orbits and Floquet multipliers, deflation; hand-written C via `CustomMultiReturnExpression`; Discontinuous Galerkin and facet/skeleton unknowns; ALE internals and remeshing mechanics. |

## Where to look for more (in this repo)

- `docs/source/tutorial/` — the full human tutorial (rst + literalinclude'd `.py`
  example scripts); `temporal/` starts with ODEs (simplest), `spatial/` covers
  stationary PDEs, `pde/` covers spatio-temporal PDEs, `ale/` covers moving
  meshes/free surfaces, `multidom/` covers multiple coupled domains, `mcflow/`
  covers the multi-component/materials equations, `dg/` covers Discontinuous
  Galerkin, `advstab/` covers stability/bifurcation analysis, `plotting/` covers the
  built-in plotters, `parallel/` covers OpenMP and MPI, `precice/` covers coupling to
  external solvers through preCICE (`pyoomph/solvers/precice_adapter.py`, plus the
  `--generate_precice_cfg` flag), `math.rst`/`math/` lists all built-in math functions
  and keyword variables (`var("...")` names).
- `pyoomph/generic/codegen.py`, `pyoomph/generic/problem.py` — the core `Equations`/
  `Problem` machinery (source of truth for exact method signatures).
- `pyoomph/equations/*.py` — all built-in physics; read the target file directly for
  full constructor signatures/docstrings before writing code against it.
- Rendered API docs: https://pyoomph.readthedocs.io/en/latest/tutorial.html

**When the one-liners here are not enough, read the source.** Every class named in this
file lives in a short, readable Python module under `pyoomph/equations/`, `pyoomph/meshes/`
or `pyoomph/materials/`, and the constructor signature plus docstring there is the source
of truth. Grep for the class before writing code against it; do not guess a keyword
argument.
