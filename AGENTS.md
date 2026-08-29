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
- `define_scalar_field(name, space, scale=None, testscale=None)` — with `scale=None` the
  field takes its scale from `set_scaling(<name>=...)` at problem level, which works for
  interface fields and Lagrange multipliers exactly as for bulk ones.
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

## ODEs (`ODEEquations`)

A 0D system needs no mesh at all — `add_equations(eqs @ "name")` alone creates the domain.
Everything else works as for a PDE: `set_scaling` reaches ODE variables by name,
`scale_factor("x")` resolves for them, and each residual must be **dimensionless**, which
is what `testscale` is for. Reduce anything higher than first order in time to first order
with an auxiliary variable.

```python
class DrivenOscillator(ODEEquations):          # m x'' + c x' + k x = F0 cos(w t)
    def __init__(self, m, c, k, F0, omega):
        super().__init__()
        self.m, self.c, self.k, self.F0, self.omega = m, c, k, F0, omega

    def define_fields(self):
        # Each testscale must cancel the units of ITS OWN residual. The x-residual is a
        # velocity; the v-residual is a force, so the mass has to be divided out too.
        self.define_ode_variable("x", testscale=scale_factor("temporal")/scale_factor("x"))
        self.define_ode_variable("v", testscale=scale_factor("temporal")/(self.m*scale_factor("v")))

    def define_residuals(self):
        x, xtest = var_and_test("x")
        v, vtest = var_and_test("v")
        t = var("time")                         # var("time") is DIMENSIONAL (carries seconds)
        self.add_residual((partial_t(x) - v)*xtest)
        self.add_residual((self.m*partial_t(v) + self.c*v + self.k*x
                           - self.F0*cos(self.omega*t))*vtest)

class OscillatorProblem(Problem):
    def define_problem(self):
        m, k, c = 2*kilogram, 200*newton/meter, 4*kilogram/second
        eqs = DrivenOscillator(m, c, k, 5*newton, 8/second)
        self.set_scaling(temporal=square_root(m/k), x=1*milli*meter, v=1*milli*meter/second)
        eqs += InitialCondition(x=0, v=0)
        eqs += TemporalErrorEstimator(x=1, v=1)   # required for temporal_error= to act
        eqs += ODEFileOutput()
        self.add_equations(eqs @ "oscillator")

if __name__ == "__main__":
    with OscillatorProblem() as problem:
        problem.run(40*second, startstep=1*milli*second, outstep=20*milli*second,
                    temporal_error=1e-4)
```

- `ODEFileOutput(filename=None, first_column="time", in_units={})` writes **one** text file
  named after the domain (`oscillator.txt`), one row per output step, with a `#`-comment
  header naming the columns: the first column is the time (or whatever `first_column=`
  names, e.g. a global parameter for a continuation scan), then every ODE variable. It is
  directly `numpy.loadtxt`-able. `in_units={"x": milli*meter}` writes a column in a chosen
  unit.
- `var("time")` is dimensional, so a frequency must carry `1/second` and the argument of
  `cos`/`exp` must come out dimensionless.
- `PointMesh` is only needed if the ODE-like equations must also sit at a spatial location
  (e.g. to couple to a surrounding PDE domain); a pure ODE does not need it.

## Symbolic expression API (`from pyoomph.expressions import *`)

| Function | Purpose |
|---|---|
| `var(name)` | Bind a field or special variable: `"time"`, `"coordinate"`, `"coordinate_x"/"_y"/"_z"`, `"mesh"`, `"lagrangian"`, `"normal"`, `"velocity"`, or any user-defined field name. `var("normal")` on a boundary points **out of** the domain the equation is attached to. |
| `var_and_test(name)` | `(var(name), testfunction(name))` in one call — the standard way to start `define_residuals`. |
| `testfunction(name)` | The FE test function of a field. |
| `weak(a, b)` | `∫_Ω a·b dΩ` — the fundamental weak-form bilinear pairing; `b` is (a derivative of) a test function. |
| `grad(f)` / `div(f)` | Gradient / divergence; automatically becomes the *surface* gradient/divergence on interface (codimension>0) domains. |
| `partial_t(f, order=1, ALE="auto")` | Time derivative `∂ₜⁿf`; `ALE=True/"auto"` corrects for mesh motion on moving meshes. |
| `dot(a, b)` | Vector dot product. |
| `contract(a, b)` | Generalized dot/Frobenius/matrix-vector product depending on tensor rank. `weak(a,b)` already contracts fully, so `weak(grad(u), grad(v))` on vector `u` is the Frobenius pairing — no `contract` needed inside `weak`. |
| `grad(vector)` | Index convention `grad(u)[i][j] = du_i/dx_j`, so the convective term is `dot(grad(u), u)`. |
| `vector([...])` | Build a vector-valued expression, e.g. `vector([-var("coordinate_y"), var("coordinate_x")])`. |
| `nondim(name)` | Nondimensional version of `var(name)`. |
| `evaluate_in_past(expr)` | Value of `expr` at the previous timestep (e.g. for error estimators). |
| `time_scheme("BDF1"/"BDF2"/"TPZ"/"MPT", expr)` | Wrap a residual term to use a specific time-stepping scheme for that term. |
| `subexpression(expr)` | Compute a shared sub-expression once in the generated C and reuse it (also caches its derivatives). Wrap any moderately expensive repeated term. |
| Elementary math | `sin`, `cos`, `exp`, `log`, `square_root` (**not** `sqrt`), `absolute`, `signum`, `heaviside`, `pi`, `minimum`/`maximum`, `rational_num(n,d)`. |
| Tensor helpers | `identity_matrix`, `transpose`, `trace`, `determinant`, `inverse_matrix`, `matproduct`. |

## Units and nondimensionalization

`from pyoomph.expressions.units import *` — a **separate import**: units are in neither
`from pyoomph import *` nor `from pyoomph.expressions import *`. Base units `meter`,
`second`, `kilogram`, `kelvin`, `mol`, `ampere`, the full SI prefix set (`nano` … `mega`,
including `centi`), and the usual derived units (`newton`, `pascal`, `joule`, `volt`,
`molar`, `celsius`, `degree`, …). There is **no `mm`/`cm` shorthand** — compose from a
prefix and a base unit. Physical constants (`epsilon_0`, `k_Boltzmann`, `N_Avogadro`,
`gas_constant`, …) are in `pyoomph.expressions.phys_consts`.

**Write the physics dimensionally and let pyoomph nondimensionalize it.** This is the
default and strongly preferred; go nondimensional only when the user asks.
`problem.set_scaling(spatial=..., temporal=..., fieldname=...)` declares the scales, and
inside a reusable `Equations` class you express `scale`/`testscale` through
`scale_factor(...)` so the class works at any set of scales.

### The one rule that governs every residual

**`weak(a, b)` integrates over the nondimensionalized domain**, so every contribution must
satisfy: **`a * b` is dimensionless — nothing else.** `b` is a test function, so the recipe
is to pick `testscale` as whatever makes `a*b` dimensionless. This holds on interfaces too;
what changes there is only that the test function of a *bulk* field already carries an
extra `1/scale_factor("spatial")`, which is exactly what makes the natural boundary term
`weak(coeff*dot(grad(u),n), utest)` work with the same `coeff` as the bulk equation.

Get it wrong and pyoomph refuses at setup with *"The added residual contribution is not
dimensionless … it still carries the base unit: meter"*, then prints every scale it used
and the offending term. That message names the fix; read it rather than guessing.

Two consequences that catch people out, because they make a coefficient's units differ
from the textbook strong form: a gradient pairing forces `coeff` to carry **length²** (so
`PoissonEquation(coefficient=1)` is a hard error in a dimensional problem, and
`coefficient=1*meter**2` is right), and a source pairing forces the source to carry the
units of **`u` itself**.

**Full detail — the derivations, the interface variant with a worked Lagrange-multiplier
flux, the unit lists, and the pitfall catalogue (rational vs float exponents, `celsius`
offsets, missing scales) — is in [`agents/units.md`](agents/units.md). Read it the first
time a residual is rejected.**

## Coordinate systems

`Problem.set_coordinate_system(...)` takes either the **string** `"axisymmetric"`,
`"axisymmetric_flipped"`, `"radialsymmetric"`, `"cartesian"`, or the ready-made
**instance** of the same name. The instances live in `pyoomph.expressions` (so
`from pyoomph.expressions import *` has them) — the *classes* they are built from are in
`pyoomph/expressions/coordsys.py`, which is where a custom `BaseCoordinateSystem`
subclass goes. `coordsys=` on an individual equation or on `weak(...)` overrides it
locally.

`radialsymmetric` is **spherical** symmetry on a **1D** mesh in `r = coordinate_x`: the
Laplacian is `(1/r²) d/dr(r² d/dr)` and the measure carries the full `4*pi*r**2`, so
`IntegralObservables(V=1)` on the bulk is the real spherical-shell volume and `S=1` on a
boundary is `4*pi*r**2` there.

**Axis convention for `axisymmetric`: `coordinate_x` is the radius r and `coordinate_y`
is the axial coordinate z.** So on a `RectangularQuadMesh` the symmetry axis r=0 is the
`"left"` boundary and the outer radius is `"right"`. `axisymmetric_flipped` swaps them
(x becomes the symmetry axis). For an ordinary **axisymmetric solve** the r-weight of the
metric makes the symmetry axis a natural boundary for a scalar field — **do not pin
anything there**; `AxisymmetryBC()` then only zeroes the radial/azimuthal components of
vector fields (and `mesh_x`).

**For azimuthal stability, add `AxisymmetryBC() @ "axis"` regardless of what fields you
have**: it also supplies the m-dependent r=0 conditions that a normal-mode eigensolve
needs, and those *do* act on scalars (`|m|>=1` forces a scalar to zero on the axis). It
takes no arguments and finds the fields and the current mode number itself. An
axisymmetric problem may also be meshed **1D in r alone** (a `LineMesh`), with the
azimuthal direction supplied entirely by `azimuthal_m`.

## `Problem` API (`pyoomph/generic/problem.py`)

Override `define_problem(self)` to build the problem (mesh + equations). Key methods:

| Method | Purpose |
|---|---|
| `add_mesh(mesh_template)` | Register a `MeshTemplate` (or ready-made mesh, see below); only valid inside `define_problem`. |
| `add_equations(eqs_at_domain)` | Attach an `EquationTree` (`equations @ "domain_name"`); only valid inside `define_problem`. |
| `get_equations(path)` / `get_mesh(name)` | Retrieve previously added equations/mesh by name. |
| `solve(*, spatial_adapt=0, timestep=None, temporal_error=None, ...)` | Stationary Newton solve if `timestep=None`; otherwise a single transient step. |
| `run(endtime, timestep=None, *, outstep=None, numouts=None, spatial_adapt=0, temporal_error=None, maxstep=None, startstep=None)` | Time-march to `endtime`, calling `output()` periodically. `outstep=True`/a float sets the fixed output interval; `numouts=N` splits `[0,endtime]` into N outputs; `timestep=` fixes dt (default: taken from `outstep`); `startstep=` the first dt under temporal adaptivity (overrides `timestep`); `maxstep=` caps dt; `spatial_adapt=n` allows n adaptations per step. `temporal_error=<tol>` enables adaptive time-stepping — **but only in combination with a `TemporalErrorEstimator`, see below**. All time arguments take dimensional values. |
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

## Meshes

| Class | Domain/boundary names |
|---|---|
| `LineMesh(N=10, size=1.0, minimum=0.0, name=)` | domain `"domain"`, boundaries `"left"`/`"right"` |
| `RectangularQuadMesh(size=1.0, N=10, lower_left=[0,0], name=)` | `"left"/"right"/"top"/"bottom"` |
| `CircularMesh(radius=1, segments="all", domain_name=)` | `"circumference"`; `segments=["NE","SE"]` gives a half-disc |
| `CuboidBrickMesh`, `CylinderMesh`, `SphericalOctantMesh`, `PointMesh` (all `domain_name=`) | see [`agents/meshes.md`](agents/meshes.md) |

`size`, `N` and `lower_left` take a scalar or a per-axis list, and accept **dimensional**
values — so `set_scaling` must come before `add_mesh`. Note `LineMesh`/`RectangularQuadMesh`
use `name=` while the others use `domain_name=`.

For unstructured or multi-domain geometry, subclass `GmshTemplate` and build it from
`point`/`line`/`circle_arc`/`spline`/`plane_surface`; for a fully hand-built mesh subclass
`MeshTemplate` and add elements by node index. Domain and boundary names defined by the
mesh are exactly the strings used with `@`, and an interface between two domains becomes
its own `InterfaceMesh` automatically.

**[`agents/meshes.md`](agents/meshes.md)** has the full template list, the hand-built
`MeshTemplate` API (node/element/facet calls and their orderings), how to make two domains
share a real interface, and the moving-mesh / free-surface / remeshing machinery.

## Generic building-block equations

In `pyoomph/equations/generic.py` (boundary conditions — use with `@"boundary_name"`; they used to
live in `pyoomph/meshes/bcs.py`, which no longer exists):

| Class | Purpose |
|---|---|
| `DirichletBC(**fields)` | Strong Dirichlet condition, e.g. `DirichletBC(u=0, v=1)@"left"`. |
| `NeumannBC(**fluxes)` | Natural/flux BC matching the bulk equation's integration-by-parts choice. |
| `EnforcedBC(**constraints)` | Arbitrary constraint enforced via a Lagrange multiplier, e.g. `EnforcedBC(u=var("u")-var("v"))` adjusting `u` to match `u=v` |
| `EnforcedDirichlet(**fields)` | Dirichlet enforced weakly (Lagrange multiplier) instead of strong pinning. **This is how you read a boundary flux out of a Dirichlet condition**: the multiplier field is named `"_lagr_enf_bc_" + fieldname` and equals `-coeff*dot(grad(u), n)` in the bulk equation's normalisation, so e.g. `IntegralObservables(lam=var("_lagr_enf_bc_c"), surf=1) @ "left"` gives the flux. (`dot(grad(u), var("normal"))` does *not* work on an interface — `grad` there is the **surface** gradient.) |
| `PeriodicBC(other_interface, offset=None)` | Periodic boundary matching, applied to **all continuous fields** at once. Attach it to **one** of the two boundaries and name the partner: `PeriodicBC("left", offset=[-Lx, 0*meter]) @ "right"`. `offset` is **dimensional** and is what you add to a node's position *on this boundary* to land on its counterpart, so it points from this boundary to the partner. The mesh must already have matching nodes on both sides. |
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
| `SpatialErrorEstimator(*fluxes, **fields)` | Drives h-adaptivity from jumps across elements. **In a dimensional problem use the keyword form**, `SpatialErrorEstimator(velocity=1, temperature=1)` — the positional flux form takes a raw expression that must be *dimensionless*, and `grad(var("u"))` is not (nor is `grad(nondim("u"))`, since `grad` still divides by a dimensional coordinate). A dimensional flux compiles to C referring to an undeclared `second`/`meter` and the build fails. |
| `RefineToLevel` | Mesh-refinement control (`RefineMaxElementSize`/`RefineAccordingToElement` are in `pyoomph/equations/additional.py`). |
| `RemeshWhen(...)` | Trigger automatic 2D remeshing on mesh distortion. |
| `ProjectExpression(scale=1, space="C2", field_type="scalar", **projs)` | L2-project an arbitrary expression onto a field, for output/diagnostics — the usual way to get a quantity `grad` cannot give you on an interface (project it in the bulk, read its trace on the boundary). **`scale=` defaults to 1 and is *not* taken from `set_scaling`**, so in a dimensional problem pass it explicitly (a string is resolved as `scale_factor(that_name)`). `field_type="vector"/"tensor"/"symmetric_tensor"` projects a whole tensor at once. |
| `TemporalErrorEstimator(**fieldfactors)` | Drives adaptive time-stepping. **Required** for `run(..., temporal_error=...)` to do anything: without it the tolerance is silently ignored and the run marches at a fixed step. Weight each field that should count, e.g. `TemporalErrorEstimator(x=1, v=1)` — a larger factor makes that field's error count for more. The `temporal_error=` tolerance is measured on the **nondimensional** solution, so with fields scaled to O(1) it behaves as a relative tolerance; `1e-3` … `1e-6` is the useful range and tightening it does converge (measured: `1e-4` → 1.6e-3 error, `1e-6` → 1.1e-4 on a driven oscillator). |
| `IntegralObservables(**exprs)` / `ExtremumObservables(...)` | Track domain integrals / min-max of fields. **`IntegralObservables` integrates over the physical domain**; `ExtremumObservables` is read back separately — see below. |
| `IntegralConstraint`/`AverageConstraint` | Enforce an integral/average constraint via a Lagrange multiplier (e.g. fixed total mass). |
| `ConnectFieldsAtInterface(fields)` | Enforce continuity of one or more fields across an interface via Lagrange multipliers, e.g. `ConnectFieldsAtInterface("T")`, `ConnectFieldsAtInterface(["u","v"])`, or `ConnectFieldsAtInterface({"c_liq": "c_gas"})` when the two sides name the field differently. Attach to **one** side only. |
| `LocalExpressions(**exprs)` | Named auxiliary expressions available to sibling/child equations. |

## Output and plotting

Output equations are added to a domain with `+=` and written on `problem.output()`:
`MeshFileOutput()` (VTU for ParaView), `TextFileOutput()` (nodal dump),
`TextFileOutputAlongLine`, `GridFileOutput`, `ODEFileOutput()` (ODE domains),
`IntegralObservableOutput()` (the observable time series). For a scan or summary file use
`problem.create_text_file_output(fname, header=[...])` + `.add_row(...)` — **only after the
problem is initialised**, since it opens the file in the output directory at once.

Diagnostics come from `IntegralObservables(**exprs)` (which integrates over the
**physical, dimensional** domain — unlike `weak`), `ExtremumObservables(...)` for min/max,
and `mesh.evaluate_at_points(coords)` to read the solution at arbitrary points.

Plot during the run by assigning a subclass of `pyoomph.output.plotting.MatplotlibPlotter`
(2D), `plotting1d.MatplotlibPlotter1D` or `plotting3d.PyVistaPlotter` to `problem.plotter`.

**[`agents/output.md`](agents/output.md)** has the file names and formats, the observable
measure and unit conventions, the `evaluate_at_points` row layout (and its blind spot for
DG fields), how to read a boundary flux out of a Dirichlet condition, and the plotter API.

## Built-in physics equation libraries (`pyoomph/equations/*.py`)

Ready-to-use `Equations` classes — **prefer these over a hand-rolled weak form when the
physics matches**. Imported explicitly, e.g.
`from pyoomph.equations.navier_stokes import NavierStokesEquations`.

| Module | Covers |
|---|---|
| `poisson.py` | `PoissonEquation`, `DiffusionEquation`, far-field conditions |
| `advection_diffusion.py` | scalar transport, with optional SUPG-type `stabilization=` |
| `navier_stokes.py` | `StokesEquations`/`NavierStokesEquations` (+ `bulkforce=`, `gravity=`, `boussinesq=`), free surfaces, contact angles, slip |
| `stabilized_ns.py`, `stabilization.py` | residual-based stabilization for equal-order spaces |
| `ALE.py` | moving-mesh / mesh-smoothing equations |
| `solid.py` | deformable and linear-elastic solids, tractions, `FSIConnection` |
| `viscoelastic.py` | Oldroyd-B, Giesekus, PTT, FENE — log-conformation by default |
| `multi_component.py`, `NSCH.py`, `cahn_hilliard.py`, `low_order_NSCH.py` | multi-species flow and phase-field two-phase flow |
| `surfactants.py`, `contact_angle.py` | interfacial surfactant transport, dynamic contact lines |
| `electrostatics.py`, `electrohydrodynamics.py` | potentials, double layers, ion transport, Maxwell stress |
| `salt_transport.py`, `darcy.py`, `lubrication.py`, `helmholtz.py`, `potential_flow.py` | dissolved salts, porous media, thin films, acoustics, inviscid flow |
| `kuramoto_sivashinsky.py`, `harmonic_oscillator.py`, `ode.py` | pattern formation and ODE utilities |
| `tracers.py`, `topological_changes.py` | tracer particles, automatic pinch-off/merging |
| `generic.py`, `additional.py` | the building blocks below |

**[`agents/physics.md`](agents/physics.md)** lists every class with its constructor
keywords, field names and the traps (which class goes alongside which, what needs
`scale_for_FSI=True`, how a Boussinesq buoyancy is written, …).

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

A moving (ALE) mesh is one more equation on the same domain: add a mesh-motion equation
from `pyoomph/equations/ALE.py` (`LaplaceSmoothedMesh()`, `PseudoElasticMesh()`,
`HyperelasticSmoothedMesh()`, …) and the nodal positions become unknowns.

```python
eqs  = NavierStokesEquations(dynamic_viscosity=mu, mass_density=rho)
eqs += LaplaceSmoothedMesh()
eqs += NavierStokesFreeSurface(surface_tension=sigma) @ "top"
eqs += DirichletBC(mesh_y=0) @ "bottom"        # True = pin at the current value
```

The mesh position is the field `"mesh"` (`mesh_x`/`mesh_y`/`mesh_z`), `var("lagrangian")`
is the undeformed reference, and `partial_t(f)` already subtracts the mesh velocity.

**With more than one moving domain you must connect the meshes explicitly.** Each domain
owns its own nodal positions at a shared interface, so if only one side is driven — by a
free surface, a prescribed motion, or its own smoothing — **the two meshes silently drift
apart, with no error and no warning**. Add `ConnectMeshAtInterface()` on **one** side:

```python
lower += NavierStokesFreeSurface(surface_tension=sigma) @ "interface"
lower += ConnectMeshAtInterface()   @ "interface"   # upper mesh co-moves with the lower
lower += ConnectVelocityAtInterface() @ "interface" # and, usually, continuous velocity
```
Measured on a two-domain box whose interface is pushed from 0.5 to 0.7 from the lower side
only: *with* the connection the upper domain's interface follows to 0.7; *without* it, it
stays at 0.5 and a gap opens. The same applies to any field that must be continuous across
the interface — use `ConnectFieldsAtInterface`, or `ConnectVelocityAtInterface` for flow.
If pyoomph complains *"Cannot deduce the coordinate space of domain X"*, add an
`ElementSpace("C2")` to that domain's equations.

`ConnectVelocityAtInterface` **also transfers the interfacial traction**, so it is the
complete fluid-fluid coupling — but only **when both domains use the same velocity and
pressure scale and the same test scale**. Give the two sides different scales and the
multiplier enters the two residuals with different normalisations and the traction balance
is silently wrong, so set the flow scales once at problem level rather than per domain.

Which connectors a ready-made interface class still needs differs, and getting it wrong
is silent either way:

| Interface class | `ConnectMeshAtInterface` | `ConnectVelocityAtInterface` |
|---|---|---|
| `FSIConnection` | **no** — it supplies the mesh coupling itself | **no** — it supplies velocity + traction itself |
| `MultiComponentNavierStokesInterface` | **yes, required** between two moving-mesh domains | **no** — it carries its own velocity coupling and traction transfer |
| a hand-built fluid–fluid interface | yes | yes |

Adding one that is already covered double-counts the coupling; omitting the mesh
connection that `MultiComponentNavierStokesInterface` needs lets the two meshes drift
apart as described above.

Remeshing (2D) needs `mesh.remesher = Remesher2d(mesh)` **on the template** plus a
`RemeshWhen(...)` equation — without the former the latter silently does nothing.

**[`agents/meshes.md`](agents/meshes.md)** covers the smoothing equations, pinning rules,
remeshing options and sizes, and how remeshing interacts with adaptivity;
[`agents/advanced.md`](agents/advanced.md) §4 has the ALE internals.

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
| [`agents/units.md`](agents/units.md) | Choosing scales, writing a `testscale`, or **writing any term by hand on an interface** — its "On interfaces and boundaries" section has the worked Lagrange-multiplier flux and is what stops you getting the units wrong in the first place. Also the first place to look when setup fails with *"residual contribution is not dimensionless"*. |
| [`agents/physics.md`](agents/physics.md) | Picking or configuring a built-in equation class — constructor keywords, field names, which classes combine with which. |
| [`agents/meshes.md`](agents/meshes.md) | Anything beyond a plain rectangle: hand-built `MeshTemplate`s, `GmshTemplate` geometry, two domains sharing an interface, moving meshes, remeshing. |
| [`agents/output.md`](agents/output.md) | Getting numbers out: observables, point evaluation, boundary fluxes, file formats, plotting. |
| [`agents/examples.md`](agents/examples.md) | A working skeleton to start from: 2D BVP with observables and a parameter scan, two coupled domains, a free-surface/ALE flow, continuation, custom gmsh geometry, spatial+temporal adaptivity, save/resume. |
| [`agents/materials.md`](agents/materials.md) | Real fluids/solids by name, mixtures, interfaces, surfactants, evaporation/mass transfer, electrolytes, UNIFAC/AIOMFAC. Anything taking a `fluid_props`/`interface_props` object. |
| [`agents/advanced.md`](agents/advanced.md) | Eigenvalues, stability, bifurcation tracking, branch switching, periodic orbits, deflation; hand-written C via `CustomMultiReturnExpression`; Discontinuous Galerkin and facet unknowns; ALE internals. |
| [`agents/parallel.md`](agents/parallel.md) | OpenMP threads, MPI, `--distribute`, rank determinism, linear solvers under `mpirun`. |

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
