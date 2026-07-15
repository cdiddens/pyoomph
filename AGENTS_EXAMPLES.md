# pyoomph — example recipes for AI coding assistants

Companion to [`AGENTS.md`](AGENTS.md) (read that first for the core mental model and
API). This file collects idiomatic, code-verified recipes for common problem shapes,
each as a short, runnable script pulled from (or closely following) the actual
tutorial tree at `docs/source/tutorial/`. Prefer copying the pattern that matches the
user's problem over inventing a new structure.

For materials/multi-component recipes, see [`AGENTS_MATERIALS.md`](AGENTS_MATERIALS.md).
For bifurcation/stability, custom C code, DG, and ALE-internals recipes, see
[`AGENTS_ADVANCED.md`](AGENTS_ADVANCED.md).

| # | Recipe | Key API exercised |
|---|---|---|
| 1 | 2D BVP with named boundaries, integral observables, parameter scan | `RectangularQuadMesh`, `IntegralObservables`, `AverageConstraint`, `create_text_file_output` |
| 2 | Multiple domains coupled at a shared interface | `MeshTemplate`, `InterfaceEquations`, `get_opposite_side_of_interface` |
| 3 | Free surface / moving-mesh flow | `NavierStokesEquations`, `LaplaceSmoothedMesh`, hand-rolled kinematic/dynamic BC |
| 4 | Parameter continuation (natural vs. arclength) | `define_global_parameter`, `go_to_param`, `arclength_continuation` |
| 5 | Custom unstructured geometry | `GmshTemplate`, `define_geometry`, `point`/`circle_arc`/`spline`/`plane_surface` |
| 6 | Combined spatial + temporal adaptivity | `SpatialErrorEstimator`, `run(..., spatial_adapt=1, temporal_error=...)` |
| 7 | Save and resume a simulation | `save_state`/`load_state` |

## 1. 2D boundary value problem: named boundaries, observables, parameter scan

A `NavierStokesEquations` problem on a square, with all boundaries pinned except a
prescribed shear traction, an `IntegralObservables` chain to derive a scalar output
from the solution, a pressure-nullspace fixation via `AverageConstraint`, and a
parameter scan written to a text file. Note the problem is built inline (no `Problem`
subclass) — both styles are equally idiomatic; use a subclass once `__init__`/
`define_problem` need to share state.

```python
from pyoomph import *
from pyoomph.equations.navier_stokes import *

problem = Problem()
T = problem.define_global_parameter(T=0)  # tangential traction at the top, our scan parameter

eqs = NavierStokesEquations(mass_density=10, dynamic_viscosity=1)
eqs += MeshFileOutput()
eqs += NoSlipBC() @ ["left", "right", "bottom"]          # @ accepts a list of boundary names
eqs += (DirichletBC(velocity_y=0) + NeumannBC(velocity_x=T)) @ "top"
# No in/outflow anywhere => pressure has a null space (p -> p+const); fix it:
eqs += AverageConstraint(pressure=0)

# Derive U = sqrt(<u.u>/Area) as a named observable, evaluated after each solve
eqs += IntegralObservables(U_sqr=dot(var("velocity"), var("velocity")), Area=1,
                            U=lambda U_sqr, Area: square_root(U_sqr) / Area)

problem += RectangularQuadMesh(N=25, name="domain")
problem += eqs @ "domain"
problem.solve()

out_file = problem.create_text_file_output("U_vs_T.txt", header=["T", "U"])
for T_val in numpy.linspace(0, 100, 20, endpoint=True):
    problem.go_to_param(T=T_val)
    problem.output()
    U_val = problem.get_mesh("domain").evaluate_observable("U")
    out_file.add_row(T_val, U_val)
```

Key points:
- `@` accepts either a single boundary name or a list, e.g. `NoSlipBC() @ ["left","right","bottom"]`.
- `IntegralObservables` entries can reference each other by name (including via a
  `lambda` of previously-defined observables), and are read back with
  `problem.get_mesh(domain).evaluate_observable(name)`.
- `problem.create_text_file_output(fname, header=[...])` + `.add_row(...)` is the
  idiom for writing a custom scan/summary file (separate from the per-timestep
  `TextFileOutput()`/`MeshFileOutput()` equation classes).
- With an all-Dirichlet velocity BC and no free surface, the pressure needs a
  nullspace fixation: either `eqs += AverageConstraint(pressure=0)` (pin the mean) or
  `StokesEquations.create_pressure_fixation()` (pin a single dof).

## 2. Multiple domains coupled at a shared interface

Two independent 1D domains with a custom `MeshTemplate` that builds both meshes and
marks the shared node as boundary `"interface"` on each side, plus a custom
`InterfaceEquations` that enforces field continuity through a Lagrange multiplier
(the general pattern behind "connect field X across an interface", which is also
available pre-built as `ConnectFieldsAtInterface`/`ConnectVelocityAtInterface`/
`ConnectMeshAtInterface` for the common cases).

```python
from pyoomph import *
from pyoomph.equations.poisson import *

class TwoDomainMesh1d(MeshTemplate):
    def __init__(self, Ntot=100, xI=1, L=2, left_domain_name="domainA", right_domain_name="domainB"):
        super().__init__()
        self.Ntot, self.xI, self.L = Ntot, xI, L
        self.left_domain_name, self.right_domain_name = left_domain_name, right_domain_name

    def define_geometry(self):
        xI, L = self.nondim_size(self.xI), self.nondim_size(self.L)
        NA = round(self.Ntot * xI / L)

        domainA = self.new_domain(self.left_domain_name)
        domainB = self.new_domain(self.right_domain_name)

        nodesA = [self.add_node_unique(x) for x in numpy.linspace(0, xI, NA)]
        for x0, x1 in zip(nodesA, nodesA[1:]):
            domainA.add_line_1d_C1(x0, x1)

        nodesB = [self.add_node_unique(x) for x in numpy.linspace(xI, L, self.Ntot - NA)]
        for x0, x1 in zip(nodesB, nodesB[1:]):
            domainB.add_line_1d_C1(x0, x1)

        self.add_facet_to_boundary("left", [nodesA[0]])
        self.add_facet_to_boundary("interface", [nodesB[0]])  # same physical point as nodesA[-1]
        self.add_facet_to_boundary("right", [nodesB[-1]])


class ConnectTAtInterface(InterfaceEquations):
    def define_fields(self):
        self.define_scalar_field("lambda", "C2")  # Lagrange multiplier

    def define_residuals(self):
        my_field, my_test = var_and_test("T")
        opp_field, opp_test = var_and_test("T", domain=self.get_opposite_side_of_interface())
        lagr, lagr_test = var_and_test("lambda")
        self.add_residual(weak(my_field - opp_field, lagr_test))  # constraint T_my - T_opp = 0
        self.add_residual(weak(lagr, my_test))                    # Lagrange reaction on my side
        self.add_residual(weak(-lagr, opp_test))                  # and (with opposite sign) on the other side


class TwoDomainTemperatureConduction(Problem):
    def __init__(self):
        super().__init__()
        self.conductivityA = 0.5
        self.conductivityB = 2

    def define_problem(self):
        self.add_mesh(TwoDomainMesh1d())

        eqsA = TextFileOutput() + PoissonEquation(name="T", coefficient=self.conductivityA, source=0)
        eqsA += DirichletBC(T=0) @ "left"

        eqsB = TextFileOutput() + PoissonEquation(name="T", coefficient=self.conductivityB, source=0)
        eqsB += DirichletBC(T=1) @ "right"

        eqsA += ConnectTAtInterface() @ "interface"  # add to exactly ONE side of the interface

        self.add_equations(eqsA @ "domainA")
        self.add_equations(eqsB @ "domainB")


if __name__ == "__main__":
    with TwoDomainTemperatureConduction() as problem:
        problem.solve()
        problem.output()
```

Key points:
- `var(name, domain=...)`/`var_and_test(name, domain=...)` reads a field from a
  *different* domain than the one the current `Equations` object lives on —
  `self.get_opposite_side_of_interface()` gives exactly that domain for an interface.
- A coupling `InterfaceEquations` is attached to only one side of the shared
  boundary; it reaches across to the other side via `domain=`/
  `get_opposite_side_of_interface()`.
- For matching whole vector fields (e.g. velocity or mesh position) across an
  interface, prefer the ready-made `ConnectFieldsAtInterface`/
  `ConnectVelocityAtInterface`/`ConnectMeshAtInterface` over hand-rolling this pattern.

## 3. Free surface / moving-mesh (ALE) flow

Navier-Stokes on a shallow rectangle with a free top surface: `LaplaceSmoothedMesh`
solves for the mesh position, and a hand-rolled kinematic condition (mesh follows the
normal fluid velocity) plus a dynamic condition (surface tension, via the
surface-divergence identity) close the system. `NavierStokesFreeSurface` is the
ready-made equivalent for the common case — write this by hand only when you need a
custom kinematic/dynamic condition.

```python
from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *

class KinematicBC(InterfaceEquations):
    def define_fields(self):
        self.define_scalar_field("_kin_bc", "C2")  # Lagrange multiplier

    def define_residuals(self):
        n, u = var(["normal", "velocity"])
        l, eta = var_and_test("_kin_bc")
        x, chi = var_and_test("mesh")
        # normal mesh velocity follows normal fluid velocity
        self.add_residual(weak(dot(n, u - mesh_velocity()), eta) - weak(l, dot(n, chi)))

    def before_assigning_equations_postorder(self, mesh):
        # pin the multiplier wherever the mesh there is already fully Dirichlet-pinned
        self.pin_redundant_lagrange_multipliers(mesh, "_kin_bc", "mesh")


class DynamicBC(InterfaceEquations):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def define_residuals(self):
        v = testfunction("velocity")
        # grad/div auto-become surface operators on a codim>0 (interface) domain,
        # so this single term yields both the Laplace-pressure and Marangoni contributions
        self.add_residual(weak(self.sigma, div(v)))


def FreeSurface(sigma):
    return KinematicBC() + DynamicBC(sigma)


class SurfaceRelaxationProblem(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=[80, 4], size=[1, 0.05]))
        eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=1)
        eqs += LaplaceSmoothedMesh()
        eqs += DirichletBC(mesh_x=True)  # shallow domain: fix all x-coordinates of the mesh
        eqs += MeshFileOutput()
        eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=0) @ "bottom"
        eqs += DirichletBC(velocity_x=0) @ "left"
        eqs += DirichletBC(velocity_x=0) @ "right"
        eqs += FreeSurface(sigma=1) @ "top"

        X, Y = var(["lagrangian_x", "lagrangian_y"])
        eqs += InitialCondition(mesh_y=Y * (1 + 0.25 * cos(2 * pi * X)))  # perturbed initial surface
        self.add_equations(eqs @ "domain")


if __name__ == "__main__":
    with SurfaceRelaxationProblem() as problem:
        problem.run(50, outstep=True, startstep=0.25)
```

Key points:
- `mesh_velocity()` is the co-moving nodal velocity (`partial_t(var("mesh"), ALE=False)`);
  don't confuse it with the fluid `velocity` field.
- `DirichletBC(mesh_x=..., mesh_y=...)` pins mesh-coordinate dofs exactly like any
  other field, since `activate_coordinates_as_dofs` makes them regular unknowns.
  `var("lagrangian_x"/"lagrangian_y")` gives the fixed reference (initial) position,
  used here to set up the initial mesh shape via `InitialCondition(mesh_y=...)`.
- See `AGENTS_ADVANCED.md` for the mechanics of `activate_coordinates_as_dofs` and the
  full catalog of mesh-smoothing equations (`PseudoElasticMesh`, `HyperelasticSmoothedMesh`, ...).

## 4. Parameter continuation

Two complementary idioms for tracing a solution branch as a parameter changes,
demonstrated on a scalar fold-normal-form ODE `dx/dt = r - x^2`:

**(a) Natural-parameter stepping** — simplest option, just re-solve after nudging the
parameter. Fails at a fold (turning point), since the branch stops being a function of
the parameter there.

```python
from pyoomph import *
from pyoomph.expressions import *

class FoldNormalForm(ODEEquations):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x, x_test = var_and_test("x")
        self.add_residual((partial_t(x) - self.r + x**2) * x_test)


class FoldProblem(Problem):
    def __init__(self):
        super().__init__()
        self.r = self.define_global_parameter(r=1)
        self.x0 = 1

    def define_problem(self):
        eq = FoldNormalForm(r=self.r)
        eq += InitialCondition(x=self.x0)
        eq += ODEFileOutput(first_column=self.r)  # write r instead of time as the first column
        self.add_equations(eq @ "fold")


if __name__ == "__main__":
    with FoldProblem() as problem:
        while True:
            problem.solve()
            problem.output_at_increased_time()
            problem.r.value -= 0.02
```

**(b) Pseudo-arclength continuation** — follows the branch through folds by
parametrizing along arclength instead of the bare parameter; `arclength_continuation`
returns a new step size to feed into the next call.

```python
from bifurcation_fold_param_change import *  # reuse FoldProblem from (a)

if __name__ == "__main__":
    with FoldProblem() as problem:
        problem.r.value = 1
        problem.get_ode("fold").set_value(x=1)
        problem.solve()
        problem.output()

        ds = -0.02  # first step decreases r
        while problem.get_ode("fold").get_value("x", as_float=True) > -1:
            ds = problem.arclength_continuation(problem.r, ds, max_ds=0.025)
            problem.output()
```

For scanning towards a specific target value of a parameter (rather than open-ended
stepping), use `problem.go_to_param(r=target_value)` — internally arclength-based, so
it also survives folds (see recipe 1 above for a `go_to_param` scan).
See `AGENTS_ADVANCED.md` for eigenvalue-based stability analysis and bifurcation
tracking (`solve_eigenproblem`, `activate_bifurcation_tracking`), which build on
these same primitives.

## 5. Custom unstructured geometry via `GmshTemplate`

Subclass `GmshTemplate`, override `define_geometry()` using `point`/`circle_arc`/
`spline`/`create_lines` to build outlines tagged with boundary names, then
`plane_surface(*outline_names, name=..., holes=[[...]])` to fill them in. Per-point
`size=` gives local mesh refinement; `mesh.mesh_mode="quads"` prefers quadrilaterals.

```python
from pyoomph import *
from pyoomph.equations.poisson import *
from pyoomph.expressions.units import *

class GmshFishMesh(GmshTemplate):
    def __init__(self, size=1, mouth_angle=45*degree, fin_angle=50*degree,
                 mouth_depth_factor=0.5, fin_length_factor=0.45, fin_height_factor=0.8,
                 domain_name="fish"):
        super().__init__()
        self.size, self.mouth_angle, self.mouth_depth_factor = size, mouth_angle, mouth_depth_factor
        self.fin_angle, self.fin_length_factor, self.fin_height_factor = fin_angle, fin_length_factor, fin_height_factor
        self.domain_name = domain_name

    def define_geometry(self):
        S = self.size  # gmsh nondimensionalizes internally, no manual scaling needed here
        p_mouth_center = self.point(-(1 - self.mouth_depth_factor) * S, 0)
        p_upper_jaw = self.point(-cos(self.mouth_angle / 2) * S, sin(self.mouth_angle / 2) * S)
        p_lower_jaw = self.point(-cos(self.mouth_angle / 2) * S, -sin(self.mouth_angle / 2) * S)
        p_upper_body_fin = self.point(cos(self.fin_angle / 2) * S, sin(self.fin_angle / 2) * S)
        p_lower_body_fin = self.point(cos(self.fin_angle / 2) * S, -sin(self.fin_angle / 2) * S)
        p_upper_fin_corner = self.point((cos(self.fin_angle / 2) + self.fin_length_factor) * S, self.fin_height_factor * S)
        p_lower_fin_corner = self.point((cos(self.fin_angle / 2) + self.fin_length_factor) * S, -self.fin_height_factor * S)

        self.create_lines(p_lower_jaw, "mouth", p_mouth_center, "mouth", p_upper_jaw)
        self.create_lines(p_lower_body_fin, "fin", p_lower_fin_corner, "fin", p_upper_fin_corner, "fin", p_upper_body_fin)
        self.circle_arc(p_lower_jaw, p_lower_body_fin, center=[0, 0], name="curved")
        self.circle_arc(p_upper_jaw, p_upper_body_fin, center=[0, 0], name="curved")

        eye_size, eye_center_x, eye_center_y = 0.125 * S, -S * 0.25, S * 0.3
        eye_resolution = self.default_resolution * 0.2  # finer mesh near the eye
        p_eye_center = self.point(eye_center_x, eye_center_y, size=eye_resolution)
        p_eye_north = self.point(eye_center_x, eye_center_y + eye_size, size=eye_resolution)
        p_eye_west = self.point(eye_center_x - eye_size, eye_center_y, size=eye_resolution)
        p_eye_east = self.point(eye_center_x + eye_size, eye_center_y, size=eye_resolution)
        p_eye_south = self.point(eye_center_x, eye_center_y - 0.25 * eye_size, size=eye_resolution)
        self.circle_arc(p_eye_west, p_eye_north, center=p_eye_center, name="eye")
        self.circle_arc(p_eye_north, p_eye_east, center=p_eye_center, name="eye")
        self.spline([p_eye_west, p_eye_south, p_eye_east], name="eye")

        self.plane_surface("mouth", "fin", "curved", name=self.domain_name, holes=[["eye"]])


class MeshTestProblem(Problem):
    def __init__(self):
        super().__init__()
        self.fish_size = 0.5 * meter
        self.resolution = 0.1
        self.mesh_mode = "quads"

    def define_problem(self):
        mesh = GmshFishMesh(size=self.fish_size)
        mesh.default_resolution = self.resolution
        mesh.mesh_mode = self.mesh_mode
        self.add_mesh(mesh)
        self.set_scaling(spatial=self.fish_size)

        eqs = MeshFileOutput() + PoissonEquation(name="u", source=1, coefficient=1*meter**2)
        eqs += DirichletBC(u=0) @ ["fin", "mouth", "curved"]
        eqs += NeumannBC(u=1*meter) @ "eye"
        self.add_equations(eqs @ "fish")


if __name__ == "__main__":
    with MeshTestProblem() as problem:
        problem.solve()
        problem.output_at_increased_time()
```

Key points:
- Boundary names are assigned per-line/arc/spline segment (`name="mouth"` etc.),
  matching several segments to the same name is fine (they become one boundary).
- `holes=[["eye"]]` cuts a hole in the surface bounded by the `"eye"`-named loop —
  the same mechanism is used for multiply-connected domains (e.g. a droplet with an
  internal bubble).
- Inside `GmshTemplate.define_geometry`, the owning `Problem`'s parameters are
  reachable via `cast(MyProblem, self.get_problem()).some_attr` if the geometry needs
  to depend on problem-level state rather than the mesh template's own constructor args.

## 6. Combined spatial + temporal adaptivity

A rotating-flow convection-diffusion problem where both mesh refinement
(`spatial_adapt=1`, driven by a `SpatialErrorEstimator`) and time-step size
(`temporal_error=1`) adapt automatically during `run(...)`.

```python
from pyoomph import *
from pyoomph.expressions import *

class ConvectionDiffusionEquation(Equations):
    def __init__(self, u, D, advection_in_partial_integration, space="C2"):
        super().__init__()
        self.u, self.D, self.space = u, D, space
        self.advection_in_partial_integration = advection_in_partial_integration

    def define_fields(self):
        self.define_scalar_field("c", self.space)

    def define_residuals(self):
        c, phi = var_and_test("c")
        advection = -weak(self.u * c, grad(phi)) if self.advection_in_partial_integration else weak(div(self.u * c), phi)
        # TPZ/MPT time-stepping can outperform BDF2 for pure-advection-dominated transport
        self.add_residual(time_scheme("TPZ", weak(partial_t(c), phi) + advection + weak(self.D * grad(c), grad(phi))))


class ConvectionDiffusionProblem(Problem):
    def __init__(self):
        super().__init__()
        self.u = 2*pi*vector([-var("coordinate_y"), var("coordinate_x")])  # solid-body rotation, one turn at t=1
        self.D = 0.001
        self.L, self.N, self.max_refinement_level = 1, 4, 5
        self.advection_in_partial_integration = False

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(lower_left=-self.L/2, size=self.L, N=self.N))
        eqs = ConvectionDiffusionEquation(self.u, self.D, self.advection_in_partial_integration)
        eqs += MeshFileOutput()

        bump_pos = vector([-self.L/5, 0])
        bump_width = 0.005 * self.L
        xdiff = var("coordinate") - bump_pos
        eqs += InitialCondition(c=exp(-dot(xdiff, xdiff) / bump_width))

        for b in ["top", "left", "right", "bottom"]:
            eqs += DirichletBC(c=0) @ b

        # jumps of grad(c) across element boundaries, both now and at the previous timestep, drive refinement
        eqs += SpatialErrorEstimator(grad(var("c")), evaluate_in_past(grad(var("c"))))
        self.add_equations(eqs @ "domain")


if __name__ == "__main__":
    with ConvectionDiffusionProblem() as problem:
        problem.run(1, outstep=0.01, maxstep=0.0025, spatial_adapt=1, temporal_error=1)
```

Key points:
- `max_refinement_level` (a `Problem` attribute) caps how deep `spatial_adapt=1` may
  refine any single element.
- `SpatialErrorEstimator` takes any number of expressions whose *jump across element
  boundaries* is used as the refinement indicator — not just gradients of a field.
- `temporal_error=<tol>` in `run(...)` enables adaptive time-stepping (rejecting/
  shrinking steps whose estimated local error exceeds `tol`); pair with
  `TemporalErrorEstimator(**fieldfactors)` to weight which fields contribute to the
  error estimate if the default (all fields equally) isn't appropriate.

## 7. Save and resume a simulation

pyoomph calls this "state" saving rather than "restart"/"checkpoint": `save_state`
dumps the full problem (mesh, dofs, output-step counter, continuation/eigen data) to
a binary file; `load_state` reconstructs it. Common uses: branch a parameter scan from
a converged base state without resolving it each time, or resume a long transient run.

```python
with MyProblem() as problem:
    problem.run(10*second, outstep=1*second)
    problem.save_state("checkpoint.dump")

# ... later, possibly in a different script/process, after rebuilding the same Problem:
with MyProblem() as problem:
    problem.initialise()                 # build mesh/equations without solving
    problem.load_state("checkpoint.dump")  # restores dofs, mesh, and output numbering
    problem.run(20*second, outstep=1*second)  # continues from where it left off
```

For a parameter scan that always branches off the *same* converged state (rather than
continuing forward from it), reload before each `go_to_param`/`arclength_continuation`
call, e.g.:
```python
problem.solve()
problem.save_state("start.dump")
for theta_deg in [60, 90, 120]:
    problem.load_state("start.dump", ignore_outstep=True)
    problem.go_to_param(theta=theta_deg*degree, reset_pars=False)
```
`load_state(..., ignore_outstep=True)` avoids clobbering/reusing the original output
file's step numbering when branching like this. `ignore_eigendata=True`/
`ignore_continuation_data=True` are available if a saved state carries stability/
continuation data you don't want restored.
