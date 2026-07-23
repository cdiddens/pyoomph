# pyoomph — advanced topics for AI coding assistants

Companion to [`AGENTS.md`](AGENTS.md). Covers four advanced areas: bifurcation/
stability analysis, embedding custom (non-symbolic or piecewise) C code into a weak
form, Discontinuous Galerkin discretizations, and the internals of ALE/moving-mesh
and remeshing. These are all built on the same `Equations`/`Problem`/weak-form
machinery described in `AGENTS.md` — nothing here changes that model, it only adds
more powerful ways to use it. If in doubt, grep the cited source file directly.

## 1. Bifurcation analysis and linear stability

All methods below are on `Problem` (`pyoomph/generic/problem.py`) and operate on
whatever residuals/Jacobian the currently assembled equations produce — no special
setup beyond building the problem normally. See `AGENTS_EXAMPLES.md` recipe 4 for the
simpler parameter-continuation primitives (`go_to_param`, `arclength_continuation`)
that these build on.

### Eigenvalue problems

```python
eigvals, eigvects = problem.solve_eigenproblem(
    n, shift=0, which="LM", azimuthal_m=None, normal_mode_k=None, normal_mode_L=None,
    filter=None, report_accuracy=False, sort=True)
```
Solves the generalized eigenproblem of the current Jacobian/mass matrix for the `n`
eigenvalues nearest `shift` (`which` controls SciPy/SLEPc's target mode, default
largest-magnitude). Requires the system to have exactly one time-derivative order
(reduce higher-order-in-time systems to first order via auxiliary variables first).
Choose the backend with `problem.set_eigensolver("scipy")` (default) or
`problem.set_eigensolver("slepc")` (PETSc/SLEPc — more robust for large/sparse
problems, needed for `azimuthal_m`/`normal_mode_k`-based stability in practice).

**Symmetry-breaking (normal-mode) stability**, for problems where the mesh only
resolves a subset of the physical dimensions (e.g. an axisymmetric base state whose
stability to non-axisymmetric perturbations you want to check):
- `azimuthal_m=m` (int or list) — perturbations `~exp(i*m*theta)` on an axisymmetric mesh.
- `normal_mode_k=k` (or `normal_mode_L=L`, converted internally via `k=2*pi/L`) — perturbations `~exp(i*k*x)` along an extra homogeneous Cartesian direction not in the mesh.

Both route through the same `solve_eigenproblem(...)` call; mutually exclusive with
each other. `problem.refine_eigenfunction(numadapt=1, eigenindex=0)` spatially adapts
the mesh based on a chosen eigenfunction's gradient and re-solves base state + eigenproblem.

### Scanning for a bifurcation via eigenvalues

```python
for param_value, eigenvalue in problem.find_bifurcation_via_eigenvalues(
        parameter, initstep, shift=0, neigen=6, epsilon=1e-8,
        azimuthal_m=None, normal_mode_k=None, eigenindex=0):
    ...  # yields as it bisects/steps toward Re(eigenvalue)=0
```
A generator: steps `parameter` via `arclength_continuation`, recomputing the
eigenproblem each step, converging onto the parameter value where
`eigenvalue[eigenindex]` crosses zero. Raises if the starting state is already unstable.

### Locking onto a bifurcation exactly: `activate_bifurcation_tracking`

```python
problem.activate_bifurcation_tracking(parameter, bifurcation_type=None,  # "hopf"/"fold"/"pitchfork"/"azimuthal"/"cartesian_normal_mode"
                                       eigenvector=None, omega=None,
                                       azimuthal_mode=None, cartesian_wavenumber_k=None)
```
Augments the system so that subsequent `solve()`/`arclength_continuation()` calls
solve simultaneously for the base state, the critical eigenvector, and `parameter` —
converging exactly onto the bifurcation point, with `parameter` itself becoming an
extra unknown. `bifurcation_type=None` autodetects from the last eigensolve.
`parameter=None` instead tracks the current eigenbranch without forcing `Re(λ)=0`
("eigenbranch continuation"). For `"azimuthal"`/`"cartesian_normal_mode"`, pass
`azimuthal_mode`/`cartesian_wavenumber_k` explicitly, or they're picked up from the
last normal-mode eigensolve. Once active, continue the *tracked bifurcation itself*
along a second parameter with ordinary `arclength_continuation(problem.other_param, ds)`
calls — this traces out a bifurcation curve in a 2-parameter plane (e.g. a stability
boundary), which is the main reason to use this over just watching eigenvalues.

### Full example: eigenvalue scan then bifurcation tracking (Kuramoto-Sivashinsky, hexagonal pattern)

```python
from kuramoto_sivanshinsky import *  # reuse a base Problem defining DampedKuramotoSivashinskyEquation

class KSEBifurcationProblem(Problem):
    def __init__(self):
        super().__init__()
        self.param_gamma = self.define_global_parameter(gamma=0.24)
        self.param_delta = self.define_global_parameter(delta=0)
        # ... mesh / equations / initial conditions as in the base problem ...

import pyoomph.solvers.petsc  # needed for the "slepc" eigensolver backend

if __name__ == "__main__":
    with KSEBifurcationProblem() as problem:
        problem.initialise()
        problem.set_initial_condition(ic_name="hexdots")
        problem.solve(timestep=10)   # relax transient towards the stationary pattern
        problem.solve()              # stationary solve
        problem.set_eigensolver("slepc")

        def output_with_eigen():
            eigvals, eigvects = problem.solve_eigenproblem(6, shift=0)
            # ... log problem.param_gamma.value, eigvals[0] ...
            problem.output_at_increased_time()

        output_with_eigen()
        ds = 0.001
        while problem.param_gamma.value > 0.23:
            ds = problem.arclength_continuation(problem.param_gamma, ds, max_ds=0.005)
            output_with_eigen()

        # Follow-up script: lock onto the fold found above and trace it in a 2nd parameter
        # problem.activate_bifurcation_tracking(problem.param_gamma, "fold")
        # ds = 0.001
        # while ...:
        #     ds = problem.arclength_continuation(problem.param_delta, ds)
```
See `docs/source/tutorial/pde/patterns/eigen.rst` and `biftrack.rst` (with example
scripts `kuramoto_sivanshinsky_arclength_eigen.py` /
`kuramoto_sivanshinsky_bifurcation.py`) and `docs/source/tutorial/advstab/` for the
full azimuthal/Cartesian-normal-mode tutorials.

## 2. Custom (non-symbolic / piecewise) C code in a weak form

Most physics is expressed as pure GiNaC `Expression`s and differentiated
automatically. When a term is not representable that way — piecewise/branching
functions, calls into an external numerical routine, or anything needing manual
control over the generated C — use `CustomMultiReturnExpression`
(`pyoomph/expressions/cb.py`), the base used internally for e.g. safe division, tensor
inversion/exponential, spline interpolation, UNIFAC activity coefficients, and
piecewise phase-field potentials (`NSCH.py`'s `PiecewiseNSCHPotential`).

Subclass contract:
```python
class MyExpr(CustomMultiReturnExpression):
    def get_num_returned_scalars(self, nargs: int) -> int:
        ...  # required: length of result_list

    def eval(self, flag: int, arg_list, result_list, derivative_matrix) -> None:
        ...  # required Python fallback; if flag, also fill derivative_matrix[i*nargs+j] = d result_i / d arg_j

    def generate_c_code(self) -> str:
        ...  # optional: raw C snippet spliced into the generated element code and JIT-compiled
             # (uses the same arg_list/result_list/derivative_matrix/flag/nargs names).
             # End with FILL_MULTI_RET_JACOBIAN_BY_FD(1e-8) to get a finite-difference Jacobian for free.
```
Optional refinements: `process_args_to_scalar_list`/`process_result_list_to_results`
(pack/unpack tensors into the flat scalar buffer), `use_symbolic_derivative(arg_list, i, j)`
(supply an exactly-known derivative entry, e.g. 0, instead of relying on the general
mechanism), `use_c_code` (`"auto"`/`True`/`False` — whether to prefer the C path),
`set_debug_python_vs_c_epsilon(eps)` (cross-check the C and Python evaluations while developing).

Concrete pattern (trimmed from `pyoomph/equations/NSCH.py`'s `PiecewiseNSCHPotential`,
returning both a potential and its derivative for a double-well with a quadratic
tail outside `[-1,1]` to avoid phase-field overshoot):
```python
class PiecewiseNSCHPotential(CustomMultiReturnExpression):
    def get_num_returned_scalars(self, nargs):
        return 2  # (potential, dpotential/dphi)

    def eval(self, flag, arg_list, result_list, derivative_matrix):
        phi = arg_list[0]
        if -1 <= phi <= 1:
            result_list[0] = (phi**2 - 1)**2
            if flag: derivative_matrix[0] = 4*phi*(phi**2 - 1)
        else:
            # quadratic extension matching value+slope at +-1
            ...
        if flag:
            derivative_matrix[1*1+0] = 0  # d(dpotential)/dphi entry, filled analogously

    def generate_c_code(self):
        return "if(arg_list[0]>=-1 && arg_list[0]<=1){ ... } else { ... }"
```
A finite-difference Jacobian (whether via Python `eval` or `FILL_MULTI_RET_JACOBIAN_BY_FD`)
is noticeably slower and slightly less accurate than an analytic one, and is
**incompatible with bifurcation tracking** in some cases (see the UNIFAC note in
`AGENTS_MATERIALS.md`) — prefer filling in the analytic Jacobian by hand when the
derivative is easy, and reserve FD only for genuinely awkward cases.

The single-return analogue is `CustomMathExpression` (`cb.py`): override
`eval(self, arg_array) -> float`, optionally `derivative(self, index) -> CustomMathExpression`
for an exact symbolic derivative (defaults to a finite-difference derivative), and
`get_argument_unit`/`get_result_unit` if the function has physical units.

### Compiler selection

`Problem.set_c_compiler(name_or_instance)` chooses the backend used to compile
generated element code: `"tcc"` (bundled TinyC — fast in-memory compile, no
optimization; the default when no system compiler is detected), or `"system"`
(optimizing compiler via `distutils.ccompiler`, roughly `-O3 -fPIC -march=native`).
`get_default_c_compiler()` auto-ranks available backends. Set the environment
variable `PYOOMPH_DEBUG=1` to instead compile with `-O0 -g3` symbols for
debugging the generated C directly. CLI flags `--tcc`/`--distutils`/`--fast-math`
toggle the same knobs from the command line. Independently of the backend, a
content-addressed JIT code cache (`pyoomph/generic/jit_cache.py`, see
`python -m pyoomph cache usage`/`cache clear`) reuses compiled shared libraries
across runs whenever the generated code is unchanged - `--no-cache` disables it.

## 3. Discontinuous Galerkin (DG) methods

Use a `"D1"/"D2"/"D1TB"/"D2TB"` (or `"DL"`/`"D0"`) space instead of `"C1"/"C2"` for
a field to make it discontinuous across elements — see `AGENTS.md`'s space table.
Two extra ingredients are then needed: interior-facet (jump) residual terms, and a
choice of weak vs. strong Dirichlet BC handling.

**Facet-term plumbing** (`pyoomph/generic/codegen.py`):
- Set `self.requires_interior_facet_terms = True` (usually in `__init__`, e.g.
  `self.requires_interior_facet_terms = is_DG_space(self.space)`) so pyoomph builds
  the interior-facet "skeleton" mesh needed to assemble jump terms.
- `self.add_interior_facet_residual(expr)` — like `add_residual`/`add_weak`, but
  assembled on that facet skeleton instead of the bulk mesh.
- Facet-aware expression helpers (`pyoomph/expressions/generic.py`):
  `jump(f, at_facet=False)` = `f⁺ − f⁻` across a facet, `avg(f, at_facet=False)` =
  `(f⁺+f⁻)/2`. Pass `at_facet=True` when `f` itself involves the facet normal (e.g.
  an upwind flux), so it's evaluated consistently on both sides of the facet.
  `var("normal")` inside facet terms is the facet normal (pointing from the `+` to
  the `-` element); `var("cartesian_element_length_h")`/`var("element_length_h")`
  give the local element size `h` for penalty scaling.
- A `DG_alpha` constructor kwarg (seen on e.g. `NavierStokesEquations`) is the
  convention for a user-tunable interior-penalty coefficient — follow the same
  naming when adding DG support to a custom equation.

**Weak vs. strong Dirichlet BCs**: `Equations.get_weak_dirichlet_terms_for_DG(fieldname, value)`
can be overridden to supply Nitsche-type facet terms for imposing a Dirichlet value
weakly (falls back to strong pinning if it returns `None`); `DirichletBC(..., prefer_weak_for_DG=True)`
(the default) uses this automatically when the field's space is a DG space.

Full example — 1D DG advection-diffusion with upwinding and a symmetric-interior-penalty
diffusion term (`docs/source/tutorial/dg/convection_diffusion.py`):
```python
from pyoomph import *
from pyoomph.equations.SUPG import *  # for is_DG_space / element-size helpers

class ConvectionDiffusionEquation(Equations):
    def __init__(self, u, D, space="C2", alpha_DG=2):
        super().__init__()
        self.u, self.D, self.space, self.alpha_DG = u, D, space, alpha_DG
        self.requires_interior_facet_terms = is_DG_space(self.space)

    def define_fields(self):
        self.define_scalar_field("c", self.space)

    def define_residuals(self):
        c, ctest = var_and_test("c")
        self.add_weak(partial_t(c), ctest)
        self.add_weak(self.D*grad(c) - self.u*c, grad(ctest))

        if is_DG_space(self.space):
            h_avg = avg(var("cartesian_element_length_h"))
            n = var("normal")
            un_upwind = (dot(self.u, n) + absolute(dot(self.u, n))) / 2
            facet_terms = weak(self.D*(self.alpha_DG/h_avg)*jump(c), jump(ctest))   # penalty
            facet_terms += -weak(self.D*jump(c)*n, avg(grad(ctest)))                # symmetrization
            facet_terms += -weak(self.D*avg(grad(c)), jump(ctest)*n)               # consistency
            facet_terms += weak(jump(un_upwind*c, at_facet=True), jump(ctest))      # upwind advection
            self.add_interior_facet_residual(facet_terms)
```
(Written here with `+=` throughout for clarity; double-check any `=` re-assignment
you see in an existing script before copying it — a `facet_terms = ...` line
overwrites rather than accumulates previous terms, which is easy to introduce by
accident when editing a SIP-diffusion formulation like this one.)

## 4. ALE / moving-mesh internals and remeshing

### How `activate_coordinates_as_dofs` actually works

`BaseMovingMeshEquations` (`pyoomph/equations/ALE.py`), the base of all mesh-motion
equation classes, calls `self.activate_coordinates_as_dofs(coordinate_space=...)`
in `define_fields()`. This turns the mesh's nodal positions into genuine Newton
unknowns (field name `"mesh"`), rather than a fixed geometry. Two coordinate
variables then coexist:
- `var("mesh")` — the current, *Eulerian* nodal position (the actual unknown).
- `var("lagrangian")` — the fixed *reference/material* position (the mesh at t=0, or
  whatever it was last reset to); unaffected by solving.

Mesh-motion equations are typically written as a PDE for the *displacement*
`x - X`, e.g. (the doc-comment example on `activate_coordinates_as_dofs` itself):
```python
def define_residuals(self):
    x, xtest = var_and_test("mesh")
    X = var("lagrangian")
    self.add_weak(grad(x - X, lagrangian=True), grad(xtest, lagrangian=True), lagrangian=True)
```
`SetLagrangianToEulerianAfterSolve` resets `X := x` after every successful Newton
solve (an "updated Lagrangian" scheme) — needed for smoothing PDEs where the natural
reference configuration should track the converged state rather than stay pinned at
`t=0`.

Ready-made mesh-motion equations, all `BaseMovingMeshEquations` subclasses in `ALE.py`:

| Class | Mesh-motion PDE |
|---|---|
| `LaplaceSmoothedMesh(factor=..., symmetrize=False)` | `laplace(x - X) = 0` — cheapest option |
| `SingleDirectionLaplaceSmoothedMesh(direction, ...)` | Laplace-smooth one Cartesian component only, Dirichlet-pin the rest |
| `PseudoElasticMesh(E=..., nu=...)` | linear-elasticity: `div(sigma(x-X)) = 0` |
| `HyperelasticSmoothedMesh(mu=1, kappa=1)` | minimizes a Neo-Hookean-like energy (`add_functional_minimization`) — more robust for large deformation |
| `YeohSmoothedMesh(kappa=1, C1=1, C2=10, C3=0)` | minimizes a 3-term Yeoh hyperelastic energy — more tunable nonlinear stiffening |
| `PrescribedMovingMesh(umesh, lagrangian=False)` | directly prescribes mesh velocity, no smoothing PDE at all |

Supporting equations: `ConnectMeshAtInterface` (match node positions of two
independently-moving domains across a shared interface, Lagrange-multiplier based —
same pattern as recipe 2 in `AGENTS_EXAMPLES.md` but for the mesh field);
`EnforcedInterfacialLaplaceSmoothing` (keep interface nodes equidistant along
arclength as the interface deforms — important near a moving contact line to avoid
element pile-up; has `.with_corners(*boundary_names)`); `EnforceVolumeByPressure`/
`VolumeEnforceStorage`/`VolumeEnforcingBoundary` (hold an enclosed volume fixed by
adjusting internal pressure, e.g. an isolated droplet); `ConstrainPositionsToC1Space`/
`UnconstrainPositionsFromC1Space` (reduce higher-order mesh nodes to lie on the linear
interpolant, to cut mesh-dof count, with an optional spatial predicate).

### Remeshing (2D only)

`RemeshWhen(remeshing_opts=None, *, max_expansion=None, min_expansion=None, min_quality_decrease=None, ...)`
(`pyoomph/equations/generic.py`), added to a bulk domain's equations, monitors each
element's size/quality ratio against its state when the mesh was last (re)built and
flags the domain for remeshing once any element's current/initial size ratio exceeds
`max_expansion`/undershoots `min_expansion`, or its quality ratio drops below
`min_quality_decrease` (catches shape distortion even at constant area).
`RemeshMeshSize(size)` (attach `@"boundary"` or `@"boundary/corner"`) controls target
element size near a given boundary/corner during remeshing.

Requires the mesh template to carry a remesher instance:
```python
mesh = RectangularQuadMesh(N=6)
mesh.remesher = Remesher2d(mesh)   # pyoomph.meshes.remesher.Remesher2d
self.add_mesh(mesh)
```
`Remesher2d` re-triangulates/re-quads a 2D domain from its current (deformed)
boundary nodes via Gmsh. Full example (`docs/source/tutorial/ale/remeshing.py`):
```python
from laplace_smoothed_mesh import *   # a prior example defining a moving-mesh Problem
from pyoomph.meshes.remesher import *

class RemeshingProblem(Problem):
    def __init__(self):
        super().__init__()
        self.remesh_options = RemeshingOptions(max_expansion=2, min_expansion=0.3, min_quality_decrease=0.2)

    def define_problem(self):
        mesh = RectangularQuadMesh(N=6)
        mesh.remesher = Remesher2d(mesh)
        self.add_mesh(mesh)

        eqs = LaplaceSmoothedMesh() + MeshFileOutput()
        eqs += DirichletBC(mesh_x=0, mesh_y=True) @ "left"
        eqs += DirichletBC(mesh_x=True, mesh_y=0) @ "bottom"
        eqs += DirichletBC(mesh_y=1) @ "top"
        xi = var("lagrangian")
        eqs += DirichletBC(mesh_x=1 + 0.5*xi[1]*var("time")) @ "right"  # moving boundary drives the deformation
        eqs += RemeshWhen(self.remesh_options)
        eqs += RemeshMeshSize(size=0.2) @ "left"
        eqs += RemeshMeshSize(size=0.02) @ "right/top"  # finer sizing at a specific corner
        self.add_equations(eqs @ "domain")

if __name__ == "__main__":
    with RemeshingProblem() as problem:
        problem.run(10, outstep=True, startstep=0.5, maxstep=0.5)
```

