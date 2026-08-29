# pyoomph — advanced topics for AI coding assistants

Companion to [`AGENTS.md`](../AGENTS.md); for running any of this in parallel see
[`parallel.md`](parallel.md). Covers four advanced areas: bifurcation/stability
analysis (eigenvalues, bifurcation tracking, branch switching, periodic orbits,
deflation), embedding custom (non-symbolic or piecewise) C code into a weak form,
Discontinuous Galerkin discretizations, and the internals of ALE/moving-mesh
and remeshing. These are all built on the same `Equations`/`Problem`/weak-form
machinery described in `AGENTS.md` — nothing here changes that model, it only adds
more powerful ways to use it. If in doubt, grep the cited source file directly.

## 1. Bifurcation analysis and linear stability

All methods below are on `Problem` (`pyoomph/generic/problem.py`) and operate on
whatever residuals/Jacobian the currently assembled equations produce — no special
setup beyond building the problem normally. See `agents/examples.md` recipe 4 for the
simpler parameter-continuation primitives (`go_to_param`, `arclength_continuation`)
that these build on.

### Before anything else: `setup_for_stability_analysis`

```python
problem.setup_for_stability_analysis(analytic_hessian=True, azimuthal_stability=None,
                                      additional_cartesian_mode=None)
```
Call this in `define_problem` (or before the first solve) whenever you intend to do
bifurcation tracking, branch switching or normal-mode stability. It tells the code
generator to emit the extra derivatives those augmented systems need — above all the
**analytic Hessian** (second derivatives of the residual), and the azimuthal/Cartesian
normal-mode variants of the residual when `azimuthal_stability=True` /
`additional_cartesian_mode=True`. Without it the Hessian is taken by finite differences
where it is available at all, which is slower and markedly less robust: bifurcation
tracking that stalls or converges to nonsense is very often a missing call here.
`problem.debug_analytic_hessian_by_fd()` cross-checks the analytic Hessian against
finite differences when you suspect it.

### Eigenvalue problems

```python
eigvals, eigvects = problem.solve_eigenproblem(
    n, shift=0, which="LM", azimuthal_m=None, normal_mode_k=None, normal_mode_L=None,
    filter=None, report_accuracy=False, sort=True)
```
Solves the generalized eigenproblem of the current Jacobian/mass matrix for the `n`
eigenvalues, shift-inverted about `shift` — so `shift=0` is what you want near a steady
bifurcation, and it is the default. (`which` selects the target of the *shifted* problem;
leave it alone unless you know you need something else.) **The returned list is not
guaranteed to be ordered by real part**, so pick the leading mode yourself with
`max(range(len(ev)), key=lambda i: ev[i].real)` rather than trusting `eigvals[0]`.
Constraint rows with no time derivative (pressure, Lagrange multipliers) are fine — they
simply contribute infinite eigenvalues that the shift-invert filters out. Requires the system to have exactly one time-derivative order
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
each other. Declare the intent up front with
`setup_for_stability_analysis(azimuthal_stability=True)` (or
`additional_cartesian_mode=True`), put `AxisymmetryBC() @ "<axis boundary>"` on the
symmetry axis, and pass one `m` at a time — `azimuthal_m` accepts a list, but the return
shape for a list is not worth guessing. A worked check: the decay modes of the diffusion
equation on a unit disc, meshed 1D in `r` with `set_coordinate_system("axisymmetric")`,
reproduce `lambda = -(j_{m,1})**2` (-5.7832, -14.6820, -26.3746 for m=0,1,2) to 1e-10.
Serial azimuthal eigensolves work with the default scipy backend; complex PETSc/SLEPc is
about robustness and size, not correctness. `problem.refine_eigenfunction(numadapt=1, eigenindex=0)` spatially adapts
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
extra unknown. `bifurcation_type=None` autodetects from the last eigensolve; a pitchfork of a trivial
branch is commonly reported as `"fold"`, which tracks the same point correctly. After
`solve()` the converged critical value is read back from `parameter.value`.
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
            problem.output(increase_time_for_PVD=True)

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

### Classifying a bifurcation and switching onto the new branch

Once `activate_bifurcation_tracking(parameter)` has converged onto the point,
`problem.classify_bifurcation(parameter)` computes the normal form and returns a dict
whose `["type"]` is `"fold"`, `"transcritical"`, `"pitchfork"`, `"hopf"`, ... For every
type except a fold there is a *second* branch through the point, and

```python
ds = problem.switch_branch(parameter, normal_form=normal_form, direction=-1)
```
steps onto it and returns a first arclength step size — deliberately small, and it grows
again by itself during the following `arclength_continuation` calls. `direction=±1`
chooses which of the two ways along the new branch to go. `problem.backup_state()` /
`problem.load_state(state)` are the idiom for parking the state just before a
bifurcation, scanning the rest of the current branch, and then coming back to it
(`docs/source/tutorial/temporal/stability/bifurcation_branch_switching.py` does exactly
this loop). `activate_eigenbranch_tracking(...)` follows an eigenbranch without
demanding `Re(λ)=0`.

### Periodic orbits and Floquet multipliers

A Hopf bifurcation gives birth to a limit cycle, and pyoomph solves for the whole orbit
at once: the time dependence over one period is discretized (collocation, periodic
B-splines, ...), the period `T` becomes an unknown, and a phase constraint removes the
time-shift invariance. This needs the residual to be **first order in time** (introduce
auxiliary dofs otherwise), writable as `M(x)·∂ₜx + R₀(x) = 0`.

```python
problem.solve()
problem.solve_eigenproblem(n=1)                       # give the tracker an eigenvector
problem.activate_bifurcation_tracking("rho", "hopf")
problem.solve()                                        # sit exactly on the Hopf

with problem.switch_to_hopf_orbit(NT=100) as orbit:    # period + initial guess derived for you
    print("supercritical:", orbit.starts_supercritically(), "T =", orbit.get_T())
    orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))
    ds = orbit.get_init_ds()                           # sign from the Lyapunov coefficient
    while problem.rho.value > 16:
        ds = problem.arclength_continuation("rho", ds)  # continue the ORBIT in the parameter
        orbit.output_orbit("orbit_at_rho_{:.4f}".format(problem.rho.value))
```

- `switch_to_hopf_orbit(...)` is a context manager returning a `PeriodicOrbit`. Use
  `problem.activate_periodic_orbit_handler(T, mode=..., NT=..., order=...)` instead when
  you already have a guess (e.g. one period sampled from a time integration) rather than
  a Hopf point.
- `mode` selects the time discretization: `"collocation"` (default), `"bspline"`,
  `"central"`, `"BDF2"`, `"floquet"`. `T_constraint` is `"phase"` (default) or `"plane"`.
- `PeriodicOrbit` methods: `get_T`, `get_init_ds`, `starts_supercritically`,
  `output_orbit(subdir)`, `iterate_over_samples`, `evaluate_observable_time_integral`,
  `change_sampling(...)` (re-discretize in time without losing the orbit),
  `load_from_state` (restart a tagged orbit from a saved state).
- **Stability of the orbit** comes from the Floquet multipliers:
  `problem.get_floquet_multipliers(n=..., method="condensed"|"periodic_schur"|"eigenproblem")`.
  A periodic orbit always has a trivial multiplier at exactly 1 (the time-shift
  direction); `ignore_periodic_unity` drops it, but **never discard it by a tolerance** on
  `|μ−1|` — a genuine second multiplier passing near 1 is precisely the bifurcation you
  are looking for. The orbit is stable iff every non-trivial `|μ| < 1`.

### Disconnected solutions: deflation

Arclength continuation and branch switching only ever reach solutions *connected* to
the one you started from. Deflation (Farrell, Beentjes & Birkisson, arXiv:1603.00809)
finds the rest: after converging to a solution, the residual is divided by a factor that
blows up there, so Newton cannot return to it and must find something else.

```python
# all solutions at the current parameter value
for dofs in problem.iterate_over_multiple_solutions_by_deflation(deflation_alpha=0.1,
                                                                 deflation_p=2,
                                                                 num_random_tries=5):
    problem.output()

# or: sweep a parameter, reporting (branch index, parameter value, dofs) as branches appear
for branch, value, dofs in problem.deflated_continuation(max_branches=10, r=[-1, 1]):
    ...
```
`set_deflation_operator(op)`/`get_deflation_operator()` give manual control.
`deflated_solve_by_eigenperturbation(eigenindex=...)` pushes off along an eigenvector
instead of a random perturbation, which is the better first move just past a
bifurcation. All of this works under `mpirun` and `--distribute`. The random
perturbations use a fixed `random_seed=0` by default, which is also what keeps the ranks
in step — pass `random_seed=None` only in a serial run.

### Interactive exploration: the bifurcation GUI

For building a bifurcation diagram by hand rather than scripting every step:
```python
from pyoomph.utils.bifurcation_gui import BifurcationGUI
with MyProblem() as problem:
    gui = BifurcationGUI(problem, "my_parameter")
    ...
```
A tkinter front-end that drives continuation, eigenvalues, bifurcation tracking, branch
switching, orbits and deflation on the live problem, and tags states so a run can be
resumed. See `docs/source/tutorial/advstab/bifgui.rst`.

### Other stability utilities

- `pyoomph.utils.periodic_driving_response` — frequency response of a periodically
  driven system (`docs/source/tutorial/advstab/response.rst`).
- `pyoomph.utils.lyapunov` — Lyapunov exponents of a chaotic trajectory.
- `pyoomph.utils.paramscan.ParallelParameterScan` — run many independent parameter
  values as separate processes; unrelated to MPI, and the right tool when the sweep,
  not the single solve, is what is slow.

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
**Using it.** Instantiate once, then *call* the instance with the arguments; it returns a
**tuple** of `get_num_returned_scalars(...)` symbolic expressions (a tuple even when there
is only one), which drop straight into a weak form:

```python
class PiecewiseConductivity(CustomMultiReturnExpression):
    def get_num_returned_scalars(self, nargs): return 1
    def eval(self, flag, arg_list, result_list, derivative_matrix):
        T = arg_list[0]                       # plain Python floats, so `if` works
        if T <= 1.0:
            result_list[0] = 1.0
            if flag: derivative_matrix[0] = 0.0
        else:
            result_list[0] = 1.0 + self.a*(T - 1.0)
            if flag: derivative_matrix[0] = self.a

k_of_T = PiecewiseConductivity()             # build it ONCE, outside define_residuals
...
T, Ttest = var_and_test("T")
k, = k_of_T(T)                                # unpack the 1-tuple
self.add_residual(weak(k*grad(T), grad(Ttest)))
```
`CustomMultiReturnExpression` and `CustomMathExpression` are re-exported by
`from pyoomph.expressions import *`. `flag` is a truthy int meaning "derivatives are
wanted"; `derivative_matrix[i*nargs + j]` is `d result_i / d arg_j`. `generate_c_code()`
returns a bare C statement block using those same names — give any local you declare an
unlikely name, since it is spliced into the generated element code.

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
`agents/materials.md`) — prefer filling in the analytic Jacobian by hand when the
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

**Do not hand-write that override naively.** `grad` on a boundary domain is the *surface*
gradient, so the consistency and symmetrisation terms of a textbook Nitsche condition
(`weak(D*grad(c), ctest*n)` and friends) silently evaluate to zero there, leaving an
inconsistent pure-penalty method. Measured on the 1D Pe=50 advection-diffusion boundary
layer with `"D2"` and N=40: the hand-written Nitsche override gives a maximum error of
2.4e-1, while simply *not* overriding it — i.e. letting `DirichletBC` fall back to strong
pinning of the boundary dof — gives 4.2e-3. **Leave it alone unless you have a specific
reason**; strong pinning of a nodal DG dof is consistent and is the right default.

Full example — 1D DG advection-diffusion with upwinding and a symmetric-interior-penalty
diffusion term (`docs/source/tutorial/dg/convection_diffusion.py`):
```python
from pyoomph import *
from pyoomph.expressions import is_DG_space  # also reachable via "from pyoomph import *"

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

### Unknowns on the facet skeleton (HDG traces, mortar multipliers)

The skeleton is a mesh in its own right and can carry its own dofs — attach equations
to the reserved subdomain `"_internal_facets_"` of a bulk mesh and declare fields in
them (`docs/source/tutorial/dg/facetfields.rst`, tests in
`tests/test_internal_facet_fields.py`):
```python
class MortarCoupling(Equations):
    def define_fields(self):
        self.define_scalar_field("lam", "D0")      # only discontinuous spaces here

    def define_residuals(self):
        lam, mu = var_and_test("lam")
        u, v = var("u"), testfunction("u")         # the bulk field, seen from the facet
        self.add_residual(weak(lam, jump(v)) + weak(jump(u), mu))
        self.set_facet_recovery("lam", -dot(var("normal"), avg(grad(u))))

eqs += MortarCoupling() @ "_internal_facets_"
```
- A bulk class that already writes its facet weak form with
  `add_interior_facet_residual` can declare the facet unknown in place instead, with
  `at_internal_facets=True` on `define_scalar_field` / `define_vector_field` /
  `define_tensor_field` / `set_facet_recovery`. The field then lives *only* on the
  skeleton. It needs `self.requires_interior_facet_terms = True` in `__init__` (which
  `add_interior_facet_residual` requires anyway): `define_fields()` runs long after the
  skeleton domain would have had to be created, so the keyword fills an existing skeleton
  rather than making one, and says so if there is none.
- Declaring a field on `"_internal_facets_"` directly implies the skeleton, so
  `requires_interior_facet_terms` need not be set in that case. Allowed spaces: `"D0"` (one constant per facet), `"DL"` (constant +
  one gradient mode per facet direction), and the nodal DG spaces
  `"D1"/"D2"/"D1TB"/"D2TB"`. Continuous spaces raise `NotImplementedError` — the dofs
  live in the facet element's own internal `Data`, and interior nodes are plain
  `pyoomph::Node`, not `BoundaryNode`, so a shared per-node facet dof cannot exist. One
  set of dofs per facet: facets meeting at a vertex share nothing.
- Inside facet equations, a *bulk* field still needs `jump()`/`avg()`, but the facet
  field itself is used bare: it is single-valued, so `var("lam", domain="|-")` and
  `jump/avg(var("lam"), at_facet=True)` are a codegen-time `RuntimeError` (they used to
  read the never-numbered opposite dummy element and silently return zero).
  `DirichletBC`, `InitialCondition`, `IntegralObservables` and `MeshFileOutput` work on
  the skeleton as on any other domain (`p.get_mesh("domain/_internal_facets_")`).
- Adaptivity: the skeleton is destroyed and rebuilt, and the values are carried across by
  an Eulerian snapshot plus a least-squares refit, all history levels — for *every*
  discontinuous space, the nodal `Dx` ones included; only the basis each is fitted in
  differs. Facets born *inside* a refined element have no counterpart in the old skeleton
  at all: they keep zero plus a one-time warning, unless
  `Equations.set_facet_recovery(field, expr)` says how to rebuild them from the bulk — the
  right default for a trace/flux, and written to every time level so no spurious
  `partial_t` appears. `mesh.get_discontinuous_unrestored_elements()` lists what stayed
  empty (it describes the last transfer, not the current state). Triangle/tet skeletons
  still only survive *uniform* refinement.
- Remeshing: `InternalInterpolator` pulls each new facet's values from the closest old
  facet within the same old bulk element (`ProjectionInternalInterpolator` re-applies
  the same transfer after its projection solve — skeleton fields are not L2-projected).
  Exact only for an identical remesh, otherwise O(nearest-facet distance × gradient), so
  `set_facet_recovery` is again the recommended default.
- State files: the facet values are in them, per facet element, addressed by the bulk
  element the facet belongs to plus its face index there. That is partition-independent,
  so a state written serially loads distributed and vice versa. Before this they were not
  in the file at all — not even `DL`/`D0` — and a load silently refitted whatever the
  loading process happened to hold on its facets, which no test that solves after loading
  can notice.
- MPI: works under `--distribute` for every discontinuous facet space. A facet shared by
  two processes is owned by the one that assembles it, and the other holder's copy is a
  halo, so it is numbered once and its equation numbers and values are copied across
  (`Problem.setup_interior_facet_halo_scheme()`, called for you after distributing,
  adapting, remeshing, loading a state file and load-balancing).
- **Pitfall**: a multiplier on *every* facet enforcing `jump(u) = 0` is rank deficient in
  2D/3D — facets meeting at a shared vertex/edge re-impose continuity there, so the
  constraints are linearly dependent and the saddle-point system is singular. That is a
  property of naive mortar methods, not of the facet fields (1D is fine, which is why the
  tutorial and the test use `LineMesh`). Use SIP-DG for plain continuity, and facet
  unknowns for genuine traces, e.g. `weak(p - avg(u), testfunction(p))` with `p` on
  `"DL"`, which reproduces the trace exactly. `DG_alpha = 1` is enough on every element
  family, tetrahedra included — the belief that tets needed `≈ 10–40` for coercivity came
  from the tetrahedron winding described in `dev_docs/mesh_construction.md` §6, which
  pointed their face normals inwards and made the scheme inconsistent, and which
  `add_tetra_3d_C1/C2` now repair.

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
same pattern as recipe 2 in `agents/examples.md` but for the mesh field);
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

