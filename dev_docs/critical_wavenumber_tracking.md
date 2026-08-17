# Tracking the critical wavenumber of a normal mode

`CriticalWavenumberTracker` (`pyoomph/generic/bifurcation_tools.py`) solves the co-dimension-2
problem behind the phrase "the instability sets in at `k_c`": not just `Re(lambda)=0` at a wavenumber
you pick, but the point where the neutral curve `gamma_c(k)` has its minimum, so that `k` is found
rather than prescribed. It handles both mode families — the Cartesian mode `~exp(I k z)` and the
azimuthal mode `~exp(I m phi)`, the latter with `m` treated as a real number (see below).

This is a companion to `bifurcation_loci.md`, which covers two-parameter continuation of ordinary
bifurcations.

## What was already there, and what was not

`NormalModeBifurcationTracker` (Python) and `AzimuthalSymmetryBreakingHandler` (C++, reached by
`activate_bifurcation_tracking(param, "cartesian_normal_mode"/"azimuthal", ...)`) both solve
`Re(lambda)=0` at a **fixed** wavenumber. Sweeping it and repeating gives the neutral curve; nothing
found its minimum, and nothing made the wavenumber an unknown.

Three facts made it possible to add this entirely in Python:

1. The mode number is an ordinary global parameter — `normal_mode_k` or `azimuthal_m`.
   `CartesianCoordinateSystemWithAdditionalNormalMode` / `AxisymmetryBreakingCoordinateSystem`
   substitute its symbol into the two generated eigen contributions, while the base residual has it
   substituted to zero. **The base state therefore does not depend on it at all**, which is what
   makes the formulation below legitimate.
2. Because it is a global parameter occurring in those residual forms, `codegen.cpp` already emits
   analytic `dJ/dk` and `dM/dk` per contribution, reachable as
   `MultiAssembleRequest.dJdp("normal_mode_k", contribution)`.
3. `DofAugmentations.add_parameter` pushes the parameter's value pointer straight into the dof
   vector, so making it an unknown is one line in `define_augmented_dofs`.

Below, `k` stands for whichever of the two is in play.

## The formulation

Conventions are the existing ones: `J_c = JR + i JI`, `M_c = MR + i MI` from the real and imaginary
contributions, eigenproblem `(J_c + lambda M_c) V = 0`, perturbation `~exp(lambda t)`.

Let `V' = dV/dk` and `lambda' = dlambda/dk`, both at **fixed** base state and fixed parameter.
Differentiating the eigenproblem in `k`:

    (J_c' + lambda M_c') V  +  lambda' M_c V  +  (J_c + lambda M_c) V'  =  0

Criticality is imposed the way every other tracker here imposes `Re(lambda)=0`: by not introducing
`Re(lambda)` and `Re(lambda')` as unknowns at all. So `lambda = i*omega` and `lambda' = i*mu`.

| | oscillatory neutral mode | stationary neutral mode |
|---|---|---|
| unknowns | `u`, `Vr`, `Vi`, `Wr`, `Wi`, `gamma`, `omega`, `k`, `mu` (`5N+4`) | `u`, `V`, `W`, `gamma`, `k` (`3N+2`) |
| rows | `R`; Re/Im of the eigen equation; Re/Im of its `k`-derivative; four normalisations | `R`; `JR V`; `JR' V + JR W`; two normalisations |

`W = dV/dk` is pinned by the `k`-derivative of the eigenvector normalisation, `<V0,W> = 0`, which is
exactly the two (or one) extra normalisation rows. In the stationary case `lambda = lambda' = 0` and
the mass matrix drops out entirely.

## Where the finite difference is, and why

Write `E` for the eigen row block. The tangency rows are, structurally,

    F  =  E_k  +  E_V W  +  E_omega mu

so for any variable `X`, `dF/dX = d/dk[E_X] + (analytic terms in W and mu)`. The `d/dk[E_X]` group
needs `d2J/dU dk`, `d2J/dgamma dk` and `d2J/dk2` — **pyoomph generates none of these**. They are
obtained by assembling the corresponding first-derivative blocks a second time at `k+delta` and
differencing (`_at_k_offset`, `k_fd_step`).

The **residual is exact**: `F` is built from the analytic `dJ/dk` and `dM/dk`, so Newton converges to
the exact critical point and only the convergence rate is affected. `exact_k_derivative_jacobian=False`
drops the differenced blocks entirely, which is useful to test the analytic ones in isolation.

Consequences worth knowing:

- Two multi-assembly passes per Newton step instead of one (three quantities are re-assembled at the
  shifted `k`: the Hessian blocks, the `gamma`-derivative blocks and the `k`-derivative blocks).
- The differenced blocks are accurate to `~1e-6` relative. That is a Jacobian, so it costs nothing in
  accuracy; `tests/test_critical_wavenumber_tracker.py` checks every column against a central
  difference of the residual at `1e-5`.

## Initial guess

`solve_eigenproblem` is refused once a Python augmentation is installed, so `W` and `mu` have to be
in hand before `set_custom_assembler`. The constructor gets them from one bordered solve on the
converged eigenpair (`_guess_k_derivative`):

    [ J_c + lambda M_c    M_c V ] [ W       ]   [ -(J_c' + lambda M_c') V ]
    [      V0^T             0   ] [ lambda' ] = [            0            ]

`Re(lambda')` that comes out is the tangency residual, i.e. how far the starting `k` is from
critical; it is printed for that reason. A failure here falls back to `W = 0, mu = 0`.

## The azimuthal case: a real `m`

Everything above carries over verbatim with `azimuthal_m` in place of `normal_mode_k`. The one thing
that does not carry over is that `m` is physically an integer.

Making it real is harmless for the *algebra* — `m` enters the generated eigen contributions
analytically, and `dJ/dm` is generated like any other parameter derivative — so the answer means "the
instability sets in between these two integer modes", and the usual next step is to run the ordinary
tracker at each neighbour. What it is **not** harmless for is the **axis**: `AxisymmetryBC` pins a
different set of dofs for `m == 0`, `|m| == 1` and `|m| > 1`, and
`_get_forced_zero_dofs_for_eigenproblem` truncates `m` towards zero before branching. A continuous
`m` has no way to express that, and left alone the mask would silently change under the solver the
moment `m` crossed an integer.

The tracker therefore **assumes the `|m| > 1` regime throughout**:

- `get_forced_to_zero_dofs` is overridden to compute the eigen mask at `m = 2`
  (`HIGH_AZIMUTHAL_MODE_REGIME`) rather than at the current `m`, and it is frozen there for the whole
  run. Starting at `|m| <= 1` is refused rather than approximated.
- The base mask is still taken at `m = 0`, as for any normal-mode tracker.

Two consequences worth knowing:

- An eigensolve after such a run needs `m` put back to an integer:
  `Problem.setup_forced_zero_dof_list_for_eigenproblems` and `actions_before_eigen_solve` both refuse
  a non-integer one. Nothing on the *tracking* path needs an integer, which is why
  `_NormalModeBifurcationTrackerBase` now types `azimuthal_m` as a float.
- `actions_after_successful_newton_solve` writes the real `m` into `_last_eigenvalues_m` without
  rounding. Rounding would report a mode that was never solved for.

For a scalar-only problem the `|m| == 1` and `|m| > 1` masks coincide, so the distinction only bites
once vector fields are present — which is exactly why the test for it needs one.

## Traps

- **`k = 0` is refused** (Cartesian), and so is **`|m| <= 1`** (azimuthal).
  `_get_forced_zero_dofs_for_eigenproblem` returns a different set of forced-zero dofs there, and the
  tracker freezes that set when it is created.
- **A complex eigenvalue does not imply a complex `J_c`.** The imaginary contribution only exists
  when the residual has terms of odd order in `k`. A problem with only Laplacians has oscillatory
  modes and no imaginary contribution, and asking the multi-assembly for a contribution the problem
  does not have is a hard error, so every such request is guarded (`has_imag_contribution`) and
  replaced by a zero matrix.
- **Parameter derivatives must not get the Jacobian's identity patch.** `patch_matrices(eigen=True)`
  puts a 1 on the diagonal of forced-zero eigen rows, which turns that row into the equation
  `V_j = 0`. The `k`- or parameter-derivative of that equation is zero, so every derivative matrix
  goes through the `M` slot (rows zeroed, no identity) instead.
- **Frozen sparsity refuses problems with an imaginary contribution.** The Hessian products of that
  contribution reach entries outside the Jacobian's symbolic pattern; set
  `problem.use_frozen_sparsity = False`. Pre-existing and not specific to this tracker —
  `NormalModeBifurcationTracker` is refused identically.
- **Serial only**, like every Python-side tracker (see `mpi_augmented_systems.md` §7-§10), and no
  spatial adaptivity while it is installed.
- `_retune_arclength_theta` silently no-ops while a Python augmentation is active, so the arclength
  scaling is whatever it was before.

## Fixes that came out of this

### `NormalModeBifurcationTracker`'s stationary branch had never run

Reached only by a normal-mode problem with a **stationary** neutral mode **and** no imaginary
contribution — scalar fields only; anything with a vector field takes the complex branch. It had four
independent defects, found by pointing the same finite-difference instrument at it:

- `dR/dparameter` was neither assembled nor placed, so the base rows had no parameter column at all
  and Newton could not move the parameter;
- the augmented residual carried the **matrix** `JR` instead of `JR@Vr`, which makes `numpy.hstack`
  produce an object array and `Problem.get_custom_residuals_jacobian` assert — i.e. the branch could
  not complete a single step;
- the Hessian and `dJ/dp` were asked of the base residual, but the row they differentiate is
  `J_real*V`, so both have to be the real contribution's (the `dparameter` branch beside it already
  got this right);
- `dJ_real/dp` was patched like a Jacobian, putting a 1 on the diagonal of every forced-zero eigen
  row — that row is the equation `V_j = 0`, whose parameter derivative is zero.

`tests/test_critical_wavenumber_tracker.py::test_stationary_normal_mode_tracker_jacobian_is_exact`
covers it, and the azimuthal end-to-end test depends on it.

### `Problem::get_jacobian` segfaulted on the custom-assembler path

Its custom-assembler branch never built the matrix distribution before calling
`CRDoubleMatrix::build(ncol, value, column_index, row_start)`, which takes the row count from that
distribution. Every call arriving from a Newton solve hands in a matrix oomph-lib has already
distributed, which is why it went unnoticed; `Problem.assemble_jacobian()` from Python passes a fresh
`CRDoubleMatrix` and **segfaulted whenever any Python-side custom assembler was installed** — a
`CustomBifurcationTracker`, deflation, anything. Fixed in `src/problem.cpp`.

## Continuing the critical point

Once converged, `problem.arclength_continuation(other_parameter, ds)` traces the critical point
through a two-parameter plane. Only `dResidual/dparameter` is ever asked of a custom assembler (see
`Problem.get_custom_residuals_jacobian`), which is what oomph-lib's bordered continuation solves
against, and the tracker supplies it by the same two-pass scheme.

`tests/test_critical_wavenumber_tracker.py` checks this against a closed-form locus: for a
Brusselator on a `PointMesh`, `B_c = (1 + A/sqrt(d))^2` and `k_c = sqrt(A)/d^(1/4)`, so every
continuation step in `d` has a known answer.

## How each branch is checked

| | closed form | consistency |
|---|---|---|
| Cartesian, stationary | Brusselator on a point: `B_c=(1+A/sqrt(d))^2`, `k_c=sqrt(A)/d^(1/4)`; also the locus in `d` | every column of the `3N+2` Jacobian vs a central difference |
| Cartesian, oscillatory | a rotating pair with an auxiliary field for the `k^4` term: `lambda(k)=r-(k^2-q0^2)^2 +- I(w0+c k^2)`, so `r_c=0`, `k_c=q0`, `omega=w0+c q0^2`, `mu=+-2 c q0` | a nonlinear advection problem with an odd-in-`k` term, so no block of the `5N+4` Jacobian is structurally zero |
| Azimuthal | none — the annulus is not separable | the ordinary tracker at three fixed real `m` around the answer: `B_c(m)` must be minimal at `m_c`, and the parabola's vertex must sit there |

The consistency tests make their eigenvector up and never call an eigensolver: residual/Jacobian
consistency does not depend on where the state came from, and it keeps them off the complex PETSc
build that a genuinely complex `J` would otherwise need.
