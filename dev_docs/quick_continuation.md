# Quick continuation: spotting bifurcations without solving an eigenproblem

`BifurcationController` normally solves the eigenproblem at every continuation point, which is what
makes a sweep expensive - a shift-inverted solve for `neigen=30..50` costs many factorisations against
the two or three a Newton step needs. Quick mode (`BifurcationController.quick_mode`) continues without
one and watches two test functions instead.

Everything below was established by probing analytic problems, not inferred from the APIs.

## Why the free route is correct

oomph-lib's `newton_solve_continuation`
(`src/thirdparty/oomph-lib/include/problem.cc:9787`) calls `Linear_solver_pt->enable_resolve()` and
solves **twice with one factorisation**, doing the arclength bordering itself. It does not factorise an
`(n+1)` matrix. So during a plain continuation step the matrix the solver has factorised is the plain
`n x n` Jacobian - exactly the one whose determinant vanishes at a fold or a branch point - and its sign
can be read off work that has already been paid for.

Two guards on that:

- **Only when the system is unaugmented.** During bifurcation tracking the factorised matrix is bordered
  and its determinant does *not* vanish at the bifurcation. `Problem._get_n_unaugmented_dofs() == 0` is
  oomph's "not augmented" sentinel (`src/problem.cpp:2645`) - note it is **0**, not `ndof()`.
- **Only changes are meaningful.** The recorded sign carries whatever overall convention and row scaling
  the assembled system has: on the probe below it comes out consistently *opposite* to the mathematical
  `det J`. `get_determinant_sign` says so in as many words; do not present it as a determinant.

## The two test functions, and why both

| crossing | dparameter/ds | det(J) |
| --- | --- | --- |
| fold | **reverses** | **flips** |
| pitchfork / transcritical | unchanged | **flips** |
| Hopf | unchanged | unchanged |

Measured, on `du/dt = mu - u^2 - b*u` (fold at `u=-b/2`, `mu_c=-b^2/4`) and on `du/dt = mu*u - u^3`
(pitchfork at `mu=0` on the trivial branch `u=0`): at the fold both flip in the same bracket; at the
pitchfork the determinant flips while `dp/ds` stays at `+1.000` throughout. **A dp/ds-only quick mode
would pass a pitchfork in silence**, which is why `quick_mode_detector="folds_only"` exists but is not
the default.

### Transcritical and pitchfork share one signature

Both are a single real eigenvalue crossing zero on a branch that does **not** turn in the parameter, so
the two test functions cannot tell them apart and both are reported as `"branch_point"`. That is a
statement about the test functions, not a gap: refining the bracket with the ordinary
`locate_bifurcation` and `classify_bifurcations=True` then names it, because `NormalFormCalculator`
distinguishes fold / transcritical / pitchfork from the normal-form coefficients.

Measured on `du/dt = mu*u - u^2` (branches `u=0` and `u=mu` crossing at `mu=0`): quick mode reports a
`branch_point` in the bracket containing `mu=0` with `dp/ds` constant at `+1.000` throughout, the
propagated unstable count steps 0 -> 1 across it, and refining lands at `mu = 0.000e+00` with the normal
form naming it `transcritical`.

### What a sign test cannot see

**An EVEN number of crossings in one step.** The determinant's sign returns to where it started, so two
bifurcations passed in a single step - or a pitchfork and a fold together - are reported as none at all.
This is inherent to any sign-based test function and the remedy is the usual one: a smaller `ds` where
the diagram looks suspicious, or a back-filled spectrum along the stretch in question.

**A Hopf is invisible to both.** A complex pair crossing leaves the determinant's sign alone and does not
turn the parameter. That is the price of not eigensolving, and it is the reason inferred stability is
labelled inferred rather than drawn as fact.

## Which solvers can report a sign

| solver | available | how |
| --- | --- | --- |
| scipy/SuperLU (`superlu`) | **free** | `_current_LU` is already kept; `sign = parity(perm_r) * parity(perm_c) * sign(prod diag U)`, L being unit-diagonal. Leaving the permutations out - as a commented-out determinant print in `solvers/scipy.py` did - gives a sign that flips whenever pivoting reorders, i.e. spurious detections. |
| PETSc + MUMPS | possible, not yet implemented | `ICNTL(33)=1`, then `RINFOG(12)` and `INFOG(34)`. petsc4py has `setMumpsIcntl`/`getMumpsRinfog`, and `solvers/petsc.py:_mumps_infog_from_pc` already does the guarded access. |
| Pardiso (MKL) | **no** | `iparm` is reachable, but MKL reports a determinant or inertia only for symmetric indefinite matrices (`mtype = -2, -4, 6`); pyoomph assembles a general Jacobian as `mtype=11`. |
| iterative | no | there is no factorisation to read. |

`GenericLinearSystemSolver.get_determinant_sign()` returns `None` by default, so a solver reports a sign
only if it genuinely has one - the same convention as `requires_explicit_diagonal`. Quick mode probes the
**class**, not the current value (`determinant_sign_supported`), because `get_determinant_sign` also
returns `None` when no factorisation exists yet, which at enable time is the normal state.

With `detector="auto"` on a solver that cannot report a sign, quick mode **refuses to start** and names
`superlu`, `petsc`+`use_mumps()` and the `folds_only` opt-out. A mode that quietly stops seeing
pitchforks is worse than one that will not start.

## Points without a spectrum

A quick-mode point stores `det_sign` and `dparam_ds` and has `eig_value_Re = NaN` - deliberately **not**
zero, which is how a *located* bifurcation is flagged and would make every quick point look like one.
Every `== 0` test in the existing code therefore answers correctly without knowing about quick mode,
since comparisons against NaN are False.

Stability comes from `BifurcationGUISolutionPoint.stability_indicator()`, which the branch segmentation
reads instead of `eig_value_Re` directly. For a measured point it returns exactly that real part, so an
ordinary diagram is unchanged; for a quick point it returns the sign of the propagated unstable count, or
NaN while that is unknown, which the segmentation turns into the neutral style it already uses where
stability changes.

`propagate_stability()` carries the unstable count along a branch from each measured point and flips it
at every recorded sign change, since a sign flip is exactly an odd number of real eigenvalues crossing.
Where a *later* measured point disagrees with the propagated value it says so: that is a Hopf the
determinant could not see, and it is worth reporting rather than hiding.

`compute_spectrum(point)` and `compute_spectrum_for_branch()` fill spectra in afterwards from the state
dumps every point already has, so a cheap sweep can be given its eigenvalues later without redoing the
continuation. Verified: the same branch swept with and without quick mode visits the **same parameter
values to 1e-12**, the propagated stability equals the measured one, and after back-filling all 13 points
of the probe have spectra and the stability still agrees.

## Cost

Only worth measuring on a real problem. On the two-dof probe both arms take ~0.04 s, so the numbers there
say nothing; what they do show is that quick mode changes *what is measured* and not the continuation
itself.
