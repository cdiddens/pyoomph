# Diagnosing non-convergence: what exists, what is undocumented, what is missing

Status: **planning document.** Nothing here is a description of new code. It records what pyoomph
already offers for "my Newton solve does not converge", which parts of it are effectively invisible
because they are undocumented, and which diagnostics would be worth building.

The motivation is that non-convergence is the single most common way a pyoomph run fails, and the
user-facing story for it is currently thin: the tutorial teaches how to set a problem up, and the
solver tells you it diverged, but almost nothing connects the two. A user whose Stokes problem has
pure Dirichlet boundaries and no pressure constraint gets "MAXIMUM RESIDUALS EXCEEDS PREDEFINED
MAXIMUM" and no hint that their problem is singular by construction.

---

## 1. Documentation owed

### 1.1 A tutorial chapter on the adaptive-resolve recovery

`AdaptiveResolveRecovery` (see `dev_docs/adaptive_resolve_recovery.md`) has no tutorial coverage at
all, so nobody will find it. It needs a section that says:

- what the failure looks like - a run that dies during the re-solve after an adaptation, with the
  pre-adapt mesh already gone, which is why it cannot simply be caught;
- the one-liner that switches it on
  (`problem.adaptive_resolve_recovery = AdaptiveResolveRecovery()`);
- what the strategies mean and why `accept_unadapted` is the one that cannot fail: the state before
  an `adapt()` is a converged solution, or a completed timestep;
- the cost - up to three full state snapshots held in memory while an adaptive solve runs;
- that `SpatialAdaptResolveError` is catchable and leaves the problem usable, unlike the failure it
  replaces;
- the limitations from §8 of that document: arclength continuation not covered, and a rank-local
  linear-solver failure declined under `--distribute`.

A good worked example would be a case that genuinely fails - the tutorial harness needs it to be
deterministic, so something with a sharp front where unrefinement across it kills the re-solve,
rather than an artificially injected failure.

### 1.2 `--largest_residuals`

Implemented (`Problem.debug_largest_residual`, `problem.py:3927`; the command-line flag is registered
at `problem.py:2499`) and documented nowhere. It is one of the most useful things in the codebase for
this class of problem and nobody knows it exists.

`--largest_residuals N` prints the `N` largest entries of the residual vector before every Newton
solve (`actions_before_newton_solve`) and after every Newton step (`actions_after_newton_step`), and
for each one resolves the equation number back to something meaningful:

```
========MAX. RESIDUALS========
Highest residual 1  with a value of 3.2e+04 Eqn number: 1871
   belongs to domain/velocity_x
   found at <node position>. Type: <node/element>
```

That mapping - equation number to `mesh/field` to the node or element it sits on - is exactly what
turns "it diverged" into "it diverged *here*", and it is what the documentation should lead with. It
also works under azimuthal bifurcation tracking, where it splits the augmented dof vector back into
base/real/imaginary parts before looking a dof up.

Worth saying explicitly in the docs: the flag is cheap, so reaching for it *first* is the right
instinct when a solve misbehaves.

### 1.3 A troubleshooting chapter for non-convergence

The broader gap. A chapter that works through the question "the Newton solver is not converging,
what now?" and that at minimum covers:

- reading the residual history that pyoomph prints, and what a residual that stalls, oscillates or
  blows up each says;
- `--largest_residuals` (§1.2) as the first move;
- the difference between *diverging* and *hitting `max_newton_iterations`*, and why
  `newton_relaxation_factor` costs iterations, so relaxing without also raising the iteration cap
  makes a solve look like it failed when it was only cut short (the same trap the recovery policy
  guards against, `adaptive_recovery.py`);
- when `globally_convergent_newton` helps and when it cannot - it only helps if the residual
  descends towards a nearby root, so it does nothing for an initial state that is outside the basin,
  which is the usual situation after a bad unrefinement;
- transient rescue: reducing `dt`, `temporal_error`, or taking pseudo-timesteps towards a stationary
  solution;
- adaptivity-specific failure, pointing at `AdaptiveResolveRecovery`;
- singular-by-construction problems (§2.2) and how to recognise them;
- what a linear-solver failure looks like as opposed to a Newton failure, and the fact that pyoomph
  reports a backend failure as an ordinary Newton failure on purpose so that callers able to retry
  with a smaller step get the chance (`src/nanobind/solver.cpp`).

---

## 2. Diagnostics worth building

None of these exist. Roughly in order of value-per-effort.

### 2.1 Condition number estimation

An estimate of the Jacobian's condition number, reported on demand or automatically when a solve
fails. It separates the two failure modes users cannot currently tell apart: "my problem is fine but
my initial guess is bad" (well-conditioned, Newton just needs help) from "my problem is singular or
nearly so" (no amount of relaxation or line searching will help).

Cheap options rather than a full SVD: a few steps of power iteration on `J` and `J^{-1}` using the
factorisation already computed, or the incremental condition estimator the direct solvers can
provide. MKL Pardiso and MUMPS can both report an estimate almost for free; MUMPS also reports a
null-space dimension, which is directly the answer for §2.2.

### 2.2 Structural inconsistency analysis on the equation tree

The most pyoomph-specific idea, and the one with no equivalent elsewhere: pyoomph *knows* the
symbolic weak form and the whole equation tree before anything is assembled, so a class of "this
problem cannot be solved" can be detected at setup time instead of as a divergence twenty minutes in.

Canonical case: **Stokes (or Navier-Stokes) with Dirichlet conditions on the entire boundary and no
pressure constraint.** The pressure is then determined only up to a constant, the Jacobian is exactly
singular, and the user sees a linear-solver failure or a wandering residual. It is entirely
detectable up front: every boundary of the domain carries a velocity Dirichlet condition, the
pressure appears in the weak form only through its gradient, and no constraint pins it.

Other members of the same family worth detecting:

- a field that appears in no residual at all, or whose Jacobian row/column is structurally empty -
  pyoomph already pins those (see `assign_eqn_numbers` in `src/problem.cpp`), but silently, and a
  report would be more useful than a silent pin;
- an incompressibility constraint together with Dirichlet velocities whose net flux through the
  boundary is nonzero - inconsistent, and checkable symbolically plus one integral;
- a Lagrange multiplier with no equation to determine it, or two constraints enforcing the same
  thing (§2.3);
- pure Neumann conditions on a Poisson-like field with no constraint fixing the constant, which is
  the same defect as the Stokes case in simpler clothes.

The output should name the field and the domain, and say what would fix it ("add a pressure
constraint, e.g. `DirichletBC(pressure=0) @ 'domain/corner'`, or an integral constraint"), rather
than merely reporting that the matrix is singular.

### 2.3 Mismatched constraints and Lagrange multipliers

A narrower, more mechanical version of §2.2, and probably the right thing to build first because it
needs counting rather than symbolic reasoning: check that the number of Lagrange multipliers matches
the number of constraints they enforce, and that each multiplier actually appears in some residual.

Typical failures this would catch: a multiplier added on an interface that no equation references
(under-determined, a zero row); the same constraint imposed twice from two different equation
classes attached to the same domain (over-determined, dependent rows); a multiplier whose constraint
is imposed on a boundary that turns out to be empty after adaptation or a remesh.

The structural sparsity machinery already knows which dofs appear in which elemental blocks
(`Problem::sparsity_mask_for_element`, `dev_docs/structural_assembly.md`), so the row/column
occupancy part of this is largely already computed.

### 2.4 Reporting the near-null-space

When a problem *is* singular, showing the user the near-null vector - as a field they can plot - says
what is undetermined far more directly than any scalar. A constant pressure mode looks unmistakable.
This is a small step beyond §2.1 once an eigen/SVD path to the smallest singular triplet exists, and
pyoomph already has the eigensolver infrastructure to do it.

---

## 3. Note on where this belongs

§1 is user documentation (tutorial and troubleshooting chapters); §2 is feature work. They are kept
together here because the troubleshooting chapter is the thing that would *use* the diagnostics, and
writing it will show which of §2 is actually worth building: if a section of the chapter reads "and
here you are on your own", that is the specification for the next diagnostic.
