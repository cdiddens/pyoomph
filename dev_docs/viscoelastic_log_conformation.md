# Viscoelastic flow with the log-conformation representation

Implementation notes for `pyoomph/equations/viscoelastic.py` and `tests/test_viscoelastic.py`, both
added by this work. Nothing outside those two files was changed, but building them turned up a
series of defects and gotchas in code they sit on top of; those are recorded in sections 6, 8.6 and
9.3 with what was done about them; the main one, 6.2, has since been fixed at source.

The starting point was that `pyoomph/expressions/tensor_funcs.py` already contained everything hard
about log-conformation — an eigendecomposition with analytic derivatives, the Fattal-Kupferman
decomposition of the velocity gradient in both 2d Cartesian and axisymmetric form, and a symmetric
matrix exponential — and that **none of it was imported anywhere in the repository**. There was no
equation class to use it from. This adds one.

All file/line references are to the state of the tree at the time of writing.

---

## 1. What is solved

The conformation tensor `C` is dimensionless and equal to the identity at equilibrium. Every model
is written as

    upper_convected(C) = -g(C)/lambda
    polymer_stress     = (eta_p/lambda) * h(C)

with the model supplying `g` and `h`. `ViscoelasticEquations` adds `weak(polymer_stress, grad(v))`
to the momentum equation, so it composes with the existing (Navier-)Stokes classes, whose
`dynamic_viscosity` then plays the role of the *solvent* viscosity:

```python
eqs = NavierStokesEquations(dynamic_viscosity=eta_s, mass_density=rho)
eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=lam, polymer_viscosity=eta_p)
```

Five models are implemented: `OldroydB`, `Giesekus`, `PTT` (linear and exponential), `FENE_CR`,
`FENE_P`. Two formulations are implemented: the log-conformation one (default) and the plain
conformation one, the latter mostly so that the two can be cross-checked where both converge.

### 1.1 Conventions, and the one that had to be verified

pyoomph's `grad` of a vector is `grad(u)[i,j] = d(u_i)/d(x_j)`
(`CartesianCoordinateSystem.vector_gradient`, `pyoomph/expressions/coordsys.py:358`). With that
convention the upper-convected derivative is

    dt(C) + (u.grad)C - grad(u)*C - C*grad(u)^t

Under `Psi = log(C)` this becomes the Fattal-Kupferman equation

    dt(Psi) + (u.grad)Psi - (Omega*Psi - Psi*Omega) - 2B + C^-1 g(C)/lambda = 0

where `Omega` and `B` are the rotational and extensional parts of `grad(u)` in the eigenframe of
`C`, produced by `LogConfTensorDecompositionCartesian2d` / `...Axisymmetric`.

Those two classes take `grad(u)` as an argument and were written before any caller existed, so
whether their `omega = (Lambda2*m01 + Lambda1*m10)/(Lambda2-Lambda1)` matched pyoomph's `grad`
convention or its transpose could not be settled by reading the code. **It matches, unchanged.**
What settles it is the start-up-of-shear test (section 7.1): the first normal stress difference
builds up as `1-exp(-s)(1+s)` while the shear stress builds up as `1-exp(-s)`, and a flipped sign
reproduces the steady state but not that transient. This is why the suite contains a transient test
at all rather than only steady ones.

### 1.2 Why the relaxation term is written with C and C^-1

`C^-1 g(C)` is an isotropic function of `C`, so the obvious implementation evaluates it on the
eigenvalues that the decomposition produces anyway, and rebuilds it as `R*diag(...)*R^t`. That is
what the first version did. It is wrong for a reason that has nothing to do with the residual — see
section 6.2 — and the models are therefore written to supply

    relaxation_matrix(C, trace, identity)                -> g(C)          (conformation formulation)
    log_relaxation_matrix(C, Cinv, trace, identity)      -> C^-1 g(C)     (log formulation)
    stress_matrix(C, trace, identity)                    -> h(C)

For all five models `C^-1 g(C)` happens to be a plain linear combination of `identity`, `C` and
`C^-1` with `tr(C)`-dependent scalar coefficients — no matrix products anywhere. Giesekus is the
only one where this is not immediate:

    C^-1[(C-I) + alpha(C-I)^2] = (I - C^-1) + alpha(C - 2I + C^-1)

That the two declared forms really are the same function is not left to inspection; it is checked in
`test_model_relaxation_forms_agree`, which evaluates `relaxation_matrix` and
`C * log_relaxation_matrix` symbolically on an arbitrary conformation tensor. Without that check the
reference integration used by the model tests (section 7.3) would be validating the log form against
itself, since it goes through `log_relaxation_matrix` to stay numpy-evaluable.

---

## 2. Everything is 3x3

The first version built the in-plane tensor at its "natural" size — 2x2 in 2d Cartesian, 3x3 in
axisymmetric — and hit `matrix::mul(): incompatible matrices` at code generation.

pyoomph's `vector()` always pads to three components (`pyoomph/expressions/generic.py:979`), and
`matrix()` pads to 3x3 unless `fill_to_max_vector_dim=False`. `vector_gradient` builds one row per
vector component and then calls `matrix()`, so **`grad(u)` is 3x3 in every coordinate system**,
including 2d Cartesian where the third row and column are zero.

Matching that turned out to be the better design anyway. With `Psi` carried as a 3x3 whose third row
and column are zero:

* `C = exp(Psi)` has `C_zz = 1` automatically, which is the correct planar value;
* `tr(C)` therefore already contains the out-of-plane contribution, which the trace-dependent models
  (PTT, FENE) need and which an explicit `+1` offset would otherwise have to supply;
* `B` and `Omega` come back from the decomposition classes already padded to 3x3 with the right
  zeros, so nothing has to be resized;
* the polymer stress contracts with `grad(v)` directly.

The one place this does not happen for free is the eigendecomposition. In Cartesian coordinates
`DiagonalizeSymmetricTensor` diagonalizes only the in-plane block, and with `fill_to_max_vector_dim`
on it returns `R[2,2] = 0`. Using that would give `C_zz = 0` rather than `exp(Psi_zz)`. The class is
therefore called with `fill_to_max_vector_dim=False` and the third eigendirection is put back by
hand as `R[2,2] = 1`. The axisymmetric branch of the same class already returns a full 3x3 rotation
with `R[2,2] = 1` and does not need this.

---

## 3. The out-of-plane component

In a planar flow, `C_zz` obeys

    dt(C_zz) + (u.grad)C_zz - 2*grad(u)[2,2]*C_zz + g_zz/lambda = 0

with `grad(u)[2,2] = 0`. So `C_zz = 1` solves this identically **provided the model's relaxation
function vanishes at an eigenvalue of 1**. It does for Oldroyd-B, Giesekus, PTT and FENE-CR, all of
which carry a factor `(Lambda - 1)`.

It does not for FENE-P, whose relaxation is `f(tr C)*Lambda - a` with `a = f(3)`: away from
equilibrium `f != a`, so `g_zz != 0` and `C_zz` is genuinely dragged off 1. Numerically, at
eigenvalues `(1.5, 0.8, 1.0)` and `tr(C) = 3.3` with `L = 4`, the five models give `g` at the
eigenvalue 1.0 as `0, 0, 0, 0, 0.0291` respectively.

`ViscoelasticEquations` therefore adds an extra scalar field `<name>_zz` when, and only when, the
model asks for it via `requires_out_of_plane_component`. The default
`solve_out_of_plane_component="auto"` defers to the model; `True`/`False` override it. In
axisymmetric coordinates the azimuthal component is always an unknown and
`define_tensor_field` already provides it as `<name>_aa`, so the flag has no effect there.

`test_nonlinear_models_in_planar_extension` asserts the distinction directly: FENE-P must leave
`C_zz` measurably away from 1, every other model must hold it at 1 to within 1e-10.

---

## 4. Field layout

`define_tensor_field(name, space, symmetric=True)` is used for the in-plane block. This appears to be
its first use in the repository — the only other reference is the reimplementation in
`pyoomph/utils/rectangular_polar_mapping.py:169`. It gives `<name>_xx`, `<name>_xy`, `<name>_yy` in
Cartesian coordinates and additionally `<name>_aa` in axisymmetric ones, with `<name>_xy` shared
between the (0,1) and (1,0) slots, so a symmetric 2d tensor costs three degrees of freedom per node.

Both the tensor and its test function are then reassembled in Python from the component variables
rather than read back through `var(name)`. `var(name)` returns an unsubstituted `field(name)`
placeholder whose indexing produces a deferred `double_index` node; that resolves at code
generation, but `grad()` of one does not do what is wanted. Building the matrices from
`var(name + "_xx")` and friends keeps every symbolic manipulation on real matrices.

Scales: `Psi` and `C` are dimensionless, so the field scale is 1. This is not cosmetic —
`DiagonalizeSymmetricTensor` and the decomposition classes compare eigenvalue differences against an
**absolute** epsilon (`eigen_epsilon`, default 1e-7), so a scale other than 1 would silently move the
degeneracy threshold. The test scale is `scale_factor("spatial")/scale_factor(velocity_name)`, i.e.
a time, which makes the residual integrand dimensionless; this mirrors what
`AdvectionDiffusionEquations` does.

---

## 5. Decisions taken without being asked

* **Default space is `C2`**, matching Taylor-Hood velocity. Configurable. No DEVSS-G or SUPG
  stabilisation is added; the constitutive equation is pure advection with no diffusion, so this
  is available but off by default (section 9).
* **PTT rejects a nonzero slip parameter** rather than ignoring it. `xi != 0` replaces the
  upper-convected derivative by the Gordon-Schowalter one, for which the decomposition in
  `tensor_funcs.py` is simply not the right object. Raising `NotImplementedError` seemed better than
  quietly solving affine PTT under a non-affine name.
* **3d is refused with an explicit message**, not attempted. Both decomposition classes are 2d
  Cartesian / axisymmetric only, and a 3d version needs a general symmetric 3x3 eigensolver with
  analytic derivatives — a separate piece of work.
* **`AxisymmetryBreakingCoordinateSystem` is refused** for the same reason `tensor_funcs.py` refuses
  it.
* **The conformation formulation is kept** rather than dropped as redundant. It costs little, and it
  is the only independent check on the log transform inside pyoomph itself.

---

## 6. What building it turned up

Three defects in code this sits on. All three are worked around inside `viscoelastic.py`; **none of
them is fixed at source**, because two are in shared code and the fix should be a deliberate change
rather than a side effect of adding an equation class.

### 6.1 `partial_t` of a matrix assumes the matrix is 3x3

`BaseCoordinateSystem.directional_tensor_derivative`
(`pyoomph/expressions/coordsys.py:379-386`) builds its result with

    res = [[0]*3 for _x in range(3)]
    for i in range(3):
        for j in range(3):
            ...  diff(T[i,j], coord) ...

Three hardcoded, regardless of the size of `T`. It is reached from the ALE correction inside
`partial_t`, so `partial_t(M)` raises `ValueError: matrix::operator(): index out of range` for any
matrix that is not 3x3. Since `define_tensor_field` had no users, this had never been exercised.

This was first worked around by taking the time derivative component-wise, which goes through the
scalar path.

**Not fixed, and it no longer needs to be.** The right answer was that `viscoelastic.py` should not
have been hand-rolling this at all: `material_derivative(A, u)` is `dt(A) + (u.grad)A` for a tensor
argument, and `upper_convected_derivative(A, u)` is that minus `grad(u)*A + A*grad(u)^t` - exactly
the conformation equation's transport term, in pyoomph's own `grad(u)[i,j] = d(u_i)/d(x_j)`
convention. Both are used now, which also removed the component-wise assembly of the advection term.
They produce and consume the padded 3x3 form, which is the convention everywhere else, so the 2x2
case never arises. The limitation stands for anyone who assembles a tensor with
`fill_to_max_vector_dim=False` and hands it to `partial_t`; `tests/test_tensor_fields.py` documents
that and covers the 3x3 path on a static and a genuinely moving mesh.

### 6.2 `DiagonalizeSymmetricTensor` zeroes its whole Jacobian on the near-diagonal branch

This is the one that mattered.

The class short-circuits whenever the off-diagonal entry is tiny (`tensor_funcs.py:134`, `:317`, and
in the emitted C at `:214`, `:393`), returning `R = identity` and the diagonal entries as
eigenvalues. That part is fine. The Jacobian it reports in the same branch
(`tensor_funcs.py:164-165` and `:344-345`) is

    derivative_matrix[:,:] = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,1]]

i.e. **every derivative of `R` is zero**, and the eigenvalues are taken to be independent of the
off-diagonal entry. Both are wrong whenever the diagonal entries differ: the rotation rate is
`1/(M11-M22)`, finite and nonzero. The branch condition tests only `M12`, never `M11-M22`.

For residuals this is invisible. For the Jacobian it is not, and at rest it is fatal. At `Psi = 0`
every term of the log-conformation equation that was built from `R` — the relaxation term and the
polymer stress — ends up with no dependence on `Psi_xy` at all, so the `Psi_xy` row and column of the
Jacobian are exactly empty. A transient run survives, because its time derivative fills the diagonal.
**A stationary solve from the rest state hits an exactly singular matrix**, and Pardiso reports a
zero pivot on the first Newton step. That is what the first working version did.

The fix inside `viscoelastic.py` is to keep the eigendecomposition out of every term that carries the
diagonal of the Jacobian:

* `C` and `C^-1` are taken from `SymmetricMatrixExponential` applied to the in-plane block of `+Psi`
  and `-Psi`. That class handles its own degenerate case properly: at `mu = 0` its Jacobian branch
  (`tensor_funcs.py:1443-1469`) reports `d(exp(Psi))_xy/d(Psi_xy) = exp((a11+a22)/2)`, which is 1 at
  `Psi = 0` — correct and nonzero.
* `DiagonalizeSymmetricTensor` is then used *only* to produce the eigenframe fed to the
  Fattal-Kupferman decomposition. That is safe, because the decomposition's own degenerate branch
  (`tensor_funcs.py:516`, `:912`) returns `B = sym(grad u)`, `Omega = 0` without using `R` at all.
  The truncated Jacobian that branch reports costs Newton iterations; it does not remove any row.

With that, a stationary solve of coupled Poiseuille flow straight from `C = identity` reaches a
residual of 9.8e-10 in four Newton steps (1.5e-3, 1.7e-4, 2.5e-6, 9.8e-10).

Two side notes. `SymmetricMatrixExponential` turns out to be needed after all — the original plan for
this work had concluded it was avoidable — but only for a 2x2 block, so its unimplemented
axisymmetric branch is never reached. It is constructed with an explicitly Cartesian coordinate
system even in the axisymmetric case, since the exponential of a plain symmetric 2x2 block is the
same operation either way; `_in_plane_exponential` says so.

**This has since been fixed at source** - see section 6.2.1. The workaround above stays, because the
fix does not make the rest state differentiable; nothing can.

#### 6.2.1 The fix

`DiagonalizeSymmetricTensor` now branches on the eigenvalue **gap** rather than on the off-diagonal
entry, which is the quantity that actually decides whether the eigenvectors are well conditioned. The
gap can be tiny with a large off-diagonal entry, or huge with a zero one, so the old test decided the
wrong thing in both directions: it sent well-behaved matrices into the zero-Jacobian fallback, and it
let near-degenerate ones through to the general branch, where the correct derivatives are of order
1/gap and reach 1.8e5 for a gap of 2e-6.

Widening the old condition is not enough on its own. At `M12=0` with `M11>M22` the general branch
builds its eigenvector as `(M12, Lx-M11) = (0,0)` and divides by a zero norm - which is *why* the
special case existed. The repair is to note that the eigenvector of the larger eigenvalue can be
written either as `(r+D2, b)` or as `(b, r-D2)`, with `D2=(a-d)/2` and `r=sqrt(D2^2+b^2)`. The two
are parallel, since `b^2=(r+D2)(r-D2)`, but their squared norms are `2r(r+D2)` and `2r(r-D2)`, so
whichever carries `|D2|` with a plus sign has norm bounded below by `2r^2` and is fine whenever the
eigenvalues are distinct at all. Selecting between them by `sign(D2)` also moves the unavoidable
branch cut - an eigenvector field around a degeneracy has a topological obstruction, so a cut cannot
be removed, only relocated - from `b=0` onto `D2=0, b<0`. That matters because `b=0` is exactly where
symmetry lines and extensional flows sit, whereas `D2=0` is not a set the solutions occupy. Across
the cut `R`'s first column changes sign, and `C=R*D*R^t`, `B` and `Omega` are all invariant to that.

The derivatives are hand-derived rather than regenerated with sympy, which replaced ~300 lines of
CSE'd machine output with about 25 lines that read like the mathematics; the file lost 172 lines net.
Both coordinate systems now share one routine, which also removed a second inconsistency: the
axisymmetric branch had been using the opposite sign convention for `R` from the Cartesian one.

Verified against finite differences at 1e-8 relative on both variants, over symmetry lines
(`b=0` with `a>d` and `a<d`), generic tensors of both signs, large `Psi`, and equal diagonals. Two
cases deviate and both are expected: near-degeneracy, where the mismatch tracks FD truncation and
falls as `O(h^2)` to a 1.2e-8 relative floor; and the branch cut itself, where one-sided derivatives
agree on each side and reconstruction stays exact.

**What it bought, measured rather than assumed: nothing, on this benchmark.** The cylinder needs 21
Newton steps over Wi=0.1...0.7 with the fix and 21 without, and the drag is identical to every digit
- as it must be, since only the Jacobian was ever wrong. The affected set here is just the nodes
sitting exactly on the symmetry line, a few percent of the mesh. It would matter for a problem living
in that regime, a purely extensional flow being the obvious one. The value of the change is that a
silently wrong Jacobian is gone, not that this particular case got faster.

### 6.3 The C-vs-Python debug facility reports false differences

`set_debug_python_vs_c_epsilon` cross-checks the emitted C against the Python `eval`. Turning it on
produces thousands of messages of the form

    MULTI-RET Python Vs C difference (flag=0):  Result 0 is 0 (Python) and 0.707107 (C) at arguments: ...

where the Python value is reported as 0. Calling the Python `eval` directly at those very arguments
returns 0.707107, i.e. exactly what the C produced - so the two implementations agree and the
harness is comparing against a buffer the Python side has not written. It is **pre-existing**: the
unmodified file emits 14448 such messages on a small channel run, the repaired one 14160.

Worth knowing because the facility is otherwise the natural way to validate hand-written C, and it
currently cannot be trusted in the affirmative *or* the negative without a control. Deliberately
corrupting the emitted C does make it fire, so it is not inert - just wrong about which side is which.

### 6.3.1 Nondeterministic residual assembly with a literal constant vector

Found while writing the regression tests for 6.1 and 6.4, and unrelated to either. Building a
tensor-valued term out of `dot(wind, grad(...))` entries fails **intermittently** during
`_add_residual` with

    RuntimeError: Not a 32-bit integer: 136610024783872

- a pointer-sized value - on roughly 8 runs in 10, at both 2x2 and 3x3. It is not a shape problem and
not caused by the fixes here: the 3x3 case, whose code path is untouched, fails slightly *more* often
than the 2x2 one.

The trigger is a **literal constant** vector. Over 10 runs each:

| wind | 2x2 | 3x3 |
|------|-----|-----|
| `vector(1.0, 0.0)` | 7/10 fail | 9/10 fail |
| `vector(var("coordinate_y"), 0)` | 0/10 | 0/10 |
| a substituted field | 0/10 | 0/10 |

`partial_t` is irrelevant - on its own it never fails - so it really is the advection term alone.
`ViscoelasticEquations` is not exposed to this, because its wind is `var(velocity_name)`, which is
why 35 tests and dozens of benchmark runs never tripped over it. `tests/test_tensor_fields.py` routes
its wind through a substituted field for the same reason, and says so.

### 6.4 `SpatialErrorEstimator` cannot be given a tensor field

`SpatialErrorEstimator(log_conformation=1)` raises `ValueError: matrix::operator(): index out of
range`. It takes `grad()` of whatever it is handed, and `grad()` of a tensor field is not a vector
gradient, so `vector_gradient` walks off the end of the matrix
(`pyoomph/expressions/coordsys.py:362`). The components have to be named individually:

```python
SpatialErrorEstimator(log_conformation_xx=1, log_conformation_xy=1, log_conformation_yy=1)
```

Same root cause as 6.1 — tensor fields are a path nothing else in the repository exercises.

**Fixed at source**: a named tensor field is now expanded into one criterion per component, all in
the same group, which is what naming the components by hand would have done. The component names come
from `_tensorfields` on the combined element, which `define_tensor_field` populates.

### 6.5 Minor: local function names collide with field names

`add_local_function("conformation_xx", ...)` raises when the conformation formulation is in use,
because `conformation_xx` is then a field. Guarded: those outputs are only added in the
log-conformation formulation, where `C` is not itself a field. `polymer_stress_*`,
`conformation_trace` and `polymer_N1` are added in both.

---

## 7. Tests

`tests/test_viscoelastic.py`, 31 tests, **all passing, 54 s**. Deliberately in the fast suite: it is
under a minute and it is the only coverage `tensor_funcs.py` has ever had.

Everything in sections 7.1-7.3 drives the constitutive equation with an **imposed, spatially uniform
velocity gradient** and `add_polymer_stress_to_momentum=False`, on a single element. The conformation
tensor is then homogeneous and its evolution is an ODE with a known answer. That isolates the log
transform, the eigenvalue decompositions and their analytic Jacobians from mesh resolution, momentum
coupling and stabilisation, none of which are involved.

### 7.1 Start-up of simple shear — the discriminating one

`u = (rate*y, 0)` switched on at `t=0` from equilibrium. Oldroyd-B has

    C_xy(t) = W(1 - exp(-s))
    C_xx(t) = 1 + 2W^2 [1 - exp(-s)(1+s)]      W = lambda*rate, s = t/lambda

Both formulations reproduce this. What is left is BDF2 truncation error and falls at second order:

| dt    | max error |
|-------|-----------|
| 0.004 | 4.50e-06  |
| 0.002 | 1.12e-06  |
| 0.001 | 3.26e-07  |

As noted in section 1.1, this is the test that pins the sign of `Omega`.

### 7.2 The rest of the closed-form set

Steady shear at Wi = 0.5, 2, 5; steady planar extension (`C_xx = 1/(1-2*lambda*rate)`, which also
exercises the already-diagonal branch of the decomposition); and the rest state, which must stay at
`C = identity` to 1e-12 while running entirely through the fully degenerate branch.

Two axisymmetric tests. Uniaxial extension `u = (-rate*r/2, rate*z)` is the one that exercises the
azimuthal velocity gradient — the entry that is the whole reason
`LogConfTensorDecompositionAxisymmetric` exists as a separate class. The second checks that an axial
velocity varying with radius, `u_z = rate*r`, reproduces the planar start-up answer with `r` and `z`
swapped for `y` and `x`.

That second one was written wrong first, and the failure was the test's, not the code's: the obvious
candidate `u_r = rate*z` is *not* a shear that leaves the azimuthal direction alone. It moves
material radially, so `grad(u)[2,2] = u_r/r` is nonzero and the planar answer does not apply. The
comment in the test says so, because it is an easy mistake to repeat.

### 7.3 The nonlinear models

Giesekus, PTT (both kinds), FENE-CR and FENE-P have no convenient closed form, so they are checked
against an RK4 integration of the **conformation** equation in numpy, in shear and in planar
extension. The reference does not use the log representation, so agreement confirms the log
transform, the eigendecomposition, the FE assembly and the time stepping at once. Tolerance 1e-4
relative; 5000 RK4 steps put the reference's own error near machine precision.

Plus: the FENE-P / `C_zz` distinction of section 3, and that `tr(C) < L^2` holds for FENE-CR at an
extension rate ten times past the Oldroyd-B singularity.

### 7.4 Coupled: planar Poiseuille flow

The only test that sees the polymer stress handed back to the momentum equation. Channel periodic in
the flow direction, driven by a body force, `eta_s = 0.3`, `eta_p = 0.7`, `lambda = 1`. For Oldroyd-B
the polymer contributes exactly `eta_p*rate` to the shear stress, so the profile is the Newtonian
parabola formed with the **total** viscosity.

The discriminating number is the centreline velocity: `F/(8*eta_0) = 0.125`, obtained as 0.12499944.
A sign error in the coupling would give `F/(8*(eta_s-eta_p)) = -0.3125`.

Solved stationary, from rest — i.e. this is also the regression test for section 6.2. The exact
velocity and conformation fields are quadratic and lie in the `C2` space, but `Psi = log(C)` does not,
so there is a genuine interpolation error, and it converges:

| elements across | velocity error | conformation error |
|-----------------|----------------|--------------------|
| 8               | 2.57e-06       | 2.45e-04           |
| 16              | 1.87e-07       | 2.55e-05           |
| 32              | 1.39e-08       | 2.27e-06           |

---

## 8. The confined-cylinder benchmark

The benchmark that justifies the log-conformation representation specifically, rather than the
constitutive law: flow past a cylinder confined between parallel plates at 50% blockage, Oldroyd-B,
`beta = 0.59`, creeping flow. It is where plain conformation-tensor formulations lose convergence
around Wi ~ 0.7, and it is the case for which drag coefficients are tabulated across many
independent codes.

Reference: **Claus & Phillips, JNNFM 200 (2013) 131-146, Table 3**, which also reproduces the values
of Alves, Oliveira & Pinho (2001) and Fan, Tanner & Phan-Thien (1999). Their own method is
DEVSS-G/DG with spectral/hp elements at polynomial order 12-18, i.e. nothing like what is used here,
which is what makes the agreement worth something.

It lives in `tests/test_viscoelastic_cylinder.py`. That runs a cheaper version of what is
reported below - 23k dofs for the Newtonian case, 60k for the viscoelastic one, about 20 s in total -
and is deliberately *not* marked slow, since a validation against external references is worth
little if it is skipped by default. The numbers in 8.2 onwards come from the fuller meshes.

### 8.1 Setup, and the mapping onto these classes

Geometry (their Fig. 2, with the extent settled by the mesh plot in Fig. 3a rather than the
ambiguous "20D" label): cylinder of radius `R = 1` in a channel of half-height 2, domain
`x` in `[-20, 20]`, upper half only with symmetry at `y = 0`.

Their equations are non-dimensionalised with total viscosity `eta_0 = 1`, so

| their symbol | value | here |
|--------------|-------|------|
| `beta` | 0.59 | `StokesEquations(dynamic_viscosity=0.59)` |
| `1-beta` | 0.41 | `ViscoelasticEquations(polymer_viscosity=0.41)` |
| `Wi = lambda<u>/R` | 0.1 ... 0.7 | `relaxation_time = Wi`, since `<u> = R = 1` |
| `Re = 0` | - | `StokesEquations`, not Navier-Stokes |

Their Giesekus form (Eq. 3) is written for the stress rather than the conformation tensor. Working
it through - substitute `tau = ((1-beta)/Wi)(C-I)` and use `I^UC = -2D` - the `2(1-beta)D` terms
cancel exactly and it reduces to `C^UC = -(1/Wi)[(C-I) + alpha(C-I)^2]`, i.e. `Giesekus` verbatim
with the same `alpha`. So their Table 5 is usable as-is too.

Their inflow stress condition (Eqs. 40-42) is exactly `oldroyd_b_shear_conformation(lambda*du/dy)`
composed with `symmetric_2x2_matrix_log` - the two helpers in `viscoelastic.py` that previously had
nothing pointing at them. With `u = 3/2 (1 - y^2/4)`, that is `lambda*du/dy = -3*Wi*y/4`.

The drag coefficient is `K = F_x/(eta_0 <u>)` over the whole cylinder. `var("normal")` points *out*
of the fluid, so the force on the cylinder is minus the traction integral, and the half-cylinder
that is actually solved is doubled.

### 8.2 Results

Newtonian limit first, which fixes the drag integration and the sign without involving the
constitutive model at all:

| base resolution | ndof | K |
|-----------------|------|---|
| 0.60 | 33 481 | 132.3877 |
| 0.35 | 38 830 | 132.3632 |
| 0.20 | 59 957 | 132.3588 |

against the standard value 132.36.

Then the Oldroyd-B series, by continuation in Wi in steps of 0.05, on a mesh adapted once at
Wi = 0.1 and then held fixed (119 803 dofs throughout):

| Wi | K here | Claus & Phillips (P=18) | deviation |
|------|----------|--------|-----------|
| 0.1 | 130.3695 | 130.364 | +0.004% |
| 0.2 | 126.6423 | 126.626 | +0.013% |
| 0.3 | 123.2257 | 123.192 | +0.027% |
| 0.4 | 120.6517 | 120.593 | +0.049% |
| 0.5 | 118.9219 | 118.826 | +0.081% |
| 0.6 | 117.9147 | 117.776 | +0.118% |
| 0.7 | 117.4970 | 117.316 | +0.154% |

The drag minimum near Wi ~ 0.7 comes out in the right place. Wi = 0.7 is the top of the range where
the reference itself still has mesh-converged values; beyond it their Table 3 carries `(D)` markers.

Reaching Wi = 0.7 **with no stabilisation at all** - no DEVSS-G, no SUPG, plain continuous C2 - was
better than expected. The deviation grows monotonically with Wi, i.e. it is a resolution effect
rather than a modelling error; sections 8.3 to 8.5 say what the estimator is actually doing about
it, how far refining the cylinder gets, and where the comparison against the reference stops being
resolution-limited.

Note the Alves value at Wi = 0.2 in that table, 126.32, is out by 0.3 while every other entry in
every column agrees to ~0.01. It is almost certainly a transcription error in Claus & Phillips and
should not be used as a reference.

### 8.3 Where the adaptivity actually goes, and why it does not help

The first guess about the growing deviation was that the mesh, adapted once at Wi = 0.1 and then
held fixed, was simply stale by Wi = 0.7. Re-adapting it turned out to change nothing, and finding
out why is more interesting than the original question.

Two runs of the same problem at Wi = 0.1, one adapting on error thresholds and one on a
`desired_ndof` budget, gave **identical** drag - 130.369483340 both times - and identical global
integrals (kinetic energy and dissipation agreeing to 11 digits) despite 12 279 versus 10 089
elements and 119 803 versus 108 171 dofs. Binning both meshes by region explains it:

| region | error-threshold | `desired_ndof` |
|--------|-----------------|----------------|
| r < 2, the O-grid on the cylinder | 1263 elements, min h = 0.0189 | 1263, 0.0189 |
| 2 < r < 5 | 346, 0.1344 | 346, 0.1344 |
| wake, x > 5 | 3766, 0.0020 | 3625, 0.0081 |
| **upstream, x < -5** | **6895**, 0.0010 | **4846**, 0.0081 |

The near-cylinder mesh is *identical*, so the drag is identical. Every element of the difference is
far field, and 2049 of the 2190 extra elements are upstream. **56% of the whole mesh sits upstream
of x = -5**, in the fully developed inflow region.

Refinement there cannot do anything, and the reason is specific to this equation set. The
constitutive equation has no diffusion - the only spatial operator is the advection `u.grad(Psi)`.
Far upstream `Psi` is x-independent and `u` is along x, so that term vanishes identically and what
is left,

    -(Omega*Psi - Psi*Omega) - 2B + C^-1 g(C)/lambda = 0

is a purely *algebraic* relation at each point, with no spatial derivatives at all. It is therefore
satisfied exactly at every node on any mesh. The velocity there is the Poiseuille parabola (exact in
C2) and the pressure is linear (exact in C1). So the entire inflow region is solved exactly
regardless of refinement, and the Z2 estimator refines it anyway - it measures the recovery error of
`grad(Psi)`, which is nonzero because `Psi = log(C)` is transcendental in y, even though the nodal
values are already exact and refining improves no functional of the solution.

The adaptivity does not even touch the boundary layer: the O-grid comes out at exactly its nominal
transfinite element count (2 sectors x (n_circ-1) x (n_radial-1) = 858 for the default 40/12 grid)
in every run, so every adaptive element goes into the surrounding triangles. The resolution that
decides the drag is therefore set entirely by hand, through the O-grid parameters.

### 8.4 Refining the O-grid: what it buys, and where it stops

Holding everything else at the baseline configuration and varying only the O-grid
(nodes per quadrant / nodes across the layer / radial growth ratio):

| Wi | 40/12/1.35, h=0.0189 | 60/18/1.30, h=0.0081 | 80/24/1.25, h=0.0047 |
|------|-----------|-----------|-----------|
| 0.1 | +0.0042% | +0.0035% | +0.0032% |
| 0.2 | +0.0129% | +0.0090% | +0.0084% |
| 0.3 | +0.0274% | +0.0171% | +0.0171% |
| 0.4 | +0.0487% | **+0.0285%** | +0.0306% |
| 0.5 | +0.0807% | **+0.0473%** | +0.0530% |
| 0.6 | +0.1178% | **+0.0684%** | +0.0790% |
| 0.7 | +0.1543% | **+0.0884%** | +0.1049% |

(ndof 119 803 / 151 744 / 183 570; deviations against the Claus & Phillips P=18 column.)

The first refinement confirms the diagnosis: a 2.3x finer wall spacing cuts the error by ~42% at
high Wi, and the gain **grows monotonically with Wi**, which is what a stress boundary layer that
sharpens with elasticity should do. At Wi = 0.7 the drag moves from +0.154% to +0.088%.

The second refinement does not continue the trend. It is marginally better at Wi <= 0.3 and
marginally *worse* from Wi = 0.4 upwards.

**That non-monotonicity is not the boundary layer**, and the pointwise wall shear rate of section
8.5 is what shows it: `du/dy` at the top of the cylinder moves *monotonically* across the same three
variants (13.3820 -> 13.4439 -> 13.4859 at Wi = 0.7), so the layer is being resolved progressively
better exactly as intended. The drag does not follow, so a second, uncontrolled factor is moving
with it. The likely one is stated above: the Z2 estimator normalises against a norm taken over the
whole mesh, so enlarging the O-grid changes the thresholds and hence how many elements the far field
and wake get. The three variants do **not** have the same wake resolution, and the drag comparison
therefore conflates two effects.

So the honest reading is weaker than it first looked: refining the O-grid does improve the boundary
layer, the drag error does fall by ~42% from the coarsest to the middle grid, but the two facts
cannot be linked causally from this experiment, because the wake was not held fixed. Re-running the
three variants with the far-field mesh pinned - error estimator restricted to the cylinder and near
wake, or adaptivity off entirely - is what would settle it.

Also worth noting that at Wi = 0.3 the residual +0.017% is already at the level of the disagreement
*between* the published sources (123.192, 123.210, 123.19 - a spread of 0.016%), so there is not
much signal left to chase there.

**60/18/1.30 is the best of the three on drag** and is the current default. The next thing to
control is the wake, not to refine the cylinder further.

### 8.5 Pointwise wall shear rate, and a discrepancy that does not go away

Claus & Phillips quote `du/dy` at the top of the cylinder, (0,1), in the text of section 5.1.3:
14.637 at Wi = 0.1, 13.739 at Wi = 0.5, 13.228 at Wi = 0.7. Being pointwise it is far more sensitive
to the boundary layer than the integrated drag - at Wi = 0.1 the deviation is +0.5% where the drag
deviation is +0.004%, a factor of 125.

It is evaluated exactly rather than by finite differences. The radial O-grid line at angle pi/2 runs
straight from (0,1) to (0,1.6) along x = 0, the elements are C2, so the three lowest nodes on that
line are the corner/midside/corner of a single element and the FE solution restricted to the line is
exactly the quadratic through them. Two things were checked rather than assumed: the midside node
sits at the geometric midpoint to machine precision (the grading is *between* elements, not within
one), and fits over one, two and three elements agree to 2.5e-4. The O-grid is never adaptively
refined, so this structure holds in every run.

| Wi | 40/12 | 60/18 | 80/24 | reference |
|------|---------|---------|---------|-----------|
| 0.1 | 14.7139 | 14.7141 | 14.7124 | 14.637 |
| 0.5 | 13.9003 | 13.9383 | 13.9468 | 13.739 |
| 0.7 | 13.3820 | 13.4439 | 13.4859 | 13.228 |

The values converge monotonically under refinement and settle roughly +0.5% / +1.5% / +2.0% above
the reference. This is a *converged* offset, not under-resolution: at Wi = 0.1 all three grids agree
to 1e-4 despite a 4x difference in first-element thickness.

One explanation was tested and **rejected**. Claus & Phillips tabulate `h_q`, "the distance of the
closest quadrature point to the cylinder surface", at 1.85e-2 to 2.77e-2 across their meshes; a
spectral method reporting a value from there rather than from the wall would read low, because the
shear rate falls from its wall value towards zero at the velocity maximum in the gap. But the offset
that reproduces 13.739 at Wi = 0.5 is about 0.006, three to four times smaller than their smallest
`h_q`. (Beware when redoing this: a high-order polynomial fitted over the first few nodes spans only
~0.01 of the layer, and evaluating it at 0.019 or beyond is extrapolation - it returns values in the
hundreds and is meaningless.)

What is left is that the discrepancy **grows with Wi**, from +0.5% to +2.0%, which is the signature
of a difference in the viscoelastic treatment rather than of any geometric or evaluation artifact -
a fixed evaluation-point offset would be roughly Wi-independent. Their scheme is decoupled DEVSS-G
with a discontinuous stress; this one is fully coupled with a continuous C2 log-conformation field.
Which is closer to the truth is not established here, and note the asymmetry in evidential weight:
the drag is tabulated across four independent codes that agree to ~0.01, whereas these `du/dy`
figures are single-source, quoted in passing to support a qualitative argument about velocity
inflection rather than offered as benchmark values.

### 8.6 What building it turned up

* **A cold start fails at Wi >= 0.2.** The first Newton step overshoots into a `Psi` large enough
  that `exp(Psi)` overflows, and the residual is reported as `inf`. Wi therefore enters as a global
  parameter and the sweep continues from the previous solution. (Global parameters do propagate into
  `DirichletBC` values - checked separately, since a frozen inlet stress would have produced
  plausible but wrong answers at every Wi above the first.)
* **`add_integral_function` does not apply the integration measure.** `IntegralObservables`
  multiplies by `get_dx()` before calling it (`pyoomph/equations/generic.py:858`); calling it
  directly without that integrates against the reference measure instead and gave K = 9861 rather
  than 132. Nothing warns. The Newtonian limit is what caught it.
* **Prescribing the velocity on the whole boundary makes the Stokes system singular**, which is what
  the reference setup does (it prescribes `u = u_in` at the outflow as well as the inflow). It needs
  an explicit `create_pressure_fixation()`. Worth stating because at one resolution Pardiso returned
  a meaningless vector rather than erroring, and only the wrong drag gave it away.
* **`solve()` defaults to `spatial_adapt=0`**, so a continuation loop that just calls `solve()` in
  sequence adapts once and then freezes the mesh - the cause of the monotonic drift above.
* **`desired_ndof` is the wrong tool for re-adapting a continuation sweep at fixed cost.** It targets
  a dof *count*, and `desired_ndof_tolerance` is a 10% relative dead band inside which nothing is
  refined or unrefined - deliberately, so that adaptation loops terminate instead of oscillating
  about the target (`pyoomph/generic/problem.py:588-590`). A sweep that lands 9.9% below its budget
  therefore adapts exactly zero elements at every subsequent step, however much the solution has
  changed, and looks like a working adaptive run in every respect except that the mesh never moves.
  Error-threshold adaptation (leaving `desired_ndof` unset) is what actually redistributes.

---

## 9. Stabilisation

### 9.1 Past the end of the reference table, without any

The Wi sweep of section 8.2 stops at 0.7 because that is where Claus & Phillips stop having
mesh-converged values. Continuing it on the 60/18/1.30 O-grid, with plain Galerkin and no
stabilisation whatsoever:

| Wi | K here | Claus & Phillips | Alves | Fan |
|------|----------|----------|----------|--------|
| 0.75 | 117.3838 | - | - | - |
| 0.80 | 117.4510 | 117.368 *(D)* | 117.357 | 117.36 |
| 0.85 | 117.6121 | - | - | - |
| 0.90 | 117.8759 | 117.812 *(D)* | 117.851 | 117.79 |
| 0.95 | 118.2736 | - | - | - |
| 1.00 | 118.7975 | 118.550 *(D)* | 118.518 | 118.49 |

*(D)* is their marker for "computations diverge after an apparent converged drag value is reached".
At Wi = 1.0 every column of their Table 3 carries it, for all four codes they compare. This
formulation - fully coupled, continuous C2 log-conformation, no stabilisation - reaches a steady
solution there without difficulty, and is within +0.02% of Alves at Wi = 0.9 and +0.24% at Wi = 1.0.
The drag minimum comes out near Wi = 0.75, which matches the upturn their table shows.

Whether the run would keep going past 1.0 was not tested. The point of the exercise was that
stabilisation turned out not to be needed to cover the benchmark range at all.

### 9.2 SUPG, and what it cost to get right

It is implemented anyway, off by default, because the equation genuinely has no diffusion - the only
spatial operator is `u.grad(Psi)` - so plain Galerkin is unstabilised advection and will oscillate
once the element Peclet number gets large enough. On a coarser mesh, or further up in Wi, it should
matter.

    h   = var("cartesian_element_length_h")
    tau = supg_factor / sqrt(4*dot(u,u)/h^2 + (1/lambda)^2)

Three things about that, each of which had a reason:

* **`cartesian_element_length_h`, not `element_length_h`.** The latter raises the *Eulerian* element
  size to the power 1/dim, and in axisymmetric coordinates that size carries the 2*pi*r factor, so it
  is not a length and tau would pick up a spurious radius dependence.
* **tau falls back on the relaxation time, not on a velocity floor.** The textbook `h/(2|u|)` blows
  up at stagnation points and on the no-slip cylinder - exactly where this problem is hardest. The
  form above gives `h/(2|u|)` where advection dominates and `lambda` where the flow is slow, and
  `1/lambda` is this equation's own reaction rate, so no arbitrary epsilon has to be invented.
* **`dot(u,u)` under the single root, not `(2*|u|/h)^2`.** Same number, different expression:
  squaring `square_root(dot(u,u))` leaves the inner root in place for GiNaC to differentiate, and
  `d|u|/du = u/|u|` is 0/0 at a stagnation point. That put NaNs in the Jacobian and the first Newton
  step on the cylinder went to 1e105.

The scheme is residual-based, so it perturbs the test function along the streamline but multiplies
the *strong* residual, which vanishes at the exact solution. Switching it on at full strength moves
the cylinder drag by 4e-6.

**The test that should have caught the third point did not.** `test_supg_does_not_move_a_converged_solution`
runs channel flow, and a channel has no stagnation point: `u.grad(w)` vanishes wherever `u` does, so
the singular factor is always multiplied by zero. `test_supg_survives_a_stagnation_point` was added
for it, using planar extension on the unit square, which puts an exact stagnation point at a corner.

### 9.3 A failure that was not the stabilisation's fault

Even with tau fixed, a stationary solve *from rest* fails for `supg_factor >= 0.1`, while the same
solve converges without stabilisation. A factor scan separates magnitude from structure: 1e-4 and
1e-2 converge to a drag identical to the unstabilised one, 0.1 and above fail. So the term is
correct and consistent, just fatal at strength.

Making `supg_factor` a global parameter and ramping it from 0 on an already-converged solution
reaches 1.0 without trouble - 0.01, 0.03, 0.1, 0.3, 0.5, 1.0 all converge, with the drag drifting
from 130.3686 to 130.3681. That identifies the cause, and it is section 6.2 again: at rest the
decomposition of grad(u) runs entirely through its degenerate branch, whose reported Jacobian is
truncated, and SUPG multiplies that same residual by `tau*(u.grad w)`, amplifying an error that was
already there rather than introducing one of its own.

So the workaround is to ramp, and the real fix is 6.2. Worth remembering that the fingerprint of
that defect is "converges without X, diverges with X, for X that should be harmless".

---

## 10. Not done

* ~~**A tutorial page for the benchmark.**~~ **Done**: `docs/source/tutorial/pde/navier/viscoelastic.rst`
  is the narrative version, with `viscoelastic_cylinder.py` alongside it, and it carries the drag table
  against Claus & Phillips as well as the stress profiles. The bibliography entries came with it
  (`Oldroyd1950`, `Giesekus1982`, `FattalKupferman2004`, `Hulsen2005`, `Alves2001`, `Claus2013`), so the
  claim below that there are none is superseded too.
* **The wake.** Section 8.4 cannot separate the O-grid from the far field, because the estimator
  redistributes the far field whenever the O-grid changes. Pinning the wake - estimator restricted
  to the cylinder and near wake, or adaptivity off - is the experiment that would settle it, and the
  wake is also where Claus & Phillips report losing convergence. Their tau_xx wake profiles
  (Figs. 6, 7) are graph-readable only, so there is no tabulated target there.
* **The `du/dy` discrepancy of section 8.5**, +0.5% to +2.0% and growing with Wi, is unexplained.
* **Giesekus against their Table 5** (alpha = 0.001, 0.01, 0.1), for which the parameter mapping is
  already worked out in section 8.1.
* **DEVSS-G.** SUPG exists (section 9) but DEVSS-G does not, and it is the other half of what the
  reference uses. Section 9.1 suggests neither is needed to cover the benchmark range.
* **The two source-level fixes** of sections 6.1 and 6.2.
* **3d**, per section 5.
* **Inflow boundary conditions** beyond the two helpers `symmetric_2x2_matrix_log` and
  `oldroyd_b_shear_conformation`, which together give the fully developed log-conformation tensor for
  a prescribed shear rate. Nothing assembles a channel inflow profile from them yet.

**Already closed from that list, so it is not re-attempted:** 6.2 is fixed (see 6.2.1 — the workaround
in `_in_plane_exponential` stays, because the fix does not make the genuinely degenerate rest state
differentiable, and nothing can); the cylinder benchmark is `tests/test_viscoelastic_cylinder.py`;
6.4 is fixed and 6.1 is sidestepped by using the library's convected derivatives. **The order to pick
it up in:** the wake experiment of §8.4, since it is the one open question about the *numbers*, and it
is now the first item rather than the second - the tutorial page and bibliography that used to head this
list are written.
