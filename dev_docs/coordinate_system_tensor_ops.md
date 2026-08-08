# Coordinate-system tensor operations: what is implemented, what is verified, what is left

Written after the run of commits that moved `contract` and the divergence of a rank-2 tensor onto the
standard adjacent-index convention.

Both normal-mode coordinate systems implement all four operators on a bulk mesh, so
`NavierStokesEquations(GCL=True)` works with `azimuthal_stability=True` and with
`additional_cartesian_mode=True`, and so does viscoelastic normal-mode stability analysis. Three wrong
entries turned up in `AxisymmetricCoordinateSystem.directional_tensor_derivative` while deriving the
azimuthal one; they are fixed, and §5 records what they were.

§6 is the part worth reading even if none of the gaps matter to you: it records how these operators
can be tested at all, which is less obvious than it looks and cost most of the effort.

---

## 1. Where things stand

Per coordinate system, for the four operators that take spatial derivatives:

| system | `scalar_gradient` | `vector_gradient` / `vector_divergence` | `tensor_divergence` | `directional_tensor_derivative` |
|---|---|---|---|---|
| `CartesianCoordinateSystem` | ok | ok | ok | ok |
| `AxisymmetricCoordinateSystem` | ok | ok, **swirl-free** (§4) | ok | ok (§5) |
| `AxisymmetryBreakingCoordinateSystem` | ok | ok | ok, bulk only | ok, bulk only |
| `CartesianCoordinateSystemWithAdditionalNormalMode` | ok | ok | ok, bulk only | ok, bulk only |
| `RadialSymmetricCoordinateSystem` | ok | ok | **raises** (§7) | raises |
| `BaseDifferentialGeometryCoordinateSystem` | `TODO` (§8) | `vector_gradient` fixed but unexercised (§8) | inherits, raises | inherits, raises |

"ok" means the three identities of §6.1 hold to machine zero, and there is a test. "bulk only" means
`ndim == edim`; the surface and point cases raise, deliberately, for the reason in §2.3.

---

## 2. The Cartesian normal-mode tensor operations (closed)

[coordsys.py](pyoomph/expressions/coordsys.py) — `CartesianCoordinateSystemWithAdditionalNormalMode`
now implements `tensor_divergence` and `directional_tensor_derivative` for `ndim == edim`.

### 2.1 What it blocked

- `NavierStokesEquations(GCL=True)` takes the divergence of a momentum flux tensor, so the
  conservative ALE formulation could not be combined with
  `setup_for_stability_analysis(additional_cartesian_mode=True)` at all.
- The viscoelastic module advects a tensor field through `material_derivative`, which routes to
  `directional_tensor_derivative`, so viscoelastic Cartesian-wavenumber stability analysis was out too.

Both work now. `docs/source/tutorial/advstab/cartesiannormal/rivulet.py` runs with `GCL=True` and
reproduces its eigenvalue curve to all fourteen printed digits — see §6.3 for why that agreement is
much weaker evidence than it looks.

### 2.2 The derivation: `exp(i*k*z)` against `exp(i*m*phi)`

The whole difference is that a Cartesian frame does not rotate. The azimuthal system carries four
first-order operators because `e_r` and `e_phi` turn with `phi` and because of the `1/r` factor; here
there is **no connection term anywhere and no `over_r`**, and the entire content of the coordinate
system is three operators. With `x_phys = x + eps*Xk*exp(I*k*z)`, `y_phys = y + eps*Yk*exp(I*k*z)` and
the additional direction `z` itself unperturbed, inverting to first order in `eps` gives, for any
scalar `q`,

    d/dx|_(y,z) q  =  q_x - (Xk_x*q_x + Yk_x*q_y)
    d/dy|_(x,z) q  =  q_y - (Xk_y*q_x + Yk_y*q_y)
    d/dz|_(x,y) q  =  q_z - I*k*(Xk*q_x + Yk*q_y)

The `I*k` on the last is the counterpart of the `I*m` in the azimuthal `d_dphi_over_r`; the
`-Xp*q_phi/x**2` term beside it there comes from the `1/r` and has no analogue. The tensor operations
are then those three applied entry by entry, with nothing else added.

Two things about the mechanics that are easy to get wrong. The `z`-derivative must be written
`diff(q, xadd)` and not `I*k*q`: `exp(I*k*z)` is a `FakeExponentialMode`, which prints as `1` in the
generated code but differentiates by the chain rule, so the `I*k` appears by itself on exactly those
terms that carry the mode. The explicit `I*k` inside the correction is right for the opposite reason —
that term always multiplies `Xk` or `Yk`, which carry the mode by construction. And test functions
carry `exp(-I*k*z)`, the complex conjugate, so the field-times-test pairing is `z`-independent and the
residual is a genuine two-dimensional integral; that is handled once in
`get_mode_expansion_of_var_or_test` and the operators need to know nothing about it.

### 2.3 Why the earlier objections to filling this did not hold

This section used to argue the generalisation was not defensible, on two grounds. Both were wrong, and
the reason is worth keeping because it is the kind of mistake that is easy to repeat.

The three operators above are **not a new derivation**. They are, term for term, what the class
already computed in `scalar_gradient` (both branches), `vector_gradient` (both branches, all nine
entries in 2d) and the `ndim == edim` branches of `vector_divergence`. That was checked entry by entry
before a line was written, and it is what makes the new operators checkable against the old ones.

- The `mmterm` with its `diff(0, s1)` products is in the `ndim==2, edim==1` **surface** branch, which
  a bulk-only tensor divergence never reaches — the azimuthal `tensor_divergence` refuses `ndim!=edim`
  for the same reason. Treating it as the template for the bulk case was a category error.
- `# XXX Not according to Duarte` on the `ndim==1` branch of `scalar_gradient` *is* `d_dz` for
  `ndim==1`, and the identical expression appears at three other sites in the same class. It is
  internally consistent; whatever the disagreement was, any error in it is already carried by every
  `grad` and `div` that every existing Cartesian-wavenumber tutorial relies on, so it neither blocks
  nor is aggravated by the tensor operations.

Still open, and still only about the branches the tensor operations refuse: what the `diff(0, s1)`
factors were meant to be, and the `edim==0` branches of `vector_divergence`, which do **not** follow
the operator form either — one of them reads `+ mm*(I*Xk*k*diff(arg[0], xadd))` where `d_dz(arg[1])`
would give `- mm*I*k*Xk*diff(arg[1], x)`, i.e. a different index and the opposite sign. Both already
carry `# TODO: Test this`.

---

## 3. The azimuthal tensor divergence (closed by §6.5)

`AxisymmetryBreakingCoordinateSystem.tensor_divergence` was implemented but verified only in its base
mode: the older test projects a steady solution, i.e. the `eps -> 0` limit, in which the
`d_dphi_over_r` contributions and every `mm*` moving-mesh correction are structurally zero.

That half is now measured, by the eigen-mode projection of §6.5, which reads the first-order value of
an expression straight out of an eigenvector. Both normal-mode systems are covered.

Independently, the conservative ALE momentum flux gives an end-to-end cross-check that needs no new
machinery at all, because `GCL=True` and `GCL=False` are two different discretisations of the same
continuous equations that differ only by the discretely-nonzero `int rho*u_i*div(u)*v_i`. Run on
`docs/source/tutorial/advstab/azimuthal/rising_bubble.py` at `m=1`, on meshes made identical between
the two arms by dumping the base state and reloading it:

| refinement | ndof | \|dLambda\| at Bo=3 | \|dLambda\| at Bo=4 |
|---|---|---|---|
| level 3 | 14204 | 1.9e-2 | 3.4e-2 |
| level 4 | 23375 | 9.2e-4 | 2.1e-3 |
| level 5 | 29780 | 2.8e-4 | 3.0e-4 |

At level 5 the two formulations differ by less than either differs from its own level-4 value
(1.3e-5 against 2.4e-3 in the real part at Bo=4), which is what a discretisation-error difference
converging away looks like and not what a wrong connection term looks like — those move an eigenvalue
by O(1). Do pin the mesh before believing any of this: run without it, the two arms adapt to different
meshes (23652 against 23489 dofs) and the comparison measures the mesh.

---

## 4. Plain axisymmetry cannot represent swirl, which hides its azimuthal row

`AxisymmetricCoordinateSystem` has no azimuthal velocity component anywhere:

- `define_vector_field` ([coordsys.py:473](pyoomph/expressions/coordsys.py#L473)) defines `_x` and
  `_y` only.
- `vector_gradient` ([:453](pyoomph/expressions/coordsys.py#L453)) hard-zeros the azimuthal
  off-diagonals and keeps only the hoop entry `u_r/r`.
- `define_tensor_field` ([:511](pyoomph/expressions/coordsys.py#L511)) puts the azimuthal component on
  the diagonal only, as `_aa` at `[2][2]`.

So every tensor the system can build from fields has `T_rphi = T_phir = T_zphi = T_phiz = 0`, and the
azimuthal row of `tensor_divergence` is identically zero for all of them. That is how both forms of
that row stayed wrong for so long — `(T_phir - T_rphi)/r` in 2d and `(2*T_phir - T_rphi)/r` on a radial
mesh, both fixed in `89e6108` against hand-derived references.

The row is still reachable from a **hand-assembled** tensor, e.g.
`div(dyadic(vector(h,0,0), vector(0,0,k)))`, which is how it is now tested. This is worth knowing
before "simplifying" any of those expressions: the structural zeros mean the test suite cannot be
trusted to catch a regression there unless the test builds its tensors by hand.

If swirl in plain axisymmetry is ever wanted, the whole set above has to change together, and at that
point `AxisymmetryBreakingCoordinateSystem` at `m=0` is probably the better starting point than
extending this one.

---

## 5. `AxisymmetricCoordinateSystem.directional_tensor_derivative` had a wrong entry in every branch

All three branches spelled out the azimuthal frame rotation separately, and all three got it wrong,
each in a different single slot. Found while deriving the azimuthal counterpart, which needs the same
group; now all four call sites go through one `azimuthal_frame_rotation_of_tensor`.

With `d_phi(e_r) = e_phi`, `d_phi(e_phi) = -e_r`, `d_phi(e_z) = 0`, expanding `d_phi(T_ij e_i e_j)`
puts the rotation on each index in turn, i.e. `C = R.T + T.R^T` for the generator `R[azi][rad] = 1`,
`R[rad][azi] = -1`:

| branch | slots | was | should be |
|---|---|---|---|
| `ndim==2` | (r,z,phi) | `C_rz = +T_phiz` | `C_rz = -T_phiz` |
| `ndim==2`, `use_x_as_symmetry_axis` | (z,r,phi) | `C_zphi = T_rz` | `C_zphi = T_zr` |
| `ndim==1` | (r,phi) | no azimuthal term on the diagonals at all, and no radial derivative on the off-diagonals | `C_rr = -T_rphi-T_phir`, `C_phiphi = T_rphi+T_phir` |

Every one of them multiplies an azimuthal off-diagonal, and §4 explains why plain axisymmetry can
build no tensor that has one — which is exactly how all three survived, the same way the two errors in
`tensor_divergence` did. The tests use hand-assembled basis dyads, per §6.2.

The cheap check that catches all three at once, and that is worth applying to anything of this shape:
the rotation must satisfy `C(T^transpose) == C(T)^transpose`. None of the three did. It costs five
lines of sympy and needs no finite elements.

This section previously argued the `ndim==1` branch could not be arbitrated locally, because the
reference for the product-rule identity is built from `directional_derivative` of a *vector*, which in
this coordinate system is itself swirl-free, so both sides of the check would be incomplete in the
same way. That is true of the identity, and the answer is simply not to use the identity: the
orthonormal-frame derivation of §6.2 is the reference, and it settles all three branches at once.

---

## 6. How to test any of this

The reusable part of the exercise. All of it lives in
`tests/test_tensor_index_conventions.py`.

### 6.1 Three identities that hold in every coordinate system

Cheap, and between them they exercise the connection terms of all four operators:

- `trace(grad(u)) == div(u)`
- `div(f*identity_matrix()) == grad(f)` — sharper than it looks: in cylindrical coordinates the radial
  row is only right because the `(T_rr - T_phiphi)/r` hoop term cancels for an isotropic tensor. It is
  also the cheapest expression with a non-zero out-of-plane component, which is how the
  `Cannot expand the field 'coordinate_z'` family of bugs in `a493590` was found.
- `div(dyadic(a,b)) == div(b)*a + (b.grad)a` and
  `(d.grad)(dyadic(a,b)) == dyadic((d.grad)a, b) + dyadic(a, (d.grad)b)` — the product rules. Both
  references are built from operators trusted independently, so no curvilinear connection term has to
  be hand-derived. Use **non-symmetric** dyads or the index convention goes untested.

### 6.2 When an identity cannot reach it, derive by hand in the orthonormal frame

For the azimuthal row of §4 there is no in-system reference, so the expected values are worked out
from `div(a (x) b) = div(b)*a + (b.grad)a` using `d_phi(e_r) = e_phi` and `d_phi(e_phi) = -e_r`, and
written into the test beside each case. Note that only `e_r (x) e_phi` discriminates the sign — it has
`T_rphi != 0` and `T_phir == 0`, so the old expression returned exactly minus the right answer, while
`e_phi (x) e_r` gives the same result either way and would have looked like coverage.

### 6.3 Traps

- **Symbolic comparison does not work for anything involving `grad`.** A gradient of a field stays a
  held `Diff(..., nondimfield(...))` until the code generator resolves it, so `(lhs-rhs).is_zero()` is
  `False` for expressions that are identical. Everything has to go through a compiled element; the
  file projects each expression onto a C2 field and reads the nodal values back.
- **`is_zero()` on a matrix expression does not mean "zero matrix".** It asks whether the *expression*
  is the zero expression, so it answers `False` for a difference that `evalm()` prints as
  `[[0],[0],[0]]` whenever either side is an unevaluated product such as `2*[[1],[2],[3]]`. The core's
  `double_dot_eval` uses `is_zero_matrix()` for exactly this reason. Compare components as floats.
- **A free-stream GCL test needs a non-harmonic mesh velocity.** The two possible orderings of the
  conservative momentum flux differ by a field whose curl is `U*laplacian(w_y)`, so with
  `LaplaceSmoothedMesh` the difference is a pure gradient, the pressure field absorbs it, and the test
  passes whichever ordering is used. `PrescribedMovingMesh` with a deliberately non-harmonic velocity
  is what makes it discriminate.
- **A normal-mode test that projects nodal values sees the base residual only.** This is the big one,
  and it is why §2 and §3 stood unverified for so long while looking tested. `_projected` in
  `tests/test_tensor_index_conventions.py` solves a steady problem and reads the nodes, i.e. the
  `eps -> 0` limit, in which every `mm*` mesh-perturbation term and every `I*k`/`I*m` term is
  multiplied by zero. Such a test pins the index pattern and the connection terms and *nothing else*.
  §6.5 is the way round it.
- **A normal-mode moving-mesh test needs mesh dofs that are free *and* excited.** Every `mm*` term is
  multiplied by the mesh perturbation, so if the eigenvector leaves the mesh alone they all vanish and
  the test passes having checked nothing. `PrescribedMovingMesh` pins the position dofs outright. A
  smoothed mesh does not have that problem but has the opposite one: its position equations carry no
  time derivative, so the eigensolver refuses the problem with an empty mass matrix, and adding
  `partial_t` of a mesh coordinate does not fix it. What works is slaving the mesh to a field that does
  have a time derivative — `_MeshSlavedToADiffusingField` in the test file is four lines and both mesh
  components come out excited by the same eigenvector. Assert on the mesh perturbation itself, so that
  a run where it collapsed fails rather than passes.
- **`rivulet.py` with `GCL=True` is the worked example of a check that proves only that the code
  compiles.** Its base state is a *static* rivulet: `u` and the mesh velocity are identically zero, so
  `div(rho*dyadic(u,u-w))` is quadratic in the perturbation and contributes an identically zero block
  to the base residual *and* to the eigenproblem. The eigenvalues match to fourteen digits and not one
  `mm*` term has been evaluated. Contrast `rising_bubble.py` in §3, whose base flow is non-zero.
- **At k=0 the in-plane mesh corrections drop out.** Measured: removing the `mm*` corrections of `d_dx`
  and `d_dy` entirely leaves a `k=0` normal-mode eigensolve bit-identical. So the otherwise attractive
  "compare the normal-mode system at k=0 against plain Cartesian" reduction — which is exact, see
  §6.6 — cannot validate those terms, only the index pattern and the derivative structure. A test at
  `k != 0` is needed as well. Why the two separate has not been established, only observed.

### 6.4 Reverting is the only proof a test bites

Every fix in these commits was checked by putting the old expression back and confirming a named test
fails. Worth keeping up: several of the bugs here are in code paths whose operands are structurally
zero, where a test can look thorough and assert nothing.

It is worth scripting rather than doing by hand — patch, run, restore, per regression — because the
interesting output is *which* tests fail, not merely that some do. That is what showed the three tests
here have genuinely different reach: the `I*k` sign is caught only by §6.5, a transposed tensor index
by all three, and the `d_dx` mesh coefficient by §6.5 alone and not by §6.6 (see the last trap in
§6.3, which is exactly how that was found).

### 6.5 Reading a first-order value straight out of an eigenvector

The way past the first normal-mode trap in §6.3, and the thing that closes §3.

A projection residual `weak(unknown - expr, test)` has no mass-matrix row, so its row of
`J v = lambda M v` reads `d(unknown) - d(expr) = 0`. The eigenvector entry for that field therefore
**is** the first-order-in-eps value of `expr` at the requested wavenumber — every `mm*` and every
`I*k`/`I*m` term included. Solve the eigenproblem, push the eigenvector back in with
`set_current_dofs`, and read the nodes exactly as a steady projection would.

Points that matter in practice:

- Read the real and the imaginary part separately. The entry is a complex number and the eigensolver
  is free to return the whole vector times any phase, which can leave either part at zero.
- Only compare *identically zero* expressions against zero. Anything the C2 space cannot represent
  exactly leaves a non-zero base residual, and the first-order change of the integration measure then
  feeds into the entry as well. Identity residuals are exactly zero and therefore clean; raw operator
  values are only good for magnitude checks, or for comparing two runs against each other as in §6.6.
- The problem needs no flow and no physics — `var("coordinate_x")` already expands to
  `x + eps*Xk*exp(I*k*z)` once coordinates are dofs. A 3x3 mesh and a diffusing field are enough.

### 6.6 The k=0 reduction, for a reference outside the coordinate system

Every identity in §6.1 compares the tensor operations against `grad` and `div` of the *same* class, so
a defect shared by both sides cancels. The one available independent reference is the ordinary
eigensolve: mesh sensitivity there comes from the C++ shape-derivative machinery, whereas for the
normal-mode contributions that machinery is switched off — `problem.py` calls
`set_ignore_dpsi_coord_diffs_in_jacobian` for them — and replaced by the `mm*` terms plus the
first-order measure in `integral_dx`. Two independent implementations of the same derivative.

At `k=0` the additional direction decouples and the in-plane spectra coincide: measured agreement is
1e-12 on the eigenvalues and 1e-13 relative on the projected operator values of §6.5. `k` is an
ordinary global parameter, so `k=0` is a plain numerical evaluation and nothing degenerates.

Compare raw operator values, not identity residuals, or the independence is thrown away. Phase-fix
each eigenvector by dividing through its own entry at the node where some reference field is largest,
and use only the leading eigenvalues — higher relaxation modes come in near-degenerate pairs where the
basis within the pair is undetermined and a componentwise comparison means nothing. And see the last
trap in §6.3 for what this does *not* reach.

---

## 7. `RadialSymmetricCoordinateSystem.tensor_divergence` raises

[coordsys.py:1577](pyoomph/expressions/coordsys.py#L1577). Unlike §2 this is not blocked by uncertain
provenance, but it is not a mechanical fill-in either.

The system is **spherical** — `vector_gradient` ([:1546](pyoomph/expressions/coordsys.py#L1546))
returns `diag(d_r u, u/r, u/r)` and `vector_divergence` gives `d_r u + 2u/r`. The polar and azimuthal
rows of a spherical tensor divergence carry `cot(theta)` terms, and this system models no `theta`
coordinate at all. The rows only vanish when `T_thetatheta == T_phiphi`, i.e. for a genuinely
spherically symmetric tensor, for which

    (div T)_r = d_r T_rr + (2*T_rr - T_thetatheta - T_phiphi)/r

and the other two rows are zero. That much is derivable and checks out against
`div(dyadic(a,b)) = div(b)*a + (b.grad)a` on the basis dyads.

The open question is what to do with input outside that class. There is no `define_tensor_field` for
this coordinate system — it inherits the base one, which raises — so there is no "natural" tensor to
take as the contract. Options: implement the diagonal case and raise on anything else; implement it
and document that off-diagonal entries are ignored; or leave it. Needs a decision about what
spherically symmetric tensors are for here before code.

---

## 8. The differential-geometry gradient is reasoned, not measured

`BaseDifferentialGeometryCoordinateSystem.vector_gradient`
([coordsys.py:1766](pyoomph/expressions/coordsys.py#L1766)) returned the transpose of what every other
coordinate system returns, `d(u_j)/d(x_i)` instead of `d(u_i)/d(x_j)`. `8c6fbfb` swapped the `dyadic`
arguments to fix it, and reducing to Cartesian (`g^ab = delta_ab`, `t_a = e_a`) makes the old form's
error immediate: `sum_a e_a[i] * d arg_j/d x_a` is `d arg_j/d x_i`.

Nothing in the repository subclasses this coordinate system, so the path is unexercised and the fix is
untested — reasoning, not measurement, and the comment in the method says so. The sibling
`scalar_gradient` just above it still carries the original `# TODO: This is likely not right!`
([:1763](pyoomph/expressions/coordsys.py#L1763)); that one was left alone, since a scalar gradient has
no index order to get wrong and the `TODO` is presumably about the augmented-dimension handling.

Testing this needs a concrete manifold subclass implementing
`get_real_position_vector_from_mesh_coordinates`. The helix in
`docs/source/tutorial/spatial/mesh/curvedline.rst` is the natural candidate, but it only ever takes
gradients of a *scalar*, which has no index order to get wrong.

---

## 9. One thing that is not a coordinate-system issue

`8c6fbfb` changed `contract(A,b)` and `A @ b` from `A_ji*b_j` to `A_ij*b_j`. Scripts outside this
repository that wrote the advection term as `contract(u, grad(u))` will now silently compute
`grad(|u|^2/2)` instead. Nothing in the repository was affected, and the CHANGELOG carries the
migration, but external user scripts cannot be swept — worth timing deliberately relative to the next
merge into `main`.
