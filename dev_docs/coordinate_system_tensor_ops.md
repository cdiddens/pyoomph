# Coordinate-system tensor operations: what is implemented, what is verified, what is left

Written 2026-08-07 on branch `develop`, after the run of commits that moved `contract` and the
divergence of a rank-2 tensor onto the standard adjacent-index convention (`8c6fbfb`, `89e6108`,
`aa8ec89`, `a493590`, `926fe0c`).

That work is finished and tested. This document is for what it *did not* close: five gaps in
`pyoomph/expressions/coordsys.py` that are either unimplemented, unreachable-and-therefore-unverified,
or verified only in part. None of them blocks anything the tutorials or the test suite do, which is
exactly why they need writing down — each survived for years behind a structural zero.

§6 is the part worth reading even if none of the gaps matter to you: it records how these operators
can be tested at all, which is less obvious than it looks and cost most of the effort.

---

## 1. Where things stand

Per coordinate system, for the four operators that take spatial derivatives:

| system | `scalar_gradient` | `vector_gradient` / `vector_divergence` | `tensor_divergence` | `directional_tensor_derivative` |
|---|---|---|---|---|
| `CartesianCoordinateSystem` | ok | ok | ok | ok |
| `AxisymmetricCoordinateSystem` | ok | ok, **swirl-free** (§4) | ok | ok on the diagonal (§5) |
| `AxisymmetryBreakingCoordinateSystem` | ok | ok | ok, base mode verified only (§3) | raises |
| `CartesianCoordinateSystemWithAdditionalNormalMode` | ok, one `XXX` (§2.3) | ok | **raises** (§2) | **raises** (§2) |
| `RadialSymmetricCoordinateSystem` | ok | ok | **raises** (§7) | raises |
| `BaseDifferentialGeometryCoordinateSystem` | `TODO` (§8) | `vector_gradient` fixed but unexercised (§8) | inherits, raises | inherits, raises |

"ok" means the three identities of §6.1 hold to machine zero, and there is a test.

---

## 2. `CartesianCoordinateSystemWithAdditionalNormalMode` has no tensor operations

[coordsys.py:893](pyoomph/expressions/coordsys.py#L893) and
[:896](pyoomph/expressions/coordsys.py#L896) both raise `Implement the ... for this coordinate
system`.

### 2.1 What it blocks

Not hypothetical. Both were run and both fail:

- `NavierStokesEquations(GCL=True)` takes the divergence of a momentum flux tensor
  ([navier_stokes.py:640](pyoomph/equations/navier_stokes.py#L640)), so the conservative ALE
  formulation cannot be combined with
  `setup_for_stability_analysis(additional_cartesian_mode=True)` at all.
- The viscoelastic module advects a tensor field through `material_derivative`, which routes to
  `directional_tensor_derivative`, so viscoelastic Cartesian-wavenumber stability analysis is out too.

The azimuthal counterpart implements both, so neither restriction applies to
`azimuthal_stability=True`. `tests/test_tensor_index_conventions.py` asserts the gap *is* a gap, so
filling it will fail a test rather than pass silently.

### 2.2 Why it was not filled

It looks like it should be mechanical — Cartesian has no connection terms, so a tensor divergence
ought to be "replace `d_z` with `I*k`" applied row by row. It is not, because the class also carries
first-order mesh-perturbation corrections, and those are where the difficulty is. Compare
`vector_divergence` ([:768](pyoomph/expressions/coordsys.py#L768)): the `edim==2` branch is one line
of `mm*(...)` corrections, but the `edim==1` branch is a `mmterm` of some thirty products built from
`diff(X01, s1)`, `diff(Xk1, s1)` and the local coordinate `s1`.

### 2.3 Two things to resolve before generalising it

Both are pre-existing and neither was touched:

- **`diff(0, s1)` inside that `mmterm`** ([:801](pyoomph/expressions/coordsys.py#L801)) — the
  derivative of a literal zero, so those four products vanish identically. That is the signature of a
  symbolic derivation in which some quantity was substituted as `0` and the result pasted in without
  cleaning. Harmless arithmetically; the worry is what the `0` was *meant* to be. Until that is known,
  the expression cannot be trusted as the template for a rank-2 version.
- **`# XXX Not according to Duarte`** on the `ndim==1` branch of `scalar_gradient`
  ([:757](pyoomph/expressions/coordsys.py#L757)) — a disagreement someone already flagged and left.

Deriving a tensor divergence on top of terms with that provenance is not defensible. Either sort out
the `mmterm` first, or — the cheaper option — leave the operations unimplemented and improve the error
message to name the blocked combinations (`GCL=True`, viscoelastic) instead of the bare "Implement the
tensor_divergence for this coordinate system", which gives a user no idea why their problem is refused.

---

## 3. The azimuthal tensor divergence is verified only in its base mode

`AxisymmetryBreakingCoordinateSystem.tensor_divergence`
([coordsys.py:1413](pyoomph/expressions/coordsys.py#L1413)) is implemented, and `89e6108` rewrote its
index pattern for the second-index convention.

What the test in `tests/test_tensor_index_conventions.py` establishes: with
`setup_for_stability_analysis(azimuthal_stability=True)` the three identities of §6.1 hold, and the
projected values agree with the plain axisymmetric system. That comparison is the load-bearing part —
it pins the index pattern and the connection terms, which is what the rewrite changed. The mode
machinery is genuinely engaged and not skipped: the generated code is about 53 kB against 38 kB for
the plain axisymmetric run.

What it does not establish: projected values are the **base residual**, the `eps -> 0` limit. The
`d_dphi_over_r` contributions and every `mm*` moving-mesh correction live in the eigen-residual, and a
projection cannot see them. So the mode-dependent half of that method remains unverified. It was
unverified before the rewrite too, and the rewrite was a term-by-term transposition checked against
the canonical cylindrical formulas — but that is reasoning, not measurement.

To close it, something has to compare an actual eigen-residual against an independent reference. The
obvious candidate is a manufactured azimuthal mode with a known analytic answer, run through
`get_last_eigenvalues`/`get_last_eigenvectors`; none of the existing azimuthal tutorials help, because
none takes the divergence of a tensor.

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

## 5. The 1d axisymmetric directional tensor derivative drops off-diagonal terms

The `ndim==1` branch of `AxisymmetricCoordinateSystem.directional_tensor_derivative`
([coordsys.py:606](pyoomph/expressions/coordsys.py#L606)) writes only four entries and omits, on the
off-diagonals, the radial-derivative term `d_0 * d_r T_01` that the `ndim==2` branch below it includes
for every `i,j`. It also omits the azimuthal connection on the diagonals.

Every omitted term is multiplied by an off-diagonal entry, and §4 explains why those are structurally
zero on a radial mesh, so nothing reachable is affected — the product-rule identity of §6.1 passes
there.

It is left alone because it cannot be arbitrated locally: the reference for the identity is built from
`directional_derivative` of a *vector*, i.e. `matproduct(grad(a), b)`, and that system's
`vector_gradient` is itself swirl-free. Both sides of the check would be incomplete in the same way.
Fixing it means hand-deriving the terms as §6.2 did for the divergence.

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

### 6.3 Three traps

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

### 6.4 Reverting is the only proof a test bites

Every fix in these commits was checked by putting the old expression back and confirming a named test
fails. Worth keeping up: several of the bugs here are in code paths whose operands are structurally
zero, where a test can look thorough and assert nothing.

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
