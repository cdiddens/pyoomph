# The first Lyapunov coefficient, and switching onto the emerging orbit

`Problem.switch_to_hopf_orbit()` turns a converged Hopf point into a periodic-orbit guess. The number
it needs is the first Lyapunov coefficient, computed by `get_hopf_lyapunov_coefficient()`
(`pyoomph/generic/bifurcation_tools.py`) — Kuznetsov's real-form algorithm, generalised to a mass
matrix.

§1 is the audit of that routine (it had no test of any kind); §2 what the audit changed; §3 the
reference values now pinned; §4 where it stands under MPI.

---

## 1. The algorithm, checked term by term

For `M x' = f(x)` with `f = A x + ½B(x,x) + ⅙C(x,x,x)`, the right and left eigenvectors at the Hopf
point satisfy `A q = iω₀ M q` and `Aᵀ p = -iω₀ Mᵀ p`, normalised so that `⟨p,Mq⟩ = 1`.

| code | is | check |
|---|---|---|
| `a`, `b`, `c` | `B(qR,qR)`, `B(qI,qI)`, `B(qR,qI)` | polarisation, exact |
| `r` | `A⁻¹B(q,q̄)` | `B(q,q̄) = a+b`; its imaginary part cancels by symmetry |
| `sv` | `(2iω₀M − A)⁻¹B(q,q)` = `h20` | `B(q,q) = a−b+2ic` |
| `sig` | `Re⟨p,B(q,r)⟩` | `= pR·B(qR,r) + pI·B(qI,r)` |
| `d0` | `Re⟨p,B(q̄,s)⟩` | `= d1+d2+d3−d4` |
| `g0` | `Re⟨p,C(q,q,q̄)⟩` | `C(q,q,q̄) = C(uuu)+C(uvv) + i[C(uuv)+C(vvv)]`, u=qR, v=qI |
| `ga` | `(g0 − 2·sig + d0)/(2ω₀)` | Kuznetsov's invariant form |

**It is correct.** Three points worth recording, because each had a `TODO` or a comment doubting it:

* **The mass matrix belongs in exactly two places** — the `h20` operator and `⟨p,Mq⟩=1`. Neither
  right-hand side carries an `M`. pde2path puts `M` on both; the two commented-out lines in the file
  are that variant. The independent `NormalFormCalculator.get_normal_form_hopf` in the same file
  (ported from BifurcationKit) agrees with the form used here.
* **`⟨p,Mq̄⟩ = 0` is automatic** for `ω₀ ≠ 0`: `p̄ᵀA = iω₀p̄ᵀM` and `Aq̄ = -iω₀Mq̄` give
  `2iω₀⟨p,Mq̄⟩ = 0`. Correctly checked rather than enforced.
* **The `pI` vs `pR` question in `sig2`, `d3` and `d4` resolves to `pI`** — it falls out of expanding
  `Re[(pR − i·pI)·(...)]`. The three `# TODO: In the book, it is pI, in pde2path is is pR` comments
  are gone, replaced by the expansion.
* **The `θ` rotation is right.** `θ = angle(conj(p)·Mq)` then `p ← p·e^{iθ}` gives
  `conj(p·e^{iθ})·Mq = e^{-iθ}|⟨p,Mq⟩|e^{iθ}`, real and positive. Measured `Im⟨p,Mq⟩ = 2e-17`.

**`ga` is not a mesh-independent number.** Under `q → cq` (with `p → p/c̄` to keep the normalisation)
`g0` scales as `|c|²`, so spreading the same uniform mode over a finer mesh shrinks it — measured
exactly `ga_ode/41` on a 41-node mesh — while `al` grows by the same factor. **The orbit amplitude
`al·q` is the invariant combination**, which is why the end-to-end test asserts a radius and not a
coefficient.

## 2. What was wrong around it

None of these touched the result on the systems tested (the Lorenz tutorial returns
`al = 3.4389365750199032` before and after, to every digit), but each is a real defect:

* **The three `CHECKING r/sR/sI` diagnostics were stale.** They carried an `M@` on the right-hand
  side from the pde2path variant two lines above, so they printed the residual of an equation that
  was never solved and reported a large error for a correct solve. They now check what is solved, and
  read 0 to 4e-16.
* **`use_hopf_tracker_for_adjoint` was inverted** — `or` where `and not` was meant, so asking for the
  Hopf tracker selected the eigensolver.
* **Three "should be zero" checks had no `abs`** before `amax`, so a large *negative* residual passed;
  and one `A*qI` where every sibling used `A@qI` (identical for `spmatrix`, elementwise for
  `sparray`). They are now a named table, and a warning fires when the worst exceeds
  `residual_tolerance`.
* **`activate_eigenbranch_tracking(eigenvector=q)` passed the caller's `q`,** which is `None` whenever
  the auto-solve branch produced the eigenvector. Now `q_resolved`.
* **`if evalT*omega0>0`** compared an array against a scalar and would raise.
* **A near-zero `⟨p,Mq⟩` printed 27 diagnostic lines and then divided by it anyway.** It now raises:
  that is the adjoint failing to pair with `q`, usually the wrong member of the conjugate pair, and
  dividing produces a confident wrong answer.
* **`#TODO: Is csr or csc?`** ×3 — answered: the binding's docstring says compressed sparse *row*, so
  `A`/`AT` were never swapped.
* Dead code removed (`qRus/pRus`, `d3f`'s discarded `direct_scale`, the `mu0` branch behind
  `if False`, `f0`). The two finite-difference cross-checks of the analytic Hessian were **not**
  deleted but put behind `check_derivatives_by_fd=True` — they are the right tool when a new
  element's Hessian is suspect.
* The prints are now behind `verbose` (default `True`, so nothing changes unasked).
* `PeriodicOrbit.iterate_over_samples` now restores the handler's dof backup in a `finally`. Without
  it, an exception in the caller's loop body left the backup in place, and the *next* `backup_dofs()`
  raised "the dofs have already been backed up" — which is what the user saw instead of their own
  error, including on the way out of the `with` block.
* In `switch_to_hopf_orbit`: the manual branch's `sign` was `0` rather than `-1`, and the
  collapse check compared a norm against a squared radius.

## 3. What is pinned now

`tests/test_hopf_lyapunov.py`, 16 cases. There was previously **no numerical test of `ga`, `c1`, `al`
or `dlam` anywhere** — the tutorials only check that they run.

**The Hopf normal form in Cartesian coordinates.** Its nonlinearity is cubic, so `B ≡ 0` and the
coefficient comes entirely from the `C` term — the half that has to be finite-differenced out of the
analytic Hessian, and the half with no other coverage.

| | expected | measured |
|---|---|---|
| `ga` | `2σ` (by hand, `Re⟨p,C(q,q,q̄)⟩/(2ω₀) = 4σ/2`) | `2σ` to 1e-15 |
| `ga` with `m1=m2=2` | unchanged, `ω₀ → 1/m` | unchanged; `ω₀ = 0.5` |
| orbit radius | `sqrt(-µ/σ)` exactly | 1.4e-3 relative, independent of `eps`, `m` and mesh |
| period | `2πm` exactly | 3.6e-8 relative |

**The Brusselator** supplies the quadratic term the normal form lacks: measured `|r| = 0.30`,
`|s| = 1.1` at `A=1.5` against `0` for the normal form, so it is the case that reaches the `h11`/`h20`
solves and the `sig`/`d0` terms at all. Its Hopf is at `B = 1+A²` with `ω₀ = A`, both exact, and `ga`
comes out as exact rationals (`-1/2`, `-17/66`, `-1/6` at `A = 1, 1.5, 2`) — characterisation values,
not derived.

**The amplitude test is the one that isolates `al`.** A wrong coefficient still converges to the true
limit cycle, so only the *guess* is wrong: comparing the guess amplitude `2·eps·al` against the
solved one gives 4.3% at `eps=0.1` and 2.1% at `eps=0.05`, halving as it must. This works on any
system, including ones with no closed-form cycle.

## 4. MPI

Still to do; see the plan. `switch_to_hopf_orbit` refuses `--distribute` because this routine
gathers the global pencil and calls scipy, and because
`Problem::get_second_order_directional_derivative` is memory-unsafe when distributed (it indexes the
caller's direction by *global* equation number while requiring it to be `nrow_local` long, skips no
halo elements and never reduces — the same shape as the `set_history_dofs` overrun fixed in
`a56b6ce`). It is correct under a plain replicated `mpirun`, where `nrow_local == ndof()`.
