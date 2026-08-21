# Salts: transport, evaporation and Marangoni without any electrostatics

**State:** implemented and tested (`tests/test_salt_transport.py`, 22 tests). The electroneutral model,
the interface condition that keeps a salt in an evaporating liquid, and the surface-tension coupling
are complete for the cases listed under "Verified" below. Everything here is measured.

Companion to [electrohydrodynamics.md](electrohydrodynamics.md), which covers the case where the
potential *is* solved for. The two share their field names on purpose; §4 says why.

## 1. The problem this closes

`water.add_salt("NaCl", 1*milli*molar)` put ions into the material, and `CompositionFlowEquations`
did not look at them. `CompositionAdvectionDiffusionEquations` builds its field list from
`fluid_props.required_adv_diff_fields`, which is `components - {passive_field}`, i.e. the *solvents*;
the ion table is a different dictionary and nothing read it. A salted mixture therefore generated
exactly the same system as an unsalted one, silently. That is the trap `salts="auto"` closes: the
salt transport is added whenever the material carries a salt, and an unsalted material is unaffected
because there is nothing to add.

## 2. One field per salt, not per ion

Without a potential the two ions of a salt cannot have separate degrees of freedom. They would
diffuse apart at their own rates — Na+ at 1.33e-9 and Cl- at 2.03e-9 m²/s — and nothing in the system
would pull them back, because what pulls them back in reality is the electric field of the charge
separation they would create.

So the unknown is the salt, and the ion concentrations are `define_field_by_substitution` multiples of
it. Electroneutrality is then structural: `get_charge_density()` returns literal zero, not a small
number. The rate the pair moves at is the **ambipolar diffusivity**

    D_s = (z+ - z-) D+ D- / (z+ D+ - z- D-)

which is derived from the ion table rather than tabulated, and reproduces the measured salt
diffusivities at 25 °C:

| | NaCl | KCl | CaCl2 | LiCl | Na2SO4 | HCl |
|---|---|---|---|---|---|---|
| from the ion table | 1.610 | 1.994 | 1.335 | 1.367 | 1.230 | 3.336 |
| measured | 1.610 | 1.990 | 1.335 | 1.370 | 1.230 | 3.340 |

in 1e-9 m²/s. HCl is the case that shows this is a prediction and not a fitted mean: its two ions
differ in diffusivity by a factor of 4.6, and the salt still moves at one well-defined rate about a
third of the proton's.

## 3. The interface condition, and why it is three different terms

**A salt does not evaporate, but that is not the same as "no flux".** The natural boundary condition
of the assembled weak form is zero *diffusive* flux, and a receding surface then sweeps the salt out
with the vapour: the concentration never changes and the dissolved amount falls as the film thins.
What is needed is zero flux **relative to the moving interface**, and which term produces that depends
on how the bulk equation was assembled — which is exactly what the GCL switch changes:

| bulk form | natural condition of the weak form | interface term |
|---|---|---|
| advection not by parts | zero diffusive flux | `-c*j_total/rho` |
| by parts, no GCL | zero flux in the lab frame | `+c*u_mesh.n` |
| GCL (conservative ALE) | zero flux through the moving boundary | nothing |

The three are one statement seen from three sides: with the kinematic condition `(u-u_mesh).n = j/rho`,
the first two differ by exactly the surface term that integrating the advection by parts leaves
behind. The middle row was derived by *calibrating against* the first rather than from scratch, after
a sign error there destroyed three quarters of the salt (`N/N0 = 0.256` where it should be 1).

The GCL row is the interesting one. Its bulk form is the derivative of the whole integral plus
advection with the velocity **relative to the mesh**, so "nothing leaves through a moving boundary"
is what the weak form already says. Measured on a 1d film evaporating to half its height:

| | 10 steps | 20 steps | 40 steps | 80 steps |
|---|---|---|---|---|
| non-conservative | 1.0e-2 | 2.6e-3 | 6.8e-4 | 1.7e-4 |
| GCL | | 3.2e-15 | | 2.2e-14 |

relative error in the total dissolved amount. The non-conservative forms converge at second order,
i.e. the salt balance is only as good as the time stepping; GCL conserves to machine precision at any
step size, and still does at a Péclet number of 14 where the enrichment layer is steep.

**A GCL caveat that is not ours.** In the same runs the *interface position* is 2.5% off at 20 steps
and 0.6% at 80, converging at first order, while the non-conservative branch lands on the analytic
position to 6 digits. That is the GCL continuity formulation, not the salt, but it means a test that
wants the position must measure it rather than assume `L0/2`.

## 4. Sharing the field names with Nernst-Planck

`c_Na_p` is a substituted field here and a solved dof under `PoissonNernstPlanck`. That is deliberate:
a surface tension law, a `DirichletBC`, an observable or an output written against it does not know
which model is running, so moving a problem from the electroneutral route to the full one changes the
equations and nothing else. `ion_fieldname_stem` is shared, so the two cannot sanitize a name
differently.

What must not happen is both at once — a substituted `c_Na_p` shadowed by a solved `c_Na_p` would run
and be wrong — so `SaltTransportEquations.define_fields` refuses a domain that carries both.

**The two models agree where they should.** An electroneutral gradient relaxing in a 1 µm box
(λ_D = 0.96 nm, so the double layer has no room to matter): the first Fourier mode decays to 0.610553
of its amplitude under the electroneutral model and 0.610554 under full Poisson-Nernst-Planck with
two ion fields and a potential — a relative difference of 2e-6, against the analytic
`exp(-π²D_ambipolar t/L²) = 0.610498`. The two models agree with each other far more closely than
either agrees with the continuum answer, which is what "the same physics, discretised twice" looks
like.

## 5. Surface tension and the direction of the Marangoni flow

`SaltProperties.surface_tension_increment` is dσ/dc at 25 °C, tabulated where the literature has it
(Weissenborn & Pugh; Ozdemir et al.) and zero otherwise, so that a salt without a measurement
contributes no Marangoni stress rather than a guessed one.

**It is positive**, because ions are pushed *away* from the surface: an ion there would give up part
of its hydration shell and is repelled by its own image charge in the low-permittivity vapour, so the
surface is slightly purer than the bulk and costs more to make. The strong acids are the exception
and are negative — the proton does sit at the surface.

The sign has a consequence worth stating plainly: **salt Marangoni pulls the surface towards the
enriched region**, the opposite of a surfactant. An evaporating drop drives its surface *towards*
wherever it is drying fastest. Measured on a 2d pool with an imposed surface gradient: mean surface
velocity +1.30e-3 m/s towards the salt with the tabulated increment, -1.30e-3 m/s with the sign
flipped (symmetric to 5 digits), and 1e-13 — ten orders down — with the increment set to zero.

**Where the shift is applied.** `LiquidGasInterfaceProperties.surface_tension` is a property whose
getter adds `Σ (dσ/dc)_s c_s` to whatever was stored. A property rather than an assignment in one
constructor, because a registered interface class may set σ from its own correlation at any point in
its `__init__`, and the shift has to survive that.

**It is written against the ion concentrations, not the salt field**, because `c_Na_p` is a name both
electrolyte models have and `c_NaCl` is not — under Nernst-Planck there is no salt field at all. The
per-salt increment is carried by an ion that no *other* dissolved salt contributes to (the anion by
preference, since sharing a cation is the common case), and a salt sharing both of its ions with the
others is refused with a message rather than silently attributed.

That makes the fields the surface tension depends on exist under either model — but not when neither
is present, and a script that captures `interf.surface_tension` while building its equations freezes
the expression before any equation could decide. So `salts=False` on a salted material does not add
*nothing*: it adds `FrozenSaltConcentrations`, which defines the same names as constants at their
bulk values, and which stands down if it finds an electrolyte model on the domain when
`define_fields` runs. That is what lets `salts=False` be the route to Poisson-Nernst-Planck and the
route to a frozen salt at the same time. A lazily-evaluated flag on the interface was tried first and
does not work, for the reason above.

`add_salt` also records the concentrations in `initial_condition` under the field names — the salt's
and both ions'. Without them, every scale that is computed by evaluating an expression "at the IC" —
and the surface tension is now one of those expressions — would fail on a salted liquid with an
unresolvable field.

One consistency point that was a bug first: a substituted field is registered nondimensionally, so
its scale has to be set separately or `var("c_Na_p")` is a bare number under the electroneutral model
and a dimensional concentration under Nernst-Planck. An expression written for both would then be
right in only one of them.

## 6. Dilute solute, and where that stops being true

The salt does not enter the mass fractions and does not change ρ or μ. At 1 mM a salt is 6e-5 of the
solution by mass, and pretending it displaces some of the water would be a larger error than ignoring
it. `DissolvedSpeciesComponent.mass_fraction_in` is how a script checks where it stands.

This is wrong for a drop drying to saturation: NaCl saturates at 26 wt%, where the assumption fails
completely and crystallisation starts. Nothing here warns about that, and the extension point is a
ρ(c)/μ(c) hook rather than promoting the salt to a mixture component — which would collide with the
fraction bookkeeping and needs mixture correlations the library does not have.

## 7. Verified

`tests/test_salt_transport.py`, 22 tests: the ambipolar diffusivities against six measured values;
the `add_salt` overload and two salts sharing an ion; auto-pickup and its opt-out; the shared field
names and the refusal to run both electrolyte models; conservation in all three ALE forms with the
convergence orders above; the enrichment layer against the quasi-steady bound, mesh converged to
0.06%; the surface tension of the enriched surface; the Marangoni direction and its two controls; the
agreement with Poisson-Nernst-Planck; the surface tension law working
under Nernst-Planck as well; the frozen fallback with the transport switched off; and a salted
glycerol-water mixture transporting its solvent fractions and its salt side by side.

Not covered: axisymmetric and 3d, adaptivity, MPI, and any of it under `--distribute`; a salt at an
interface between two liquids; and the combination with a Maxwell stress, which is
`electrohydrodynamics.md`'s territory and needs the potential.
