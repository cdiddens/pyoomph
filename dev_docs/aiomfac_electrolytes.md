# AIOMFAC with ions: activity coefficients of a salt solution

**State:** implemented and checked against AIOMFAC itself (`tests/test_activity_aiomfac.py`, 25
tests). The short-range parameters were regenerated from the current AIOMFAC source, and the
middle- and long-range parts were added in all three back-ends. Everything below is measured.

Companion to [salt_transport.md](salt_transport.md), which moves the salt about, and to
[electrohydrodynamics.md](electrohydrodynamics.md), which is about the potential.

## 1. Why: a drying brine evaporated like pure water

Salt was transported and it changed the surface tension, but not the *activity* of the solvent — so
the vapour pressure over a brine was that of pure water, and an evaporating drop dried at the wrong
rate. AIOMFAC is the only model in the library with an electrolyte generalisation, and only half of
it was present: the short-range (UNIFAC) part, including the ions' `R` and `Q`, but neither the
middle-range ion interactions nor the long-range Debye–Hückel term, and no ion parameters at all.

## 2. What the parameter audit found

`citools/generate_aiomfac_parameters.py` parses the AIOMFAC Fortran source and regenerates
`aiomfac.py` and `aiomfac_electrolyte.py`; `citools/aiomfac_param_import.py` is the reader. Run it
against a clone and it reports what differs — the previous tables had been imported once by a script
nobody kept, from a version nobody recorded. Against AIOMFAC 3.13 (commit 492b091) it found:

* **Two ions attached to the wrong species.** Subgroup 246 was called `CH3COO-` and is `IO3-`; 247
  was called `SCN-` and is `OH-`. Not a naming quibble: every middle-range table is indexed by these
  ids, so the names decide which parameters a species gets.
* **Ion molar masses in g/mol** while every neutral subgroup was in kg/mol — 22.99 against 0.015034.
  Harmless only for as long as nothing read them, and `define_sub_group` was in fact *discarding*
  the value it was given. The middle-range part weights a main group's contribution by its mass, so
  it stopped being harmless.
* **7 ions with placeholder `R=Q=1`** (Ba²⁺, Sr²⁺, Co²⁺, Ni²⁺, Cu²⁺, Zn²⁺, Hg²⁺) that AIOMFAC does
  not name and has no parameters for. Dropped: a solution using them would have produced numbers.
* **155 interaction entries that differed** and **528 nonzero ones that were missing**, plus a whole
  main group (76, CO₂) and 14 subgroups. Regenerated.
* **246 pairs AIOMFAC marks as never determined.** They live in the tables as a marker (−8.89e5,
  where a real parameter is of order 1e3). Importing them verbatim gives `exp(-888889/T)` = 0, i.e.
  every activity coefficient identically zero — which is how they were found. They are now listed in
  `undetermined_interactions` and a mixture that needs one is refused, as AIOMFAC refuses it.

**What is deliberately *not* imported: `BRR` and `CRR`.** AIOMFAC carries three interaction matrices,
but the two extra ones belong to the temperature form of Ganbavale et al. (2015),
`Psi = exp(-A/T + B(1/T0 - 1/T) + C((T0-T)/T + ln(T/T0)))`, which `ModSRunifac.f90` selects only for
particular fit datasets; every ordinary AIOMFAC calculation takes the default branch,
`Psi = exp(-A/T)`. pyoomph's `B` and `C` mean something else again (`exp(-(A/T + B + C*T))`), so
importing AIOMFAC's values into them would not be a different parameterisation but a wrong one.

**Regression:** with `A` alone the regenerated tables reproduce the previous activity coefficients to
1e-8 for glycerol–water and ethanol–water. The residual difference is that the old table had been
round-tripped through single precision (697.20001221 for 697.2).

## 3. The three contributions

Nomenclature as in Zuend et al. and in `ModMRpart.f90`; molalities `m_i` in mol per kg of solvent,
`I = ½Σm_i z_i²`, `Z = Σm_i|z_i|`.

* **SR**, the UNIFAC part, with the ions present as ordinary species carrying their `R` and `Q`.
  AIOMFAC has *no* short-range energetic parameters for ions (main group 51 is empty by design), so
  ions enter the short range only through volume, surface and the group fractions they displace.
* **MR**, `B_ki(I) = b + c·exp(-ω√I)` for a neutral main group against an ion (ω = 1.2 throughout),
  and `B_ca`, `C_nca` for a cation–anion pair with their own two ω. This is where the fitted
  electrolyte parameters are, and where most of the effect on a solvent comes from.
* **LR**, Pitzer–Debye–Hückel, evaluated with the properties of **pure water** (ρ = 997 kg/m³,
  ε = 78.54) whatever the actual solvent is. That is what AIOMFAC does; matching it is the point.

Neutrals get all three. The claim that they get only the short range — which a first reading of
`ModCalcActCoeff.f90` supports, since `gnmrln` and `gnlrln` are assigned in a different file — is
wrong, and it is the whole mechanism by which salt lowers water activity.

## 4. Written once, rendered three ways

The middle- and long-range maths is written once, in
`pyoomph/materials/activity_electrolyte.py`, against the expression-generator interface that the
short-range part already used. Three back-ends supply the values:

| generator | values are | used for |
|---|---|---|
| `UNIFACPyoomphExpressionGenerator` | GiNaC expressions | the symbolic path, exact derivatives |
| `FloatExpressionGenerator` | Python floats | `eval` of the multi-return expression |
| `CCodeExpressionGenerator` | `CExpr` strings | the generated C |

The short-range part had historically been written three times over — once symbolically, once as C
strings, once in numpy — and adding the electrolyte parts the same way would have been three chances
to get one of them subtly wrong. `subexpression` is what makes the C generator work: it appends a
`const double` and returns its name, so a shared subexpression is computed once in the emitted code
exactly as it is shared in the symbolic one.

The short-range code needed no changes at all. An ion is a molecule with one subgroup, and
`_FixedMoleFractionGenerator` answers `get_molefrac_var` from a dictionary of all-species mole
fractions, so the entire existing UNIFAC implementation serves the electrolyte case unchanged.

Two things that were bugs first: a negative literal must be parenthesised in C, or `a - -0.29` is
the decrement operator and does not compile; and the ionic strength is floored at 1e-30, because the
derivative of `B(I)` carries a `1/√I` that diverges at zero salt. Every *use* of that derivative is
finite — it always appears multiplied by `I` or by a molality — but the intermediate is not, and a
symbolic expression cannot branch the way AIOMFAC's `if (SI > tiny)` does.

## 5. The convention that had to be got right

pyoomph's mole fractions stay **salt-free** (`molefrac_water` means what it always meant), while
AIOMFAC's are over all species including ions. Since `set_vapor_pressure_by_raoults_law` computes
`gamma * molefrac * p_pure`, what pyoomph must return is

    gamma_pyoomph = gamma_AIOMFAC * x_all/x_saltfree = gamma_AIOMFAC / (1 + Σ_i ξ_i)

with `ξ_i = m_i·M̄` the moles of ion per mole of solvent. Getting this wrong is invisible in a
salt-free mixture and wrong by a few percent in a brine, which is why the water-activity comparison
against AIOMFAC is the test that pins it.

Ions are reported on the molality scale with infinite dilution in pure water as reference — the
convention AIOMFAC prints and the literature tabulates — which needs the unsymmetric short-range
reference `lnGaCinf` and the `Tmolal` conversion.

## 6. Agreement with AIOMFAC

AIOMFAC 3.13 built with gfortran and driven through its own input format. Water + NaCl at 298.15 K:

| m [mol/kg] | a(H₂O) pyoomph | a(H₂O) AIOMFAC | γ(Na⁺) pyoomph | γ(Na⁺) AIOMFAC |
|---|---|---|---|---|
| 0.1 | 0.996651 | 0.996649 | 0.776755 | 0.776755 |
| 1.0 | 0.966822 | 0.966825 | 0.660712 | 0.660712 |
| 5.0 | 0.807195 | 0.807195 | 0.921238 | 0.921236 |

and water + glycerol + NaCl, which is what exercises the main-group machinery, agrees to six digits
in both solvent activities and both ion coefficients — including a point where γ(Na⁺) = 6.15, far
from any regime where a mistake could hide. The reference numbers are frozen into the test file, so
the suite needs no Fortran. Independently: γ± of aqueous NaCl reproduces the literature minimum near
1 molal (0.778 / 0.657 / 0.714 at 0.1 / 1 / 3), and the dilute limit follows the Debye–Hückel
limiting law, which is the part of the model with no fitted parameters at all.

The three back-ends agree with each other to 3e-12 (compiled C against numpy, pure floating-point
reassociation) and 4e-16 (symbolic against numpy, i.e. the same expression tree evaluated twice).
An earlier draft of this document reported 1e-6 for the latter and blamed the molality round trip in
the test; that was wrong on both counts. The round trip is exact, and the 1e-6 came from a molar mass
typed into the test instead of read from the material -- a mole fraction wrong in its sixth digit
makes an activity coefficient wrong in its sixth digit, which looks exactly like two back-ends
disagreeing.

## 7. What is not covered

* The dissociation equilibria AIOMFAC can solve — bisulfate, ammonia/ammonium — and with them the
  `Qcca`/`Rcc` three-ion terms, which AIOMFAC uses only for `[NH4+|H+|HSO4-]` mixtures.
* The PEG special case in the middle-range parameters.
* AIOMFAC's viscosity and its water-diffusivity modules.
* Only 15 of the ion library's 28 ions have AIOMFAC parameters; the rest are refused by name rather
  than approximated. `AIOMFAC_SUBGROUP_OF_ION` in `pyoomph/materials/ions.py` is the mapping, which
  exists mostly because AIOMFAC writes a double charge as `++` where the library writes `2+`.
