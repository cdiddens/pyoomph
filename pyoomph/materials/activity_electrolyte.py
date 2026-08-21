from __future__ import annotations
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================
"""
The middle- and long-range parts of AIOMFAC, i.e. everything about an activity coefficient that only
exists because there are ions in the solution.

Plain UNIFAC has no such thing. AIOMFAC adds two contributions to the short-range (UNIFAC) one:

  * **LR**, a Pitzer-Debye-Hueckel term. It is what makes a dilute salt solution obey the limiting
    law, and it is evaluated with the properties of *pure water* -- not of the actual solvent --
    because that is what AIOMFAC does and matching it is the point.
  * **MR**, ion-organic and ion-ion interactions with an ionic-strength dependence,
    `B(I) = b + c*exp(-omega*sqrt(I))`. This is where the fitted electrolyte parameters live.

Both act on the solvents as well as on the ions, which is how a salt lowers water activity and
therefore the vapour pressure over a brine.

**Written once, evaluated three ways.** The same code produces the symbolic expression, the numpy
evaluation and the generated C, by taking an expression generator
(:py:class:`~pyoomph.materials.activity.UNIFACExpressionGeneratorBase`) and using nothing but
arithmetic, `exp`, `ln` and `subexpression` on the values it hands out. Writing this three times, as
the short-range part historically was, would be three chances to get one of them subtly wrong.

The formulas follow ModMRpart.f90 of https://github.com/andizuend/AIOMFAC (GPL v3); cite the AIOMFAC
publications when publishing results based on them.
"""

import math

from ..expressions.cb import CustomMultiReturnExpression
from ..expressions import ExpressionNumOrNone, var
from ..typings import *

if TYPE_CHECKING:
    from .activity import UNIFACExpressionGeneratorBase, UNIFACLikeActivityModel

#: Molar mass of water in kg/mol, the reference solvent of the molality scale.
M_WATER = 0.01801528
#: Density and static permittivity of pure water at 298.15 K. AIOMFAC evaluates the long-range term
#: with these regardless of what the solvent actually is ("all LR properties set to those of pure
#: water"), so anything else would disagree with it by construction.
RHO_WATER_298 = 997.0
EPS_WATER_298 = 78.54
#: R and Q of the water subgroup, and its UNIFAC ``l``. The infinite-dilution reference of an ion is
#: taken in pure water, so these are constants of that reference rather than of the mixture.
R_WATER, Q_WATER = 0.92, 1.40
L_WATER = 5.0 * (R_WATER - Q_WATER) - R_WATER + 1.0     # = -2.32


class ElectrolyteContributions:
    """
    The ion-dependent parts of AIOMFAC for one fixed set of species.

    Everything that depends only on *which* species are present is worked out once here; what
    depends on the composition is done in :py:meth:`evaluate`, against whichever expression
    generator it is given.

    Args:
        server: The activity model, for the subgroup tables.
        molecule_subgroups: ``{molecule: {subgroup: count}}`` for the neutral components.
        ion_subgroups: The names of the ion subgroups present, e.g. ``["Na+", "Cl-"]``.
    """

    def __init__(self, server: "UNIFACLikeActivityModel", molecule_subgroups: dict[str, dict[str, int]],
                 ion_subgroups: list[str]):
        from .UNIFAC import aiomfac_electrolyte as par
        self.server = server
        self.par = par
        # sorted throughout: these orders end up in generated sums, and a set's iteration order is
        # randomized per process by PYTHONHASHSEED.
        self.molecules = sorted(molecule_subgroups)
        self.molecule_subgroups = {m: dict(sorted(molecule_subgroups[m].items())) for m in self.molecules}
        self.ions = sorted(ion_subgroups, key=lambda n: server.subgroups[n].index or 0)
        for n in self.ions:
            if server.subgroups[n].index is None:
                raise RuntimeError("The ion subgroup '"+n+"' has no index, so it cannot be "
                                   "matched against AIOMFAC's electrolyte tables.")

        missing = [n for n in self.ions if (server.subgroups[n].index or 0) not in par.ION_CHARGE]
        if missing:
            raise RuntimeError("AIOMFAC has no electrolyte parameters for " + ", ".join(missing) +
                               ". The ions it can do are: " + ", ".join(sorted(par.ION_NAME.values())))
        self.ion_index = {n: cast(int, server.subgroups[n].index) for n in self.ions}
        self.charge = {n: par.ION_CHARGE[self.ion_index[n]] for n in self.ions}
        self.cations = [n for n in self.ions if self.charge[n] > 0]
        self.anions = [n for n in self.ions if self.charge[n] < 0]

        # Which neutral subgroups are present, and which main group each belongs to.
        self.subgroups = sorted({sg for m in self.molecules for sg in self.molecule_subgroups[m]})
        self.main_of = {sg: server.subgroups[sg].maingroup.index for sg in self.subgroups}
        self.main_groups = sorted({mg for mg in (self.main_of[sg] for sg in self.subgroups)
                                   if mg is not None})
        self.subgroup_mass = {sg: self._subgroup_mass(sg) for sg in self.subgroups}
        #: Molar mass of each neutral molecule in kg/mol, which is what the long-range term weights
        #: its contribution by.
        self.molecule_mass = {m: sum(cnt * self.subgroup_mass[sg]
                                     for sg, cnt in self.molecule_subgroups[m].items())
                              for m in self.molecules}
        #: How many subgroups of each main group a molecule carries.
        self.main_group_count = {m: {mg: sum(cnt for sg, cnt in self.molecule_subgroups[m].items()
                                             if self.main_of[sg] == mg)
                                     for mg in self.main_groups} for m in self.molecules}

    def _subgroup_mass(self, sg: str) -> float:
        mass = self.server.subgroups[sg].molar_mass
        if mass is None:
            raise RuntimeError("The subgroup '" + sg + "' has no molar mass, which the middle-range "
                               "part of AIOMFAC needs: it weights each main group's contribution by "
                               "its mass.")
        return mass

    # ---- parameter lookups -------------------------------------------------------------------

    def _b_neutral_ion(self, kind: str, main_group: int, ion: str) -> float:
        table = self.par.MR_B_ION if kind == "b" else self.par.MR_C_ION
        row = table.get(self.ion_index[ion], {})
        if main_group not in row:
            raise RuntimeError("AIOMFAC has no middle-range parameter between main group " +
                               str(main_group) + " and " + ion + ", so it cannot describe this "
                               "mixture. It stops on exactly these rather than assuming zero.")
        return row[main_group]

    def _ca(self, key: str, cation: str, anion: str) -> float:
        entry = self.par.MR_CATION_ANION.get((self.ion_index[cation], self.ion_index[anion]))
        if entry is None:
            raise RuntimeError("AIOMFAC has no middle-range parameters for the pair " + cation +
                               " / " + anion + ".")
        return entry.get(key, self.par.MR_CATION_ANION_DEFAULTS[key])

    # ---- the calculation ---------------------------------------------------------------------

    def evaluate(self, gen: "UNIFACExpressionGeneratorBase", x_saltfree: dict[str, Any],
                 molalities: dict[str, Any], T: Any) -> dict[str, Any]:
        r"""
        The middle- and long-range contributions at one composition.

        Args:
            gen: The expression generator; everything below is built with its ``ln``, ``exp`` and
                ``subexpression``, so the same code serves the symbolic, numpy and C back-ends.
            x_saltfree: Mole fraction of each neutral molecule on the **salt-free** basis, i.e.
                summing to one over the solvents alone.
            molalities: Molality of each ion in mol per kg of solvent.
            T: Temperature in Kelvin.

        Returns:
            ``ln_gamma_MR`` and ``ln_gamma_LR`` per molecule and per ion, plus the ionic strength
            and ``Tmolal``, the term that converts an ion from the mole-fraction to the molality
            scale.
        """
        sub = gen.subexpression
        ln, exp = gen.ln, gen.exp                                    # type: ignore[attr-defined]

        # -- composition of the solvent, on the salt-free basis ---------------------------------
        # The subgroup mole fractions the middle-range part works with are salt-free: the ions
        # enter through their molalities, not through this.
        sg_counts = {sg: sum(self.molecule_subgroups[m].get(sg, 0) * x_saltfree[m]
                             for m in self.molecules) for sg in self.subgroups}
        sg_total = sub(sum(sg_counts.values()))
        x_sg = {sg: sg_counts[sg] / sg_total for sg in self.subgroups}
        x_mg = {mg: sub(sum(x_sg[sg] for sg in self.subgroups if self.main_of[sg] == mg))
                for mg in self.main_groups}
        # Mass of a main group: the mass-weighted mean over the subgroups that make it up, so it
        # depends on the composition whenever a main group has more than one subgroup in play.
        mass_mg = {mg: sub(sum(self.subgroup_mass[sg] * x_sg[sg]
                               for sg in self.subgroups if self.main_of[sg] == mg) / x_mg[mg])
                   for mg in self.main_groups}
        mean_mg_mass = sub(sum(mass_mg[mg] * x_mg[mg] for mg in self.main_groups))
        mean_solvent_mass = sub(sum(self.molecule_mass[m] * x_saltfree[m] for m in self.molecules))

        # -- ionic strength and charge sum ------------------------------------------------------
        # Floored, because the derivative of the middle-range B(I) carries a 1/sqrt(I) that diverges
        # at zero salt. Every *use* of it is finite -- it always appears multiplied by I or by a
        # molality, both of which vanish at least as fast -- but the intermediate does not, and a
        # symbolic expression cannot branch the way AIOMFAC's "if (SI > tiny)" does. At 1e-30 the
        # floor is far below any concentration that means anything and changes nothing that does.
        I = sub(0.5 * sum(molalities[i] * self.charge[i] ** 2 for i in self.ions) + 1.0e-30)
        sqrtI = sub(gen.pow(I, 0.5))
        Z = sub(sum(molalities[i] * abs(self.charge[i]) for i in self.ions))
        total_molality = sub(sum(molalities[i] for i in self.ions))

        # -- long range: Pitzer-Debye-Hueckel, with the properties of pure water ------------------
        A = sub(1.327757e5 * RHO_WATER_298 ** 0.5 / gen.pow(EPS_WATER_298 * T, 1.5))
        b = sub(6.359696 * RHO_WATER_298 ** 0.5 / gen.pow(EPS_WATER_298 * T, 0.5))
        bb = sub(1.0 + b * sqrtI)
        lr_neutral_common = sub(2.0 * A / (b * b * b) * (bb - 1.0 / bb - 2.0 * ln(bb)))
        ln_lr = {m: self.molecule_mass[m] * lr_neutral_common for m in self.molecules}
        lr_ion_common = sub(A * sqrtI / bb)
        for i in self.ions:
            ln_lr[i] = -self.charge[i] ** 2 * lr_ion_common

        # -- middle range: the ionic-strength dependent interaction coefficients -----------------
        # B(I) = b + c*exp(-omega*sqrt(I)), and B'(I) = dB/dI = -omega/(2 sqrt(I)) * (B - b).
        dfac = sub(-0.5 / sqrtI)
        B_ki: dict[tuple[int, str], Any] = {}
        Bs_ki: dict[tuple[int, str], Any] = {}
        omega_ni = self.par.OMEGA_NEUTRAL_ION
        for mg in self.main_groups:
            decay = sub(exp(-omega_ni * sqrtI))
            for i in self.ions:
                b0, c0 = self._b_neutral_ion("b", mg, i), self._b_neutral_ion("c", mg, i)
                B_ki[(mg, i)] = sub(b0 + c0 * decay)
                Bs_ki[(mg, i)] = dfac * omega_ni * (B_ki[(mg, i)] - b0)

        B_ca: dict[tuple[str, str], Any] = {}
        Bs_ca: dict[tuple[str, str], Any] = {}
        C_ca: dict[tuple[str, str], Any] = {}
        Cs_ca: dict[tuple[str, str], Any] = {}
        for c in self.cations:
            for a in self.anions:
                b0, c0 = self._ca("b", c, a), self._ca("c", c, a)
                cn1, cn2 = self._ca("cn1", c, a), self._ca("cn2", c, a)
                om, om2 = self._ca("omega", c, a), self._ca("omega2", c, a)
                B_ca[(c, a)] = sub(b0 + c0 * exp(-om * sqrtI))
                Bs_ca[(c, a)] = dfac * om * (B_ca[(c, a)] - b0)
                C_ca[(c, a)] = sub(cn1 + cn2 * exp(-om2 * sqrtI))
                Cs_ca[(c, a)] = dfac * om2 * (C_ca[(c, a)] - cn1)

        # -- middle range for the neutrals -------------------------------------------------------
        sum_bm = {mg: sub(sum(B_ki[(mg, i)] * molalities[i] for i in self.ions))
                  for mg in self.main_groups}
        sum_kion1 = sub(sum((B_ki[(mg, i)] + I * Bs_ki[(mg, i)]) * x_mg[mg] * molalities[i]
                            for mg in self.main_groups for i in self.ions))
        sum_ca1 = sub(sum((B_ca[(c, a)] + I * Bs_ca[(c, a)] + 2.0 * Z * C_ca[(c, a)]
                           + Z * I * Cs_ca[(c, a)]) * molalities[c] * molalities[a]
                          for c in self.cations for a in self.anions))
        g_mg = {mg: sub(sum_bm[mg] - mass_mg[mg] / mean_mg_mass * sum_kion1 - mass_mg[mg] * sum_ca1)
                for mg in self.main_groups}
        ln_mr = {m: sum(self.main_group_count[m][mg] * g_mg[mg] for mg in self.main_groups)
                 for m in self.molecules}

        # -- middle range for the ions -----------------------------------------------------------
        sum_kion2 = sub(sum(Bs_ki[(mg, i)] * x_mg[mg] * molalities[i]
                            for mg in self.main_groups for i in self.ions))
        sum_bs_ca = sub(sum(Bs_ca[(c, a)] * molalities[c] * molalities[a]
                            for c in self.cations for a in self.anions))
        sum_cnca = sub(sum(C_ca[(c, a)] * molalities[c] * molalities[a]
                           for c in self.cations for a in self.anions))
        sum_csca = sub(sum(Cs_ca[(c, a)] * molalities[c] * molalities[a]
                           for c in self.cations for a in self.anions))
        for i in self.ions:
            z, z2 = abs(self.charge[i]), self.charge[i] ** 2
            sum_bx = sum(B_ki[(mg, i)] * x_mg[mg] for mg in self.main_groups)
            if self.charge[i] > 0:
                sum_pair = sum((B_ca[(i, a)] + Z * C_ca[(i, a)]) * molalities[a] for a in self.anions)
            else:
                sum_pair = sum((B_ca[(c, i)] + Z * C_ca[(c, i)]) * molalities[c] for c in self.cations)
            ln_mr[i] = (sum_bx / mean_mg_mass + z2 / (2.0 * mean_mg_mass) * sum_kion2 + sum_pair
                        + 0.5 * z2 * sum_bs_ca + z * sum_cnca + 0.5 * z2 * Z * sum_csca)

        # -- the molality-scale conversion for the ions ------------------------------------------
        # An ion's activity coefficient is reported on the molality scale with infinite dilution in
        # pure water as its reference, while the short-range part produces a mole-fraction one.
        tmolal = sub(ln(M_WATER / mean_solvent_mass + M_WATER * total_molality))

        return {"ln_gamma_MR": ln_mr, "ln_gamma_LR": ln_lr, "ionic_strength": I, "Tmolal": tmolal,
                "mean_solvent_molar_mass": mean_solvent_mass}

    # ---- the short-range reference of an ion ---------------------------------------------------

    def combinatorial_reference(self, ion: str) -> float:
        r"""
        :math:`\ln\gamma_i^{\mathrm{C},\infty}`, the combinatorial part of an ion's short-range
        activity coefficient at infinite dilution in **pure water**.

        An ion has no pure liquid state to reference, so the unsymmetric convention is used and this
        is what gets subtracted. A constant, since the reference is pure water rather than the
        mixture at hand.
        """
        sg = self.server.subgroups[ion]
        r, q = sg.R, sg.Q
        l = 5.0 * (r - q) - r + 1.0
        import math
        B = 5.0 * q * math.log(q / Q_WATER * R_WATER / r) + l - r / R_WATER * L_WATER
        return math.log(r / R_WATER) + B


class FloatExpressionGenerator:
    """An expression generator whose values are plain numbers.

    What the multi-return expression evaluates with, and what a test compares the symbolic path
    against. ``subexpression`` is the identity here: there is nothing to name when the value is
    already a number.
    """
    def __init__(self, temperature_in_K: float = 298.15):
        self._T = temperature_in_K

    def ln(self, x: Any) -> Any:
        return math.log(x)

    def exp(self, x: Any) -> Any:
        return math.exp(x)

    def pow(self, a: Any, b: Any) -> Any:
        return a ** b

    def subexpression(self, expr: Any) -> Any:
        return expr

    def get_temperature_in_kelvin(self) -> Any:
        return self._T

    def get_molefrac_var(self, name: str) -> Any:
        raise RuntimeError("mole fractions are supplied explicitly to this generator")


def _c(value: Any) -> str:
    """One value as C source. Floats go through repr so that the C constant is the same double."""
    if isinstance(value, CExpr):
        return value.code
    if isinstance(value, bool):
        raise TypeError("a bool has no place in generated arithmetic")
    v = float(value)
    # A negative literal has to be parenthesised: "a - -0.29" is the decrement operator in C, and
    # the compiler rejects it rather than computing what it looks like.
    return "(" + repr(v) + ")" if v < 0.0 else repr(v)


class CExpr:
    """A piece of C arithmetic, built up by the ordinary Python operators.

    Parenthesised at every step rather than tracking precedence: the C compiler folds it away, and
    a missing parenthesis in generated code is a bug that only shows up as a wrong number.
    """
    __slots__ = ("code",)

    def __init__(self, code: str):
        self.code = code

    def _bin(self, other: Any, op: str, flip: bool = False) -> "CExpr":
        a, b = (_c(other), self.code) if flip else (self.code, _c(other))
        return CExpr("(" + a + op + b + ")")

    def __add__(self, o: Any) -> "CExpr": return self._bin(o, "+")
    def __radd__(self, o: Any) -> "CExpr": return self._bin(o, "+", True)
    def __sub__(self, o: Any) -> "CExpr": return self._bin(o, "-")
    def __rsub__(self, o: Any) -> "CExpr": return self._bin(o, "-", True)
    def __mul__(self, o: Any) -> "CExpr": return self._bin(o, "*")
    def __rmul__(self, o: Any) -> "CExpr": return self._bin(o, "*", True)
    def __truediv__(self, o: Any) -> "CExpr": return self._bin(o, "/")
    def __rtruediv__(self, o: Any) -> "CExpr": return self._bin(o, "/", True)
    def __neg__(self) -> "CExpr": return CExpr("(-" + self.code + ")")
    def __pow__(self, o: Any) -> "CExpr": return CExpr("pow(" + self.code + "," + _c(o) + ")")
    def __rpow__(self, o: Any) -> "CExpr": return CExpr("pow(" + _c(o) + "," + self.code + ")")
    def __repr__(self) -> str: return "CExpr(" + self.code + ")"


class CCodeExpressionGenerator:
    """An expression generator that emits C.

    ``subexpression`` is what makes this worth doing: it appends a ``const double`` to
    :py:attr:`lines` and returns its name, so a shared subexpression is computed once in the
    generated code exactly as it is shared in the symbolic one. Without it the same expression tree
    would be pasted out in full at every use and the emitted function would explode.
    """
    def __init__(self, temperature: str = "T_in_K", prefix: str = "aiom"):
        self.lines: list[str] = []
        self.prefix = prefix
        self._count = 0
        self._T = CExpr(temperature)

    def ln(self, x: Any) -> CExpr:
        return CExpr("log(" + _c(x) + ")")

    def exp(self, x: Any) -> CExpr:
        return CExpr("exp(" + _c(x) + ")")

    def pow(self, a: Any, b: Any) -> CExpr:
        if b == 0.5:
            return CExpr("sqrt(" + _c(a) + ")")
        return CExpr("pow(" + _c(a) + "," + _c(b) + ")")

    def subexpression(self, expr: Any) -> CExpr:
        name = self.prefix + "_t" + str(self._count)
        self._count += 1
        self.lines.append("const double " + name + " = " + _c(expr) + ";")
        return CExpr(name)

    def get_temperature_in_kelvin(self) -> CExpr:
        return self._T

    def get_molefrac_var(self, name: str) -> Any:
        raise RuntimeError("mole fractions are supplied explicitly to this generator")


class _FixedMoleFractionGenerator:
    """An expression generator that answers ``get_molefrac_var`` from a dictionary.

    The short-range code asks the generator for each species' mole fraction by name. Handing it one
    of these, filled with the *all-species* mole fractions, is what lets the entire existing UNIFAC
    implementation serve the electrolyte case unchanged -- an ion is a molecule with one subgroup,
    and nothing else about the short-range part has to know that ions exist.
    """
    def __init__(self, base: "UNIFACExpressionGeneratorBase", values: dict[str, Any]):
        self._base = base
        self._values = values

    def get_molefrac_var(self, name: str) -> Any:
        return self._values[name]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)


class AIOMFACElectrolyteMixture:
    """
    An AIOMFAC mixture of solvents and dissolved ions: short range, middle range and long range.

    The short-range part is the ordinary UNIFAC one with the ions included as species, so it is the
    existing :py:class:`~pyoomph.materials.activity.UNIFACMixture` doing the work; what this adds is
    the composition conversion and the two electrolyte contributions.

    Args:
        server: The activity model. Must be AIOMFAC: no other model in the library has electrolyte
            parameters at all.
        molecule_subgroups: ``{molecule: {subgroup: count}}`` for the neutral components.
        ion_subgroups: The ion subgroup names present, e.g. ``["Na+", "Cl-"]``.
    """

    def __init__(self, server: "UNIFACLikeActivityModel", molecule_subgroups: dict[str, dict[str, int]],
                 ion_subgroups: list[str]):
        from .activity import UNIFACMixture, UNIFACMolecule
        self.server = server
        self.contributions = ElectrolyteContributions(server, molecule_subgroups, ion_subgroups)
        self.molecules = self.contributions.molecules
        self.ions = self.contributions.ions
        species: list[UNIFACMolecule] = []
        for name in self.molecules:
            mol = UNIFACMolecule(name, server)
            for sg, cnt in self.contributions.molecule_subgroups[name].items():
                mol.add_subgroup(sg, cnt)
            species.append(mol)
        for name in self.ions:
            ion = UNIFACMolecule(name, server)
            ion.add_subgroup(name, 1)
            species.append(ion)
        self.sr_mixture = UNIFACMixture(*species)
        self._ion_reference = {n: self.contributions.combinatorial_reference(n) for n in self.ions}

    def mole_fractions_from_molalities(self, gen: "UNIFACExpressionGeneratorBase",
                                       x_saltfree: dict[str, Any],
                                       molalities: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        r"""
        The all-species mole fractions, and the factor :math:`1+\sum_i \xi_i` that relates them to
        the salt-free ones.

        A molality is moles of ion per kg of solvent, so :math:`\xi_i=m_i\bar{M}` is moles of ion per
        mole of solvent, and every solvent mole fraction is divided by one plus their sum.
        """
        mean_mass = gen.subexpression(sum(self.contributions.molecule_mass[m] * x_saltfree[m]
                                          for m in self.molecules))
        xi = {i: molalities[i] * mean_mass for i in self.ions}
        denom = gen.subexpression(1 + sum(xi.values()))
        x_all: dict[str, Any] = {m: x_saltfree[m] / denom for m in self.molecules}
        for i in self.ions:
            x_all[i] = xi[i] / denom
        return x_all, denom

    def activity_coefficients(self, gen: "UNIFACExpressionGeneratorBase", x_saltfree: dict[str, Any],
                              molalities: dict[str, Any], T: Any) -> dict[str, Any]:
        r"""
        Every species' activity coefficient at one composition.

        For a solvent this is a mole-fraction based coefficient with the pure component as reference,
        **defined against the salt-free mole fraction**: pyoomph's Raoult law multiplies it by
        ``molefrac_<component>``, which stays salt-free when a salt is dissolved, whereas AIOMFAC's
        own coefficient goes with the all-species mole fraction. The two differ by exactly the factor
        :math:`1+\sum_i \xi_i`, and dividing by it here is what keeps the vapour pressure right.

        For an ion it is the molality-based coefficient with infinite dilution in pure water as
        reference, which is what AIOMFAC reports and what the literature tabulates.
        """
        sub = gen.subexpression
        exp = gen.exp                                                # type: ignore[attr-defined]
        x_all, denom = self.mole_fractions_from_molalities(gen, x_saltfree, molalities)
        sr_gen = cast("UNIFACExpressionGeneratorBase", _FixedMoleFractionGenerator(gen, x_all))
        self.sr_mixture.set_expression_generator(sr_gen)
        extra = self.contributions.evaluate(gen, x_saltfree, molalities, T)
        ln_mr, ln_lr = extra["ln_gamma_MR"], extra["ln_gamma_LR"]

        res: dict[str, Any] = {}
        for name in self.molecules:
            compo = self.sr_mixture._get_component_by_name(name)
            ln_sr = sub(self.sr_mixture.get_ln_combinatorial_gamma(compo)
                        + self.sr_mixture.get_ln_residual_gamma(compo))
            res[name] = sub(exp(ln_sr + ln_mr[name] + ln_lr[name]) / denom)
        for name in self.ions:
            compo = self.sr_mixture._get_component_by_name(name)
            # The unsymmetric convention: an ion has no pure liquid state, so its short-range part is
            # referenced to infinite dilution in pure water, and Tmolal moves it from the mole
            # fraction scale to the molality scale.
            ln_sr = sub(self.sr_mixture.get_ln_combinatorial_gamma(compo)
                        + self.sr_mixture.get_ln_residual_gamma(compo)
                        - self._ion_reference[name])
            res[name] = sub(exp(ln_sr + ln_mr[name] + ln_lr[name] - extra["Tmolal"]))
        return res

    def mean_ionic_activity_coefficient(self, coefficients: dict[str, Any], cation: str, anion: str,
                                        nu_cation: int, nu_anion: int) -> Any:
        r""":math:`\gamma_\pm=(\gamma_+^{\nu_+}\gamma_-^{\nu_-})^{1/(\nu_++\nu_-)}`, the combination
        that is actually measurable -- a single ion's activity coefficient is not."""
        total = nu_cation + nu_anion
        return (coefficients[cation] ** nu_cation * coefficients[anion] ** nu_anion) ** (1.0 / total)


class AIOMFACElectrolyteMultiReturnExpression(CustomMultiReturnExpression):
    """
    The salted AIOMFAC as a multi-return expression: one call returns every species' activity
    coefficient, in generated C with a finite-difference Jacobian.

    The alternative is the symbolic route, which builds one GiNaC expression per species and
    differentiates them exactly. That is what a two-component mixture uses and what bifurcation
    tracking needs; for a mixture of several solvents and a salt the expressions get large, and this
    computes all of them together instead.

    Both the C and the numpy evaluation come from
    :py:meth:`AIOMFACElectrolyteMixture.activity_coefficients` -- the same code, given a different
    expression generator -- so the two cannot drift apart, which is the failure mode that matters
    when one of them is only exercised in generated code.

    Args:
        mixture: The species and their parameters.
        constant_temperature: Fix the temperature instead of taking it as the last argument, which
            removes it from the argument list and lets the temperature-dependent parts fold away.
    """

    def __init__(self, mixture: "AIOMFACElectrolyteMixture",
                 constant_temperature: "ExpressionNumOrNone" = None):
        super().__init__()
        from ..expressions.units import kelvin
        self.mixture = mixture
        self.constant_temperature = constant_temperature
        self._T_const = float(constant_temperature / kelvin) if constant_temperature is not None else None
        #: The order of the arguments: every solvent's mole fraction, then every ion's molality,
        #: then the temperature unless it is fixed.
        self.argument_order = list(mixture.molecules) + list(mixture.ions)
        #: The order of the results, which is the same species order.
        self.result_order = list(mixture.molecules) + list(mixture.ions)
        self.FD_epsilon = 1e-9

    def get_num_returned_scalars(self, nargs: int) -> int:
        expected = len(self.argument_order) + (0 if self._T_const is not None else 1)
        if nargs != expected:
            raise RuntimeError("AIOMFAC with ions must be called with " + str(expected) +
                               " arguments: the mole fractions " + str(self.mixture.molecules) +
                               ", the molalities " + str(self.mixture.ions) +
                               (" and the temperature" if self._T_const is None else "") +
                               ". Got " + str(nargs) + ".")
        return len(self.result_order)

    def _split(self, values: "list[Any]") -> "tuple[dict[str,Any],dict[str,Any],Any]":
        n = len(self.mixture.molecules)
        x = {m: values[i] for i, m in enumerate(self.mixture.molecules)}
        molal = {ion: values[n + i] for i, ion in enumerate(self.mixture.ions)}
        T = self._T_const if self._T_const is not None else values[len(self.argument_order)]
        return x, molal, T

    def generate_c_code(self) -> str:
        gen = CCodeExpressionGenerator(temperature="T_in_K")
        args = [CExpr("arg_list[" + str(i) + "]") for i in range(len(self.argument_order))]
        if self._T_const is None:
            args.append(CExpr("arg_list[" + str(len(self.argument_order)) + "]"))
        x, molal, T = self._split(args)
        res = self.mixture.activity_coefficients(cast(Any, gen), x, molal, T)
        lines = ["const double T_in_K = " + (repr(self._T_const) if self._T_const is not None
                                             else "arg_list[" + str(len(self.argument_order)) + "]") + ";"]
        lines += gen.lines
        for i, name in enumerate(self.result_order):
            lines.append("result_list[" + str(i) + "] = " + _c(res[name]) + ";")
        lines.append("FILL_MULTI_RET_JACOBIAN_BY_FD(" + repr(self.FD_epsilon) + ")")
        return "\n            ".join(lines)

    def eval(self, flag: int, arg_list: Any, result_list: Any, derivative_matrix: Any) -> None:
        x, molal, T = self._split([float(v) for v in arg_list])
        gen = FloatExpressionGenerator(T)
        res = self.mixture.activity_coefficients(cast(Any, gen), x, molal, T)
        for i, name in enumerate(self.result_order):
            result_list[i] = res[name]
        if flag:
            # Finite differences, as the short-range multi-return does: the expressions here are
            # long and their exact derivatives would be longer still, while the Newton solver only
            # needs the Jacobian well enough to converge.
            nargs = len(arg_list)
            base = [result_list[i] for i in range(len(self.result_order))]
            for j in range(nargs):
                shifted = [float(v) for v in arg_list]
                h = self.FD_epsilon * max(1.0, abs(shifted[j]))
                shifted[j] += h
                xs, ms, Ts = self._split(shifted)
                rs = self.mixture.activity_coefficients(cast(Any, FloatExpressionGenerator(Ts)), xs, ms, Ts)
                for i, name in enumerate(self.result_order):
                    derivative_matrix[i * nargs + j] = (rs[name] - base[i]) / h

    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> "list[ExpressionOrNum]":
        from ..expressions.units import kelvin
        res = [a for a in args]
        if self._T_const is None:
            res[-1] = res[-1] / kelvin
        return res

    def get_activity_coefficients(self, mole_fractions: "dict[str,Any]", molalities: "dict[str,Any]",
                                  temperature: "Any" = None) -> "dict[str,Any]":
        """One call for every species, as expressions.

        ``mole_fractions`` are the solvents' salt-free mole fractions and ``molalities`` the ions'
        molalities in mol/kg, both as dimensionless expressions and keyed the way the mixture names
        them."""
        call = [mole_fractions[m] for m in self.mixture.molecules]
        call += [molalities[i] for i in self.mixture.ions]
        if self._T_const is None:
            call.append(temperature if temperature is not None else var("temperature"))
        out = self.__call__(*call)
        return {name: out[i] for i, name in enumerate(self.result_order)}


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
