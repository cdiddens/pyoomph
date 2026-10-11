from __future__ import annotations
#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
r"""
Estimating the Fickian diffusivity matrix of a liquid mixture.

Measured diffusivities of liquid mixtures are rare, but where an activity model (UNIFAC, AIOMFAC) is
set, the *thermodynamic* part of the diffusivity is already known -- only the Maxwell-Stefan part has
to be correlated. That is what this module does, and what
:py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_estimate_diffusivities` exposes.

With :math:`n` components, the last of which is the passive one, and the independent set
:math:`i=1\dots n-1`, the matrix written into the mixture is

.. math:: [D] = \frac{1}{\bar{M}}\,[C]\,[B]^{-1}\,[\Gamma]\,[A]

so that :math:`j_i=-\rho\sum_j D_{ij}\nabla w_j` in the mass-average reference frame, which is the
frame :py:meth:`~pyoomph.materials.generic.BaseMixedProperties.get_diffusive_mass_flux_for` assembles
in. The four factors are

.. math::
    \bar{M}&=\left(\sum_k w_k/M_k\right)^{-1} \\
    B_{ii}&=\frac{x_i}{\mathcal{D}_{in}}+\sum_{k\neq i}\frac{x_k}{\mathcal{D}_{ik}},\qquad
    B_{ij}=-x_i\left(\frac{1}{\mathcal{D}_{ij}}-\frac{1}{\mathcal{D}_{in}}\right) \\
    \Gamma_{ij}&=\delta_{ij}+x_i\frac{\partial\ln\gamma_i}{\partial x_j}
      \quad\text{with } x_n=1-\textstyle\sum_{k<n}x_k \\
    A_{ij}&=\frac{\partial x_i}{\partial w_j}
      =\bar{M}\left(\frac{\delta_{ij}}{M_i}-x_i\left(\frac{1}{M_j}-\frac{1}{M_n}\right)\right),\qquad
    C_{ij}=M_i\delta_{ij}-w_i(M_j-M_n)

:math:`[C]` converts a molar flux in the molar-average frame into a mass flux in the mass-average
frame, and :math:`[A]` converts mass-fraction gradients into mole-fraction gradients. Both follow
from the definitions alone, so the only modelling is in :math:`[\mathcal{D}]` and :math:`[\Gamma]`.

Two exact identities follow and are used as tests: a binary collapses to
:math:`D=\mathcal{D}_{12}\Gamma`, and :math:`\bar{M}^{-1}[C][A]=[I]`, so equal Maxwell-Stefan diffusivities
with an ideal mixture give an isotropic matrix.

References:

    * Wilke, C. R. and Chang, P. 1955. AIChE J., 1: 264
    * Vignes, A. 1966. Ind. Eng. Chem. Fundam., 5: 189
    * Darken, L. S. 1948. Trans. AIME, 175: 184
    * Taylor, R. and Krishna, R. 1993. *Multicomponent Mass Transfer*, chapters 2-4
    * Wesselingh, J. A. and Krishna, R. 2000. *Mass Transfer in Multicomponent Mixtures*
"""

import warnings

from .. import _pyoomph_core as _pyoomph
from ..expressions import var, log, exp, subexpression, square_root, rational_num, maximum, is_zero
from ..expressions import ExpressionOrNum, ExpressionNumOrNone, Expression
from ..expressions.units import *
from ..expressions.phys_consts import k_Boltzmann, N_Avogadro
from ..typings import *

if TYPE_CHECKING:
    from .generic import MixtureLiquidProperties, PureLiquidProperties


#: Association factors of the Wilke-Chang correlation, looked up by component name whenever
#: :py:attr:`~pyoomph.materials.generic.PureLiquidProperties.association_factor_for_Wilke_Chang_eq`
#: is not set. Anything not listed here is unassociated, i.e. 1.0.
WILKE_CHANG_ASSOCIATION_FACTORS: dict[str, float] = {"water": 2.6, "methanol": 1.9, "ethanol": 1.5}

#: The unit every Maxwell-Stefan diffusivity is normalized by before a *variable* exponent is applied
#: to it. GiNaC cannot take the unit out of ``D**x`` with a field ``x``, so Vignes is built from
#: dimensionless factors and the unit is multiplied back afterwards.
_D_UNIT = meter**2 / second


####################################################################################################
# The correlations
####################################################################################################

def get_association_factor(pure: "PureLiquidProperties") -> float:
    """The Wilke-Chang association factor of a pure liquid acting as the solvent."""
    phi = pure.association_factor_for_Wilke_Chang_eq
    if phi is not None:
        return phi
    return WILKE_CHANG_ASSOCIATION_FACTORS.get(pure.name, 1.0)


def get_molar_volume_at_boiling_point(pure: "PureLiquidProperties") -> tuple[ExpressionOrNum, bool]:
    """
    The molar volume at the normal boiling point, and whether it had to be estimated.

    The Wilke-Chang correlation wants the volume at the normal boiling point, usually from the
    additive Le Bas increments. When
    :py:attr:`~pyoomph.materials.generic.PureLiquidProperties.molar_volume_for_Wilke_Chang_eq` is not
    set, the liquid volume ``molar_mass/mass_density`` is used instead. That is the volume at room
    temperature and hence too small -- by 5% for water, 1% for ethanol, but 32% for glycerol, i.e.
    the more hydrogen bonds the worse. With the exponent 3/5 this turns into a 1-17% *over*\\ estimate
    of the diffusivity, which is why the caller warns about it.
    """
    V = pure.molar_volume_for_Wilke_Chang_eq
    if V is not None:
        return V, False
    if not hasattr(pure, "mass_density"):
        raise RuntimeError("Cannot estimate the molar volume of '" + pure.name + "': it has neither "
                           "molar_volume_for_Wilke_Chang_eq nor a mass_density. Set either one.")
    return pure.molar_mass / pure.mass_density, True


def wilke_chang_diffusivity(solute: "PureLiquidProperties", solvent: "PureLiquidProperties",
                            viscosity: ExpressionOrNum,
                            temperature: ExpressionOrNum = var("temperature"),
                            molar_volume: ExpressionNumOrNone = None) -> ExpressionOrNum:
    r"""
    The Wilke-Chang estimate of the diffusivity of a solute at infinite dilution in a solvent,

    .. math:: \mathcal{D}^0_{AB}=7.4\cdot10^{-8}\frac{\sqrt{\phi_B M_B}\,T}{\mu\,V_A^{0.6}}

    in :math:`\mathrm{cm^2/s}` with :math:`M_B` in g/mol, :math:`T` in K, :math:`\mu` in cP and
    :math:`V_A` in :math:`\mathrm{cm^3/mol}`. Accurate to roughly 10-20% for the systems it was
    fitted on.

    Args:
        solute: The component that is infinitely dilute; only its molar volume enters.
        solvent: The component it is dilute in; its molar mass and association factor enter.
        viscosity: The viscosity to use. Passing the *mixture* viscosity is what makes the estimate
            composition dependent; passing the pure solvent viscosity gives the literal correlation.
        temperature: Temperature, by default the temperature field.
        molar_volume: Overrides the solute's molar volume at the normal boiling point.
    """
    if molar_volume is None:
        molar_volume, _ = get_molar_volume_at_boiling_point(solute)
    phi = get_association_factor(solvent)
    TK = temperature / kelvin
    muCP = viscosity / (milli * pascal * second)
    MB = solvent.molar_mass / (gram / mol)
    VA = molar_volume / ((centi * meter) ** 3 / mol)
    # 0.6 is exactly 3/5: a rational exponent, not a float one, so the units stay intact
    D = 7.4e-8 * square_root(phi * MB) * TK / (muCP * VA ** rational_num(3, 5))
    return D * (centi * meter) ** 2 / second


def hydrodynamic_radius(pure: "PureLiquidProperties",
                        molar_volume: ExpressionNumOrNone = None) -> ExpressionOrNum:
    r"""
    The radius of the sphere of the same volume as one molecule,
    :math:`r=(3V/(4\pi N_\mathrm{A}))^{1/3}`, for the Stokes-Einstein estimate.
    """
    if molar_volume is None:
        molar_volume, _ = get_molar_volume_at_boiling_point(pure)
    Vmolecule = molar_volume / N_Avogadro / meter**3
    return (3 * Vmolecule / (4 * pi)) ** rational_num(1, 3) * meter


def stokes_einstein_diffusivity(radius: ExpressionOrNum, viscosity: ExpressionOrNum,
                                temperature: ExpressionOrNum = var("temperature")) -> ExpressionOrNum:
    r"""The Stokes-Einstein diffusivity :math:`k_\mathrm{B}T/(6\pi\mu r)`."""
    return k_Boltzmann * temperature / (6 * pi * viscosity * radius)


def vignes_ms_diffusivity(D0_ij: ExpressionOrNum, D0_ji: ExpressionOrNum,
                          x_i: ExpressionOrNum, x_j: ExpressionOrNum) -> ExpressionOrNum:
    r"""
    The Wesselingh-Krishna generalization of the Vignes interpolation,

    .. math:: \mathcal{D}_{ij}=\left(\mathcal{D}^0_{ij}\right)^{(1+x_j-x_i)/2}\left(\mathcal{D}^0_{ji}\right)^{(1+x_i-x_j)/2}

    which is symmetric in :math:`i\leftrightarrow j` and reduces to the classical binary form
    :math:`(\mathcal{D}^0_{12})^{x_2}(\mathcal{D}^0_{21})^{x_1}` when :math:`x_1+x_2=1`.

    Args:
        D0_ij: Diffusivity of ``i`` at infinite dilution in ``j``.
        D0_ji: Diffusivity of ``j`` at infinite dilution in ``i``.
        x_i, x_j: The mole fractions of the two components.
    """
    e_ij = (1 + x_j - x_i) / 2
    e_ji = (1 + x_i - x_j) / 2
    # Two things force the exp/log form rather than the literal powers. The exponents are fields, so
    # the quantities they act on have to be made dimensionless first (the two exponents sum to
    # exactly one, which is what puts the unit back). And the division by the unit only cancels
    # symbolically when the base is a single product: with a mixture viscosity that is a sum over
    # the components, GiNaC leaves factors of kilogram^(<field expression>) standing, and unit
    # extraction then reaches GiNaC::power::to_polynomial, which recurses forever on a negative
    # integer power of a power with a symbolic exponent and takes the process down with it.
    # citools/patches/ginac-to-polynomial-symbolic-exponent.patch fixes that, but this form is kept
    # regardless: it costs nothing, and a wheel may be built against an unpatched GiNaC. See
    # dev_docs/code_generation.md 2.3.
    return _D_UNIT * exp(e_ij * log(D0_ij / _D_UNIT) + e_ji * log(D0_ji / _D_UNIT))


def darken_ms_diffusivity(D_self_i: ExpressionOrNum, D_self_j: ExpressionOrNum,
                          x_i: ExpressionOrNum, x_j: ExpressionOrNum) -> ExpressionOrNum:
    r"""The Darken relation :math:`\mathcal{D}_{ij}=x_jD_i^\mathrm{self}+x_iD_j^\mathrm{self}`."""
    return x_j * D_self_i + x_i * D_self_j


####################################################################################################
# The matrices that follow from the definitions alone
####################################################################################################

def mixture_molar_mass(mixture: "MixtureLiquidProperties") -> ExpressionOrNum:
    r"""The mixture molar mass :math:`\bar{M}=(\sum_k w_k/M_k)^{-1}`."""
    return 1 / sum(var("massfrac_" + c) / mixture.pure_properties[c].molar_mass
                   for c in sorted(mixture.components))


def mole_fraction_jacobian(mixture: "MixtureLiquidProperties", indep: list[str],
                           Mbar: ExpressionOrNum) -> list[list[ExpressionOrNum]]:
    r"""
    :math:`A_{ij}=\partial x_i/\partial w_j` over the independent components, with the passive mass
    fraction eliminated. Closed form -- no symbolic differentiation is needed, since
    :math:`\partial(\sum_k w_k/M_k)/\partial w_j=1/M_j-1/M_n`.

    This assumes the molar masses are composition independent, which the caller checks.
    """
    passive = mixture.passive_field
    assert passive is not None
    Mn = mixture.pure_properties[passive].molar_mass
    M = {c: mixture.pure_properties[c].molar_mass for c in indep}
    x = {c: var("molefrac_" + c) for c in indep}
    return [[Mbar * ((1 / M[ci] if ci == cj else 0) - x[ci] * (1 / M[cj] - 1 / Mn))
             for cj in indep] for ci in indep]


def mass_frame_conversion_matrix(mixture: "MixtureLiquidProperties",
                                 indep: list[str]) -> list[list[ExpressionOrNum]]:
    r"""
    :math:`C_{ij}=M_i\delta_{ij}-w_i(M_j-M_n)`, which turns a molar flux in the molar-average frame
    into a mass flux in the mass-average frame. It follows from
    :math:`j_i=M_iJ_i+\rho w_i(u-\bar{v})` together with :math:`\sum_ij_i=0` and
    :math:`J_n=-\sum_{i<n}J_i`.
    """
    passive = mixture.passive_field
    assert passive is not None
    Mn = mixture.pure_properties[passive].molar_mass
    M = {c: mixture.pure_properties[c].molar_mass for c in indep}
    w = {c: var("massfrac_" + c) for c in indep}
    return [[(M[ci] if ci == cj else 0) - w[ci] * (M[cj] - Mn) for cj in indep] for ci in indep]


def maxwell_stefan_matrix(ms_diffusivities: dict[frozenset[str], ExpressionOrNum],
                          components: list[str], indep: list[str],
                          passive: str) -> list[list[ExpressionOrNum]]:
    r"""
    The Maxwell-Stefan matrix :math:`[B]`,

    .. math:: B_{ii}=\frac{x_i}{\mathcal{D}_{in}}+\sum_{k\neq i}\frac{x_k}{\mathcal{D}_{ik}},\qquad
              B_{ij}=-x_i\left(\frac{1}{\mathcal{D}_{ij}}-\frac{1}{\mathcal{D}_{in}}\right)

    over the independent components, with ``n`` the passive one.

    Args:
        ms_diffusivities: The symmetric Maxwell-Stefan diffusivities, keyed by the unordered pair.
        components: All component names.
        indep: The independent (non-passive) component names, in the order of the matrix.
        passive: The passive component name.
    """
    def D(a: str, b: str) -> ExpressionOrNum:
        return ms_diffusivities[frozenset((a, b))]
    x = {c: var("molefrac_" + c) for c in components}
    B: list[list[ExpressionOrNum]] = []
    for ci in indep:
        row: list[ExpressionOrNum] = []
        for cj in indep:
            if ci == cj:
                entry = x[ci] / D(ci, passive)
                for ck in components:
                    if ck != ci:
                        entry = entry + x[ck] / D(ci, ck)
            else:
                entry = -x[ci] * (1 / D(ci, cj) - 1 / D(ci, passive))
            row.append(entry)
        B.append(row)
    return B


def determinant(M: list[list[ExpressionOrNum]]) -> ExpressionOrNum:
    """The determinant of a small symbolic matrix by Laplace expansion along the first row."""
    m = len(M)
    if m == 1:
        return M[0][0]
    if m == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    res: ExpressionOrNum = 0
    for j in range(m):
        minor = [[M[r][c] for c in range(m) if c != j] for r in range(1, m)]
        res = res + (-1) ** j * M[0][j] * determinant(minor)
    return res


def cofactor_inverse(M: list[list[ExpressionOrNum]]) -> tuple[list[list[ExpressionOrNum]], ExpressionOrNum]:
    r"""
    Returns :math:`(\operatorname{adj}M,\det M)`, so that :math:`M^{-1}=\operatorname{adj}M/\det M`.

    Gaussian elimination is not an option symbolically -- it needs pivoting, and symbolic magnitudes
    cannot be compared. The adjugate needs no division at all, which lets the caller keep the single
    :math:`1/\det` as one factor in front of the whole product instead of spreading it over every
    entry.
    """
    m = len(M)
    if m == 1:
        return [[1]], M[0][0]
    adj: list[list[ExpressionOrNum]] = []
    for i in range(m):
        row: list[ExpressionOrNum] = []
        for j in range(m):
            # adj[i][j] is the cofactor C_ji, i.e. the transpose of the cofactor matrix
            minor = [[M[r][c] for c in range(m) if c != i] for r in range(m) if r != j]
            row.append((-1) ** (i + j) * determinant(minor))
        adj.append(row)
    return adj, determinant(M)


def _matmul(A: list[list[ExpressionOrNum]], B: list[list[ExpressionOrNum]]) -> list[list[ExpressionOrNum]]:
    m, k, n = len(A), len(B), len(B[0])
    return [[sum((A[i][l] * B[l][j] for l in range(k)), start=cast(ExpressionOrNum, 0))
             for j in range(n)] for i in range(m)]


####################################################################################################
# The thermodynamic factor
####################################################################################################

def thermodynamic_factor_matrix_symbolic(mixture: "MixtureLiquidProperties",
                                         indep: list[str]) -> list[list[ExpressionOrNum]]:
    r"""
    :math:`\Gamma_{ij}=\delta_{ij}+x_i\,\partial\ln\gamma_i/\partial x_j` by symbolic
    differentiation of the activity coefficients.

    The mole fractions are not independent, so the passive one is eliminated as
    :math:`x_n=1-\sum_{k<n}x_k` *before* differentiating -- differentiating with all
    ``molefrac_`` fields held independent gives a partial derivative at constant :math:`x_n`, which
    is not the thermodynamic factor.

    The substitutions go through ``GiNaC_subs`` rather than ``GiNaC_subsfields``, because the latter
    tries to extract the unit of every subexpression it passes and UNIFAC activity coefficients wrap
    temperature-dependent subexpressions that have no unit of their own.
    """
    E = _pyoomph.Expression
    passive = mixture.passive_field
    assert passive is not None
    X = {c: _pyoomph.GiNaC_new_symbol("__thermofactor_x_" + c) for c in indep}
    xsum: ExpressionOrNum = 0
    for c in indep:
        xsum = xsum + X[c]
    Gamma: list[list[ExpressionOrNum]] = []
    for ci in indep:
        lng = E(log(mixture.activity_coefficients[ci]))
        for c in indep:
            lng = _pyoomph.GiNaC_subs(lng, E(var("molefrac_" + c)), E(X[c]))
        lng = _pyoomph.GiNaC_subs(lng, E(var("molefrac_" + passive)), E(1 - xsum))
        row: list[ExpressionOrNum] = []
        for cj in indep:
            d = _pyoomph.GiNaC_diff(lng, E(X[cj]))
            for c in indep:
                d = _pyoomph.GiNaC_subs(d, E(X[c]), E(var("molefrac_" + c)))
            row.append((1 if ci == cj else 0) + var("molefrac_" + ci) * d)
        Gamma.append(row)
    return Gamma


def thermodynamic_factor_matrix_finite_difference(mixture: "MixtureLiquidProperties",
                                                  indep: list[str],
                                                  epsilon: float = 1e-6) -> list[list[ExpressionOrNum]]:
    r"""
    :math:`[\Gamma]` by a central difference of the activity coefficients in the mole fractions.

    This is the branch for multi-return activity coefficients, which cannot be differentiated
    symbolically at all: their derivatives exist only while the Jacobian code is generated, from the
    expanded node (to first and second order, so an analytic Hessian is fine, but only
    there). Differentiating the unexpanded invocation raises
    (``python_multi_cb_function``'s partial derivative in ``src/expressions.cpp``); before that error
    existed it built a node that no printer could render, and the failure surfaced only as a C file
    the compiler rejects.

    The multi-return argument order already excludes the passive component, so shifting one argument
    lets the passive mole fraction absorb the change, which is exactly the elimination the symbolic
    branch does by hand. Two evaluations per independent component suffice, since each one returns
    all activity coefficients at once.
    """
    mr = cast(Any, mixture._unifac_multi_return)
    assert mr is not None
    if list(mr.argument_order) != list(indep):
        raise RuntimeError("The activity model orders the components as " + str(list(mr.argument_order))
                           + " but the diffusivity matrix is built for " + str(indep)
                           + ". These have to agree, since the finite difference shifts by index.")
    base: list[ExpressionOrNum] = [var("molefrac_" + c) for c in mr.argument_order]
    if mr._constant_temperature_in_K is None:
        base.append(var("temperature"))

    def log_gammas(shift_index: int, sign: int) -> dict[str, ExpressionOrNum]:
        args = list(base)
        args[shift_index] = args[shift_index] + sign * epsilon
        res = mr(*args)
        return {c: log(res[mr.argument_order_with_passive.index(c)]) for c in indep}

    plus = [log_gammas(j, +1) for j in range(len(indep))]
    minus = [log_gammas(j, -1) for j in range(len(indep))]
    return [[(1 if ci == cj else 0)
             + var("molefrac_" + ci) * (plus[j][ci] - minus[j][ci]) / (2 * epsilon)
             for j, cj in enumerate(indep)] for ci in indep]


def thermodynamic_factor_matrix(mixture: "MixtureLiquidProperties", indep: list[str],
                                mode: Literal["auto", "symbolic", "finite_difference"] = "auto",
                                epsilon: float = 1e-6) -> list[list[ExpressionOrNum]]:
    """
    :math:`[\\Gamma]`, dispatching between the symbolic and the finite-difference branch.

    ``"auto"`` takes the finite difference whenever the activity coefficients are multi-return
    expressions, which is the default of
    :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_activity_coefficients_by_unifac`
    from three components on.
    """
    is_multi_return = getattr(mixture, "_unifac_multi_return", None) is not None
    if mode == "auto":
        mode = "finite_difference" if is_multi_return else "symbolic"
    if mode == "symbolic":
        if is_multi_return:
            raise RuntimeError("The activity coefficients of '" + mixture.describe() + "' are "
                               "multi-return expressions, which cannot be differentiated "
                               "symbolically -- their derivatives exist only while the Jacobian "
                               "code is generated. Either keep the default "
                               "thermodynamic_factor_mode='auto', or set the activity coefficients "
                               "with set_activity_coefficients_by_unifac(..., "
                               "use_multi_return=False).")
        return thermodynamic_factor_matrix_symbolic(mixture, indep)
    elif mode == "finite_difference":
        if not is_multi_return:
            raise RuntimeError("thermodynamic_factor_mode='finite_difference' needs multi-return "
                               "activity coefficients, which '" + mixture.describe() + "' does not "
                               "have. Use 'auto' or 'symbolic' instead.")
        return thermodynamic_factor_matrix_finite_difference(mixture, indep, epsilon)
    else:
        raise ValueError("thermodynamic_factor_mode must be 'auto', 'symbolic' or "
                         "'finite_difference', not " + str(mode))


####################################################################################################
# Assembly
####################################################################################################

def _ms_diffusivity_table(mixture: "MixtureLiquidProperties", method: str,
                          components: list[str],
                          given: dict[tuple[str, str], ExpressionOrNum] | None,
                          infinite_dilution: dict[tuple[str, str], ExpressionOrNum] | None,
                          viscosity: ExpressionOrNum,
                          temperature: ExpressionOrNum,
                          use_subexpressions: bool) -> dict[frozenset[str], ExpressionOrNum]:
    """The symmetric Maxwell-Stefan diffusivities of every unordered pair, by the chosen method."""
    supplied: dict[frozenset[str], ExpressionOrNum] = {}
    for (a, b), D in (given or {}).items():
        for name in (a, b):
            if name not in mixture.components:
                raise RuntimeError("maxwell_stefan_diffusivities names '" + name + "', which is not "
                                   "a component of " + str(sorted(mixture.components)))
        key = frozenset((a, b))
        if key in supplied and not is_zero(supplied[key] - D):
            raise RuntimeError("Both orders of the Maxwell-Stefan pair " + str(sorted(key)) + " are "
                               "given with different values. They are symmetric by definition.")
        supplied[key] = D

    def D0(solute: str, solvent: str) -> tuple[ExpressionOrNum, bool]:
        """Diffusivity of solute at infinite dilution in solvent, and whether V_b was estimated."""
        if infinite_dilution is not None and (solute, solvent) in infinite_dilution:
            return infinite_dilution[(solute, solvent)], False
        V, estimated = get_molar_volume_at_boiling_point(mixture.pure_properties[solute])
        return wilke_chang_diffusivity(mixture.pure_properties[solute],
                                       mixture.pure_properties[solvent], viscosity, temperature,
                                       molar_volume=V), estimated

    estimated_volumes: set[str] = set()
    res: dict[frozenset[str], ExpressionOrNum] = {}
    for i, ci in enumerate(components):
        for cj in components[i + 1:]:
            key = frozenset((ci, cj))
            if key in supplied:
                res[key] = supplied[key]
                continue
            if method == "given":
                raise RuntimeError("method='given' needs a Maxwell-Stefan diffusivity for every "
                                   "pair, but " + str(sorted(key)) + " is missing.")
            x_i, x_j = var("molefrac_" + ci), var("molefrac_" + cj)
            if method == "vignes":
                Dij, e1 = D0(ci, cj)
                Dji, e2 = D0(cj, ci)
                if e1:
                    estimated_volumes.add(ci)
                if e2:
                    estimated_volumes.add(cj)
                D = vignes_ms_diffusivity(Dij, Dji, x_i, x_j)
            elif method == "stokes_einstein":
                radii: list[ExpressionOrNum] = []
                for c in (ci, cj):
                    V, est = get_molar_volume_at_boiling_point(mixture.pure_properties[c])
                    if est:
                        estimated_volumes.add(c)
                    radii.append(hydrodynamic_radius(mixture.pure_properties[c], molar_volume=V))
                D = stokes_einstein_diffusivity((radii[0] + radii[1]) / 2, viscosity, temperature)
            elif method == "darken":
                selfD: list[ExpressionOrNum] = []
                for c in (ci, cj):
                    pure = mixture.pure_properties[c]
                    if pure.self_diffusivity is not None:
                        selfD.append(pure.self_diffusivity)
                        continue
                    V, est = get_molar_volume_at_boiling_point(pure)
                    if est:
                        estimated_volumes.add(c)
                    selfD.append(stokes_einstein_diffusivity(
                        hydrodynamic_radius(pure, molar_volume=V), viscosity, temperature))
                D = darken_ms_diffusivity(selfD[0], selfD[1], x_i, x_j)
            else:
                raise ValueError("Unknown method '" + str(method) + "', use one of 'vignes', "
                                 "'stokes_einstein', 'darken' or 'given'.")
            res[key] = subexpression(D) if use_subexpressions else D

    if estimated_volumes:
        warnings.warn("No molar volume at the normal boiling point is set for "
                      + ", ".join(sorted(estimated_volumes)) + ", so molar_mass/mass_density was "
                      "used instead. That is the room-temperature liquid volume and too small (by "
                      "5% for water but 32% for glycerol), which overestimates the diffusivity by "
                      "up to about 17%. Set molar_volume_for_Wilke_Chang_eq on the pure component "
                      "to avoid this.")
    return res


def fickian_mass_diffusivity_matrix(mixture: "MixtureLiquidProperties",
                                    method: str = "vignes", *,
                                    maxwell_stefan_diffusivities: dict[tuple[str, str], ExpressionOrNum] | None = None,
                                    infinite_dilution_diffusivities: dict[tuple[str, str], ExpressionOrNum] | None = None,
                                    thermodynamic_factor: bool = True,
                                    thermodynamic_factor_mode: Literal["auto", "symbolic", "finite_difference"] = "auto",
                                    fd_epsilon: float = 1e-6,
                                    min_thermodynamic_factor: float | None = None,
                                    thermodynamic_factor_shift: float = 0.0,
                                    viscosity: ExpressionOrNum | Literal["mixture", "pure"] = "mixture",
                                    temperature: ExpressionOrNum = var("temperature"),
                                    use_subexpressions: bool = True,
                                    max_components: int = 6) -> dict[tuple[str, str], ExpressionOrNum]:
    r"""
    Assembles :math:`[D]=\bar{M}^{-1}[C][B]^{-1}[\Gamma][A]` for the non-passive components.

    See :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_estimate_diffusivities`,
    which is the intended entry point and documents every argument. This function only builds the
    matrix; it does not write it into the mixture.
    """
    components = sorted(mixture.components)
    passive = mixture.passive_field
    assert passive is not None
    indep = [c for c in components if c != passive]
    if len(components) < 2:
        raise RuntimeError("Cannot estimate diffusivities of a mixture with fewer than 2 components")
    if len(components) > max_components:
        raise RuntimeError("This mixture has " + str(len(components)) + " components, and the "
                           "symbolic inverse of the Maxwell-Stefan matrix grows steeply beyond "
                           + str(max_components) + ". Either raise max_components and accept the "
                           "expression size, or supply the diffusivities directly with "
                           "set_diffusion_coefficient.")
    for c in components:
        pure = mixture.pure_properties[c]
        if not hasattr(pure, "molar_mass"):
            raise RuntimeError("The component '" + c + "' has no molar_mass, which every part of "
                               "the estimate needs.")
        if isinstance(pure.molar_mass, Expression) and not is_zero(
                _pyoomph.GiNaC_subs(_pyoomph.Expression(pure.molar_mass),
                                    _pyoomph.Expression(var("massfrac_" + c)),
                                    _pyoomph.Expression(0)) - pure.molar_mass):
            raise RuntimeError("The molar mass of '" + c + "' depends on the composition. The "
                               "closed form of the mole-fraction Jacobian assumes it does not.")

    if viscosity == "mixture":
        if not hasattr(mixture, "dynamic_viscosity"):
            raise RuntimeError("The mixture has no dynamic_viscosity, which the diffusivity "
                               "estimate needs. Set it, e.g. by "
                               "set_by_weighted_average('dynamic_viscosity').")
        mu: ExpressionOrNum = mixture.dynamic_viscosity
    elif viscosity == "pure":
        mu = None  # type: ignore[assignment] # resolved per pair below
    else:
        mu = cast(ExpressionOrNum, viscosity)

    if viscosity == "pure":
        # The Wilke-Chang solvent viscosity of the pair, i.e. no mixture-level viscosity at all
        Dms: dict[frozenset[str], ExpressionOrNum] = {}
        for i, ci in enumerate(components):
            for cj in components[i + 1:]:
                sub = _ms_diffusivity_table(mixture, method, [ci, cj], maxwell_stefan_diffusivities,
                                            infinite_dilution_diffusivities,
                                            mixture.pure_properties[cj].dynamic_viscosity,
                                            temperature, use_subexpressions)
                Dms[frozenset((ci, cj))] = sub[frozenset((ci, cj))]
    else:
        Dms = _ms_diffusivity_table(mixture, method, components, maxwell_stefan_diffusivities,
                                    infinite_dilution_diffusivities, mu, temperature,
                                    use_subexpressions)

    m = len(indep)
    if thermodynamic_factor:
        if not mixture.activity_coefficients:
            warnings.warn("No activity coefficients are set for '" + mixture.describe() + "', so "
                          "the thermodynamic factor is the identity and the estimate is the one of "
                          "an ideal mixture. Call set_activity_coefficients_by_unifac first, or "
                          "pass thermodynamic_factor=False to say so explicitly.")
            Gamma: list[list[ExpressionOrNum]] = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
        else:
            Gamma = thermodynamic_factor_matrix(mixture, indep, thermodynamic_factor_mode, fd_epsilon)
    else:
        Gamma = [[1 if i == j else 0 for j in range(m)] for i in range(m)]

    if min_thermodynamic_factor is not None:
        if m != 1:
            raise NotImplementedError("min_thermodynamic_factor only applies to binary mixtures. "
                                      "Bounding a matrix means bounding its eigenvalues, which is "
                                      "not expressible symbolically; use "
                                      "thermodynamic_factor_shift instead.")
        Gamma[0][0] = maximum(Gamma[0][0], min_thermodynamic_factor)
    if thermodynamic_factor_shift:
        Gamma = [[Gamma[i][j] + (thermodynamic_factor_shift if i == j else 0) for j in range(m)]
                 for i in range(m)]
    if use_subexpressions:
        Gamma = [[subexpression(e) if isinstance(e, Expression) else e for e in row] for row in Gamma]

    Mbar = mixture_molar_mass(mixture)
    if use_subexpressions:
        Mbar = subexpression(Mbar)
    A = mole_fraction_jacobian(mixture, indep, Mbar)
    C = mass_frame_conversion_matrix(mixture, indep)
    B = maxwell_stefan_matrix(Dms, components, indep, passive)
    adjB, detB = cofactor_inverse(B)
    if use_subexpressions:
        adjB = [[subexpression(e) if isinstance(e, Expression) else e for e in row] for row in adjB]
        detB = subexpression(detB) if isinstance(detB, Expression) else detB
    prefactor = 1 / (Mbar * detB)
    if use_subexpressions:
        prefactor = subexpression(prefactor)

    D = _matmul(_matmul(C, adjB), _matmul(Gamma, A))
    res: dict[tuple[str, str], ExpressionOrNum] = {}
    for i, ci in enumerate(indep):
        for j, cj in enumerate(indep):
            e = prefactor * D[i][j]
            res[(ci, cj)] = subexpression(e) if use_subexpressions else e
    return res


def scan_thermodynamic_factor(mixture: "MixtureLiquidProperties", npoints: int = 21, *,
                              temperature: ExpressionOrNum = 20 * celsius,
                              matrix: dict[tuple[str, str], ExpressionOrNum] | None = None,
                              thermodynamic_factor_mode: Literal["auto", "symbolic", "finite_difference"] = "auto",
                              fd_epsilon: float = 1e-6) -> list[tuple[dict[str, float], float]]:
    r"""
    Evaluates the smallest eigenvalue of :math:`[\Gamma]` (or of a given diffusivity matrix) on a
    grid over the composition simplex and returns the points where it is not positive.

    A negative eigenvalue is not a numerical artefact: it says the mixture is inside its miscibility
    gap there, where diffusion is genuinely backward-parabolic and no solver will get through. This
    is the diagnostic to run when a solve with estimated diffusivities blows up.

    Args:
        mixture: The mixture to scan.
        npoints: Number of grid points per independent mass fraction.
        temperature: Temperature to evaluate at.
        matrix: Scan this matrix (e.g. the return value of
            :py:meth:`~pyoomph.materials.generic.MixtureLiquidProperties.set_estimate_diffusivities`)
            instead of the bare thermodynamic factor.
        thermodynamic_factor_mode: Passed on when the thermodynamic factor is built here.
        fd_epsilon: Passed on when the thermodynamic factor is built here.

    Returns:
        A list of ``(composition, smallest eigenvalue)`` for the grid points that are not positive
        definite, sorted by the eigenvalue.
    """
    import numpy
    import itertools
    passive = mixture.passive_field
    assert passive is not None
    indep = sorted(mixture.components - {passive})
    if matrix is None:
        M = thermodynamic_factor_matrix(mixture, indep, thermodynamic_factor_mode, fd_epsilon)
        entries = {(ci, cj): M[i][j] for i, ci in enumerate(indep) for j, cj in enumerate(indep)}
    else:
        entries = dict(matrix)
    bad: list[tuple[dict[str, float], float]] = []
    for combo in itertools.product(numpy.linspace(0, 1, npoints), repeat=len(indep)):
        if sum(combo) > 1:
            continue
        cond: dict[str, ExpressionOrNum] = {"massfrac_" + c: float(v) for c, v in zip(indep, combo)}
        cond["temperature"] = temperature
        try:
            evaluated = numpy.array([[float(mixture.evaluate_at_condition(entries[(ci, cj)], cond))
                                      for cj in indep] for ci in indep])
        except Exception:
            continue
        smallest = float(numpy.min(numpy.real(numpy.linalg.eigvals(evaluated))))
        if smallest <= 0:
            bad.append(({c: float(v) for c, v in zip(indep, combo)}, smallest))
    bad.sort(key=lambda e: e[1])
    return bad


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
