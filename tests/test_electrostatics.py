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

# pyoomph.equations.electrostatics: the potential formulation of Gauss's law, and the boundary
# conditions that go with it.
#
# What actually has to hold, and what is checked here:
#
#   1. THE SIGN OF A SURFACE CHARGE. Everything else in the module keys off it -- the two-bulk
#      coupling is built so that removing the opposite domain degenerates into SurfaceChargeBC --
#      and it is not checkable by inspection, because it depends on which way the bulk residual was
#      integrated by parts. test_surface_charge_sign pins it against the closed-form 1D solution,
#      including the statement that a positively charged wall has a POSITIVE potential.
#   2. THE SUBSTITUTED ELECTRIC FIELD IS THE FIELD. Substituted fields are expanded as
#      get_scaling(name)*<substitution>, so a substitution written dimensionally is off by exactly
#      one factor of its own scale -- which is invisible as long as that scale is 1, i.e. in every
#      nondimensional test. test_electric_field_matches_gradient runs with a scale that is not 1.
#   3. SCALES ARE COSMETIC. The dimensional answer may not depend on the potential or permittivity
#      scale chosen. That is what makes it safe to demand ONE shared scale across coupled domains.
#
# Each Problem gets its OWN output directory: several Problems in one directory share the JIT cache
# and variants that differ only in constructor flags then silently reuse the first one's code.

import itertools

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import *
from pyoomph.equations.electrostatics import *
from pyoomph.equations.stabilization import ScalarTransportStabilization
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh

_run_counter = itertools.count()


def _fresh(problem):
    """Own output directory per Problem, see the note at the top."""
    problem.set_output_directory("run%d" % next(_run_counter))
    problem.set_c_compiler("system")
    return problem


# ---------------------------------------------------------------------------------------------
# Constants and units
# ---------------------------------------------------------------------------------------------

def test_faraday_and_boltzmann_are_consistent():
    # PB and PNP are written interchangeably with e/(k_B T) and F/(R T); if k_B*N_A != R the two
    # forms disagree in the 8th digit and any test comparing them has an unexplained floor.
    assert float(k_Boltzmann * N_Avogadro / gas_constant) == 1.0
    assert float(faraday_constant / (elementary_charge * N_Avogadro)) == 1.0
    assert float(faraday_constant / (coulomb / mol)) == pytest.approx(96485.33212, rel=1e-9)
    assert float(thermal_voltage(298.15 * kelvin) / (milli * volt)) == pytest.approx(25.6926, rel=1e-5)
    # Textbook values: 0.304 nm for a 1 M 1:1 electrolyte, 0.71 nm in water at room temperature.
    assert float(debye_length(78.4 * epsilon_0, 1 * mol / liter, 298.15 * kelvin) / (nano * meter)) \
        == pytest.approx(0.304, rel=1e-2)
    assert float(bjerrum_length(78.4 * epsilon_0, 298.15 * kelvin) / (nano * meter)) \
        == pytest.approx(0.715, rel=1e-2)


@pytest.mark.parametrize("expr,expected", [
    ("1e-12*coulomb", "pC"),
    ("1e-3*siemens", "mS"),
    ("1e-3*siemens/meter", "mS/m"),
    ("1e5*volt/meter", "MV/m"),
    ("1e-3*coulomb/meter**2", "mC/m^2"),
    ("1e-3*coulomb/meter**3", "mC/m^3"),
    ("1e-12*farad", "pF"),
    ("epsilon_0", "pF/m"),
    ("0.1*farad/meter**2", "F/m^2"),          # Stern layer / Lippmann capacitance
    ("1e3*ohm", "kOhm"),
    ("1*ohm*meter", "Ohm m"),
    ("faraday_constant", "MC/mol"),
    ("1e-30*coulomb*meter", "aC m"),
    ("1e-2*siemens*meter**2/mol", "mS m^2/mol"),
    ("1e-3*henry", "mH"),
    ("mu_0", "uH/m"),
    ("1*volt", "V"),
    ("1*ampere/meter**2", "A/m^2"),           # falls out of the base units, nothing registered
    # Must NOT be relabelled by the electric entries:
    ("1*pascal*second", "Pas"),
    ("1/second", "1/s"),
    ("1e-19*joule", "aNm"),
    ("1*watt", "W"),
    ("70e-3*newton/meter", "N/m"),
    ("1000*kilogram/meter**3", "Mg/m^3"),
    ("1*farad/meter*(volt/meter)**2", "Pa"),  # a Maxwell stress really is a pressure
])
def test_electric_unit_strings(expr, expected):
    assert unit_to_string(eval(expr), estimate_prefix=True)[0] == expected


_SI_PREFIXES = {"y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9,
                "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
                "P": 1e15, "E": 1e18, "Z": 1e21, "Y": 1e24}

_NAMED_ELECTRIC = {
    "C": "coulomb", "S": "siemens", "S/m": "siemens/meter", "F": "farad", "F/m": "farad/meter",
    "F/m^2": "farad/meter**2", "V": "volt", "V/m": "volt/meter", "C/m^2": "coulomb/meter**2",
    "C/m^3": "coulomb/meter**3", "Ohm": "ohm", "Ohm m": "ohm*meter", "C/mol": "coulomb/mol",
    "C m": "coulomb*meter", "S m^2/mol": "siemens*meter**2/mol", "H": "henry", "H/m": "henry/meter",
}


@pytest.mark.parametrize("label,expr", sorted(_NAMED_ELECTRIC.items()))
@pytest.mark.parametrize("magnitude", [1e-12, 1e-6, 1e-3, 1.0, 1e3, 1e6, 3.7e-8, 96485.33])
def test_electric_units_round_trip(label, expr, magnitude):
    """The printed number and the printed unit must multiply back to the input.

    This is the failure mode the farad comment in units.py records: the prefix is written in front
    of the whole numerator, so it binds to the first symbol TOGETHER WITH ITS EXPONENT. A simplified
    name short-circuits the exponent search and is always treated as exponent 1, so a label starting
    with an exponentiated base unit (e.g. "m^2/(V s)" for ion mobility) would be silently wrong by
    three orders per prefix step. That is why no such label is registered, and why this test exists
    rather than a comparison of strings.
    """
    unit = eval(expr)
    name, mag, factor = unit_to_string(magnitude * unit, estimate_prefix=True)
    assert name.endswith(label)
    prefix = name[:len(name) - len(label)]
    assert prefix in _SI_PREFIXES, "unrecognised prefix %r in %r" % (prefix, name)
    assert mag * factor * _SI_PREFIXES[prefix] == pytest.approx(magnitude, rel=1e-9)


def test_ion_mobility_is_left_in_base_units():
    # Deliberately NOT registered, see the docstring above: its natural label would start with m^2.
    # The generic base-unit path does handle the exponent, so this still round-trips.
    name, mag, factor = unit_to_string(1e-8 * meter ** 2 / (volt * second), estimate_prefix=True)
    assert "V" not in name  # i.e. it did not get a derived-unit label
    assert float(mag) == pytest.approx(1e-8, rel=1e-12)


def test_coulomb_alias():
    assert (coulomb - coulomb).is_zero()


# ---------------------------------------------------------------------------------------------
# The bulk equation
# ---------------------------------------------------------------------------------------------

class _Capacitor(Problem):
    """Charge-free 1D slab between two electrodes: phi is linear, E = -V/h."""

    def __init__(self, h, V, eps_r, potential_scale=None, permittivity_scale=epsilon_0):
        super().__init__()
        self.h, self.V, self.eps_r = h, V, eps_r
        self.potential_scale = potential_scale if potential_scale is not None else V
        self.permittivity_scale = permittivity_scale

    def define_problem(self):
        self.set_scaling(spatial=self.h)
        set_electrostatic_scaling(self, potential=self.potential_scale,
                                  permittivity=self.permittivity_scale)
        self.add_mesh(LineMesh(N=8, size=self.h))
        eqs = ElectricPotentialEquations(relative_permittivity=self.eps_r,
                                         permittivity_scale=self.permittivity_scale)
        eqs += ElectrodeBC(0) @ "left"
        eqs += ElectrodeBC(self.V) @ "right"
        eqs += IntegralObservables(_length=1,
                                   Ex=var("electric_field")[0],
                                   gradphi=-grad(var("phi"))[0],
                                   Dx=self.eps_r * epsilon_0 * var("electric_field")[0],
                                   phi=var("phi"))
        self.add_equations(eqs @ "domain")


def _solve(problem):
    with _fresh(problem) as p:
        p.solve()
        obs = p.get_mesh("domain").evaluate_all_observables()
    return {k: v / obs["_length"] for k, v in obs.items() if k != "_length"}


def test_parallel_plate_capacitor():
    h, V, eps_r = 10 * micro * meter, 2 * volt, 3.0
    m = _solve(_Capacitor(h, V, eps_r))
    assert float(m["Ex"] / (volt / meter)) == pytest.approx(float(-V / h / (volt / meter)), rel=1e-12)
    assert float(m["phi"] / volt) == pytest.approx(1.0, rel=1e-11)
    assert float(m["Dx"] / (coulomb / meter ** 2)) \
        == pytest.approx(float(-eps_r * epsilon_0 * V / h / (coulomb / meter ** 2)), rel=1e-11)


def test_electric_field_matches_gradient():
    # Point 2 at the top: with a potential scale of 1 V and a length scale of 1 m both scales are 1
    # and a substitution off by its own scale looks perfectly correct. These numbers are not 1.
    m = _solve(_Capacitor(10 * micro * meter, 2 * volt, 3.0))
    assert float((m["Ex"] - m["gradphi"]) / (volt / meter)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("potential_scale", [1 * volt, 1 * milli * volt, 17 * volt])
@pytest.mark.parametrize("permittivity_scale", [epsilon_0, 78 * epsilon_0])
def test_scales_are_cosmetic(potential_scale, permittivity_scale):
    # Point 3 at the top. If this fails, no cross-domain coupling in this module can be trusted.
    h, V, eps_r = 10 * micro * meter, 2 * volt, 3.0
    m = _solve(_Capacitor(h, V, eps_r, potential_scale=potential_scale,
                          permittivity_scale=permittivity_scale))
    assert float(m["Ex"] / (volt / meter)) == pytest.approx(float(-V / h / (volt / meter)), rel=1e-10)


# ---------------------------------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------------------------------

class _Manufactured(Problem):
    r"""phi = x*(1-x)*(1+y) lies in C2 on a quad mesh, so -div(eps grad phi) = rho_e with
    rho_e computed symbolically is solved exactly, up to round-off."""

    def define_problem(self):
        eps = 2.5
        x, y = var(["coordinate_x", "coordinate_y"])
        exact = x * (1 - x) * (1 + y)
        self.add_mesh(RectangularQuadMesh(N=4))
        eqs = ElectricPotentialEquations(permittivity=eps, charge_density=-div(eps * grad(exact)),
                                         permittivity_scale=1, consider_scaling=True)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(phi=exact) @ b
        eqs += IntegralObservables(err=subexpression((var("phi") - exact) ** 2), _area=1)
        self.add_equations(eqs @ "domain")


def test_manufactured_solution_is_exact():
    with _fresh(_Manufactured()) as p:
        p.solve()
        obs = p.get_mesh("domain").evaluate_all_observables()
    # A squared-error observable, so this is an L2 error of ~4e-9 -- the Newton tolerance, not
    # machine epsilon. The point is that it does not depend on the mesh: the solution is IN the
    # space, so there is no discretization error to converge away.
    assert float(obs["err"] / obs["_area"]) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------------------------
# Surface charge
# ---------------------------------------------------------------------------------------------

class _ChargedWall(Problem):
    r"""Charge-free slab, grounded at x=h, surface charge sigma_s on the wall at x=0.

    n.D = -sigma_s with n = -e_x gives eps*dphi/dx = -sigma_s, hence
    phi(x) = sigma_s*(h-x)/eps and in particular phi(0) = sigma_s*h/eps > 0 for sigma_s > 0.
    """

    def __init__(self, h, sigma_s, eps_r):
        super().__init__()
        self.h, self.sigma_s, self.eps_r = h, sigma_s, eps_r

    def define_problem(self):
        eps = self.eps_r * epsilon_0
        self.set_scaling(spatial=self.h)
        set_electrostatic_scaling(self, potential=self.sigma_s * self.h / eps)
        self.add_mesh(LineMesh(N=8, size=self.h))
        eqs = ElectricPotentialEquations(relative_permittivity=self.eps_r)
        eqs += SurfaceChargeBC(self.sigma_s) @ "left"
        eqs += ElectrodeBC(0) @ "right"
        eqs += IntegralObservables(_length=1, phi=var("phi"), Ex=var("electric_field")[0])
        self.add_equations(eqs @ "domain")


@pytest.mark.parametrize("sign", [+1, -1])
def test_surface_charge_sign(sign):
    # Point 1 at the top. The magnitude alone would not catch a flipped sign, hence both signs and
    # an explicit statement about which way the potential goes.
    h, eps_r = 10 * micro * meter, 3.0
    sigma_s = sign * 1 * milli * coulomb / meter ** 2
    eps = eps_r * epsilon_0
    m = _solve(_ChargedWall(h, sigma_s, eps_r))
    # phi is linear from sigma_s*h/eps down to 0, so its mean is half the wall value.
    assert float(m["phi"] / volt) == pytest.approx(float(sigma_s * h / eps / 2 / volt), rel=1e-10)
    assert float(m["Ex"] / (volt / meter)) == pytest.approx(float(sigma_s / eps / (volt / meter)), rel=1e-10)
    assert numpy.sign(float(m["phi"] / volt)) == sign


# ---------------------------------------------------------------------------------------------
# Far field
# ---------------------------------------------------------------------------------------------

class _PointCharge(Problem):
    r"""A charge Q on a sphere of radius a, in radial symmetry, with the mesh truncated at b.

    Exact: phi = Q/(4 pi eps r), so the truncation is where a far-field condition earns its keep --
    a plain Dirichlet phi=0 at r=b would be wrong by Q/(4 pi eps b).
    """

    def __init__(self, a, b, Q, eps_r, use_farfield):
        super().__init__()
        self.a, self.b, self.Q, self.eps_r, self.use_farfield = a, b, Q, eps_r, use_farfield

    def define_problem(self):
        eps = self.eps_r * epsilon_0
        self.set_coordinate_system(radialsymmetric)
        self.set_scaling(spatial=self.a)
        set_electrostatic_scaling(self, potential=self.Q / (4 * pi * eps * self.a))
        self.add_mesh(LineMesh(N=200, size=self.b - self.a, minimum=self.a))
        eqs = ElectricPotentialEquations(relative_permittivity=self.eps_r)
        eqs += SurfaceChargeBC(self.Q / (4 * pi * self.a ** 2)) @ "left"
        if self.use_farfield:
            eqs += ElectricFarFieldCondition(0) @ "right"
        else:
            eqs += ElectrodeBC(0) @ "right"
        eqs += IntegralObservables(_dummy=1, err=subexpression(
            (var("phi") - self.Q / (4 * pi * eps * var("coordinate_x"))) ** 2))
        self.add_equations(eqs @ "domain")


def test_far_field_beats_truncation():
    # Also the only coverage the extracted farfield_monopole_residual helper in poisson.py has:
    # nothing in the tree used PoissonFarFieldMonopoleCondition before.
    a, b, Q, eps_r = 1 * micro * meter, 20 * micro * meter, 1e-15 * coulomb, 3.0
    eps = eps_r * epsilon_0
    ref = float(Q / (4 * pi * eps * a) / volt) ** 2 * float(4 * pi / 3 * a ** 3 / meter ** 3)

    def rms(use_ff):
        with _fresh(_PointCharge(a, b, Q, eps_r, use_ff)) as p:
            p.solve()
            o = p.get_mesh("domain").evaluate_all_observables()
        return float(o["err"] / (volt ** 2 * meter ** 3)) / ref

    ff, trunc = rms(True), rms(False)
    assert ff < 1e-4, "far field should reproduce the monopole almost exactly"
    assert trunc > 100 * ff, "the truncated Dirichlet problem must be visibly worse"


# ---------------------------------------------------------------------------------------------
# Two coupled bulk domains
# ---------------------------------------------------------------------------------------------

class _TwoDielectrics(Problem):
    r"""Two slabs with different permittivity, a free surface charge on the interface between them,
    and electrodes at the two outer ends.

    No bulk charge, so phi is piecewise linear and the interface value follows from
    n.(D_B - D_A) = sigma_s with n = +e_x:

        phi_i = (eps2*V + sigma_s*L/2) / (eps1 + eps2)

    Note that neither permittivity appears in the interface equation -- the Lagrange multiplier
    picks each side's own D up from its own bulk residual. That is the property being tested.
    """

    def __init__(self, L, V, eps_r1, eps_r2, sigma_s, potential_scale=None,
                 layer_capacitance=None, permittivity_scale=(epsilon_0, epsilon_0)):
        super().__init__()
        self.L, self.V, self.eps_r1, self.eps_r2, self.sigma_s = L, V, eps_r1, eps_r2, sigma_s
        self.potential_scale = potential_scale if potential_scale is not None else V
        self.layer_capacitance = layer_capacitance
        self.permittivity_scale = permittivity_scale

    def phi_interface(self):
        eps1, eps2 = self.eps_r1 * epsilon_0, self.eps_r2 * epsilon_0
        return (eps2 * self.V + self.sigma_s * self.L / 2) / (eps1 + eps2)

    def define_problem(self):
        self.set_scaling(spatial=self.L)
        set_electrostatic_scaling(self, potential=self.potential_scale)
        self.add_mesh(LineMesh(N=16, size=self.L, name=lambda x: "A" if x < 0.5 else "B"))
        eqsA = ElectricPotentialEquations(relative_permittivity=self.eps_r1,
                                          permittivity_scale=self.permittivity_scale[0])
        eqsA += ElectrodeBC(0) @ "left"
        if self.layer_capacitance is None:
            eqsA += ElectricPotentialConnection(surface_charge_density=self.sigma_s) @ "A_B"
        else:
            eqsA += ThinDielectricLayer(capacitance=self.layer_capacitance) @ "A_B"
        eqsA += IntegralObservables(_length=1, phi=var("phi"))
        eqsB = ElectricPotentialEquations(relative_permittivity=self.eps_r2,
                                          permittivity_scale=self.permittivity_scale[1])
        eqsB += ElectrodeBC(self.V) @ "right"
        self.add_equations(eqsA @ "A")
        self.add_equations(eqsB @ "B")


def _mean_phi_in_A(problem):
    with _fresh(problem) as p:
        p.solve()
        o = p.get_mesh("A").evaluate_all_observables()
    return o["phi"] / o["_length"]


@pytest.mark.parametrize("sigma_s_mC", [0.0, +0.1, -0.1])
@pytest.mark.parametrize("eps_pair", [(2.0, 8.0), (78.0, 1.0)])
def test_two_dielectrics_with_surface_charge(sigma_s_mC, eps_pair):
    L, V = 10 * micro * meter, 2 * volt
    sigma_s = sigma_s_mC * milli * coulomb / meter ** 2
    prob = _TwoDielectrics(L, V, eps_pair[0], eps_pair[1], sigma_s)
    # phi is linear from 0 to phi_i across domain A, so its mean is half the interface value.
    assert float(_mean_phi_in_A(prob) / volt) \
        == pytest.approx(float(prob.phi_interface() / 2 / volt), rel=1e-10)


# The realistic span, from the thermal voltage to a high-voltage EHD experiment. "Cosmetic" is a
# statement about the physics, not about conditioning: a scale that is orders of magnitude off makes
# the nondimensional dofs huge and Newton stalls above tolerance, which is a real limitation of any
# nondimensionalisation and not something to paper over here.
@pytest.mark.parametrize("potential_scale", [1 * volt, 25 * milli * volt, 300 * volt])
def test_two_dielectrics_scales_are_cosmetic(potential_scale):
    L, V = 10 * micro * meter, 2 * volt
    sigma_s = 0.1 * milli * coulomb / meter ** 2
    prob = _TwoDielectrics(L, V, 2.0, 8.0, sigma_s, potential_scale=potential_scale)
    assert float(_mean_phi_in_A(prob) / volt) \
        == pytest.approx(float(prob.phi_interface() / 2 / volt), rel=1e-9)


def test_inconsistent_permittivity_scale_is_rejected():
    # The trap this guard exists for: a per-domain scale does not make the answer a bit worse, it
    # silently breaks the flux continuity altogether. So it has to be an error, not a warning.
    prob = _TwoDielectrics(10 * micro * meter, 2 * volt, 2.0, 8.0, 0,
                           permittivity_scale=(epsilon_0, 78 * epsilon_0))
    with pytest.raises(RuntimeError, match="disagree on the"):
        with _fresh(prob) as p:
            p.initialise()


def test_thin_dielectric_layer_is_a_series_capacitance():
    L, V, e1, e2 = 10 * micro * meter, 2 * volt, 2.0, 8.0
    C = 0.05 * farad / meter ** 2
    eps1, eps2 = e1 * epsilon_0, e2 * epsilon_0
    # Three capacitors in series; D is uniform, so phi in A rises linearly to D*(L/2)/eps1.
    C_tot = 1 / ((L / 2) / eps1 + 1 / C + (L / 2) / eps2)
    expected_mean = C_tot * V * L / (4 * eps1)
    prob = _TwoDielectrics(L, V, e1, e2, 0, layer_capacitance=C)
    assert float(_mean_phi_in_A(prob) / volt) == pytest.approx(float(expected_mean / volt), rel=1e-10)


# ---------------------------------------------------------------------------------------------
# Poisson-Boltzmann and Debye-Hueckel
# ---------------------------------------------------------------------------------------------

class _DiffuseLayer(Problem):
    r"""A charged flat wall against an electrolyte occupying 0 < x < L.

    The far end carries the natural boundary condition, i.e. dphi/dx = 0, so the exact
    Debye-Hueckel solution on this finite domain is

        phi(x) = zeta * cosh((L-x)/lambda) / cosh(L/lambda)

    which also fixes the wall charge, sigma_s = eps*zeta/lambda * tanh(L/lambda) -- the textbook
    relation in the semi-infinite limit. Both are checked, because the profile alone would not
    notice a wrong prefactor in the charge density and the charge alone would not notice a wrong
    profile shape.
    """

    def __init__(self, *, model, zeta, lambda_D, eps_r=78.4, N=400, L_over_lambda=20.0,
                 valence=1, T=298.15 * kelvin, extra_observables=None):
        super().__init__()
        self.model, self.zeta, self.lambda_D, self.eps_r = model, zeta, lambda_D, eps_r
        self.N, self.L, self.valence, self.T = N, L_over_lambda * lambda_D, valence, T
        self.extra_observables = extra_observables

    def bulk_concentration(self):
        """The c_inf that produces the requested Debye length, for the ion-based models."""
        eps = self.eps_r * epsilon_0
        I = eps * gas_constant * self.T / (2 * faraday_constant ** 2 * self.lambda_D ** 2)
        return I / self.valence ** 2  # I = z^2*c for a symmetric z:z electrolyte

    def exact(self, x):
        lam, L = self.lambda_D, self.L
        return self.zeta * cosh((L - x) / lam) / cosh(L / lam)

    def define_problem(self):
        eps = self.eps_r * epsilon_0
        self.set_scaling(spatial=self.lambda_D, temperature=self.T)
        set_electrostatic_scaling(self, potential=thermal_voltage(self.T), temperature=self.T,
                                  ion_concentration=self.bulk_concentration())
        self.add_mesh(LineMesh(N=self.N, size=self.L))
        common = dict(relative_permittivity=self.eps_r, temperature=self.T)
        if self.model == "debye_huckel_explicit":
            eqs = DebyeHuckelEquations(debye_length=self.lambda_D, **common)
        elif self.model == "debye_huckel_ions":
            eqs = PoissonBoltzmannEquations(bulk_concentration=self.bulk_concentration(),
                                            valence=self.valence, linearized=True, **common)
        elif self.model == "poisson_boltzmann":
            eqs = PoissonBoltzmannEquations(bulk_concentration=self.bulk_concentration(),
                                            valence=self.valence, **common)
        else:
            raise ValueError(self.model)
        x = var("coordinate_x")
        obs = dict(_length=1, charge=eqs.get_charge_density(),
                   err=subexpression((var("phi") - self.exact(x)) ** 2),
                   ref=subexpression(self.exact(x) ** 2))
        if self.extra_observables is not None:
            obs.update(self.extra_observables(self))
        eqs = eqs + ElectrodeBC(self.zeta) @ "left"
        eqs += IntegralObservables(**obs)
        eqs += InitialCondition(phi=self.exact(x))
        self.add_equations(eqs @ "domain")


def _diffuse_layer(**kw):
    prob = _DiffuseLayer(**kw)
    with _fresh(prob) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
    return prob, o


@pytest.mark.parametrize("model", ["debye_huckel_explicit", "debye_huckel_ions"])
def test_debye_huckel_profile(model):
    lam = 3 * nano * meter
    prob, o = _diffuse_layer(model=model, zeta=0.5 * thermal_voltage(298.15 * kelvin), lambda_D=lam)
    rel_l2 = float(o["err"] / o["ref"]) ** 0.5
    assert rel_l2 < 1e-6, "relative L2 error %g" % rel_l2


@pytest.mark.parametrize("model", ["debye_huckel_explicit", "debye_huckel_ions"])
def test_debye_huckel_surface_charge(model):
    # Global electroneutrality: the diffuse layer holds exactly minus the wall charge. With the
    # exact profile that is sigma_s = eps*zeta/lambda*tanh(L/lambda).
    lam, eps_r = 3 * nano * meter, 78.4
    zeta = 0.5 * thermal_voltage(298.15 * kelvin)
    prob, o = _diffuse_layer(model=model, zeta=zeta, lambda_D=lam, eps_r=eps_r)
    eps = eps_r * epsilon_0
    expected = eps * zeta / lam * tanh(prob.L / lam)
    assert float(-o["charge"] / (coulomb / meter ** 2)) \
        == pytest.approx(float(expected / (coulomb / meter ** 2)), rel=1e-5)


def test_debye_huckel_converges_third_order():
    lam = 3 * nano * meter
    zeta = 0.5 * thermal_voltage(298.15 * kelvin)
    errs = []
    for N in (25, 50, 100):
        _, o = _diffuse_layer(model="debye_huckel_explicit", zeta=zeta, lambda_D=lam, N=N)
        errs.append(float(o["err"] / o["ref"]) ** 0.5)
    orders = [numpy.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(orders) > 2.7, "C2 should give ~3rd order in L2, got %s" % orders


def test_poisson_boltzmann_gouy_chapman():
    # Gouy-Chapman: tanh(psi/4) = tanh(psi_0/4)*exp(-x/lambda) with psi = zF*phi/(RT). At psi_0 = 4
    # the linearized model is off by tens of percent, so this really does test the nonlinearity.
    T, lam, eps_r = 298.15 * kelvin, 3 * nano * meter, 78.4
    VT = thermal_voltage(T)
    psi0 = 4.0
    def gouy_chapman(prob):
        x = var("coordinate_x")
        psi = var("phi") / VT
        gc = 4 * atanh(numpy.tanh(psi0 / 4) * exp(-x / prob.lambda_D))
        return dict(gc_err=subexpression((psi - gc) ** 2), gc_ref=subexpression(gc ** 2))

    prob = _DiffuseLayer(model="poisson_boltzmann", zeta=psi0 * VT, lambda_D=lam, eps_r=eps_r, N=800,
                         extra_observables=gouy_chapman)
    with _fresh(prob) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
    # Not exact: the mesh is truncated at 20*lambda with dphi/dx=0 rather than semi-infinite.
    rel_l2 = float(o["gc_err"] / o["gc_ref"]) ** 0.5
    assert rel_l2 < 2e-3, "relative L2 error against Gouy-Chapman: %g" % rel_l2


def test_poisson_boltzmann_reduces_to_debye_huckel():
    # The linearization is only justified below the thermal voltage; at zeta = 0.01*RT/F the two
    # must agree to the square of that, i.e. ~1e-4.
    T, lam = 298.15 * kelvin, 3 * nano * meter
    zeta = 0.01 * thermal_voltage(T)
    _, o_pb = _diffuse_layer(model="poisson_boltzmann", zeta=zeta, lambda_D=lam, N=200)
    _, o_dh = _diffuse_layer(model="debye_huckel_ions", zeta=zeta, lambda_D=lam, N=200)
    assert float(o_pb["charge"] / o_dh["charge"]) == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------------------------
# Nernst-Planck / PNP
# ---------------------------------------------------------------------------------------------

class _PNPDiffuseLayer(Problem):
    r"""Blocking charged wall at x=0, ion reservoir at x=L.

    In 1D steady state div(J_i)=0 makes each flux constant, and the blocking wall makes that
    constant zero -- so J_i = 0 everywhere, which is exactly the Boltzmann distribution
    c_i = c_inf*exp(-z_i F phi/RT). Steady PNP against a reservoir therefore *is* Poisson-Boltzmann,
    and comparing them is the sharpest available check on both: they share no code at all, one
    solves a transport equation and the other an algebraic closure.
    """

    def __init__(self, *, zeta, lambda_D, eps_r=78.4, N=300, L_over_lambda=20.0, valence=1,
                 T=298.15 * kelvin, D=1e-9 * meter ** 2 / second, stabilization=None,
                 stab_factor=1, dc_factor=1):
        super().__init__()
        self.zeta, self.lambda_D, self.eps_r, self.valence, self.T, self.D = \
            zeta, lambda_D, eps_r, valence, T, D
        self.N, self.L = N, L_over_lambda * lambda_D
        self.stabilization, self.stab_factor, self.dc_factor = stabilization, stab_factor, dc_factor

    def bulk_concentration(self):
        eps = self.eps_r * epsilon_0
        I = eps * gas_constant * self.T / (2 * faraday_constant ** 2 * self.lambda_D ** 2)
        return I / self.valence ** 2

    def define_problem(self):
        cinf = self.bulk_concentration()
        self.set_scaling(spatial=self.lambda_D, temperature=self.T,
                         temporal=self.lambda_D ** 2 / self.D)
        set_electrostatic_scaling(self, potential=thermal_voltage(self.T), temperature=self.T,
                                  ion_concentration=cinf)
        self.add_mesh(LineMesh(N=self.N, size=self.L))
        # A stabilized Nernst-Planck is advective even at wind=0, because the migration drift is
        # not zero, so it does reference the velocity scale. D/lambda_D is that drift.
        self.set_scaling(velocity=self.D / self.lambda_D)
        ions = symmetric_electrolyte(cinf, self.valence, cation_diffusivity=self.D,
                                     anion_diffusivity=self.D)
        stab = self.stabilization
        if stab is not None:
            stab = ScalarTransportStabilization(stab, stab_factor=self.stab_factor,
                                                dc_factor=self.dc_factor)
        eqs = PoissonNernstPlanck(ions, relative_permittivity=self.eps_r, wind=0,
                                  temperature=self.T, stabilization=stab)
        eqs += ElectrodeBC(self.zeta) @ "left"
        eqs += (ElectrodeBC(0) + DirichletBC(c_cation=cinf, c_anion=cinf)) @ "right"
        # Boltzmann reference, evaluated on the SOLVED potential: if the transport equation is right
        # the concentrations must equal it pointwise.
        VT = thermal_voltage(self.T)
        boltz_p = cinf * exp(-self.valence * var("phi") / VT)
        boltz_m = cinf * exp(+self.valence * var("phi") / VT)
        eqs += IntegralObservables(
            _length=1, charge=var("charge_density"),
            err=subexpression((var("c_cation") - boltz_p) ** 2 + (var("c_anion") - boltz_m) ** 2),
            ref=subexpression(boltz_p ** 2 + boltz_m ** 2),
            n_cation=var("c_cation"), n_anion=var("c_anion"))
        self.add_equations(eqs @ "domain")


def _pnp_boltzmann_mismatch(psi0, N, T=298.15 * kelvin, lam=3 * nano * meter):
    prob = _PNPDiffuseLayer(zeta=psi0 * thermal_voltage(T), lambda_D=lam, T=T, N=N)
    with _fresh(prob) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
    return float(o["err"] / o["ref"]) ** 0.5


@pytest.mark.parametrize("psi0,tol", [(0.5, 3e-7), (2.0, 3e-5), (4.0, 1e-3)])
def test_pnp_equilibrium_is_poisson_boltzmann(psi0, tol):
    # The residual mismatch is DISCRETIZATION, not a difference of models: c_i is a C2 interpolant
    # while exp(-phi/VT) of a C2 phi is not, so the two cannot coincide on a finite mesh. The
    # tolerance therefore grows with the steepness of the layer, and the statement that this is
    # merely the mesh is made by the convergence test below, not by these numbers.
    assert _pnp_boltzmann_mismatch(psi0, 300) < tol


def test_pnp_boltzmann_mismatch_is_only_the_mesh():
    errs = [_pnp_boltzmann_mismatch(4.0, N) for N in (150, 300, 600)]
    orders = [numpy.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(orders) > 2.6, "should converge at ~3rd order for C2, got %s" % orders


def test_pnp_charge_matches_poisson_boltzmann_solution():
    # The same statement one level up: the integrated diffuse charge, which is what actually enters
    # a force or a capacitance, must agree with the Poisson-Boltzmann model.
    T, lam, psi0 = 298.15 * kelvin, 3 * nano * meter, 2.0
    zeta = psi0 * thermal_voltage(T)
    prob = _PNPDiffuseLayer(zeta=zeta, lambda_D=lam, T=T)
    with _fresh(prob) as p:
        p.solve()
        q_pnp = p.get_mesh("domain").evaluate_all_observables()["charge"]

    pb = _DiffuseLayer(model="poisson_boltzmann", zeta=zeta, lambda_D=lam, N=300)
    # The PB problem grounds the far end with dphi/dx=0; the PNP one pins phi=0 there. At 20 Debye
    # lengths the difference is below the tolerance asked for here.
    with _fresh(pb) as p:
        p.solve()
        q_pb = p.get_mesh("domain").evaluate_all_observables()["charge"]
    assert float(q_pnp / q_pb) == pytest.approx(1.0, rel=1e-4)


def test_pnp_conserves_ions_in_a_transient():
    # Blocking walls everywhere: the natural boundary condition of the weak form is zero TOTAL flux,
    # so nothing should need an interface equation for this to hold.
    T, lam, D = 298.15 * kelvin, 3 * nano * meter, 1e-9 * meter ** 2 / second
    eps_r = 78.4
    eps = eps_r * epsilon_0
    cinf = eps * gas_constant * T / (2 * faraday_constant ** 2 * lam ** 2)
    L = 20 * lam

    class _Closed(Problem):
        def define_problem(self):
            self.set_scaling(spatial=lam, temperature=T, temporal=lam ** 2 / D)
            set_electrostatic_scaling(self, potential=thermal_voltage(T), temperature=T,
                                      ion_concentration=cinf)
            self.add_mesh(LineMesh(N=100, size=L))
            ions = symmetric_electrolyte(cinf, 1, cation_diffusivity=D, anion_diffusivity=D)
            eqs = PoissonNernstPlanck(ions, relative_permittivity=eps_r, wind=0, temperature=T)
            eqs += SurfaceChargeBC(-0.01 * coulomb / meter ** 2) @ "left"
            eqs += ElectrodeBC(0) @ "right"
            eqs += IntegralObservables(n_cation=var("c_cation"), n_anion=var("c_anion"))
            self.add_equations(eqs @ "domain")

    with _fresh(_Closed()) as p:
        p.initialise()
        p.solve()
        n0 = p.get_mesh("domain").evaluate_all_observables()
        p.run(0.5 * float(lam ** 2 / D / second) * second, numouts=2, startstep=0.05 * lam ** 2 / D,
              temporal_error=None, out_initially=False)
        n1 = p.get_mesh("domain").evaluate_all_observables()
    for k in ("n_cation", "n_anion"):
        assert float(n1[k] / n0[k]) == pytest.approx(1.0, rel=1e-9), k


# ---------------------------------------------------------------------------------------------
# The gas/liquid pairing: a cheap model on one side, a resolved one on the other
# ---------------------------------------------------------------------------------------------

class _GasLiquid(Problem):
    r"""A charge-free gas gap next to an electrolyte whose Debye layer is solved for.

    The point of the pairing is that neither side knows what the other solves: the connection only
    requires an ElectricPotentialEquations, of which PoissonNernstPlanck's potential part, the
    Poisson-Boltzmann models and the plain dielectric are all instances. Swapping the liquid model
    changes nothing at the interface, which is what ``liquid_model`` exercises here.

    The check is Gauss's law across the whole liquid side. With the far field of the liquid
    field-free, integrating -div(eps grad phi) = rho_e over the liquid and combining with
    n.(D_gas - D_liq) = sigma_s (n outward from the liquid, i.e. -e_x) gives

        Q_diffuse + sigma_s + D_gas,x = 0

    i.e. the diffuse charge plus the interfacial charge exactly terminates the gas field. Every
    piece of the coupling -- the Lagrange multiplier, the surface charge sign, the two different
    permittivities, and the ion transport that produces Q_diffuse -- has to be right for this to
    hold, and none of them appears in the others' equations.
    """

    def __init__(self, *, liquid_model, sigma_s, V, lambda_D=3 * nano * meter, eps_r=78.4,
                 L_gas_over_lambda=60.0, L_liq_over_lambda=25.0, N=400, T=298.15 * kelvin,
                 D=1e-9 * meter ** 2 / second):
        super().__init__()
        self.liquid_model, self.sigma_s, self.V = liquid_model, sigma_s, V
        self.lambda_D, self.eps_r, self.T, self.D = lambda_D, eps_r, T, D
        self.L_gas, self.L_liq = L_gas_over_lambda * lambda_D, L_liq_over_lambda * lambda_D
        self.N = N

    def bulk_concentration(self):
        eps = self.eps_r * epsilon_0
        return eps * gas_constant * self.T / (2 * faraday_constant ** 2 * self.lambda_D ** 2)

    def define_problem(self):
        cinf = self.bulk_concentration()
        self.set_scaling(spatial=self.lambda_D, temperature=self.T,
                         temporal=self.lambda_D ** 2 / self.D)
        set_electrostatic_scaling(self, potential=thermal_voltage(self.T), temperature=self.T,
                                  ion_concentration=cinf)
        self.add_mesh(LineMesh(N=self.N, size=self.L_gas + self.L_liq, minimum=-self.L_gas,
                               name=lambda x: "gas" if x < 0 else "liquid"))

        # --- gas: the cheap model, a charge-free dielectric ---------------------------------------
        geqs = ElectricPotentialEquations(relative_permittivity=1.0, charge_density=0)
        geqs = geqs + ElectrodeBC(self.V) @ "left"
        geqs += IntegralObservables(_gas_length=1, Dx_gas=epsilon_0 * var("electric_field")[0])
        self.add_equations(geqs @ "gas")

        # --- liquid: the detailed model ------------------------------------------------------------
        if self.liquid_model == "pnp":
            ions = symmetric_electrolyte(cinf, 1, cation_diffusivity=self.D, anion_diffusivity=self.D)
            leqs = PoissonNernstPlanck(ions, relative_permittivity=self.eps_r, wind=0,
                                       temperature=self.T)
            reservoir = ElectrodeBC(0) + DirichletBC(c_cation=cinf, c_anion=cinf)
        elif self.liquid_model == "poisson_boltzmann":
            leqs = PoissonBoltzmannEquations(bulk_concentration=cinf, relative_permittivity=self.eps_r,
                                             temperature=self.T)
            reservoir = ElectrodeBC(0)
        elif self.liquid_model == "debye_huckel":
            leqs = DebyeHuckelEquations(debye_length=self.lambda_D, relative_permittivity=self.eps_r,
                                        temperature=self.T)
            reservoir = ElectrodeBC(0)
        else:
            raise ValueError(self.liquid_model)
        rho_e = leqs.get_equation_of_type(ElectricPotentialEquations).get_charge_density() \
            if not isinstance(leqs, ElectricPotentialEquations) else leqs.get_charge_density()
        leqs = leqs + reservoir @ "right"
        leqs += ElectricPotentialConnection(surface_charge_density=self.sigma_s) @ "gas_liquid"
        leqs += IntegralObservables(_liq_length=1, Q_diffuse=rho_e)
        self.add_equations(leqs @ "liquid")


@pytest.mark.parametrize("liquid_model", ["pnp", "poisson_boltzmann", "debye_huckel"])
def test_gas_liquid_pairing_satisfies_gauss(liquid_model):
    T = 298.15 * kelvin
    sigma_s = 2 * milli * coulomb / meter ** 2
    prob = _GasLiquid(liquid_model=liquid_model, sigma_s=sigma_s, V=3 * thermal_voltage(T), T=T)
    with _fresh(prob) as p:
        p.solve()
        g = p.get_mesh("gas").evaluate_all_observables()
        l = p.get_mesh("liquid").evaluate_all_observables()
    Dx_gas = g["Dx_gas"] / g["_gas_length"]   # uniform in the charge-free gas
    Q = l["Q_diffuse"]
    unit = coulomb / meter ** 2
    residual = float((Q + sigma_s + Dx_gas) / unit)
    reference = max(abs(float(Q / unit)), abs(float(sigma_s / unit)), abs(float(Dx_gas / unit)))
    assert abs(residual) < 1e-6 * reference, \
        "Gauss across the interface is violated: Q=%g sigma_s=%g D_gas=%g" % (
            float(Q / unit), float(sigma_s / unit), float(Dx_gas / unit))
    assert reference > 1e-6, "the test would be vacuous with no charge anywhere"


def test_gas_liquid_models_are_interchangeable():
    # Same problem, three liquid models, all at a potential low enough for the linearization to be
    # good. The interface must not care which one is used.
    T = 298.15 * kelvin
    kw = dict(sigma_s=0.05 * milli * coulomb / meter ** 2, V=0.2 * thermal_voltage(T), T=T)
    out = {}
    for m in ("pnp", "poisson_boltzmann", "debye_huckel"):
        with _fresh(_GasLiquid(liquid_model=m, **kw)) as p:
            p.solve()
            out[m] = float(p.get_mesh("gas").evaluate_all_observables()["Dx_gas"]
                           / (coulomb / meter))
    assert out["pnp"] == pytest.approx(out["poisson_boltzmann"], rel=1e-4)
    assert out["pnp"] == pytest.approx(out["debye_huckel"], rel=1e-2)


# ---------------------------------------------------------------------------------------------
# Leaky dielectric (Taylor-Melcher)
# ---------------------------------------------------------------------------------------------

class _LeakyBilayer(Problem):
    r"""Two leaky-dielectric layers in series between electrodes, no flow.

    In steady state the surface charge equation reduces to current continuity, so with
    n = +e_x (outward from layer A):

        phi_i = V*(sigma_2/d2) / (sigma_1/d1 + sigma_2/d2)
        q     = n.(D_B - D_A) = -eps_2*(V - phi_i)/d2 + eps_1*phi_i/d1

    which is the Maxwell-Wagner interfacial charge. It vanishes exactly when
    eps_1/sigma_1 == eps_2/sigma_2, i.e. when the two charge relaxation times match -- a sharp,
    parameter-free check that the sign and the prefactor are both right.
    """

    def __init__(self, *, L, V, eps_r, sigma_c, N=16):
        super().__init__()
        self.L, self.V, self.eps_r, self.sigma_c, self.N = L, V, eps_r, sigma_c, N

    def analytic(self):
        d = self.L / 2
        e1, e2 = (e * epsilon_0 for e in self.eps_r)
        s1, s2 = self.sigma_c
        phi_i = self.V * (s2 / d) / (s1 / d + s2 / d)
        q = -e2 * (self.V - phi_i) / d + e1 * phi_i / d
        return phi_i, q

    def define_problem(self):
        self.set_scaling(spatial=self.L, temporal=1 * second)
        set_electrostatic_scaling(self, potential=self.V)
        self.set_scaling(electric_conductivity=self.sigma_c[0],
                         surface_charge_density=epsilon_0 * self.V / self.L)
        self.add_mesh(LineMesh(N=self.N, size=self.L, name=lambda x: "A" if x < 0.5 else "B"))
        # Gauss-driven pairing: the bulk solves Gauss's law and carries the conductivity only so
        # that the interface can build the Ohmic current from it.
        eqsA = ElectricPotentialEquations(relative_permittivity=self.eps_r[0],
                                          conductivity=self.sigma_c[0])
        eqsA += ElectrodeBC(0) @ "left"
        ifeqs = SurfaceChargeConservation(name="qs", advection_velocity=0)
        ifeqs += ElectricPotentialConnection(surface_charge_density="qs")
        ifeqs += IntegralObservables(_len=1, qs=var("qs"), phi_i=var("phi"))
        eqsA += ifeqs @ "A_B"
        eqsB = ElectricPotentialEquations(relative_permittivity=self.eps_r[1],
                                          conductivity=self.sigma_c[1])
        eqsB += ElectrodeBC(self.V) @ "right"
        self.add_equations(eqsA @ "A")
        self.add_equations(eqsB @ "B")


@pytest.mark.parametrize("eps_r,sigma_ratio", [
    ((2.0, 8.0), 1.0),      # mismatched relaxation times: charge accumulates
    ((2.0, 8.0), 4.0),      # eps_2/eps_1 == sigma_2/sigma_1: the charge must vanish exactly
    ((5.0, 1.0), 0.25),
])
def test_maxwell_wagner_interfacial_charge(eps_r, sigma_ratio):
    L, V = 1 * milli * meter, 100 * volt
    s1 = 1e-8 * siemens / meter
    prob = _LeakyBilayer(L=L, V=V, eps_r=eps_r, sigma_c=(s1, sigma_ratio * s1))
    phi_i_exact, q_exact = prob.analytic()
    with _fresh(prob) as p:
        p.solve()
        o = p.get_mesh("A/A_B").evaluate_all_observables()
    assert float((o["phi_i"] / o["_len"]) / volt) == pytest.approx(float(phi_i_exact / volt), rel=1e-9)
    unit = coulomb / meter ** 2
    q = float((o["qs"] / o["_len"]) / unit)
    if abs(float(q_exact / unit)) < 1e-14:
        # Matched relaxation times: the charge is exactly zero, so an absolute bound is the only
        # meaningful statement. Referenced against what it would be for a mismatched pair.
        assert abs(q) < 1e-12
    else:
        assert q == pytest.approx(float(q_exact / unit), rel=1e-8)


def test_ohmic_bulk_solves_current_conservation():
    # The current-driven pairing: the bulk conserves current, so the interface connection means
    # current continuity and the potential divides by the resistances rather than the capacitances.
    L, V = 1 * milli * meter, 100 * volt
    s1, s2 = 1e-8 * siemens / meter, 4e-8 * siemens / meter

    class _P(Problem):
        def define_problem(self):
            self.set_scaling(spatial=L)
            set_electrostatic_scaling(self, potential=V)
            self.set_scaling(electric_conductivity=s1)
            self.add_mesh(LineMesh(N=16, size=L, name=lambda x: "A" if x < 0.5 else "B"))
            a = OhmicConductionEquations(conductivity=s1, relative_permittivity=2.0)
            a += ElectrodeBC(0) @ "left"
            a += (ElectricPotentialConnection()
                  + IntegralObservables(_len=1, phi_i=var("phi"))) @ "A_B"
            b = OhmicConductionEquations(conductivity=s2, relative_permittivity=8.0)
            b += ElectrodeBC(V) @ "right"
            self.add_equations(a @ "A")
            self.add_equations(b @ "B")

    with _fresh(_P()) as p:
        p.solve()
        o = p.get_mesh("A/A_B").evaluate_all_observables()
    expected = V * s2 / (s1 + s2)   # equal layer thicknesses
    assert float((o["phi_i"] / o["_len"]) / volt) == pytest.approx(float(expected / volt), rel=1e-9)


def test_charge_relaxation_flag_explains_itself():
    with pytest.raises(NotImplementedError, match="stiff"):
        OhmicConductionEquations(conductivity=1 * siemens / meter, permittivity=epsilon_0,
                                 charge_relaxation=True)


# ---------------------------------------------------------------------------------------------
# Per-species SUPG
# ---------------------------------------------------------------------------------------------

def _pnp_dofs(**kw):
    prob = _PNPDiffuseLayer(zeta=2.0 * thermal_voltage(298.15 * kelvin),
                            lambda_D=3 * nano * meter, N=120, **kw)
    with _fresh(prob) as p:
        p.solve()
        return numpy.array(p.get_current_dofs()[0])


def test_pnp_stabilization_is_off_by_default_and_at_zero_factor():
    """Invariant 2 of tests/test_stabilized_transport.py, restated for Nernst-Planck.

    Bitwise, not merely close: that is what catches a term added unconditionally, or a prefactor
    that does not reach every contribution.
    """
    plain = _pnp_dofs()
    # Both factors, as the sibling test in test_stabilized_transport.py does: stab_factor scales
    # tau, dc_factor scales nu_dc, and neither scales the other.
    zeroed = _pnp_dofs(stabilization="GLS+DC", stab_factor=0, dc_factor=0)
    assert plain.shape == zeroed.shape
    assert numpy.array_equal(plain, zeroed)


class _ManufacturedPNP(Problem):
    r"""A Poisson-Nernst-Planck solution that lies in C2 AND satisfies the equations strongly.

    Both are needed for a consistency test. Lying in the space makes the Galerkin solution exact;
    satisfying the PDE pointwise makes the strong residual identically zero, which is what SUPG,
    GLS and ASGS all multiply. So switching them on must not move a single dof.

    Construction: pick phi quadratic, then phi'' is constant, so Gauss forces c_+ - c_- to be the
    constant q = -eps*phi''/F. Any common quadratic profile may be added to both species on top of
    that without disturbing it. The Nernst-Planck source terms are then whatever the chosen fields
    leave over, supplied through ``reactions``, which the strong residual subtracts again.
    """

    def __init__(self, *, stabilization=None, N=8, L=1 * micro * meter, T=300 * kelvin,
                 D=1e-9 * meter ** 2 / second, eps_r=78.4, c0=1 * mol / meter ** 3,
                 B=0.2 * mol / meter ** 3):
        super().__init__()
        self.stabilization, self.N, self.L, self.T, self.D = stabilization, N, L, T, D
        self.eps_r, self.c0, self.B = eps_r, c0, B
        self.A = thermal_voltage(T)

    def _fields(self):
        x = var("coordinate_x")
        shape = x * (self.L - x) / self.L ** 2
        eps = self.eps_r * epsilon_0
        phi = self.A * shape
        # phi'' = -2A/L^2, so Gauss needs F*(c_p - c_m) = -eps*phi'' = 2*eps*A/L^2.
        q = 2 * eps * self.A / (self.L ** 2 * faraday_constant)
        return phi, self.c0 + q / 2 + self.B * shape, self.c0 - q / 2 + self.B * shape

    def define_problem(self):
        phi_e, cp, cm = self._fields()
        self.set_scaling(spatial=self.L, temperature=self.T, temporal=self.L ** 2 / self.D,
                         velocity=self.D / self.L)
        set_electrostatic_scaling(self, potential=self.A, temperature=self.T,
                                  ion_concentration=self.c0)
        self.add_mesh(LineMesh(N=self.N, size=self.L))
        m = lambda z: z * (self.D / (gas_constant * self.T)) * faraday_constant
        src = lambda c, z: -div(self.D * grad(c)) - div(m(z) * c * grad(phi_e))
        ions = [IonSpec("cation", +1, self.D), IonSpec("anion", -1, self.D)]
        eqs = PoissonNernstPlanck(ions, relative_permittivity=self.eps_r, wind=0, temperature=self.T,
                                  stabilization=self.stabilization, set_bulk_initial_conditions=False,
                                  reactions={"cation": src(cp, +1), "anion": src(cm, -1)})
        for b in ("left", "right"):
            eqs += DirichletBC(phi=phi_e, c_cation=cp, c_anion=cm) @ b
        eqs += InitialCondition(phi=phi_e, c_cation=cp, c_anion=cm)
        eqs += IntegralObservables(
            _length=1,
            err=subexpression((var("phi") - phi_e) ** 2 / self.A ** 2
                              + (var("c_cation") - cp) ** 2 / self.c0 ** 2
                              + (var("c_anion") - cm) ** 2 / self.c0 ** 2))
        self.add_equations(eqs @ "domain")


def _manufactured_pnp(stabilization=None):
    with _fresh(_ManufacturedPNP(stabilization=stabilization)) as p:
        p.solve()
        o = p.get_mesh("domain").evaluate_all_observables()
        return numpy.array(p.get_current_dofs()[0]), float(o["err"] / o["_length"])


def test_manufactured_pnp_is_solved_exactly():
    # If this fails, the consistency test below proves nothing.
    _, err = _manufactured_pnp()
    assert err == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("stabilization", ["SUPG", "GLS", "ASGS"])
def test_pnp_stabilization_is_consistent(stabilization):
    """SUPG/GLS/ASGS all multiply the strong residual, so where it vanishes they must return the
    unstabilized answer -- including with the per-species migration wind and the migration reaction
    rate in tau, neither of which is zero here."""
    plain, _ = _manufactured_pnp()
    stabilized, err = _manufactured_pnp(stabilization)
    assert err == pytest.approx(0.0, abs=1e-14)
    rel = numpy.max(numpy.abs(stabilized - plain)) / numpy.max(numpy.abs(plain))
    assert rel < 1e-8, "stabilization moved a zero-residual solution by %g" % rel


def test_migration_wind_is_per_species():
    """The point of the per-field hook: a cation and an anion drift in opposite directions.

    Sized from the fluid velocity alone -- which is zero here -- tau would see no advection at all
    and the SUPG weight would be identically zero, which is what the base class did before
    stabilization_wind_for_field existed.
    """
    D_p, D_m = 1e-9 * meter ** 2 / second, 2e-9 * meter ** 2 / second
    ions = symmetric_electrolyte(1 * mol / meter ** 3, 1, cation_diffusivity=D_p,
                                 anion_diffusivity=D_m)
    eqs = NernstPlanckEquations(ions, wind=0, temperature=300 * kelvin, stabilization="SUPG")
    a_p = eqs.stabilization_wind_for_field("c_cation")
    a_m = eqs.stabilization_wind_for_field("c_anion")
    assert not (a_p - a_m).is_zero(), "the two species must not share a wind"
    # Opposite charges drift oppositely, and the Einstein relation makes the magnitudes scale with
    # the diffusivities, so a_m = -2*a_p exactly.
    assert (a_m + 2 * a_p).is_zero()
    # The shared part is still the fluid velocity, i.e. zero here.
    assert is_zero(convert_to_expression(eqs.stabilization_wind()))


def test_migration_contributes_a_reaction_rate_to_tau():
    # -div(m c grad(phi)) is an advective part PLUS a term linear in c, which tau has a slot for.
    # Zero for a plain advection-diffusion equation, so the default must stay zero.
    from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
    ad = AdvectionDiffusionEquations("c", diffusivity=1, stabilization="SUPG")
    assert is_zero(convert_to_expression(ad.stabilization_reaction_rate("c")))

    ions = symmetric_electrolyte(1 * mol / meter ** 3, 1,
                                 cation_diffusivity=1e-9 * meter ** 2 / second,
                                 anion_diffusivity=1e-9 * meter ** 2 / second)
    eqs = NernstPlanckEquations(ions, wind=0, temperature=300 * kelvin, stabilization="SUPG")
    r_p = eqs.stabilization_reaction_rate("c_cation")
    r_m = eqs.stabilization_reaction_rate("c_anion")
    assert not is_zero(convert_to_expression(r_p))
    assert (r_p + r_m).is_zero(), "opposite valences give opposite rates"
