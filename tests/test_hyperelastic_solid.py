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

# The hyperelastic solid laws (HyperelasticSolidConstitutiveLaw and its subclasses) and the
# Kelvin-Voigt viscous extension, against an independent reference: the stress of a homogeneous
# deformation with principal stretches l_a follows directly from the strain energy W(I1,I2,J),
#     S_a = 2 W_1 + 2 W_2 (I1 - l_a^2) + W_J J / l_a^2 - p / l_a^2,     sigma_a = l_a^2 S_a / J,
# which is differentiated here with sympy, not with pyoomph. Every coordinate system and dimension
# is covered, since each maps to a different set of principal stretches:
#   1d cartesian      : (L,1,1)  bar, uniaxial strain
#   1d axisymmetric   : (L,L,1)  disc (r only), equibiaxial in r,phi
#   1d radialsymmetric: (L,L,L)  sphere, equitriaxial
#   2d cartesian      : (L,M,1)  plane strain, uniaxial stress in x
#   2d axisymmetric   : (M,M,L)  cylinder, uniaxial stress in z
#   3d cartesian      : (L,M,M)  brick, uniaxial stress in x
# with the loaded stretch L (Cauchy traction T) and the traction-free lateral stretch M. The
# deformation is homogeneous, so the C2 discretization is exact and the agreement is to solver
# precision. Incompressible laws are only tested where they allow a deformation (not in 1d).

import numpy
import pytest
import sympy
from scipy.integrate import solve_ivp

from pyoomph import Problem, LineMesh, RectangularQuadMesh, CuboidBrickMesh, DirichletBC
from pyoomph.expressions import var, scale_factor, square_root
from pyoomph.expressions.units import milli, meter, mega, pascal, kilogram, second
from pyoomph.equations.solid import (DeformableSolidEquations, SolidTraction, HyperelasticSolidConstitutiveLaw,
                                     NeoHookeanSolidConstitutiveLaw, IncompressibleNeoHookeanSolidConstitutiveLaw,
                                     MooneyRivlinSolidConstitutiveLaw, YeohSolidConstitutiveLaw,
                                     GentSolidConstitutiveLaw, KelvinVoigtSolidConstitutiveLaw)

E_ND, NU = 1.0, 0.3
LAM = E_ND * NU / ((1 + NU) * (1 - 2 * NU))
MU = E_ND / (2 * (1 + NU))
KAPPA = 5.0


def _svk_energy(I1, I2, J, lam=LAM, mu=MU):
    # St. Venant-Kirchhoff in invariants: tr(E)=(I1-3)/2, E:E=(tr(C^2)-2 tr(C)+3)/4, tr(C^2)=I1^2-2 I2
    return lam / 2 * ((I1 - 3) / 2)**2 + mu * (I1**2 - 2 * I2 - 2 * I1 + 3) / 4


def _iso(Wiso):
    """Nearly incompressible: isochoric invariants plus kappa/2*(J-1)^2"""
    return lambda I1, I2, J: Wiso(I1 * J**sympy.Rational(-2, 3), I2 * J**sympy.Rational(-4, 3)) + KAPPA / 2 * (J - 1)**2


_W_MR = lambda I1, I2: 0.3 * (I1 - 3) + 0.1 * (I2 - 3)
_W_YEOH = lambda I1, I2: 0.2 * (I1 - 3) + 0.05 * (I1 - 3)**2 + 0.01 * (I1 - 3)**3
_W_GENT = lambda I1, I2: -0.4 * 2.0 / 2 * sympy.log(1 - (I1 - 3) / 2.0)

# name: (pyoomph law factory taking a stress scale, reference energy W(I1,I2,J) in sympy, incompressible)
MATERIALS = {
    "neo_hookean": (lambda S: NeoHookeanSolidConstitutiveLaw(E=E_ND * S, nu=NU),
                    lambda I1, I2, J: MU / 2 * (I1 - 3) - MU * sympy.log(J) + LAM / 2 * sympy.log(J)**2, False),
    "incompressible_neo_hookean": (lambda S: IncompressibleNeoHookeanSolidConstitutiveLaw(E=E_ND * S),
                                   lambda I1, I2, J: E_ND / 6 * (I1 - 3), True),
    "mooney_rivlin": (lambda S: MooneyRivlinSolidConstitutiveLaw(0.3 * S, 0.1 * S),
                      lambda I1, I2, J: _W_MR(I1, I2), True),
    "mooney_rivlin_kappa": (lambda S: MooneyRivlinSolidConstitutiveLaw(0.3 * S, 0.1 * S, kappa=KAPPA * S),
                            _iso(_W_MR), False),
    "yeoh": (lambda S: YeohSolidConstitutiveLaw(0.2 * S, 0.05 * S, 0.01 * S),
             lambda I1, I2, J: _W_YEOH(I1, I2), True),
    "yeoh_kappa": (lambda S: YeohSolidConstitutiveLaw(0.2 * S, 0.05 * S, 0.01 * S, kappa=KAPPA * S), _iso(_W_YEOH), False),
    "gent": (lambda S: GentSolidConstitutiveLaw(0.4 * S, 2.0), lambda I1, I2, J: _W_GENT(I1, I2), True),
    "gent_kappa": (lambda S: GentSolidConstitutiveLaw(0.4 * S, 2.0, kappa=KAPPA * S), _iso(_W_GENT), False),
    # the generic base class with a user-given energy (here: St. Venant-Kirchhoff, incl. the I2 dependence)
    "generic_svk": (lambda S: HyperelasticSolidConstitutiveLaw(lambda I1, I2, J: S * _svk_energy(I1, I2, J)),
                    _svk_energy, False),
}

GEOMETRIES = ["1d_cartesian", "1d_axisymmetric", "1d_radialsymmetric", "2d_cartesian", "2d_axisymmetric", "3d_cartesian"]


def _principal_stretches(geom, L, M):
    """Principal stretches (l1,l2,l3), the index of the loaded and of the traction-free direction (or None),
    and the expected positions of the reference point (1), (1,1) or (1,1,1)."""
    return {"1d_cartesian": ((L, 1, 1), 0, None, [L]),
            "1d_axisymmetric": ((L, L, 1), 0, None, [L]),
            "1d_radialsymmetric": ((L, L, L), 0, None, [L]),
            "2d_cartesian": ((L, M, 1), 0, 1, [L, M]),
            "2d_axisymmetric": ((M, M, L), 2, 0, [M, L]),
            "3d_cartesian": ((L, M, M), 0, 1, [L, M, M])}[geom]


def _cauchy_stresses(W, incompressible, ls, p, eta=0, zeta=0, ldots=(0, 0, 0)):
    I1s, I2s, Js = sympy.symbols("I1 I2 J")
    Wsym = W(I1s, I2s, 1 if incompressible else Js)
    C = [l**2 for l in ls]
    I1, I2, J = sum(C), C[0] * C[1] + C[1] * C[2] + C[0] * C[2], ls[0] * ls[1] * ls[2]
    subs = {I1s: I1, I2s: I2, Js: J}
    W1, W2, WJ = [sympy.diff(Wsym, s).subs(subs) for s in (I1s, I2s, Js)]
    Edot = [l * ld for l, ld in zip(ls, ldots)]
    res = []
    for a in range(3):
        S = 2 * W1 + 2 * W2 * (I1 - C[a]) + (-p if incompressible else WJ * J) / C[a]
        S += 2 * eta * Edot[a] + zeta * sum(Edot)
        res.append(C[a] * S / J)
    return res


def _reference_static(material, geom, T):
    _, W, incompressible = MATERIALS[material]
    L, M, p = sympy.symbols("L M p")
    ls, iload, ifree, expected = _principal_stretches(geom, L, M)
    sig = _cauchy_stresses(W, incompressible, ls, p)
    eqs, unknowns, guess = [sig[iload] - T], [L], [1]
    if ifree is not None:
        eqs.append(sig[ifree]); unknowns.append(M); guess.append(1)
    if incompressible:
        eqs.append(ls[0] * ls[1] * ls[2] - 1); unknowns.append(p); guess.append(0)
    sol = sympy.nsolve(eqs, unknowns, guess, tol=1e-25, prec=30)
    vals = {u: float(sol[i]) for i, u in enumerate(unknowns)}
    return [float(sympy.sympify(e).subs(vals)) for e in expected]


class _HyperelasticProblem(Problem):
    def __init__(self, material, geom, T, dimensional=False, viscosity=None):
        super().__init__()
        self.material, self.geom, self.T, self.dimensional, self.viscosity = material, geom, T, dimensional, viscosity
        self.newton_solver_tolerance = 1e-11  # the default 1e-8 would limit the agreement to ~1e-8

    def define_problem(self):
        dim, coordsys = self.geom.split("_")
        if self.dimensional:
            Lsc, Esc, tsc = 1 * milli * meter, 1 * mega * pascal, 1 * second
            self.set_scaling(spatial=Lsc, temporal=tsc, mass_density=Esc * tsc**2 / Lsc**2)
            self.set_scaling(pressure=Esc)  # scale of the pressure field of incompressible laws
        else:
            Lsc, Esc, tsc = 1, 1, 1
        self.set_coordinate_system(coordsys)
        if dim == "1d":
            self += LineMesh(N=3, size=Lsc, name="solid")
        elif dim == "2d":
            self += RectangularQuadMesh(N=2, size=Lsc, name="solid")
        else:
            self += CuboidBrickMesh(N=2, size=Lsc, domain_name="solid")
        law = MATERIALS[self.material][0](Esc)
        if self.viscosity is not None:
            law = KelvinVoigtSolidConstitutiveLaw(law, self.viscosity[0] * Esc * tsc, self.viscosity[1] * Esc * tsc)
        eqs = DeformableSolidEquations(law, coordinate_space="C2")
        # symmetry planes (and the axis) at x=0, y=0, z=0
        eqs += DirichletBC(mesh_x=0) @ "left"
        if dim != "1d":
            eqs += DirichletBC(mesh_y=0) @ "bottom"
        if dim == "3d":
            eqs += DirichletBC(mesh_z=0) @ "back"
        loaded = "top" if self.geom == "2d_axisymmetric" else "right"
        eqs += SolidTraction(self.T * Esc * var("normal")) @ loaded
        self += eqs @ "solid"

    def corner_position(self):
        n = int(self.geom[0])
        row = self.get_mesh("solid").evaluate_at_points([[1.0] * n], with_position=True)[0]
        return numpy.array(row[-n:])


def _static_cases():
    cases = []
    for mat, (_, _, incompressible) in MATERIALS.items():
        for geom in GEOMETRIES:
            if incompressible and geom.startswith("1d"):
                continue  # no deformation possible
            if geom == "3d_cartesian" and mat not in ("neo_hookean", "incompressible_neo_hookean", "mooney_rivlin", "generic_svk"):
                continue  # 3d is no new code path, and the isochoric laws are expensive to compile in 3d
            cases.append((mat, geom, 0.15, False))
    # with physical units
    for mat in ["neo_hookean", "incompressible_neo_hookean", "gent_kappa"]:
        for geom in ["1d_radialsymmetric", "2d_axisymmetric"] + (["3d_cartesian"] if mat != "gent_kappa" else []):
            if not (MATERIALS[mat][2] and geom.startswith("1d")):
                cases.append((mat, geom, 0.15, True))
    # compression
    for mat in ["neo_hookean", "mooney_rivlin_kappa", "incompressible_neo_hookean"]:
        for geom in ["1d_axisymmetric", "2d_cartesian", "2d_axisymmetric"]:
            if not (MATERIALS[mat][2] and geom.startswith("1d")):
                cases.append((mat, geom, -0.15, False))
    return cases


@pytest.mark.parametrize("material,geom,T,dimensional", _static_cases())
def test_hyperelastic_homogeneous(material, geom, T, dimensional):
    with _HyperelasticProblem(material, geom, T, dimensional) as problem:
        problem.solve()
        pos = problem.corner_position()
    ref = _reference_static(material, geom, T)
    assert numpy.allclose(pos, ref, rtol=0, atol=1e-9), (pos, ref)
    assert numpy.max(numpy.abs(numpy.array(ref) - 1)) > 0.005  # the test must not be trivial


def test_neo_hookean_matches_linear_elasticity_for_small_strains():
    # neo-Hookean and St. Venant-Kirchhoff both reduce to Hooke's law with the same E, nu
    T = 1e-5
    lin = 1 + T * (1 + NU) * (1 - 2 * NU) / (E_ND * (1 - NU))  # 1d cartesian: uniaxial strain, modulus lam+2mu
    with _HyperelasticProblem("neo_hookean", "1d_cartesian", T) as problem:
        problem.solve()
        assert abs(problem.corner_position()[0] - lin) < 1e-9


# ---------------------------------------------------------------- Kelvin-Voigt

ETA, ZETA = 0.5, 0.2


def _reference_creep(material, geom, T, tend):
    """Creep under a suddenly applied traction without inertia: the viscous stress is linear in the stretch
    rates, so sigma(y, ydot)=(T,0) gives an ODE for the unknown stretches y."""
    _, W, incompressible = MATERIALS[material]
    assert not incompressible
    L, M, Ld, Md = sympy.symbols("L M Ld Md")
    ls, iload, ifree, expected = _principal_stretches(geom, L, M)
    ld = tuple({L: Ld, M: Md, 1: 0}[l] for l in ls)
    sig = _cauchy_stresses(W, False, ls, 0, ETA, ZETA, ld)
    eqs, y, yd = [sig[iload] - T], [L], [Ld]
    if ifree is not None:
        eqs.append(sig[ifree]); y.append(M); yd.append(Md)
    A = sympy.Matrix([[sympy.diff(e, v) for v in yd] for e in eqs])
    b = sympy.Matrix([e.subs({v: 0 for v in yd}) for e in eqs])
    Af, bf = sympy.lambdify([y], A, "numpy"), sympy.lambdify([y], b, "numpy")
    rhs = lambda t, yy: numpy.linalg.solve(numpy.array(Af(yy), dtype=float), -numpy.array(bf(yy), dtype=float).flatten())
    sol = solve_ivp(rhs, (0, tend), [1.0] * len(y), rtol=1e-12, atol=1e-14)
    vals = {k: float(v) for k, v in zip(y, sol.y[:, -1])}
    return [float(sympy.sympify(e).subs(vals)) for e in expected]


@pytest.mark.parametrize("dimensional", [False, True], ids=["nondim", "dim"])
@pytest.mark.parametrize("geom", ["1d_cartesian", "1d_radialsymmetric", "2d_cartesian", "2d_axisymmetric"])
def test_kelvin_voigt_creep(geom, dimensional):
    T, tend = 0.15, 1.0
    ref = _reference_creep("neo_hookean", geom, T, tend)
    errs = []
    for dt in [0.02, 0.01]:
        with _HyperelasticProblem("neo_hookean", geom, T, dimensional, viscosity=(ETA, ZETA)) as problem:
            u = second if dimensional else 1
            problem.newton_solver_tolerance = 1e-10  # the viscous residual scales with 1/dt, round-off stalls at ~1e-11
            problem.run(tend * u, timestep=dt * u)
            errs.append(numpy.max(numpy.abs(problem.corner_position() - ref)))
    # second order in time (BDF2), and the creep is far from complete at tend
    assert errs[1] < 2e-5
    assert errs[0] / errs[1] > 3
