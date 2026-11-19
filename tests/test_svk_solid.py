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

# StVenantKirchhoffSolidConstitutiveLaw against analytic homogeneous finite deformations, in every
# coordinate system and with/without physical units. With the stretch L and g=(L^2-1)/2, a Cauchy
# traction T on the loaded boundary gives
#   1d cartesian      : bar, uniaxial strain          T = L*(lam+2mu)*g
#   1d axisymmetric   : disc (r only), equibiaxial    T = (2lam+2mu)*g
#   1d radialsymmetric: sphere, equitriaxial          T = (3lam+2mu)*g/L
#   2d cartesian      : plane strain, uniaxial stress T = L1/L2*S11, S22=0
#   2d axisymmetric   : cylinder, uniaxial stress     T = Lz/Lr^2*Szz, Srr=Sphiphi=0
# The deformation is homogeneous, so the C2 discretization reproduces it exactly and the solution
# matches to solver precision. The load is large enough (stretches of up to ~11%) that the result
# is far off the linear-elastic one, i.e. the nonlinear part of the law is tested as well.

import numpy
import pytest
from scipy.optimize import brentq

from pyoomph import Problem, LineMesh, RectangularQuadMesh, DirichletBC
from pyoomph.expressions import var, matrix, scale_factor, square_root
from pyoomph.expressions.units import milli, meter, mega, pascal, kilogram
from pyoomph.equations.solid import (DeformableSolidEquations, SolidTraction,
                                     StVenantKirchhoffSolidConstitutiveLaw)

E_ND, NU, T_ND = 1.0, 0.3, 0.15
LAM = E_ND * NU / ((1 + NU) * (1 - 2 * NU))
MU = E_ND / (2 * (1 + NU))

CASES = ["1d_cartesian", "1d_axisymmetric", "1d_radialsymmetric", "2d_cartesian", "2d_axisymmetric"]


def _analytic_stretches(case):
    g = lambda L: (L**2 - 1) / 2
    solve = lambda f: brentq(f, 0.9, 1.5, xtol=1e-15)
    if case == "1d_cartesian":
        return [solve(lambda L: L * (LAM + 2 * MU) * g(L) - T_ND)]
    if case == "1d_axisymmetric":
        return [solve(lambda L: (2 * LAM + 2 * MU) * g(L) - T_ND)]
    if case == "1d_radialsymmetric":
        return [solve(lambda L: (3 * LAM + 2 * MU) * g(L) / L - T_ND)]
    if case == "2d_cartesian":  # load in x
        L2 = lambda L1: numpy.sqrt(1 - 2 * LAM / (LAM + 2 * MU) * g(L1))
        L1 = solve(lambda L1: L1 / L2(L1) * 4 * MU * (LAM + MU) / (LAM + 2 * MU) * g(L1) - T_ND)
        return [L1, L2(L1)]
    if case == "2d_axisymmetric":  # x=r, y=z, load in z
        Lr = lambda Lz: numpy.sqrt(1 - LAM / (LAM + MU) * g(Lz))
        Lz = solve(lambda Lz: Lz / Lr(Lz)**2 * E_ND * g(Lz) - T_ND)
        return [Lr(Lz), Lz]
    raise ValueError(case)


class _SVKProblem(Problem):
    def __init__(self, case, dimensional=False, mode="traction"):
        super().__init__()
        self.case, self.dimensional, self.mode = case, dimensional, mode
        self.growth = 1.3

    def define_problem(self):
        dim, coordsys = self.case.split("_")
        if self.dimensional:
            Lsc, Esc = 1 * milli * meter, 1 * mega * pascal
            self.set_scaling(spatial=Lsc, mass_density=1000 * kilogram / meter**3)
            self.set_scaling(temporal=Lsc * square_root(scale_factor("mass_density") / Esc))
        else:
            Lsc, Esc = 1, 1
        self.set_coordinate_system(coordsys)
        if dim == "1d":
            self += LineMesh(N=4, size=Lsc, name="solid")
        else:
            self += RectangularQuadMesh(N=3, size=Lsc, name="solid")

        growth = 1
        active = None
        if self.mode == "growth":
            growth = self.growth
        elif self.mode == "active":
            active = matrix([[1.2, 0, 0], [0, 0.9, 0], [0, 0, 1]])
        law = StVenantKirchhoffSolidConstitutiveLaw(E=E_ND * Esc, nu=NU, active_deformation=active)
        eqs = DeformableSolidEquations(law, coordinate_space="C2", isotropic_growth_factor=growth)

        # symmetry planes (and the axis) at x=0 and y=0
        eqs += DirichletBC(mesh_x=0) @ "left"
        if dim == "2d":
            eqs += DirichletBC(mesh_y=0) @ "bottom"
        if self.mode == "traction":
            loaded = "right" if (dim == "1d" or coordsys == "cartesian") else "top"
            eqs += SolidTraction(T_ND * Esc * var("normal")) @ loaded
        self += eqs @ "solid"

    def corner_position(self):
        """Nondimensional position of the reference point (1) or (1,1)."""
        pt = [1.0] if self.case.startswith("1d") else [1.0, 1.0]
        row = self.get_mesh("solid").evaluate_at_points([pt], with_position=True)[0]
        return numpy.array(row[-len(pt):])


@pytest.mark.parametrize("dimensional", [False, True], ids=["nondim", "dim"])
@pytest.mark.parametrize("case", CASES)
def test_svk_homogeneous_traction(case, dimensional):
    with _SVKProblem(case, dimensional) as problem:
        problem.solve()
        pos = problem.corner_position()
    assert numpy.allclose(pos, _analytic_stretches(case), rtol=0, atol=1e-10)


@pytest.mark.parametrize("case,dim_eff", [("1d_cartesian", 1), ("1d_axisymmetric", 2), ("1d_radialsymmetric", 3),
                                          ("2d_cartesian", 2), ("2d_axisymmetric", 3)])
def test_svk_stress_free_growth(case, dim_eff):
    # A volumetric growth factor f without load must lead to the stress-free stretch f^(1/dim)
    with _SVKProblem(case, mode="growth") as problem:
        problem.solve()
        pos = problem.corner_position()
        expected = problem.growth**(1 / dim_eff)
    assert numpy.allclose(pos, expected, rtol=0, atol=1e-8)


def test_svk_active_deformation():
    # The reference metric is inverted numerically here; the stress-free state is x=F_a*X
    with _SVKProblem("2d_cartesian", mode="active") as problem:
        problem.solve()
        pos = problem.corner_position()
    assert numpy.allclose(pos, [1.2, 0.9], rtol=0, atol=1e-8)


def test_svk_rejects_incompressible():
    with pytest.raises(ValueError):
        StVenantKirchhoffSolidConstitutiveLaw(E=1, nu=0.5)
