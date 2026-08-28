#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
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

# Scanning SEVERAL azimuthal modes in one solve_eigenproblem call, on a problem that has an axis of
# symmetry.
#
# AxisymmetryBC enforces the axis conditions in two different ways. As long as only the base mode has
# been solved they are strong (pinned) Dirichlet conditions; the first m != 0 eigenproblem releases them
# and imposes the same zeros by a matrix manipulator instead, because for |m|=1 a different set of
# fields has to vanish on the axis. Releasing them renumbers the equations, so ndof is larger from then
# on -- 201 -> 217 for the four-by-four mesh below.
#
# A mode list starting at m=0 therefore used to solve its first block in the OLD numbering and every
# later block in the new one, and stacking them died in numpy with
#     ValueError: all the input array dimensions except for the concatenation axis must match exactly,
#     but along dimension 1, the array at index 0 has size 201 and the array at index 1 has size 217
# which is what the bifurcation GUI runs into: it always prepends the base mode to whatever modes are
# requested, so its very first mode scan on a fresh problem hit exactly this.
#
# Single-mode solves never showed it (each is self-consistent), and neither did a second scan (the
# conditions stay released once dropped), which is why it survived: it needs a FRESH problem and a list.
#
# scipy rather than SLEPc as the eigensolver, so this needs no complex PETSc build.

import numpy

from pyoomph import Problem, DirichletBC
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.generic import AxisymmetryBC
from pyoomph.expressions import vector
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _AxisymStokes(Problem):
    """Stokes flow driven by a body force in a box whose left boundary is the axis.

    The top is left free so that the natural condition there pins the pressure level; with velocities
    imposed all around, the base state would not be unique.
    """

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(RectangularQuadMesh(size=[1.0, 1.0], N=[4, 4]))
        eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1,
                                    bulkforce=vector(0, -self.get_global_parameter("g")))
        eqs += AxisymmetryBC(verbose=False) @ "left"
        eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_phi=0) @ "right"
        eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_phi=0) @ "bottom"
        self += eqs @ "domain"


def _solved(tmp_path):
    problem = _AxisymStokes()
    problem.set_output_directory(str(tmp_path))
    problem.set_linear_solver("superlu")
    problem.set_eigensolver("scipy")
    problem.quiet()
    problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
    problem.get_global_parameter("g").value = 1.0
    return problem


def test_mode_list_scan_on_a_fresh_problem(tmp_path):
    """A list of modes, base mode first, on a problem that has never solved an m != 0 eigenproblem."""
    modes = [0, 1, 2]
    neigen = 3
    with _solved(tmp_path) as problem:
        problem.solve()
        values, vectors = problem.solve_eigenproblem(neigen, azimuthal_m=modes)
        recorded = problem.get_last_eigenmodes_m()

        assert len(values) == neigen*len(modes), len(values)
        assert len(recorded) == len(values), "one mode per eigenvalue"
        assert set(int(m) for m in recorded) == set(modes), recorded
        # One numbering for the whole scan: every eigenvector has to be a vector of the problem as it
        # stands now, or it cannot be pushed into the dofs to be plotted or continued from.
        assert vectors.shape == (len(values), problem.ndof()), \
            "{:s} for {:d} dofs".format(str(vectors.shape), problem.ndof())

        # The scan must agree with the modes solved one at a time. In particular the base mode: it is
        # the one whose numbering the fix changes, and the manipulator has to impose exactly what the
        # pinning did.
        for m in modes:
            from_scan = sorted(float(numpy.real(values[i])) for i in range(len(values))
                               if int(recorded[i]) == m)
            single, _ = problem.solve_eigenproblem(neigen, azimuthal_m=m)
            alone = sorted(float(numpy.real(v)) for v in single)
            assert numpy.allclose(from_scan, alone, rtol=1e-8, atol=1e-8), \
                "m={:d}: scan {:s} but single solve {:s}".format(m, str(from_scan), str(alone))
