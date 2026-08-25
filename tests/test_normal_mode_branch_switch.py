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

# Branch switching, and the normal form behind it, at a bifurcation of an azimuthal m != 0 (or a
# Cartesian k != 0) normal mode.
#
# There is nothing to switch to. The eigenfunction that goes unstable varies as exp(i*m*phi), so the
# solution born there is genuinely three-dimensional and cannot be represented in an axisymmetric dof
# vector at any amplitude. Nothing about that announces itself: the eigenvector has the base-mode
# length, every matrix contraction goes through, and what came back was a plausible "pitchfork"
# assembled from the BASE-mode Hessian contracted with an m=1 eigenvector -- the coefficient of
# nothing -- followed by a "switch" that converged back onto the branch it started from. So it is
# refused, at Problem level, where a plain script meets it just as the bifurcation GUI does.
#
# The discrimination is what is actually under test, not the refusal: a guard that refused everything
# would look identical from the m=1 side alone. The same problem, the same call, at an m=0 eigenvalue
# must go through -- so the m=0 spectrum of the very same state is the control.
#
# u_t = laplace(u) + lam*u - u^3, axisymmetric, u=0 on the outer boundaries: the trivial state loses
# stability to the first m=1 Dirichlet mode at a lam that is a genuine symmetry-breaking bifurcation.
# The same system as tests/mpi_bifurcation_worker.py's AzimuthalReactionProblem, kept small here
# because generating the azimuthal and Hessian code is what costs the time, not solving.
#
# Only ONE Problem per process (a second segfaults in the JIT loader, see
# tests/test_multiple_problems.py), hence one test function.

import numpy
import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var_and_test, grad, partial_t, weak, axisymmetric
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class ReactionDiffusionEquations(Equations):
    """u_t = laplace(u) + lam*u - u^3. The odd nonlinearity makes every crossing a symmetry breaking."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(self.lam*u - u**3, v))


class AzimuthalReactionProblem(Problem):
    def __init__(self, N=5):
        super().__init__()
        self.N = N
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = ReactionDiffusionEquations(self.lam)
        eqs += DirichletBC(u=0) @ ["right", "top", "bottom"]
        self += eqs @ "domain"


def _dummy_normal_form(p):
    """A normal form with working predictors, so that only the guard itself can refuse the switch.

    switch_branch computes one when none is given, and would then be refused by classify_bifurcation
    rather than by its own guard - which is the path that matters here, because the bifurcation GUI
    always hands one in and would otherwise never meet the guard at all.
    """
    return {"type": "pitchfork",
            "param_predictor": lambda eps: 0.0,
            "perturbation_predictor": lambda eps: numpy.zeros(p.ndof())}


def test_normal_mode_bifurcations_refuse_branch_switching(tmp_path):
    with AzimuthalReactionProblem() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.initialise()
        p.lam.value = 10.0
        p.solve()

        # ------------------------------------------------------------------ at the m=1 bifurcation
        for _lam, _ev in p.find_bifurcation_via_eigenvalues("lam", initstep=4.0, do_solve=False,
                                                            neigen=3, azimuthal_m=1, epsilon=1e-2):
            pass
        p.activate_bifurcation_tracking("lam", bifurcation_type="azimuthal")
        p.solve()
        assert p.get_bifurcation_tracking_mode() == "azimuthal"
        assert abs(numpy.real(p.get_last_eigenvalues()[0])) < 1e-8, "the tracker converged"

        assert p._critical_normal_mode(0) == ("m", 1.0), p._critical_normal_mode(0)

        with pytest.raises(RuntimeError) as ei:
            p.classify_bifurcation("lam")
        assert "azimuthal mode m = 1" in str(ei.value), str(ei.value)

        # The GUI's path: the normal form is handed in, so classify_bifurcation is never reached and
        # switch_branch's own guard is the only thing that can refuse.
        with pytest.raises(RuntimeError) as ei:
            p.switch_branch("lam", normal_form=_dummy_normal_form(p))
        assert "azimuthal mode m = 1" in str(ei.value), str(ei.value)
        assert p.get_bifurcation_tracking_mode() == "azimuthal", \
            "a refusal must not have deactivated the tracker on its way out"

        # A Hopf's orbit is the same story - a rotating wave, not a standing oscillation in these
        # dofs - and used to come back as "Hopf tracking not activated", which is not the problem.
        with pytest.raises(RuntimeError) as ei:
            p.switch_to_hopf_orbit()
        assert "azimuthal mode m = 1" in str(ei.value), str(ei.value)

        lam_c = float(p.lam.value)
        p.deactivate_bifurcation_tracking()
        assert numpy.isclose(float(p.lam.value), lam_c), "and it is still sitting on the bifurcation"

        # ------------------------------------------------------------- the control, same state
        # Nothing above may be read as "normal-mode problems refuse everything": at an m=0 eigenvalue
        # of the very same state the guard has to stand aside, and only the modes tell the two apart.
        p.solve_eigenproblem(3, shift=0.5, azimuthal_m=[0, 1], quiet=True)
        modes = p.get_last_eigenmodes_m()
        assert modes is not None and len(modes) == len(p.get_last_eigenvalues())
        base = [i for i in range(len(modes)) if int(modes[i]) == 0]
        azim = [i for i in range(len(modes)) if int(modes[i]) == 1]
        assert base and azim, list(modes)
        for i in base:
            assert p._critical_normal_mode(i) is None, i
            p._refuse_at_normal_mode_bifurcation("Branch switching", i)     # must not raise
        for i in azim:
            assert p._critical_normal_mode(i) == ("m", 1.0), i
            with pytest.raises(RuntimeError):
                p._refuse_at_normal_mode_bifurcation("Branch switching", i)

        # A mode array left over from an earlier scan describes eigenvalues that are no longer there,
        # and it is read positionally. Trusting one would refuse a perfectly ordinary base-mode
        # bifurcation whenever the previous scan happened to put an m!=0 eigenvalue first.
        p._last_eigenvalues = numpy.array([0j])
        assert len(p._last_eigenvalues_m) != 1, "the premise of this check"
        assert p._critical_normal_mode(0) is None
