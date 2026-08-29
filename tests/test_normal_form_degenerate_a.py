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

# The fold/branch-point verdict when dR/dp is zero.
#
# classify_bifurcation decides "fold or branch point" from a_rel, the cosine of the angle between
# R01 = -dR/dp and the left null vector: a genuine fold gives O(1), a branch point 1e-5 or less. But
# a cosine needs two directions, and R01 can have none - at the transcritical of x' = mu*x - x^2 the
# derivative IS the solution component (dR/dmu = x) and x = 0 there, so R01 is zero up to whatever
# Newton left behind and the cosine becomes round-off over round-off.
#
# Measured on the macOS wheel jobs of 29th August 2026: |a| = 1.4e-17 against |R01| of the same size
# gave a_rel = 1.0 exactly - the strongest possible FOLD signature for a textbook transcritical - and
# BifurcationController.branch_switch then declined with "a fold has only one branch through it",
# returning False without raising. Both [transcritical-real] and [transcritical-stable] of
# test_bifurcation_gui failed that way, on both Mac architectures, in two consecutive runs.
#
# Linux passed the same tests only because its Newton landed on x = 0.0 exactly, which makes a
# exactly 0.0 and the ratio 0. That is luck, not robustness, and it is why this test does not just
# locate the bifurcation and classify it: it PUTS x at 1e-17 first, which is what the Macs converged
# to anyway, so that the degenerate case is reproduced deterministically on every platform.

"""A transcritical whose dR/dp is zero must not be classified as a fold."""

import numpy

from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var, testfunction, partial_t


# Away from 0 on purpose, and 0.1 specifically: see the comment at the eigensolve below and
# BifurcationController._shift_for_an_eigensolve_at_a_bifurcation, which stands 0.1 in for exactly
# this situation.
_SHIFT = 0.1


class _Eqs(ODEEquations):
    """x' = mu*x - x^2: a transcritical at mu = 0, second branch at x = mu.

    z' = -z rides along, decoupled. It is not decoration: with x alone the problem has ONE dof, and
    a one-dimensional eigenproblem leaves the solver no room - the right eigensolve came back with
    1.0 for a Jacobian that is exactly 0. The extra stable direction also matches
    secondary_real_bifurcation_worker.py, which carries one for the same reason.
    """

    def define_fields(self):
        self.define_ode_variable("x", "z")

    def define_residuals(self):
        mu = self.get_problem().mu
        x, z = var(["x", "z"])
        self.add_residual((partial_t(x) - (mu*x - x**2))*testfunction(x))
        self.add_residual((partial_t(z) + z)*testfunction(z))


class _Prob(Problem):
    def __init__(self):
        super().__init__()
        self.mu = self.define_global_parameter(mu=0.0)

    def define_problem(self):
        self += _Eqs() @ "nf"
        self += InitialCondition(x=0.0, z=0.0) @ "nf"


def _classify_at(x_value, tmp_path):
    """Classify the bifurcation with the solution component sitting exactly at ``x_value``."""
    with _Prob() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        p.mu.value = 0.0
        # No solve: mu = 0 with x = z = 0 IS the solution (the residual is mu*x - x^2 and -z, both
        # exactly zero there), and asking Newton for it means factorising an exactly singular
        # Jacobian. On macOS with the Accelerate backend that now raises rather than returning a
        # step - correctly, since src/mac_accelerate.cpp learned to report SparseMatrixIsSingular
        # instead of letting libSparse trap - so a solve here fails the test for a reason that has
        # nothing to do with what it is testing.
        # The whole point: mu = 0 and x = x_value is the bifurcation, and dR/dmu = x is then exactly
        # x_value. Set through the dof vector rather than solved for, because a solve would land back
        # on 0.0 and destroy the case being tested.
        dofs, _ = p.get_current_dofs()
        dofs = numpy.array(dofs, dtype=float)
        assert len(dofs) == 2, dofs
        dofs[0] = x_value
        p.set_current_dofs(dofs)
        # NOT shift=0: a located real bifurcation has an exactly singular Jacobian, and a
        # shift-invert there factorises it - the very defect the secondary-bifurcation worker
        # documents, which returns eigenvalues like +0.37 or -1.2e18 instead of the critical 0.
        p.solve_eigenproblem(2, shift=_SHIFT)
        return p.classify_bifurcation("mu")


def test_a_transcritical_with_zero_dRdp_is_not_a_fold(tmp_path):
    """The regression: 1e-17 in both |a| and |R01| used to read as a_rel = 1, i.e. a fold."""
    nf = _classify_at(1e-17, tmp_path / "tiny")
    assert nf["type"] == "transcritical", \
        "classified as %r with a=%r, a_rel=%r (raw cosine %r)" % (
            nf["type"], nf.get("a"), nf.get("a_rel"), nf.get("a_rel_raw"))
    # The decision quantity, not the raw cosine: the guard sets it to zero on purpose, and the raw
    # value is kept alongside precisely so that a future failure can tell the two apart.
    assert nf["a_rel"] == 0.0, nf.get("a_rel")
    # And the thing branch switching actually needs: a prediction for where the other branch goes.
    assert nf.get("param_predictor") is not None
    assert nf.get("perturbation_predictor") is not None


# There is deliberately no "x is exactly 0.0" case here. That state has an EXACTLY singular
# Jacobian, and a shift-invert eigensolve then has nothing to inverpolate against: measured with the
# scipy fallback this wheel-style environment uses, the critical eigenvalue comes back as
# 0.3678794411714423 whatever the shift, and the classification fails before reaching the guard with
# "the left eigenvector solve landed on 1.3e-23 instead of the requested 0.368". That is the separate
# hazard BifurcationController._shift_for_an_eigensolve_at_a_bifurcation documents, not this one, and
# the guard's exact-zero branch is unreachable through it. Both cases below sit a hair off the exact
# state, which is where the Macs were anyway.


def test_a_genuine_fold_is_still_a_fold(tmp_path):
    """The guard must only ever turn a fold verdict into a branch point, never the reverse.

    x' = mu - x^2 has a fold at mu = 0, and there dR/dmu = 1 is as far from zero as it gets - so a is
    O(1) beside b1 and b2 and the guard cannot fire.
    """
    from_x = 1e-9   # off the exactly singular state, as above
    class _FoldEqs(ODEEquations):
        def define_fields(self):
            self.define_ode_variable("x", "z")

        def define_residuals(self):
            mu = self.get_problem().mu
            x, z = var(["x", "z"])
            self.add_residual((partial_t(x) - (mu - x**2))*testfunction(x))
            self.add_residual((partial_t(z) + z)*testfunction(z))

    class _FoldProb(Problem):
        def __init__(self):
            super().__init__()
            self.mu = self.define_global_parameter(mu=0.0)

        def define_problem(self):
            self += _FoldEqs() @ "nf"
            self += InitialCondition(x=0.0, z=0.0) @ "nf"

    with _FoldProb() as p:
        p.set_output_directory(str(tmp_path / "fold"))
        p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        p.mu.value = 0.0
        # No solve, for the same reason as above: at mu = 0 the fold's own Jacobian is singular.
        dofs, _ = p.get_current_dofs()
        dofs = numpy.array(dofs, dtype=float)
        dofs[0] = from_x
        p.set_current_dofs(dofs)
        p.solve_eigenproblem(2, shift=_SHIFT)
        nf = p.classify_bifurcation("mu")
    assert nf["type"] == "fold", "a=%r, a_rel=%r" % (nf.get("a"), nf.get("a_rel"))
    assert nf["a_rel"] > 1e-3, nf["a_rel"]
