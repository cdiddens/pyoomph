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

"""A RemeshWhen may not remesh from inside a C++ solve call.

``oomph::Problem::adaptive_unsteady_newton_solve()`` snapshots the dofs by FLAT INDEX before the
step and, if the step is rejected on temporal error, restores them the same way. RemeshWhen used to
fire from ``actions_after_newton_solve()``, which oomph calls from inside ``newton_solve()`` -- i.e.
between those two. A remesh there changes both ``ndof`` and the meaning of every index, so the
restore writes the old mesh's values into unrelated dofs of the new one, and past the end of
``Dof_pt`` entirely whenever the remesh shrank the system.

What that did in practice, before ``Problem._perform_pending_remesh()`` existed: a converged state
came back as ``Initial Maximum residuals 2981``, the retry diverged to ``inf``, and the step was
rejected 26 more times -- halving dt and restoring the same corrupt snapshot each time -- until the
run died with "Max. residual has been exceeded". Nothing in that message mentions remeshing, which
is why this went unnoticed: ``dev_docs/mesh_construction.md`` section 5.1 recorded it as read off
the source and never reproduced.

Both tests below fail on the pre-fix tree, and they fail differently on purpose: the first one dies
with an ``OomphException``, the second one still completes but reports the remesh that happened
inside the solve. Keeping both means a fix that merely stops the run from crashing -- or one that
quietly stops remeshing altogether -- does not pass.
"""

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.generic import RemeshWhen, RemeshingOptions, RemeshMeshSize
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.meshes.remesher import Remesher2d


class _LaplaceSmoothedMesh(Equations):
    """The tutorial's Laplace-smoothed moving mesh, inlined so the test needs no tutorial import."""

    def define_fields(self):
        self.activate_coordinates_as_dofs(coordinate_space="C2")

    def define_residuals(self):
        x, xtest = var_and_test("mesh")
        self.add_residual(weak(grad(x, lagrangian=True), grad(xtest, lagrangian=True)))


class _MovingScalar(Equations):
    """A transported scalar with a fast, time-dependent source.

    The source is what gives ``global_temporal_error_norm()`` something to see; without
    ``set_temporal_error_factor`` the norm is identically zero, no step is ever rejected, and the
    branch this test is about is unreachable.
    """

    def define_fields(self):
        self.define_scalar_field("c", "C2")

    def define_residuals(self):
        c, v = var_and_test("c")
        self.add_residual(weak(partial_t(c, ALE="auto"), v) + weak(0.001 * grad(c), grad(v))
                          - weak(10 * sin(25 * var("time")) * var("coordinate_x"), v))
        self.set_temporal_error_factor("c", 1)


class _RemeshingProblem(Problem):
    def __init__(self):
        super().__init__()
        # Hair-trigger on purpose: a remesh in almost every step is what makes the coincidence with
        # a rejected step likely enough to be a deterministic test rather than a lucky one.
        self.remesh_options = RemeshingOptions(max_expansion=1.15, min_expansion=0.87,
                                               min_quality_decrease=0.95)
        self.ndof_changed_inside_solve = []
        self.remeshes = 0

    def define_problem(self):
        mesh = RectangularQuadMesh(N=4)
        mesh.remesher = Remesher2d(mesh)
        self.add_mesh(mesh)
        eqs = _LaplaceSmoothedMesh()
        eqs += DirichletBC(mesh_x=0, mesh_y=True) @ "left"
        eqs += DirichletBC(mesh_x=True, mesh_y=0) @ "bottom"
        eqs += DirichletBC(mesh_y=1) @ "top"
        xi = var("lagrangian")
        eqs += DirichletBC(mesh_x=1 + 0.35 * xi[1] * sin(4 * var("time"))) @ "right"
        eqs += RemeshWhen(self.remesh_options)
        # The size contrast is what makes a remesh actually change ndof rather than rebuild the same
        # mesh; without it the whole test is vacuous.
        eqs += RemeshMeshSize(size=0.3) @ "left"
        eqs += RemeshMeshSize(size=0.12) @ "right/top"
        eqs += _MovingScalar()
        eqs += IntegralObservables(csqr=var("c") ** 2)
        self.add_equations(eqs @ "domain")

    def actions_after_newton_solve(self):
        before = self.ndof()
        res = super().actions_after_newton_solve()
        after = self.ndof()
        if after != before:
            self.ndof_changed_inside_solve.append((before, after))
        return res

    def force_remesh(self, *args, **kwargs):
        self.remeshes += 1
        return super().force_remesh(*args, **kwargs)


def _run(tmp_path):
    with _RemeshingProblem() as p:
        p.set_output_directory(str(tmp_path / "out"))
        p.quiet()
        p.run(0.30, outstep=False, startstep=0.05, maxstep=0.05, temporal_error=1e-3)
        obs = p.get_mesh("domain").evaluate_all_observables()
    return p, obs


def test_a_remesh_never_fires_from_inside_a_solve(tmp_path):
    p, _obs = _run(tmp_path)
    # The mechanism has to have engaged, or the invariant below is satisfied trivially.
    assert p.remeshes > 0, "no remesh happened at all, so this test proves nothing"
    assert p.ndof_changed_inside_solve == [], (
        "ndof changed inside actions_after_newton_solve(), i.e. a remesh ran while oomph-lib held a "
        "flat-index dof snapshot around the step: " + repr(p.ndof_changed_inside_solve))


def test_the_transient_survives_a_remesh_in_a_rejected_step(tmp_path):
    # Pre-fix this raises RuntimeError("OomphException") out of adaptive_unsteady_newton_solve: the
    # restored-from-a-stale-snapshot state diverges, dt is halved to below Minimum_dt, and oomph
    # gives up. The assertions after it are the cheap sanity net on the state it ends in.
    p, obs = _run(tmp_path)
    assert p.remeshes > 0
    csqr = float(obs["csqr"])
    assert csqr == csqr, "the integrated solution is NaN"
    assert 0.0 < csqr < 1e3, "integrated solution out of range: " + repr(csqr)


@pytest.mark.parametrize("temporal_error", [None, 1e-3])
def test_remeshing_still_happens_with_and_without_temporal_adaptivity(tmp_path, temporal_error):
    """The deferral must not switch remeshing off on the path that never had the hazard.

    ``unsteady_newton_solve`` takes no dof snapshot, so nothing was wrong there -- but it reaches
    ``actions_after_newton_solve()`` through the same code, so it is the obvious thing for the fix
    to have broken.
    """
    with _RemeshingProblem() as p:
        p.set_output_directory(str(tmp_path / ("out_te_" + str(temporal_error))))
        p.quiet()
        p.run(0.20, outstep=False, startstep=0.05, maxstep=0.05, temporal_error=temporal_error)
    assert p.remeshes > 0
    assert p.ndof_changed_inside_solve == []
