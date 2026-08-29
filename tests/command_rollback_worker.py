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

# A command that fails must leave the problem exactly where it started.
#
# oomph-lib's steady Newton solve re-raises without restoring the dofs, so a diverging solve leaves
# them at the last iterate - and under a bifurcation tracker the continuation PARAMETER is one of
# those dofs. Measured on the ODE below before the fix: the parameter went from -0.629 to +2.371 the
# moment deactivate_bifurcation_tracking() wrote the augmented unknown back, while the diagram went
# on reporting the point that was still current. The field plots then showed the diverged state and
# the next continuation step set off from it - and failed outright.
#
# The second half is the case that is easy to get wrong: on a LOCUS the recovery must keep the
# bifurcation tracker. A state file holds the BASE problem - the tracker's augmented unknowns live
# outside the meshes and are not in it - so loading one takes the augmentation off, and a rollback
# that stopped there left ndof at 1 instead of 3. Every later step would then continue an ordinary
# branch while the diagram claimed to be following the fold.
#
# The failing solve is substituted rather than provoked: what is being tested is the recovery, and a
# real divergence would make the test depend on how far a particular Newton run happens to wander.

"""Does a failed command put the problem back - a bifurcation search, and a step along a locus?"""
import sys
from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import var_and_test, partial_t
from pyoomph.equations.generic import ODEObservables
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits


class Eqs(ODEEquations):
    """u' = mu - u^2 - b u: a fold at (mu, u) = (-b^2/4, -b/2), whose locus in b is a parabola."""
    def __init__(self, mu, b): super().__init__(); self.mu = mu; self.b = b
    def define_fields(self): self.define_ode_variable("u")
    def define_residuals(self):
        u, ut = var_and_test("u")
        self.add_weak(partial_t(u) - (self.mu - u**2 - self.b * u), ut)


class Prob(Problem):
    def define_problem(self):
        self += (Eqs(self.get_global_parameter("mu"), self.get_global_parameter("b"))
                 + InitialCondition(u=0.8) + ODEObservables(uval=lambda u: u)) @ "ode"


def _state(problem):
    return (problem.get_global_parameter("mu").value, problem.get_global_parameter("b").value,
            list(problem.get_current_dofs()[0]), problem.ndof())


def _assert_same(before, after, what):
    for a, b, name in zip(before, after, ("mu", "b", "dofs", "ndof")):
        if name == "dofs":
            assert len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) < 1e-12, \
                what + " moved the dofs: " + str(a) + " -> " + str(b)
        else:
            assert abs(a - b) < 1e-12, what + " moved " + name + ": " + str(a) + " -> " + str(b)


# Run under a __main__ guard: see the note in tag_output_worker.py - this file lives in tests/ and
# the suite hands pytest every .py in the directory by name.
def main():
    with Prob() as problem:
        problem.set_output_directory(sys.argv[1])
        problem.set_linear_solver("superlu")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        problem.get_global_parameter("mu").value = 1.0
        problem.get_global_parameter("b").value = 0.4
        problem.quiet()
        gui = BifurcationGUI(problem, "mu"); gui.neigen = 1
        c = gui.controller; c.view = _FixedViewLimits(xlim=(-5, 5), ylim=(-5, 5))
        c.start(-0.2)
        c.step()

        orig_solve = problem.solve
        def diverging_solve(**kw):
            if problem.get_bifurcation_tracking_mode():
                d, _ = problem.get_current_dofs()
                problem.set_current_dofs([x + 3.0 for x in d])   # walk off the branch...
                raise RuntimeError("pretend the tracker diverged")   # ... and then fail
            return orig_solve(**kw)

        # ---------------------------------------------------------------- a failed search
        before = _state(problem)
        ds_before = c._last_ds
        npoints_before = [len(b) for b in c.branches]
        problem.solve = diverging_solve
        try:
            c.locate_bifurcation()
            raise AssertionError("the substituted solve must have made this fail")
        except RuntimeError as e:
            print("locate failed with:", e)
        finally:
            problem.solve = orig_solve
        print("after the failed search:", _state(problem))
        _assert_same(before, _state(problem), "the failed search")
        assert c._last_ds == ds_before, "the failed search changed ds"
        assert [len(b) for b in c.branches] == npoints_before, "a failed search must record nothing"
        assert not problem.get_bifurcation_tracking_mode(), "the tracker is still installed"

        # ... and the branch continues afterwards, which is the symptom users see.
        c.step()
        print("continued to mu =", problem.get_global_parameter("mu").value)
        assert [len(b) for b in c.branches] == [n + 1 for n in npoints_before]

        # ---------------------------------------------------------------- a failed locus step
        while c.current_point.eig_value_Re != 0:
            c.step()
            assert len(c.branches[0]) < 40, "never reached the fold"
            if c.current_point.eig_value_Re > -0.2:
                c.locate_bifurcation()
        print("fold at mu =", problem.get_global_parameter("mu").value)
        c.start_locus(tracked="mu", continue_in="b")
        c.step()
        assert c.on_locus() and problem.get_bifurcation_tracking_mode(), "not on a tracked locus"

        # The tracker must SURVIVE the rollback: a locus is followed with the augmented system
        # installed, and putting the problem back by loading a state must not leave it on the plain
        # one - every later step would then continue an ordinary branch and silently compute
        # something else.
        before = _state(problem)
        mode_before = problem.get_bifurcation_tracking_mode()
        ds_before = c._last_ds
        nlocus_before = len(c.current_branch)
        orig_arc = problem.arclength_continuation
        def diverging_arc(*a, **kw):
            d, _ = problem.get_current_dofs()
            problem.set_current_dofs([x + 3.0 for x in d])
            raise RuntimeError("pretend the locus step diverged")
        problem.arclength_continuation = diverging_arc
        try:
            c.step()
            raise AssertionError("the substituted continuation must have made this fail")
        except RuntimeError as e:
            print("locus step failed with:", e)
        finally:
            problem.arclength_continuation = orig_arc
        print("after the failed locus step:", _state(problem))
        _assert_same(before, _state(problem), "the failed locus step")
        assert c._last_ds == ds_before, "the failed locus step changed ds"
        assert problem.get_bifurcation_tracking_mode() == mode_before, \
            "the rollback lost the bifurcation tracker: '" + str(mode_before) + "' -> '" + \
            str(problem.get_bifurcation_tracking_mode()) + "'"
        assert c.on_locus() and len(c.current_branch) == nlocus_before

        # ... and the locus can be followed on.
        c.step()
        assert len(c.current_branch) == nlocus_before + 1, "the locus did not continue"

        # Going BACK to an earlier locus point must not lose the tangent either. A locus point's dump
        # cannot carry one (the vector belongs to the augmented tracker), so without one computed the
        # next step is oomph's "first step": the parameter marches by the whole of ds. Measured before
        # the fix, with ds at -2.94: b went from -0.369 to -3.310, exactly one ds, instead of -2.759.
        locus = c.current_branch
        target = locus[1]
        c.load_pt(target)
        tangent = list(problem.get_arclength_dof_derivative_vector())
        b_before = problem.get_global_parameter("b").value
        ds_now = c._last_ds
        assert len(tangent) == problem.ndof() > 1, \
            "no arclength tangent after loading a locus point: " + str(len(tangent))
        c.step()
        b_after = problem.get_global_parameter("b").value
        print("locus reload: b", b_before, "->", b_after, " (ds", ds_now, ")")
        assert abs(b_after - (b_before + ds_now)) > 0.1 * abs(ds_now), \
            "the step from the reloaded locus point marched b by the whole of ds"
        print("locus continued to b =", problem.get_global_parameter("b").value,
              "mu =", problem.get_global_parameter("mu").value)
        print("ROLLBACK OK")
        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
