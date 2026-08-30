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

# The Deflation tab's widgets: that they exist, that what is typed into them reaches the controller,
# and that a refresh puts the controller's values back rather than whatever was typed last. The
# numerics are tested against a real problem in deflation_gui_worker.py; this is only the wiring.
#
# A worker rather than an in-process test, because the tab's variables do not exist until the window
# is built, i.e. tk.Tk(); out here it runs under xvfb-run like the other window-opening workers. No
# Problem is involved - the window is built against shortcut_worker's stub and never shown.

import sys

import matplotlib
matplotlib.use("Agg")

from pyoomph.utils.bifurcation_gui.controller import BifurcationController
from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter
from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp

from shortcut_worker import StubProblem


class DeflationStub(StubProblem):
    """shortcut_worker's stub plus what a full panel refresh asks the problem.

    The _commit_* methods all end in _update_panels(), which walks every tab, so this stub has to
    answer more than the one that only builds the window does.
    """

    def is_normal_mode_stability_set_up(self):
        return None

    def get_global_parameter(self, name):
        raise RuntimeError("no parameters on the stub")

    def ndof(self):
        return 0


def main() -> int:
    c = BifurcationController(DeflationStub(), "mu")
    # A full panel refresh walks every tab, and the axis selectors need an observable to name. On a
    # real problem start() discovers these; here they are the only thing the stub cannot invent.
    c._avail_observables = ["u"]
    c._current_observable = "u"
    app = BifurcationTkApp(c, BifurcationDiagramPlotter(), title="deflation tab check")
    try:
        app.root.withdraw()

        titles = [app.side.tab(i, "text") for i in range(app.side.index("end"))]
        assert "Deflation" in titles, "no Deflation tab: " + ", ".join(titles)

        for action_id in ("deflated_solve", "deflated_continuation", "deflation_forget",
                          "delete_branch"):
            assert action_id in app._actions, action_id + " is not a command"
            assert action_id in app._menu_entries, action_id + " is in no menu"

        # Typed values must reach the controller.
        app.defl_alpha_var.set("0.25")
        app.defl_p_var.set("3")
        app.defl_pert_var.set("0.75")
        app.defl_tries_var.set("5")
        app.defl_seed_var.set("17")
        app.defl_newton_var.set("12")
        app.defl_eigpert_var.set(True)
        app._commit_deflation()
        assert c.deflation_alpha == 0.25, c.deflation_alpha
        assert c.deflation_p == 3, c.deflation_p
        assert c.deflation_perturbation == 0.75, c.deflation_perturbation
        assert c.deflation_random_tries == 5, c.deflation_random_tries
        assert c.deflation_random_seed == 17, c.deflation_random_seed
        assert c.deflation_max_newton_iterations == 12, c.deflation_max_newton_iterations
        assert c.deflation_use_eigenperturbation is True

        # "auto" for the perturbation means "read one off the current solution", not zero. It is the
        # scale the whole search is measured in (alpha is dimensionless in units of it), so it is the
        # one setting where the difference between a value and no value matters most.
        app.defl_pert_var.set("auto")
        app._commit_deflation()
        assert c.deflation_perturbation is None, c.deflation_perturbation
        app._update_deflation_panel()
        assert app.defl_pert_var.get() == "auto", app.defl_pert_var.get()

        # An empty seed and an empty Newton cap mean "no seed" / "the problem's own", not zero.
        app.defl_seed_var.set("")
        app.defl_newton_var.set("")
        app._commit_deflation()
        assert c.deflation_random_seed is None, c.deflation_random_seed
        assert c.deflation_max_newton_iterations is None, c.deflation_max_newton_iterations

        # Nonsense is refused and reported, and the setting keeps its previous value.
        logged: list = []
        c.set_observer(on_log=lambda *a: logged.append(" ".join(str(x) for x in a)))
        app.defl_alpha_var.set("-1")
        app.defl_p_var.set("0")
        app._commit_deflation()
        assert c.deflation_alpha == 0.25, "a negative shift must be refused: " + repr(c.deflation_alpha)
        assert c.deflation_p == 3, "p < 1 must be refused: " + repr(c.deflation_p)
        assert any("alpha" in m for m in logged) and any("power p" in m for m in logged), logged

        # The scan settings, and "auto" for the increment.
        app.defl_steps_var.set("7")
        app.defl_dparam_var.set("-0.2")
        # Off, not on: it is ON by default, so setting it True would prove nothing about the commit.
        app.defl_scan_eig_var.set(False)
        app._commit_deflated_scan()
        assert c.deflated_scan_steps == 7, c.deflated_scan_steps
        assert c.deflated_scan_dparam == -0.2, c.deflated_scan_dparam
        assert c.deflated_scan_eigensolve is False
        app.defl_scan_eig_var.set(True)
        app._commit_deflated_scan()
        assert c.deflated_scan_eigensolve is True
        app.defl_dparam_var.set("auto")
        app._commit_deflated_scan()
        assert c.deflated_scan_dparam is None, c.deflated_scan_dparam

        # A refresh writes the controller's state back into the widgets.
        c.deflation_alpha = 0.05
        c.deflated_scan_steps = 3
        c.deflation_random_seed = None
        app._update_deflation_panel()
        assert app.defl_alpha_var.get() == "0.05", app.defl_alpha_var.get()
        assert app.defl_steps_var.get() == "3", app.defl_steps_var.get()
        assert app.defl_seed_var.get() == "", app.defl_seed_var.get()
        assert app.defl_dparam_var.get() == "auto", app.defl_dparam_var.get()
        assert "Nothing deflated yet" in app.defl_known_var.get(), app.defl_known_var.get()

        print("DEFLATION TAB OK: {:d} tabs, settings round-trip".format(len(titles)))
    finally:
        app.root.destroy()

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
