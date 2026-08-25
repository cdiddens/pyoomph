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

# The Orbit tab's widgets, the same way deflation_tab_worker.py checks the Deflation tab's: that
# they exist, that what is typed reaches the controller, and that a refresh puts the controller's
# values back. The numerics are tested against a real problem in orbit_gui_worker.py.
#
# A worker rather than an in-process test, because the tab's variables do not exist until tk.Tk()
# has been called. No Problem is involved; the window is built against the same stub and never shown.

import sys

import matplotlib
matplotlib.use("Agg")

from pyoomph.utils.bifurcation_gui.controller import BifurcationController
from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter
from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp

from deflation_tab_worker import DeflationStub


def main() -> int:
    c = BifurcationController(DeflationStub(), "mu")
    c._avail_observables = ["u"]
    c._current_observable = "u"
    app = BifurcationTkApp(c, BifurcationDiagramPlotter(), title="orbit tab check")
    try:
        app.root.withdraw()

        titles = [app.side.tab(i, "text") for i in range(app.side.index("end"))]
        assert "Orbit" in titles, "no Orbit tab: " + ", ".join(titles)

        for action_id in ("switch_to_orbit", "orbit_change_mode", "orbit_floquet_here",
                          "orbit_floquet_branch", "orbit_output"):
            assert action_id in app._actions, action_id + " is not a command"
            assert action_id in app._menu_entries, action_id + " is in no menu"

        # None of them applies away from an orbit, and switching applies only at a bifurcation.
        for action_id in ("switch_to_orbit", "orbit_change_mode", "orbit_floquet_here",
                          "orbit_floquet_branch", "orbit_output"):
            assert not app._actions[action_id].enabled_when(), \
                action_id + " is offered on a diagram with no orbit and no bifurcation"

        # Typed values must reach the controller.
        app.orbit_nt_var.set("48")
        app.orbit_order_var.set("2")
        app.orbit_eps_var.set("0.01")
        app.orbit_ampl_var.set("1.5")
        app.orbit_samples_var.set("64")
        app.orbit_mode_var.set("floquet")
        app.orbit_tconstr_var.set("plane")
        app.orbit_collapse_var.set(False)
        app.orbit_portable_var.set(True)
        app.floquet_method_var.set("periodic_schur")
        app.floquet_n_var.set("6")
        app.floquet_enabled_var.set(False)
        app.floquet_shift_var.set(False)
        app._commit_orbit()
        assert c.orbit_NT == 48, c.orbit_NT
        assert c.orbit_order == 2, c.orbit_order
        assert c.orbit_eps == 0.01, c.orbit_eps
        assert c.orbit_amplitude_factor == 1.5, c.orbit_amplitude_factor
        assert c.orbit_observable_samples == 64, c.orbit_observable_samples
        assert c.orbit_mode == "floquet", c.orbit_mode
        assert c.orbit_T_constraint == "plane", c.orbit_T_constraint
        assert c.orbit_check_collapse is False
        assert c.orbit_portable is True
        assert c.floquet_method == "periodic_schur", c.floquet_method
        assert c.floquet_n == 6, c.floquet_n
        assert c.floquet_enabled is False
        assert c.floquet_shift_invert is False

        # "auto" for the parameter step means "take what ds buys", which is the whole point of the
        # default - not zero, which switch_to_hopf_orbit would silently replace with its own eps.
        app.orbit_eps_var.set("auto")
        app.orbit_samples_var.set("auto")
        app.floquet_n_var.set("all")
        app._commit_orbit()
        assert c.orbit_eps is None, c.orbit_eps
        assert c.orbit_observable_samples is None, c.orbit_observable_samples
        assert c.floquet_n is None, c.floquet_n
        app._update_orbit_panel()
        assert app.orbit_eps_var.get() == "auto", app.orbit_eps_var.get()
        assert app.orbit_samples_var.get() == "auto", app.orbit_samples_var.get()
        assert app.floquet_n_var.get() == "all", app.floquet_n_var.get()

        # Nonsense is refused and reported, and the setting keeps its previous value.
        logged: list = []
        c.set_observer(on_log=lambda *a: logged.append(" ".join(str(x) for x in a)))
        app.orbit_nt_var.set("not a number")
        app._commit_orbit()
        assert c.orbit_NT == 48, "a bad step count must be refused: " + repr(c.orbit_NT)
        assert any("time steps" in m for m in logged), logged

        # A discretization with no degree of freedom at the end of the period has no multipliers, and
        # the tab has to say so instead of letting every continuation step fail.
        c.orbit_mode = "BDF2"
        app._update_orbit_panel()
        assert str(app.floquet_method_combo.cget("state")) == "disabled"
        assert "no Floquet multipliers" in app.floquet_info_var.get(), app.floquet_info_var.get()
        c.orbit_mode = "collocation"
        app._update_orbit_panel()
        assert str(app.floquet_method_combo.cget("state")) == "readonly"

        # A refresh writes the controller's state back into the widgets.
        c.orbit_NT = 30
        c.floquet_enabled = True
        app._update_orbit_panel()
        assert app.orbit_nt_var.get() == "30", app.orbit_nt_var.get()
        assert app.floquet_enabled_var.get() is True
        assert "Hopf" in app.orbit_state_var.get(), app.orbit_state_var.get()

        print("ORBIT TAB OK: {:d} tabs, settings round-trip".format(len(titles)))
    finally:
        app.root.destroy()

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
