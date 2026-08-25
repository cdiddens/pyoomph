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

# Every default keyboard shortcut has to reach a command. _on_key looks the accelerator up in the
# keymap, finds no action of that id and returns - silently - which is how "Grab selected point" spent
# its life reachable only from a checkbox in the settings panel.
#
# A worker rather than an in-process test, because building the window means tk.Tk(): out here it can
# be launched under xvfb-run like the other two window-opening workers in this module. No Problem is
# involved - the window is built against a stub, never shown, and nothing is solved.

import sys

import matplotlib
matplotlib.use("Agg")

from pyoomph.utils.bifurcation_gui.actions import DEFAULT_KEYMAP
from pyoomph.utils.bifurcation_gui.controller import BifurcationController
from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter
from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp


class StubProblem:
    """What BifurcationController's constructor and the menu builders touch, and nothing else."""

    _runmode = "overwrite"
    write_states = False
    continuation_data_in_states = False
    plotter = None
    _arclength_inner_product = None

    def get_global_parameter_names(self):
        return ["mu"]

    def is_initialised(self):
        return True

    def set_arclength_inner_product(self, kind):
        self._arclength_inner_product = kind


def main() -> int:
    app = BifurcationTkApp(BifurcationController(StubProblem(), "mu"),
                           BifurcationDiagramPlotter(), title="shortcut check")
    try:
        app.root.withdraw()
        missing = sorted(a for a in DEFAULT_KEYMAP if a not in app._actions)
        assert not missing, "shortcuts bound to nothing: " + ", ".join(missing)
        # The three added with the split/merge/grab commands, which is what this was written for.
        for action_id in ("toggle_move_point", "split_branch", "merge_branches"):
            assert action_id in app._actions, action_id + " is not a command"
            assert action_id in app._menu_entries, action_id + " is in no menu"
            assert app.keymap.get(action_id), action_id + " has no default shortcut"
        # Reachable from a menu, but deliberately unbound: it is a repair, not something to hit by
        # accident while moving points around.
        assert "disentangle_branch" in app._actions
        assert "disentangle_branch" in app._menu_entries
        assert not app.keymap.get("disentangle_branch")
        print("SHORTCUTS OK: {:d} bound, all reach a command".format(len(DEFAULT_KEYMAP)))
    finally:
        app.root.destroy()

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
