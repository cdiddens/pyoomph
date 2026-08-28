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

# The arclength metric is a radio group in the Continuation menu: the GUI starts in the mass-matrix
# ("l2") metric, and the dot has to say so and then follow the state rather than the click.
#
# A worker rather than an in-process test, because the menu variables only exist once the window is
# built, i.e. tk.Tk(); out here it runs under xvfb-run like the other window-opening workers. No
# Problem is involved - the window is built against shortcut_worker's stub and never shown.

import sys

import matplotlib
matplotlib.use("Agg")

from pyoomph.utils.bifurcation_gui.controller import BifurcationController
from pyoomph.utils.bifurcation_gui.plotter import BifurcationDiagramPlotter
from pyoomph.utils.bifurcation_gui.tkapp import BifurcationTkApp

from shortcut_worker import StubProblem


class MetricStub(StubProblem):
    """shortcut_worker's stub plus the two calls switching the metric makes on the problem."""

    theta = 1.0

    def set_arc_length_parameter(self, **kwargs):
        pass

    def get_arc_length_theta_sqr(self):
        return self.theta


def main() -> int:
    c = BifurcationController(MetricStub(), "mu")
    assert c.arclength_metric() == "l2", "the GUI must start in the mesh-independent metric, not " \
                                         "oomph's dof sum: " + c.arclength_metric()
    assert c.scale_arc_length is False, "the scaling and the metric would tune the same theta^2"

    app = BifurcationTkApp(c, BifurcationDiagramPlotter(), title="arclength metric check")
    try:
        app.root.withdraw()
        var = app._radio_vars["arclength_metric"]
        assert var.get() == "l2", "the dot must start where the metric is: " + var.get()
        for action_id, expect in (("arclength_metric_ndof", "ndof"),
                                  ("arclength_metric_dofsum", "dofsum"),
                                  ("arclength_metric_l2", "l2")):
            app._actions[action_id].callback()
            app._sync_radio_vars()
            assert c.arclength_metric() == expect, action_id + " did not reach the problem"
            assert var.get() == expect, action_id + " left the dot on " + var.get()
        print("ARCLENGTH METRIC OK: starts on l2, the dot follows the state")
    finally:
        app.root.destroy()

    print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
