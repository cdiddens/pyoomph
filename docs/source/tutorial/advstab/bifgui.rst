.. _advstabbifgui:

The interactive bifurcation GUI
-------------------------------

Everything in the sections on :ref:`arclength continuation <secODEarclength>`, :ref:`bifurcation tracking
<sectemporalbiftrack>` and :ref:`branch switching <sectemporalbranchswitch>` is done from a script: you write down the
continuation, the bifurcation tracking and the branch switching in advance, run it, and look at the
result afterwards. 

If you do not know anything about a system yet, i.e. when you try to explore  the bifurcation diagram 
of an unfamiliar problem, you might want to have dynamic control over arclength step sizes, refine branches by adding more points,
investigate the spectrum at points, plot different observables against the continuation parameter or continue in multiple parameters.
Moreover, you also want to select the direction when switching branches, etc, without having to restart your script-based investigation over and over again.

That is what pyoomph's bifurcation GUI is for. It drives exactly the same machinery the scripted route uses -
:py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation`,
:py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking`,
:py:meth:`~pyoomph.generic.problem.Problem.switch_branch`,
:py:meth:`~pyoomph.generic.problem.Problem.switch_to_hopf_orbit`,
:py:meth:`~pyoomph.generic.problem.Problem.deflated_continuation` - but you can interactively decide whe* to use which
from the data shown on the screen. Already computated data is fully stored: every point is a full state dump, the diagram is
written to disk as you go, and restarting the same script picks it up where you left it. When the
diagram is finished it can be exported as plain columns for a plotting tool, and the state dumps are
there to restart a scripted run from any point you tagged.

A typical boot script for the GUI is just this:

.. code:: python

	from pyoomph import *
	from pyoomph.utils.bifurcation_gui import BifurcationGUI

	with MyProblem() as problem:
		problem.setup_for_stability_analysis(analytic_hessian=True)
		gui=BifurcationGUI(problem,"my_parameter")   # which parameter is continued
		gui.neigen=20                                 # eigenvalues per point
		gui.set_initial_observable("domain/my_obs")   # what goes on the vertical axis
		if gui.must_init():        # False when a stored diagram was found, which is loaded instead
			problem.solve()        # whatever it takes to reach the first stationary solution
		gui.start(0.001)           # opens the window; the argument is the initial arclength step

:py:meth:`~pyoomph.utils.bifurcation_gui.BifurcationGUI.must_init` is what makes the script restartable:
on the first run it returns ``True`` and the block below it produces the starting solution, on every
later run it returns ``False`` because the diagram written last time is on disk and is loaded instead.
To restart the diagram from scratch, just wipe the output directory.

``analytic_hessian=True`` is required if you intend to classify bifurcations, switch branches or
step onto a periodic orbit - all three need the second derivative.

The window that opens has five parts:

* the **bifurcation diagram** itself, an ordinary matplotlib canvas with its navigation toolbar, in
  which points and branches can be clicked;
* the **field plots** beside it, one per plotter the problem carries
  (:py:attr:`~pyoomph.generic.problem.Problem.plotter`, which also accepts a list), redrawn as you move
  along a branch, so you can see what the solution actually looks like without leaving the window;
* the **toolbar** at the top with the handful of commands used constantly (*Step*, *Multi*, ``ds``
  up/down/reverse, *Find bif*, *Switch*, *Delete*) and the *Abort* button on the right, which stops a
  running sweep between steps;
* the **notebook of tabs on the right**, which holds the settings and the tables - the step size, the
  eigenvalue settings, the list of parameters, the spectrum at the current point, the branch tree, and
  the deflation and orbit settings (see :ref:`advstabbifguitabs`);
* the **log** along the bottom and a status bar, which is where a refused command says why.

Every command is reachable three ways - from the menu bar (:ref:`advstabbifguimenus`), from a keyboard
shortcut (:ref:`advstabbifguikeys`), and, for the frequent ones, from the toolbar or a button in a tab.
The menus show each command's shortcut next to it, so the keys can be learned by using the menus.

We will start by constructing the bifurcation diagram of a representative example on the next page.

.. toctree::
   :maxdepth: 5
   :hidden:

   bifgui/example.rst
   bifgui/menus.rst
   bifgui/tabs.rst
   bifgui/keys.rst
