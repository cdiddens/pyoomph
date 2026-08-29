.. _advstabbifguimenus:

The menu bar
~~~~~~~~~~~~

Every command of the GUI lives in one of these menus, and each entry shows its keyboard shortcut in
the right-hand column when it has one (:ref:`advstabbifguikeys`). A command that cannot be applied
right now is greyed out rather than hidden - *Switch branch* away from a bifurcation, the orbit
commands away from an orbit branch - so the menus also serve as a list of what the current point can
and cannot do. Commands marked *(toggle)* below are checkboxes that stay on until switched off.

File
^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Entry
     - What it does
   * - Save diagram
     - Write the diagram - every branch, point and state dump - into the problem's output directory.
       This also happens by itself as you work; the entry is there for the moment before you close
       something.
   * - Reload diagram from disk
     - Throw away what is in memory and read the stored diagram back. The way to undo a split, a merge
       or an accidental deletion of a point.
   * - Start new branch from state file...
     - Open a branch at a solution stored elsewhere, e.g. one produced by a script or by another
       diagram.
   * - Export curves...
     - Write the curves as plain text columns, one file per branch and per stability segment, plus one
       per bifurcation and per tagged point. This is what a plotting tool reads; the state dumps of
       tagged points are copied along, with the ``.msh`` files they refer to.
   * - Output the tagged points
     - Load each tagged point and run the problem's *own* output - plots, VTUs, text files - into
       ``output/tagNN/``, with a copy of the state dump beside it. The curve export only copies the
       dumps, which preserve the solution but show nothing. On a periodic orbit branch, the rest of
       the cycle travels along as a companion file, so the folder is a complete starting point for
       another script (see :ref:`advstabbifguiexample`).
   * - Save figure as...
     - Save the diagram canvas as an image.
   * - Record a frame per redraw *(toggle)*
     - Write one image per redraw, to be assembled into a video of the diagram being built.
   * - Quit
     - Close the window. The diagram on disk is up to date.

Continuation
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Entry
     - What it does
   * - Continuation parameter
     - Submenu: which global parameter the arclength continuation varies. Changing it starts a new
       diagram - a diagram is a section through parameter space, and a different parameter is a
       different section.
   * - Step
     - One arclength continuation step, followed by an eigensolve unless quick mode is on.
   * - Multistep
     - Keep stepping until the branch leaves the visible axes. *Abort* stops it between steps, and the
       step size is capped at what the first step covered so a turning branch cannot run away.
   * - Step (never grow ds)
     - One step that is not allowed to grow the step size, for getting through a difficult stretch.
   * - Increase ds / Decrease ds
     - Multiply or divide the arclength step by 1.25.
   * - Reverse direction
     - Continue the other way along the branch.
   * - Set ds...
     - Type the step size instead.
   * - Quick mode (no eigensolve per step) *(toggle)*
     - Continue without solving an eigenproblem at every point. Bifurcations are then spotted from the
       sign of the determinant and from the tangent, which is far cheaper - but a Hopf bifurcation
       cannot be seen this way at all, since no real eigenvalue changes sign there. The spectra can be
       filled in afterwards from *Bifurcation → Compute the eigenvalues along this branch*.
   * - Quick mode: watch folds only *(toggle)*
     - The cheapest variant: needs no support from the solver, but sees only folds and misses
       pitchfork and transcritical points.
   * - Continue in the selected parameter
     - Start a new diagram continuing in whichever parameter is selected in the *Parameters* tab.
   * - Set the selected parameter's value...
     - Change a parameter that is *not* being continued, i.e. move to another section through
       parameter space.
   * - Scale arclength / Do not scale arclength
     - Whether the arclength metric is retuned as the continuation proceeds.
   * - Parameter fraction of the arclength...
     - How much of one step goes into the parameter rather than into the solution. Small values make
       the step follow the solution, large values make it march in the parameter.
   * - Arclength metric: dof sum / per dof / L2 (mass matrix)
     - Radio group: what "length" means in the arclength constraint. The oomph-lib default sums over
       the degrees of freedom, so the same physical step feels different on a finer mesh; the other
       two are mesh-independent, the last one weighting by the mass matrix.

Bifurcation
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Entry
     - What it does
   * - Locate bifurcation / leave it
     - Away from a bifurcation, install a tracker on the eigenvalue nearest the imaginary axis and
       converge onto the bifurcation exactly. *At* one, it instead takes the way off that its type
       offers: a branch switch at a transcritical or a pitchfork, the periodic orbit at a Hopf, and a
       transient departure at a fold, which has no second steady branch.
   * - Locate the bifurcation of the selected eigenvalue
     - Track the eigenvalue picked in the *Points* tab instead of the one nearest the axis. This is
       what to use on a branch that is already unstable, where the mode about to cross is not the
       leading one.
   * - Locate pitchfork
     - Locate with the pitchfork tracker, which uses the problem's symmetry rather than a plain fold
       condition.
   * - Switch branch
     - Step onto the other branch through the bifurcation, using its normal form to predict where that
       branch goes. Only a transcritical or a pitchfork has one; a fold does not.
   * - Refine the detected bifurcation
     - A bifurcation that quick mode only *bracketed* between two points (drawn as an open circle) is
       located properly.
   * - Compute the eigenvalues at this point / along this branch
     - Solve the eigenproblem where it is missing - after a quick-mode sweep, or at points loaded from
       a stored diagram. Along a branch, each point's state dump is loaded in turn.
   * - Recompute the eigenvalues at this point / along this branch
     - The same, but redoing points that already have a spectrum - after changing the eigenvalue
       count, the shift or the set of normal modes.
   * - Scan the stripe for eigenvalues here / along this branch
     - Find *every* eigenvalue in a stripe :math:`|\mathrm{Re}|<r`, :math:`|\mathrm{Im}|<i` around the
       origin. Shift-invert returns the eigenvalues nearest the shift and can walk straight past a
       Hopf pair sitting far up the imaginary axis; this finds it. The stripe is set in the
       *Continuation* tab.
   * - Merge a stripe scan into the spectrum *(toggle)*
     - Whether what a stripe scan finds is added to the recorded spectrum or only reported.
   * - Deflated solve
     - Look for *another* solution at this parameter value, by deflating away the ones already known.
       One that is genuinely new opens a new branch. See the *Deflation* tab for the settings.
   * - Deflated continuation
     - Scan the continuation parameter and deflate at every value. This finds branches that are not
       connected to the current one, which arclength continuation by construction never can. *Abort*
       stops it between parameter steps.
   * - Forget the deflated solutions
     - Start the deflated search over here, instead of continuing to avoid what it has already found.
   * - Switch onto the periodic orbit
     - At a Hopf, step onto the periodic orbit it sheds and continue that instead. The step off the
       Hopf is the one the current ``ds`` buys.
   * - Re-discretize this orbit
     - Re-sample the installed orbit with the discretization set in the *Orbit* tab, keeping the orbit
       itself. This is how an orbit found with B-splines - which converge more readily off a Hopf but
       carry no Floquet multipliers - is converted to collocation, which has them.
   * - Floquet multipliers here / along this branch
     - The stability of a periodic orbit, the way an eigenvalue is a stationary solution's. Needs a
       discretization with a degree of freedom at the end of the period.
   * - Write the orbit's cycle
     - Run the problem's own output along the whole period rather than at the single phase the
       solution happens to sit at.
   * - Follow this bifurcation in...
     - Start a *locus*: hold the bifurcation by adjusting the current parameter while continuing in a
       second one, tracing the curve of the bifurcation in the plane of the two. The diagram's axes
       switch to those two parameters.
   * - Leave the locus and continue in...
     - Step off the bifurcation back onto an ordinary branch through it, continuing in a chosen
       parameter.
   * - Leave branch transiently (mode 0 / mode 1)
     - Perturb along an eigenfunction and integrate in time until the solution settles somewhere else.
       This is the way off a fold or a Hopf whose orbit cannot be started, and the way to find out
       which attractor an unstable branch loses out to.
   * - Classify bifurcations (normal form) *(toggle)*
     - Compute the normal form at each located bifurcation, which is what names it fold, transcritical,
       pitchfork or Hopf and what a branch switch predicts from. On by default.
   * - Eigenvalue settings...
     - The eigenvalue count, the shift and the normal modes, in a dialog - the same values as in the
       *Continuation* tab.

Points
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Entry
     - What it does
   * - Go to selected point
     - Load the selected point's state, making it the current one. Everything else acts on the current
       point.
   * - Select previous / next / first / last
     - Move the selection along the current branch.
   * - Delete point
     - Remove the selected point and its state dump.
   * - Tag → 0 ... 9
     - Give the point a number, drawn next to it in the diagram. A tagged point's state is copied out
       by *File → Export curves* and its fields written by *File → Output the tagged points*, which is
       how a point of interest is carried over into a figure or into a scripted run. Tagging a point
       with a number already in use moves the tag.
   * - Split branch here
     - Cut the branch in two at the selected point. For when a continuation step quietly landed on a
       different branch.
   * - Merge selected branch into current
     - Join two branches that are really one curve, ordered by which of their ends meet.
   * - Disentangle branch
     - Reorder the points of the branch so that the curve drawn through them is the shortest one
       there is, measured in the visible box. Moving a point by hand changes its coordinates but not
       its place in the branch, so the line starts doubling back on itself; this puts the order right.
       It follows the curve rather than sorting by the parameter, so a branch running around a fold
       comes out as the S it is. Nothing is recomputed, so it is undone by reloading the diagram.
   * - Delete branch...
     - Remove a whole branch and every state dump it owns. It asks first: unlike a split or a merge,
       this cannot be undone by reloading the diagram.
   * - Move-point mode *(toggle)*
     - Switch what the arrow keys and the mouse do from continuing to moving along the branch.
   * - Grab selected point *(toggle)*
     - While grabbed, the point-selection keys move the point itself along its branch instead of
       moving the selection.

View
^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Entry
     - What it does
   * - X axis / Y axis
     - Submenus: what each axis shows. Either can be any global parameter or any observable, which is
       what lets the same canvas be an ordinary bifurcation diagram or the locus of a bifurcation in a
       plane of two parameters. Orbit branches add their period and the minimum/maximum of each
       observable over a cycle.
   * - Next observable
     - Cycle the vertical axis through the available observables.
   * - Interpolated splines *(toggle)*
     - Draw the branches as splines through the computed points rather than as straight segments.
   * - Show branches from other slices *(toggle)*
     - Draw branches computed at other values of the non-continued parameters, faintly, as context.
   * - Trust stability inferred from the determinant *(toggle)*
     - Whether the stability a quick-mode sweep *infers* is drawn as known (solid/dashed) or as
       unknown (dotted).
   * - Logarithmic parameter axis / observable axis *(toggle)*
     - Log scale on the horizontal or the vertical axis.
   * - Field plots
     - Submenu: refresh the problem's own plots, switch off their automatic update, and add or remove
       an eigenfunction view. An eigenfunction view is the same plot definition with the eigenvector
       set, so it needs no extra plotter in the script. The imaginary part is offered only when the
       eigenvector is genuinely complex.
   * - Autoscale to data
     - Fit the axes around everything computed so far.
   * - Reset view around current point
     - Recentre on the current point.
   * - Fix parameter range...
     - Pin the parameter axis, which also bounds where *Multistep* and the deflated scan stop.

Custom
^^^^^^

Present only when the script defines
:py:attr:`~pyoomph.utils.bifurcation_gui.BifurcationGUI.custom_key_functions`. Each entry is one of
those functions, called with the GUI object, and reachable from the key it is registered under.

Settings and Help
^^^^^^^^^^^^^^^^^

*Settings → Keyboard shortcuts...* opens the rebinding dialog; what you change there is stored per
user and survives a restart (:ref:`advstabbifguikeys`). *Help → Shortcut reference* lists the current
bindings.
