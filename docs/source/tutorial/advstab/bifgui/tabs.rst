.. _advstabbifguitabs:

The tabs on the right
~~~~~~~~~~~~~~~~~~~~~

The notebook on the right of the window holds the settings that would otherwise be attributes set in
the script, and the tables that say where you are. Everything in it is also an attribute of the
underlying :py:class:`~pyoomph.utils.bifurcation_gui.controller.BifurcationController`, so anything
that can be adjusted here can equally be preset before :py:meth:`~pyoomph.utils.bifurcation_gui.BifurcationGUI.start`.

A typed value is committed with :kbd:`Return` or by leaving the field; nonsense is refused, said so in
the log, and the previous value kept. Several fields accept ``auto`` (or an empty string) to mean
"derive it", which is not the same as zero.

Continuation
^^^^^^^^^^^^

The tab you work in. Step size, what is solved at each point, and what the axes show.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - Step size ds
     - The arclength step, with ``/1.25``, ``x1.25`` and ``reverse`` buttons under it. Its sign is the
       direction of travel.
   * - Normal modes
     - Azimuthal :math:`m` (integers) or Cartesian :math:`k` (reals) to solve alongside the base state,
       comma-separated; empty means the base state alone. Only meaningful on a problem set up for them
       (see :ref:`azimuthalstabana`).
   * - Modes at every step
     - Solve those modes at every continuation point rather than on demand. Off by default: a scan of
       :math:`N` modes is :math:`N` extra eigensolves per point.
   * - Stripe Re,Im,max
     - The region ``|Re| < Re``, ``|Im| < Im`` that a stripe scan searches, and the maximum number of
       eigenvalues it may return. A scan that comes back with exactly the maximum probably missed some.
   * - Adapt
     - Whether the mesh is adapted during continuation: ``off``, ``when_needed`` (ask the problem's own
       ``remeshing_necessary()``) or ``every_n``.
   * - Adapt every N
     - The :math:`N` of ``every_n``.
   * - Adapt to the eigenfunction
     - Refine towards the critical eigenfunction rather than towards the base solution alone - what you
       want when the instability is finer than the state it grows on.
   * - Eigenvalues
     - How many eigenvalues are computed per point (``neigen``). More costs time but is what lets you
       see a pair coming towards the axis before it crosses.
   * - Shift
     - The shift handed to the eigensolver. Shift-invert returns the eigenvalues nearest this value, so
       it decides which part of the spectrum you get to see.
   * - x axis / y axis
     - What the two axes show - any parameter or any observable. The same as the *View* menu's
       submenus.
   * - Scale arclength
     - Whether the arclength metric is retuned as the continuation proceeds.
   * - Param. fraction
     - How much of one step goes into the parameter rather than into the solution.
   * - Interpolated splines
     - Draw branches as splines through the points instead of straight segments.
   * - Mode: Arclength / Move point
     - What the keys and the mouse do: continue the branch, or move along the points already computed.
   * - Grab selected point
     - While on, the selection keys move the point itself along its branch.

Parameters
^^^^^^^^^^

One row per global parameter of the problem, with its value and its role - which one is being
continued, and what the others are held at. A bifurcation diagram is a section through parameter
space, so this is part of the result rather than a setting, which is why it is a permanent table
rather than a dialog. The two buttons below start a new diagram continuing in the selected parameter,
or set the value of one that is not being continued - i.e. move to another section. The label at the
bottom names the section you are in, which is what tells branches computed elsewhere apart from the
ones on screen.

Points
^^^^^^

What the current and the selected point are: parameter value, observables, stability, the type of the
bifurcation if it is one, and the period of the orbit if it is on an orbit branch.

Below them sits the **spectrum** at the current point - the whole list, not just the leading
eigenvalue, because watching the others is how a Hopf pair is spotted before it crosses. Eigenvalues
with a positive real part are marked in red and the tracked one at a located bifurcation with a ``*``.
The list is *selectable*, and the button underneath locates the bifurcation of whichever eigenvalue is
picked. That is the way to work on a branch that is already unstable, where the mode about to cross is
not the leading one and tracking the leading one converges to the wrong bifurcation.

Branches
^^^^^^^^

The tree of branches and their points; double-clicking a point goes to it. The buttons split the
branch at the selected point, merge the selected branch into the current one, or delete a whole branch;
*Points → Disentangle branch* puts the order of a branch right after points have been moved by hand.
Splitting and merging are bookkeeping - they can be undone by reloading the diagram - while deleting a
branch removes its state dumps and cannot.

Deflation
^^^^^^^^^

Deflation finds solutions the diagram does not have yet: the residual is multiplied by a factor that
blows up at every solution already known, so Newton cannot converge onto one of those again and has to
go somewhere else. Unlike arclength continuation, it can find branches that are not connected to the
one you are on.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - Shift alpha
     - Shift of the deflation operator (default ``0.1``). Larger pushes the known solutions away over a
       shorter range, so a nearby second solution is easier to reach and a distant one harder.
   * - Power p
     - Order of the deflation, the power of :math:`1/\|U-W\|` (default ``2``).
   * - Perturbation
     - How far the guess is moved off a known solution before the deflated solve, or ``auto`` for a
       value read off the current solution. This carries the *scale* of the whole search - the
       deflation measures :math:`\|U-W\|` in units of it - so it is the one number that has to match
       the problem. A fixed value wrong by orders of magnitude finds a fraction of the branches.
   * - Random tries
     - Random perturbations tried per deflated solve before giving up (default ``4``). A failed attempt
       is a Newton solve that gives up early, so these are cheap next to the ones that succeed.
   * - Random seed
     - Seed of those perturbations, empty for a different sequence every run. Seeded by default so a
       search can be repeated.
   * - Max Newton it.
     - Iteration cap for a deflated solve (default ``20``). A deflated solve is *asked* to fail often -
       that is how the search terminates - so a low cap costs little.
   * - Perturb along the eigenvector
     - Also perturb along the leading eigenvector. That direction is a field rather than a random
       dof-index vector, so it means the same thing however the mesh is partitioned, and it usually
       points at the branch about to appear. Costs one eigensolve per attempt.
   * - Steps
     - Parameter steps a deflated continuation takes from the current value.
   * - d(parameter)
     - Signed parameter increment per step, or ``auto`` for ``ds``.
   * - Solve eigenproblems during the scan
     - On, like an ordinary step: a branch drawn without stability is half a bifurcation diagram. Turn
       it off where eigensolves dominate and fill the spectra in afterwards.

Orbit
^^^^^

Everything about the periodic orbits a Hopf bifurcation sheds. The top has the *Switch onto orbit* and
*Write the cycle* buttons and a line saying what this point is - and, when it cannot become an orbit,
why not.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Meaning
   * - Parameter step
     - How far off the Hopf the first orbit is placed, or ``auto`` for what the current ``ds`` buys.
       The parameter offset of a Hopf orbit is :math:`\varepsilon^2`, so this *is* that
       :math:`\varepsilon` squared.
   * - Amplitude factor
     - Extra factor on the amplitude of the starting guess, for a Hopf whose normal form is a poor
       predictor a finite step away.
   * - Check it did not collapse
     - Verify that the solved orbit did not fall back onto the stationary branch. On, because a
       collapsed "orbit" looks like a perfectly good branch of solutions until its period is read.
   * - Time steps
     - How many time points the orbit is discretized with. Raised at use time to a multiple of the
       collocation order and to an even number - with an odd number of intervals a
       differential-algebraic problem puts a spurious Floquet multiplier on exactly :math:`-1`, which
       is where a period doubling would be.
   * - Mode
     - The time discretization: ``collocation`` (the default), ``floquet``, ``central``, ``BDF2`` or
       ``bspline``. Only ``collocation`` and ``floquet`` carry a degree of freedom at the end of the
       period, which is what the Floquet multipliers are computed from; the others can be continued but
       have no stability. B-splines often converge more readily on the step off a Hopf, which is what
       *Apply to this orbit* is for.
   * - Order
     - Order of the discretization.
   * - Phase constraint
     - ``phase`` or ``plane``: which extra equation fixes the otherwise free time origin of the orbit.
   * - Observable samples
     - Samples per period used for the minimum, average and maximum of each observable, or ``auto`` for
       one per time step. Each sample is a full observable evaluation.
   * - Portable orbit files
     - Store the orbit's degrees of freedom as full state dumps, one per time point, instead of as raw
       dof vectors. Mesh- and partition-independent, at :math:`n_T` times the disk; forced on a
       distributed problem, where a raw vector means nothing outside its own partitioning.
   * - Apply to this orbit
     - Re-discretize the orbit that is *installed* with the settings above, keeping the orbit itself.
       Without it these settings only take effect at the next switch onto an orbit.
   * - Compute at every step
     - Compute the Floquet multipliers at every continuation point, like the eigensolve on an ordinary
       branch.
   * - Method
     - ``condensed`` (the default), ``periodic_schur`` or ``eigenproblem``.
   * - How many
     - How many multipliers to compute, or ``all``.
   * - Shift-invert
     - Whether the multiplier solve is shift-inverted.

The line at the bottom reports the multipliers found, or says why there are none - and, when the
installed orbit is in a mode that has none, points at *Apply to this orbit*.
