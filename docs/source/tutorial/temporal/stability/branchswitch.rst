.. _sectemporalbranchswitch:

Branch switching
~~~~~~~~~~~~~~~~

Arclength continuation follows one branch of solutions. At a bifurcation, however, another branch passes through the very same point, and the continuation will just walk past it without ever noticing that there was a turn-off. Stepping over onto that second branch is called *branch switching* and is done by :py:meth:`~pyoomph.generic.problem.Problem.switch_branch`.

Not all bifurcations have a second branch, though. At a fold, the single branch merely turns around, and the arclength continuation gets around it by itself. At a *transcritical* and at a *pitchfork* bifurcation, however, there is, so we consider these with a specifically made-up ODE system

.. math:: :label: eqodebranchswitchsys

   \begin{aligned}
   \partial_t x&=rx+x^2-x^3\\
   \partial_t y&=\left(x-\frac{3}{2}\right)y-y^3
   \end{aligned}

with the single parameter :math:`r`. Its stationary solutions can be written down by hand, which is what makes it a good test case. Since :math:`\partial_t x` does not involve :math:`y` at all, the :math:`x`-equation can be solved on its own. It is easy to see that :math:`x` has the trivial solution :math:`x=0` and a nontrivial one, :math:`x^2-x-r=0`. This gives a transcritical bifurcation at :math:`r=0` and a fold at :math:`r=-1/4`. For the :math:`y`-equation, the parameter :math:`r` does not enter, but :math:`x` does, and it forms a pitchfork normal form, where :math:`x` takes the function of a parameter (shifted  by :math:`x=3/2`), i.e. we expect a pitchfork bifurcation once :math:`x` passes :math:`3/2`.

The numerical implementation is straightforward:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: class BranchSwitchODE(ODEEquations):
   :end-at: self.add_equations(eqs@"ode")


Branch switching requires the analytically derived Hessian, since the normal form of the bifurcation, from which the position of the second branch is predicted, is built from it. It is therefore mandatory to call :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis` with ``analytic_hessian=True``. We furthermore let the state files also store the arclength tangent, which we will need further below:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Branch switching needs the analytically derived Hessian
   :end-at: problem.continuation_data_in_states=True

Then, a few helper functions are defined, in particular one that performs arclength continuation until the largest eigenvalue changes sign, i.e. until we have just stepped over a bifurcation, and one that simply carries on along a branch:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: def continue_to_bifurcation(ds
   :end-before: # Start on the trivial branch


We start on the trivial branch :math:`x=y=0` at :math:`r=-1`, where it is stable, and continue upwards until the sign change announces the bifurcation at :math:`r=0`. The trivial branch does not stop there, though - it merely becomes unstable. For a complete diagram, we therefore scan it out first and only then turn to the bifurcation. To be able to return to it, the state is stored before continuing, which is what ``continuation_data_in_states=True`` above was set for: a reloaded state must also know in which direction along the branch it was travelling, or the resumed continuation picks the wrong one and might drop off the branch. The very same pattern will be used at the fold and at the pitchfork below:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Start on the trivial branch
   :end-at: problem.load_state(near_transcritical,quiet=False) # go back to the bifurcation

Back at the bifurcation, we converge exactly onto it with :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking`, as discussed in :numref:`sectemporalbiftrack`. What is new here is :py:meth:`~pyoomph.generic.problem.Problem.classify_bifurcation`, which calculates the normal form of the bifurcation we are sitting on and thereby tells us which kind it is:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Converge exactly onto the bifurcation
   :end-at: print("Found a",normal_form["type"],"at r,x,y =",get_state())

It reports ``transcritical``, so there is a second branch, and :py:meth:`~pyoomph.generic.problem.Problem.switch_branch` steps onto it. It deactivates the bifurcation tracking by itself, since the switch needs the plain, unaugmented system, and it returns a step size to carry on the continuation with:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # It is a transcritical, so there is a second branch
   :end-at: ds=continue_to_bifurcation(ds)

The ``direction`` argument selects which of the two sides of the bifurcation is taken, here the one towards decreasing :math:`r`. The returned step size is deliberately small: right next to a bifurcation the branch is still badly conditioned, and :py:meth:`~pyoomph.generic.problem.Problem.arclength_continuation` will grow it again by itself. The bifurcation point itself is written to the output file before switching, so that the new branch is connected to it when it is plotted.


The next sign change along the new branch is the fold at :math:`r=-1/4`. Before converging onto it, we store the current state again, so that the continuation can be resumed exactly here afterwards:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # We are just past the fold
   :end-at: problem.deactivate_bifurcation_tracking()

This time, :py:meth:`~pyoomph.generic.problem.Problem.classify_bifurcation` reports ``fold`` and :py:meth:`~pyoomph.generic.problem.Problem.switch_branch` refuses to do anything, since there simply is no second branch to go to. We hence just return to the stored state and let the arclength continuation pass around the fold on its own:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Back to just past the fold
   :end-at: ds=continue_to_bifurcation(ds)


Beyond the fold, the branch runs upwards in :math:`x` again and the next sign change is the pitchfork at :math:`r=3/4`, :math:`x=3/2`. As on the trivial branch, we first scan out the part of :math:`y=0` beyond it, which has just become unstable, and then come back:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Beyond the pitchfork, y=0 has become unstable
   :end-at: problem.load_state(near_pitchfork,quiet=True)

Two things then have to be done differently from the transcritical bifurcation:

* The bifurcation must be located with the pitchfork tracker, i.e. by passing ``"pitchfork"`` to :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking`. The fold tracker used above augments the system by :math:`\mathbf{J}\vec{v}=0`, which is itself singular at a symmetry-breaking bifurcation, and the Newton solve would fail with a singular matrix.
* The bifurcation should be classified with ``assume="pitchfork"``. The distinction between a pitchfork and a transcritical is made by whether the quadratic coefficient of the normal form vanishes - and here it vanishes *identically*, by virtue of the symmetry :math:`y\to-y`, so all that is numerically left of it is round-off. Whenever the symmetry of the system already answers the question, it is better to say so than to let the numbers decide.

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # The pitchfork. It must be located
   :end-at: print("Found a",normal_form["type"],"at r,x,y =",get_state())

The two branches :math:numref:`eqodebranchswitchpf` emanating from a pitchfork lie at the *same* values of :math:`r` and differ only in the sign of the amplitude, i.e. the ``direction`` argument is the only thing that tells them apart. We therefore store the state at the bifurcation, switch onto one of them, come back and switch onto the other:

.. literalinclude:: bifurcation_branch_switching.py
   :language: python
   :start-at: # Both branches of the pitchfork sit at the same r
   :end-at: analytically y = {:+.5f}

The printed values agree with :math:numref:`eqodebranchswitchpf` to all digits shown. The complete diagram is depicted in :numref:`figodebranchswitch`.

..  figure:: branchswitch.*
	:name: figodebranchswitch
	:align: center
	:alt: Bifurcation diagram of the branch switching example
	:class: with-shadow
	:width: 100%

	Bifurcation diagram of :math:numref:`eqodebranchswitchsys`, obtained by branch switching. Solid lines are stable, dashed lines unstable. Since :math:`x` does not depend on :math:`y`, the two branches emerging from the pitchfork coincide with the second branch in the left plot.

The very same three methods, :py:meth:`~pyoomph.generic.problem.Problem.classify_bifurcation`, :py:meth:`~pyoomph.generic.problem.Problem.switch_branch` and the state files that let one revisit a bifurcation, also work for spatio-temporal problems, where the branches are not known beforehand and the diagram must be explored step by step.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_branch_switching.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
