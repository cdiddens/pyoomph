.. _secODEpendulum:

Enforcing constraints by Lagrange multipliers
---------------------------------------------

As an example, let us consider the pendulum equation. A mass :math:`m` is located at :math:`(x,y)`, but its movement is constrained by the length :math:`L` of the pendulum, i.e. by the constraint function

.. math:: g(x,y)=\sqrt{x^2+y^2}-L=0\,.

The easiest way to solve this is of course, the conventional way: instead of expressing the system by the two coordinates :math:`x(t)` and :math:`y(t)`, we introduce the angle :math:`\phi`, which is the *generalized coordinate*. This angle is measured with respect to the equilibrium at :math:`(x,y)=(0,-L)`. Thereby, one arrives at the pendulum equation

.. math:: \partial_t^2\phi+\frac{g}{L}\sin(\phi)=0\,.

If you have read this tutorial until here, implementing this equation should be a trivial task:

.. literalinclude:: pendulum_generalized_coordinate.py
   :language: python
   :start-at: from pyoomph import * # Import pyoomph
   :end-at: problem.run(endtime=100,numouts=1000)

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <pendulum_generalized_coordinate.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   		

However, in general, it is not always easy to find the generalized coordinate(s) for which the system automatically fulfills all imposed constraints. In that case, one still can enforce the constraints via Lagrange multipliers. In the given example of the pendulum, let us assume we were unable to find the generalized coordinate :math:`\phi` from the constraint :math:`g(x,y)`. We therefore would have to solve the full system, i.e. the equations of motion

.. math::

   \begin{aligned}
   m\partial_t^2 x&=0 \\
   m\partial_t^2 y&=-mg
   \end{aligned}

subject to the *scleronomic* and *holonomic* constraint :math:`g(x,y)=\sqrt{x^2+y^2}-L=0`. From the *analytical mechanics*, i.e. *Lagrangian mechanics*, we know how to implement the constraint, namely by introducing a Lagrange multiplier :math:`\lambda` and minimizing the action functional, which eventually leads to Lagrange's equations of the first kind, i.e.

.. math::

   \begin{aligned}
   \frac{\partial \mathcal{L}}{\partial x}-\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial \mathcal{L}}{\partial \dot{x}}+\lambda \frac{\partial g}{\partial x} &=0\\
   \frac{\partial \mathcal{L}}{\partial y}-\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial \mathcal{L}}{\partial \dot{y}}+\lambda \frac{\partial g}{\partial y} &=0
   \end{aligned}

with the Lagrangian

.. math:: \mathcal{L}=T-V=\frac{1}{2}m\left(\dot{x}^2+\dot{y}^2\right)-mgy

This gives rise to the equations of motion augmented by the additional force stemming from the tension of the rod of the pendulum

.. math:: :label: eqaugmotionpendy

   \begin{aligned}
   m\partial_t^2 x&=-\lambda \frac{x}{\sqrt{x^2+y^2}}\\
   m\partial_t^2 y&=-mg-\lambda \frac{y}{\sqrt{x^2+y^2}} 
   \end{aligned}

As a third equation to determine the three unknowns :math:`x`, :math:`y` and :math:`\lambda` (in fact, usually it is treated as five unknowns :math:`x`, :math:`y`, :math:`\dot x`, :math:`\dot y` and :math:`\lambda`), one has :math:`g(x,y)=0`.

We solve these equations by splitting the system into the unconstrained motion in an equation class ``NewtonsLaw2d``:

.. literalinclude:: pendulum_lagrange_multiplier.py
   :language: python
   :start-at: class NewtonsLaw2d(ODEEquations):
   :end-at: self.add_residual( (partial_t(y)-ydot)*testfunction(ydot))

and the constraint itself, which adds the additional terms stemming from the constraint to the equation of motion and solves for the unknown Lagrange multiplier :math:`\lambda` by solving the constraint equation :math:`g(x,y)=0`:

.. literalinclude:: pendulum_lagrange_multiplier.py
   :language: python
   :start-at: #Pendulum constraint: Enforcing sqrt(x**2+y**2)=L via a Lagrange multiplier
   :end-at: self.add_residual((currentL-self.L)*testfunction(lambda_pendulum)) #constraint equation to solve for the Lagrange multiplier

In the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields`, we introduce the Lagrange multiplier :math:`\lambda` as ODE variable. In the function :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`, we add the corresponding forces to the residual form of :math:numref:`eqaugmotionpendy`. Since the residual form requires putting all terms on one side, note that the sign of the additional terms proportional to :math:`\lambda` has changed. By using ``testfunction(x)`` and ``testfunction(y)``, it is ensured that this additional forcing is indeed added to the correct equation of motion. Finally, one still has the constraint equation :math:`g(x,y)=0` and the degree of freedom :math:`\lambda`. This is accounted for in the last line where the constraint equation is solved in the residual term for the Lagrange multiplier :math:`\lambda`.

Besides :py:func:`~pyoomph.expressions.square_root`, which is just the mathematical square root for pyoomph expressions, there is one additional function occurring in this snippet that has not been discussed yet, namely :py:func:`~pyoomph.expressions.generic.subexpression`. Wrapping an expression in a :py:func:`~pyoomph.expressions.generic.subexpression` does not change the results at all, however, we note that the term :math:`\sqrt{x^2+y^2}` occurs multiple times in the residuals. By wrapping it in a :py:func:`~pyoomph.expressions.generic.subexpression`, pyoomph will internally make sure to evaluate this term only once, store it in a local variable and reuse this local variable for all further occurrences of the wrapped expression. Depending on the complexity of the expression wrapped in a :py:func:`~pyoomph.expressions.generic.subexpression`, this can lead to a huge performance gain.

At a last step, the problem definition reads like this:

.. literalinclude:: pendulum_lagrange_multiplier.py
   :language: python
   :start-at: def __init__(self):
   :end-at: problem.run(endtime=100,numouts=10000)

Here, both equations, ``NewtonsLaw2d`` and ``PendulumConstraint`` get combined. While ``NewtonsLaw2d`` can be solved without the constraint (which would just result in a free fall of the mass), ``PendulumConstraint`` is only valid when combined with an equation that defines the variables :math:`x` and :math:`y`, since these are required for the constraint.

Again, we make use of the :py:class:`~pyoomph.equations.generic.ODEObservables` class to define further output quantities, which depend on the degrees of freedom. Here, we are interested e.g. in the angle :math:`\phi`, which can be calculated by :math:`\phi=\arctan(-x/y)`, or in order to accurately treat the special case :math:`y=0`, ``phi=atan2(var("x"),-var("y"))``. By adding this to the system, the output file will contain one additional column for :math:`\phi`, which is again automatically calculated at each output step. As before in :numref:`secodetimestepping`, one can e.g. also add further arguments to the constructor of :py:class:`~pyoomph.equations.generic.ODEObservables`, e.g. ``Ekin=0.5*self.mass*(partial_t(var("x"))**2+partial_t(var("y"))**2)`` for the kinetic energy. A plot of :math:`\phi(t)` and :math:`\lambda(t)` is shown in :numref:`figodependulum`.


..  figure:: pendulum.*
	:name: figodependulum
	:align: center
	:alt: Pendulum with an explicit constraint enforced by a Lagrange multiplier
	:class: with-shadow
	:width: 100%
	
	Pendulum equation with the help of a Lagrange multiplier :math:`\lambda` enforcing the rod constraint. Note how the oscillation :math:`\phi(t)` shows an anharmonic curve. Furthermore, it is apparent that :math:`\lambda` is negative whenever :math:`|\phi|` is large. In this case, the rod of the pendulum is pushing the mass outwards. Close to the points where the velocity is highest, i.e. the slope of :math:`\phi` is steepest, maxima in :math:`\lambda` can be found. This corresponds to the high centripetal force required for the high velocity.

.. only:: html
	
	.. container:: downloadbutton

		:download:`Download this example <pendulum_lagrange_multiplier.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   		



Finally, note that we need quite some time steps to get a stable and conserving scheme (using ``"BDF2"``) here. One can play around with the :py:func:`~pyoomph.expressions.generic.time_scheme` function as explained in :numref:`secODEtimescheme`, but one should not apply :py:func:`~pyoomph.expressions.generic.time_scheme` on the residuals added in the ``PendulumConstraint``, since the pendulum constraint should be always fulfilled, in particular exactly in the time step :math:`n` we are solving for, whereas time schemes like ``"TPZ"``, ``"MPT"`` or similar would include values from previous time steps.

Here, we have seen how to enforce a constraint in analytical mechanics. However, the method of Lagrange multipliers is even more powerful since it can be applied nearly anywhere to enforce constraints in a system. Let us first see what happens if we remove (or comment out) the line ``self.add_residual(lambda_pendulum*y/currentL*testfunction(y))``. In this case, there will be no additional force added to the :math:`y`-direction, i.e. the particle will exhibit a free fall in :math:`y`-direction. However, there is now still a force acting on the :math:`x`-direction to keep the constraint fulfilled. The particle will hence fall down and experience exactly that force in :math:`x`-direction, which is necessary to keep it on the circle with radius :math:`L`. Of course, since it does not obey Lagrange's equations anymore, the energy will not be conserved. Furthermore, the simulation will crash the moment the particle is reaching the south pole at :math:`(x,y)=(0,-L)`. It is still falling, but the constraint cannot be satisfied by any force acting just in :math:`x`-direction. However, this discussion shows that you can have a constraint in a system that depends on multiple variables, here :math:`x` and :math:`y`, but the dynamics is only changed in a particular degree of freedom, which is :math:`x` here.

This gives rise to the general recipe for how to use Lagrange multipliers to enforce arbitrary constraints in a system: Suppose you have a vector of unknowns :math:`\vec{U}=(U_1,\ldots,U_N)` and :math:`M` constraints, which are expressed by implicit equations :math:`g_i(\vec{U})=0` for :math:`i=1,\ldots,M`. We can enforce these constraints by adding the :math:`M` Lagrange multipliers :math:`\lambda_i` (:math:`i=1,\ldots,M`) to the system. We then add the constraints :math:`g_i(\vec{U})` times the test function of :math:`\lambda_i` for all :math:`i=1,\ldots,M` to the residuals of the system. Finally, for each degree of freedom :math:`U_j` which shall be adjusted to ensure the :math:`i^{\text{th}}` constraint holds, we add :math:`\lambda_i\partial g_i(\vec{U})/\partial_{U_j}` to the residual of :math:`U_j`, i.e. by projecting it to the corresponding test function of :math:`U_j`.


