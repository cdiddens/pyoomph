Adjusting a parameter so that a specified condition is met
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following, we discuss a neat feature of fully-implicit methods: We have a problem that outputs an observable as function of a parameter. While such problems can be easily scanned to get a curve of the observable as function of the parameter (*forward problem*), we can also automatically adjust the parameter so that the observable attains some desired value (*inverse problem*).

Forward problem
...............

Consider a rectangular cavity which is driven by tangential shear stress at the top boundary and no-slip on all other boundaries.
The tangential shear stress at the top is given by a parameter :math:`T` and we want to observe the induced velocity inside the cavity by the root-mean-square velocity 

.. math::

   \begin{aligned}
   U&=\sqrt{\frac{1}{A}\int \vec{u}^2\:\mathrm{d}^2x}
   \end{aligned}

where :math:`A` is the area of the cavity.


We can easily assemble such simple problems by using the predefined equations classes. We do not even have to define a specific :py:class:`~pyoomph.generic.problem.Problem` class, but instead assemble the problem directly:


.. literalinclude:: cavity_forward_problem.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: out_file.add_row(T, U_val)

Main steps are again to define :math:`T` as a global parameter, so that it can be scanned afterwards. Then, for the calculation of :math:`U`, we first have to split it into spatial integrals and finally express :math:`U` in terms of these integrals.

.. math::

   \begin{aligned}
   U_\text{sqr}&=\int \vec{u}^2\:\mathrm{d}^2x \\
   A&=\int 1\:\mathrm{d}^2x \\   
   U&=\sqrt{U_\text{sqr}/A}
   \end{aligned}

This can all be achieved in a single :py:class:`~pyoomph.equations.generic.IntegralObservables`, where the last step in the calculation is obtained by a ``lambda`` using the correct parameter names to get the values from the spatial integrals. :py:meth:`~pyoomph.generic.problem.Problem.create_text_file_output` generates a file in the output directory, which is then written by scanning over :math:`T` using :py:meth:`~pyoomph.generic.problem.Problem.go_to_param`. The results are plotted in :numref:`figspatialcavity`.

..  figure:: cavity.*
	:name: figspatialcavity
	:align: center
	:alt: Flow in a cavity
	:class: with-shadow
	:width: 80%

	(left) Velocity field inside the cavity for :math:`T=100` . (right) RMS velocity :math:`U` as function of :math:`T`



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <cavity_forward_problem.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   
		
		
Inverse problem
...............

Consider you now want to adjust :math:`T` so that the observable :math:`U` attains some specific value :math:`U_\text{desired}`. Of course, you can look up the appropriate value of :math:`T` from the graph in :numref:`figspatialcavity`, but this requires to first solve the forward problem for an entire range of the parameter :math:`T`. A more sophisticated approach would be bisection in :math:`T` until the condition :math:`U=U_\text{desired}` is met.

However, fully-implicit solvers, as used in pyoomph, are capable of inverting the problem and of directly solving for :math:`T` so that :math:`U=U_\text{desired}` holds. Obviously, :math:`T` is then not a free parameter anymore, but an unknown. Instead, we get :math:`U_\text{desired}` as a parameter and the residual equation for :math:`T` becomes

.. math::

   \begin{aligned}
   U-U_\text{desired}=0
   \end{aligned}

i.e. we adjust :math:`T` so that we have the desired average velocity inside the cavity.
To incorporate this equation into the problem, we first reformulate it to

.. math::

   \begin{aligned}
   R_T=\int (\vec{u}^2  - U_\text{desired}^2)\:\mathrm{d}^2x&=0
   \end{aligned} 
   
:math:`R_T` is now in a suitable residual formulation to be used directly in a weak form to find the value for :math:`T`.

So we first remove :math:`T` as parameter and introduce :math:`U_\text{desired}` instead. Then, :math:`T` must be defined as an unknown:

.. literalinclude:: cavity_inverse_problem.py
   :language: python
   :start-at: #T = problem.define_global_parameter(T=0) # T is not a parameter anymore
   :end-at: T,Ttest=problem.add_global_dof("T",initial_condition=10)

Since :math:`T` is a single degree of freedom (i.e. not a field), we add it to the problem with :py:meth:`~pyoomph.generic.problem.Problem.add_global_dof`. It is important to provide a reasonable initial condition for :math:`T`, since Newton's method can run into convergence issues otherwise.

Then, we add the equation for :math:`T`, which can be achieved by 

.. literalinclude:: cavity_inverse_problem.py
   :language: python
   :start-at: # We want to adjust T, so that the residual R=integral(u^2) - Udesired^2 = 0
   :end-at: eqs += WeakContribution(dot(var("velocity"), var("velocity")) - Udesired*Udesired, Ttest)

:py:meth:`~pyoomph.generic.codegen.WeakContribution` is just an equation class which just adds a single contribution to the weak formulation. Here, just the spatial integral of :math:`\vec{u}^2  - U_\text{desired}^2` over the domain to the residuals of :math:`T`. 

There is one caveat for this particular equation, namely that it is quadratic in the velocity, i.e. :math:`\vec{u}^2`. By default, pyoomph starts with a zero velocity field, which then produces a zero Jacobian row in the first Newton iteration. So we first must find a reasonable guess for the velocity field without trying to adjust the unknown forcing :math:`T`. We therefore remove :math:`T` from the degrees of freedom and solve for the velocity and pressure only:


.. literalinclude:: cavity_inverse_problem.py
   :language: python
   :start-at: # Newton's method cannot start with a zero velocity field, since u^2 will produce an empty Jacobian row.
   :end-at: # When leaving the 'with' block, T is an unknown again, and we can solve the full problem to adjust T to reach the desired U

:py:meth:`~pyoomph.generic.problem.Problem.select_dofs` should be used in a ``with`` statement. Different degrees of freedom can be then selected or unselected and all :py:meth:`~pyoomph.generic.problem.Problem.solve` commands within the ``with`` block are only performed on the selected (or not unselected) degrees of freedom. When leaving the ``with`` block, all degrees of freedom become active again, so that we can now solve for the full problem, including the adjustment of :math:`T`:


.. literalinclude:: cavity_inverse_problem.py
   :language: python
   :start-at: # Solve to adjust T to reach the desired U
   :end-at: print("U (calculated)=", problem.get_mesh("domain").evaluate_observable("U"))

Note that the definition of the unknown :math:`T` via :py:meth:`~pyoomph.generic.problem.Problem.add_global_dof` above stores the unknown in an ODE called ``"globals"``. Therefore, we have to access the value of :math:`T` using :py:meth:`~pyoomph.generic.problem.Problem.get_ode`.

At this point, one could of course do a scan in :math:`U_\text{desired}`, but this would just yield the same plot as in the forward problem, just as the inverse function :math:`T(U_\text{desired})`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <cavity_inverse_problem.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   
		
.. note::

	Finding a suitable initial condition for the parameter to be adjusted (:math:`T` in the discussed inverse problem) is crucial. If it is hard to guess a reasonable initial value, one can replace :math:`T \leftarrow (1-\mu)T_0+\mu T`, where :math:`T_0` is a constant (some initial guess) and :math:`\mu` is a blending parameter. One can then start with :math:`\mu=0` and gradually blend :math:`\mu\to 1`, e.g. via :py:meth:`~pyoomph.generic.problem.Problem.go_to_param`. However, to close the problem for :math:`\mu=0`, one has to add e.g. ``equation_contribution=(1-mu)*(var("T")-T0)`` to the ``add_global_dof`` definition of :math:`T` and augment the ``WeakContribution`` with the factor :math:`\mu`, e.g. by using ``mu*Ttest`` as second argument. Thereby, for :math:`\mu=0`, the residual for :math:`T` will be just :math:`T-T_0=0`, whereas for :math:`\mu=1` one recovers the original form. In particular, also the issue with Newton's method in the first step (due to the zero velocity field) is circumvented when starting with :math:`\mu<1` (e.g. :math:`\mu=0`).
		
