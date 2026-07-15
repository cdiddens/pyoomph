Coupled harmonic oscillators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| In the following, we will discuss three different routes to the very same result, namely to couple two harmonic oscillators, i.e.

  .. math:: :label: eqcoupledoscis

     \begin{split}
     \partial_t^2 y_1 + K_{11} y_1 + K_{12} y_2 &=0 \\
     \partial_t^2 y_2 + K_{21} y_1 + K_{22} y_2 &=0 
     \end{split}

   

.. container:: center

   **Method I**

We start with the naive way of implementing this, i.e. by defining a specific equation class, inherited from the generic :py:class:`~pyoomph.generic.codegen.ODEEquations` class:

.. literalinclude:: coupled_oscillators_method_1.py
   :language: python
   :start-at: from pyoomph import * # Import pyoomph
   :end-at: problem.run(endtime=100,numouts=1000)

Here, in particular the line

.. container:: center

   ``self.add_residual(residual1*testfunction(y1)+residual2*testfunction(y2))``

| is of interest. It is exactly the formulation as required in :math:numref:`eqscalresresultspec`, with :math:`\vec{U}=\left[y_1(t),y_2(t)\right]` and :math:`\vec{\mathcal{R}}` the vector containing the lhs of :math:numref:`eqcoupledoscis`, i.e. ``residual1`` and ``residual2`` in the code. The test functions :math:`V^{(j)}_i` are just obtained by the calls of :py:func:`~pyoomph.expressions.generic.testfunction`. We actually solve now the residual of the first equation in :math:numref:`eqcoupledoscis` by adjusting :math:`y_1` and the second equation by adjusting :math:`y_2` until both equations are fulfilled.  


.. only:: html  

	.. container:: downloadbutton

		:download:`Download this example <coupled_oscillators_method_1.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    

.. container:: center

   **Method II**

The previous method is straightforward, but would require rewriting a lot of code if e.g. a third oscillator should be coupled to the system. Of course, one could use ``len`` to obtain the dimension of the coupling matrix ``Kmatrix`` and perform the calls for :py:meth:`~pyoomph.generic.codegen.ODEEquations.define_ode_variable` and the residual calculation in a loop. However, pyoomph also offers another way of coupling multiple equations. We have already seen that we can augment an equation e.g. with an :py:class:`~pyoomph.equations.generic.InitialCondition` object to set an initial condition or an :py:class:`~pyoomph.output.generic.ODEFileOutput` object to make sure that output is written. By the very same way, also multiple equations can be coupled. To that end, we write an equation class for a general second order equation of the form

.. math:: \partial_t^2y+T=0\,,

where :math:`T` is a place holder for an arbitrary mathematical expression. The corresponding equation class looks like this:

.. code:: python

   from pyoomph import * # Import pyoomph 
   from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

   class SingleHarmonicOscillator(ODEEquations):
   	def __init__(self,name,terms): #Pass the name of the unknown and the terms T
   		super(SingleHarmonicOscillator,self).__init__()
   		self.name=name #Store the name of the unknown
   		self.terms=terms #and the terms to consider
   		
   	def define_fields(self):
   		self.define_ode_variable(self.name) 
   		
   	def define_residuals(self):
   		y=var(self.name) #Bind the single variable
   		# Calculate the residuals
   		residual=partial_t(y,2)+self.terms #Just add the passed terms here
   		self.add_residual(residual*testfunction(y))
   		

In the definition of the problem, we can now combine two instances of the ``SingleHarmonicOscillator`` class and by passing the correct names and additional terms, we can recreate the system :math:numref:`eqcoupledoscis`. This is achieved by changing the method :py:meth:`~pyoomph.generic.problem.Problem.define_problem` of the ``TwoCoupledHarmonicOscillatorProblem`` as follows:

.. literalinclude:: coupled_oscillators_method_2.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_equations(eqs@"coupled_oscillator")

| This is a common way to couple multiple equations in multi-physics, e.g. a Navier-Stokes equation and a temperature equation for a Rayleigh-Bénard setting, later on.  

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <coupled_oscillators_method_2.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    
	

.. container:: center

   **Method III**

The final method is very similar to the second method, but the equations are not combined to a single equation on the domain ``"coupled_oscillator"``, but we use two different domains:

.. literalinclude:: coupled_oscillators_method_3.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_equations(eqs1@"oscillator1"+eqs2@"oscillator2")

Note that the unknowns of both oscillators have the same name, namely ``"y"``. However, since we are creating two distinct equations on different domains (``"oscillator1"`` and ``"oscillator2"``), the same names do not interfere. However, we have to be careful, since the unknown obtained by ``var("y")`` will have different meanings in both domains, namely the unknown ``"y"`` defined in the particular domain. When coupling unknowns of different domains, it is therefore required to pass the keyword ``domain`` in the :py:func:`~pyoomph.expressions.generic.var` function to remove this ambiguity.

Also, the initial conditions are now separated, since initial conditions can only be set on the domain where the corresponding unknown is defined. Since there are also two instances of :py:class:`~pyoomph.output.generic.ODEFileOutput`, each domain will write its own output file, containing just the single degree of freedom within this domain. The calculated solution is depicted in :numref:`figodecoupledoscisn`.

.. _figodecoupledoscis: 

..  figure:: coupledoscis.*
	:name: figODEcoupledoscisn
	:align: center
	:alt: Coupled harmonic oscillators
	:class: with-shadow
	:width: 100%
	
	Output for the user-defined harmonic oscillator equation with damping and driving.
	
.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <coupled_oscillators_method_3.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    
		
