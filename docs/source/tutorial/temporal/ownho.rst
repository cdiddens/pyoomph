.. _secodecustomharmosci:

Defining your own harmonic oscillator equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Up to now, we have loaded the predefined harmonic oscillator equations with the line

.. code:: python

   from pyoomph.equations.harmonic_oscillator import HarmonicOscillator

However, one of the main features of pyoomph is actually the definition of arbitrary equations. Instead of including the predefined equation class, we will write our own class, inheriting from the generic :py:class:`~pyoomph.generic.codegen.ODEEquations`. This will serve as an example of how to express arbitrary ODEs within pyoomph. At the same time, let us generalize the equation to include damping :math:`\delta` and driving :math:`f(t)` as follows

.. math:: \partial_t^2 y + 2\delta\partial_t y +\omega^2 y = f

The corresponding code reads as follows:

.. code:: python

   from pyoomph import * # Import pyoomph 
   from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

   # We define a new class called HarmonicOscillator, which is inherited from the generic ODEEquations
   class HarmonicOscillator(ODEEquations):
   	# Constructor, allow us to set some parameters like the name of the variable, omega, damping and driving
   	def __init__(self,*,name="y",omega=1,damping=0,driving=0):
   		super(HarmonicOscillator,self).__init__()
   		self.name=name #Store these as members of the equation object
   		self.omega=omega
   		self.damping=damping
   		self.driving=driving
   		
   	# This function is called to define all fields in this ODE (system)
   	def define_fields(self):
   		self.define_ode_variable(self.name)
   		
   	# This function will finally define the equations
   	def define_residuals(self):
   		y=var(self.name) # bind the local variable y to the ODE variable
   		# Write the equation in residual form, i.e. lhs-rhs=0
   		residual=partial_t(y,2)+2*self.damping*partial_t(y)+self.omega**2 *y - self.driving
   		# And add the residual to the equation. Here, we have to project it on the test function.
   		self.add_residual(residual*testfunction(y))
   		

In the :py:meth:`~pyoomph.generic.codegen.ODEEquations.__init__` method, we take optional keyword arguments which have default values. These are then stored as members of the class. Of course, we have to call again the constructor of the super-class :py:class:`~pyoomph.generic.codegen.ODEEquations` using the Python builtin ``super``.

In the next step, the method :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields` is overloaded. Inside this function, all unknowns of the ODE system have to be defined. Here, it is just the single unknown :math:`y`.

Finally, the nitty-gritty, the equation has to be defined. This happens in the method :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`. To that end, the equation first has to be cast to a residual form, which can be done by putting all terms on one side of the equation:

.. math:: \partial_t^2 y + 2\delta\partial_t y +\omega^2 y - f =0

Of course, only the lhs of this equation is of relevance. Before adding it to the system of equations with the :py:meth:`~pyoomph.generic.codegen.BaseEquations.add_residual` method, it must be multiplied by a so-called *test function*, which can be obtained by the function :py:func:`~pyoomph.expressions.generic.testfunction`. This all happens in the line ``self.add_residual(residual*testfunction(y))``. Here, we only have one equation to be solved, i.e. the harmonic oscillator. In general, one will have a system of ODEs, i.e. multiple equations for multiple unknowns. This is where test functions become important, which will be explained in the next section.

The remainder of the code is very similar to before, but now also damping and driving will be considered:

.. code:: python


   # The remainder is almost the same is in the example nondim_harmonic_osci.py
   class HarmonicOscillatorProblem(Problem):

   	def __init__(self):
   		super(HarmonicOscillatorProblem,self).__init__() 
   		self.omega=1
   		self.damping=0.1 #But we add some default damping here
   		t=var("time")
   		self.driving=0.2*cos(0.2*t) #and some driving
   	

   	def define_problem(self):
   		eqs=HarmonicOscillator(omega=self.omega,damping=self.damping,driving=self.driving,name="y") #We also pass the damping and driving here
   		eqs+=InitialCondition(y=1-var("time"))
   		eqs+=ODEFileOutput() 
   		self.add_equations(eqs@"harmonic_oscillator") 
   		

   if __name__=="__main__":
   	with HarmonicOscillatorProblem() as problem:
   		problem.run(endtime=100,numouts=1000)

The output is plotted in :numref:`fignondimhocustom`.

.. _fignondimhocustom:

..  figure:: nondimhocustom.*
    :name: figcustomnondimho
    :align: center
    :alt: User defined harmonic oscillator
    :class: with-shadow
    :width: 100%
    
    Output for the user-defined harmonic oscillator equation with damping and driving.
    
.. only:: html    

	.. container:: downloadbutton

		:download:`Download this example <custom_harmonic_oscillator.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`    
