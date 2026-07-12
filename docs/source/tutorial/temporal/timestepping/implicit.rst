.. _secodetimesteppingsimple:

Testing different time stepping methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A good way to test whether the chosen time stepping method is appropriate is to compare the numerical results of a simple test problem with the corresponding analytical solution. Furthermore, one can test for conserved quantities, e.g. the total energy of an undamped harmonic oscillator. This will be done for all time stepping methods in the following.

To do so, the harmonic oscillator equation class will be modified to be able to calculate with all time stepping methods implemented in pyoomph. For the ``"Newmark2"`` method, we use the second order ODE

.. math:: \partial_t^2 y+\omega^2 y=0

whereas for the other methods, i.e. the ones which evaluate only first order time derivatives, ``"BDF1"`` and ``"BDF2"``, this equation is separated into a system of two first order equations

.. math::

   \begin{aligned}
   \partial_t z+\omega^2 y&=0\\
   \partial_t y-z&=0
   \end{aligned}

i.e. where :math:`z=\partial_t y`. The code for the oscillator equation that allows one to select the time stepping scheme is the following:

.. code:: python

   class HarmonicOscillator(ODEEquations):
   	def __init__(self,*,omega=1,scheme="Newmark2"): #Passing a time stepping scheme
   		super(HarmonicOscillator,self).__init__()
   		allowed_schemes={"Newmark2","BDF1","BDF2"} #Possible values
   		if not (scheme in allowed_schemes): #Test for valid input
   			raise ValueError("Unknown time stepping scheme: "+str(scheme)+". Allowed: "+str(allowed_schemes))
   		self.scheme=scheme
   		self.omega=omega
   		
   	def define_fields(self):
   		self.define_ode_variable("y")
   		if self.scheme!="Newmark2":
   			self.define_ode_variable("dot_y") #Additional variable for first order ODE system
   		
   	def define_residuals(self):
   		y=var("y") 
   		if self.scheme=="Newmark2": #One second order ODE
   			residual=(partial_t(y,2)+self.omega**2 *y)*testfunction(y)  
   		else:	#Two first order ODEs
   			dot_y=var("dot_y")
   			residual=(partial_t(dot_y,scheme=self.scheme)+self.omega**2*y)*testfunction(dot_y)
   			residual+=(partial_t(y,scheme=self.scheme)-dot_y)*testfunction(y)
   		self.add_residual(residual)

A string ``scheme`` is passed to the constructor to select the time stepping scheme. First, the validity of the passed argument is checked. In the method :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields`, we define the variable :math:`y` and, if necessary, i.e. if ``"BDF1"`` or ``"BDF2"`` are selected, the variable :math:`z`, which is called ``"dot_y"`` in the code.

In :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`, we add the single second order ODE if ``scheme=="Newmark2"`` was selected, otherwise the two first order ODEs are added to the residual. Note that it is arbitrary here, which of the two equations is projected on which test function, i.e. we could also project the first equation on ``testfunction(y)`` and the second one on ``testfunction(dot_y)`` instead.

Important is the keyword argument ``scheme`` in the :py:func:`~pyoomph.expressions.generic.partial_t` function: a second order time derivative is calculated by ``partial_t(...,2)`` (always with the ``"Newmark2"`` method) and for first order derivatives, we can pass the optional keyword argument ``scheme`` which can either take the value ``"BDF2"`` (default), ``"BDF1"`` or ``"Newmark2"``. The latter calculates the first order derivative which is internally used for the second derivative calculation in ``"Newmark2"``.

In the problem class, also some modifications are necessary:

.. code:: python

   class HarmonicOscillatorProblem(Problem):
   	def __init__(self,scheme="Newmark2"): # Passing scheme here
   		super(HarmonicOscillatorProblem,self).__init__() 
   		self.omega=1
   		self.scheme=scheme

   	def define_problem(self):
   		eqs=HarmonicOscillator(omega=self.omega,scheme=self.scheme)
   		
   		t=var("time") # Time variable
   		Ampl, phi=1, 0 #Amplitude and phase
   		y0=Ampl*cos(self.omega*t+phi) #Initial condition with full time depencency
   		dot_y0 = -self.omega*Ampl * sin(self.omega * t + phi) #derivative of it
   		eqs+=InitialCondition(y=y0) #Set initial condition for y(t) at t=0
   		if self.scheme!="Newmark2":
   			eqs += InitialCondition(dot_y=dot_y0)  # And if required also for dot_y

   		#Calculate the total energy
   		y=var("y")
   		total_energy=1/2*partial_t(y,scheme=self.scheme)**2+1/2*(self.omega*y)**2
   		eqs+=ODEObservables(Etot=total_energy) # Add the total energy as observable

   		eqs+=ODEFileOutput() 
   		self.add_equations(eqs@"harmonic_oscillator") 

First of all, we take an argument ``scheme`` already in the constructor of the problem class. This is stored as member of the problem and passed later on to the equation class, namely in the method :py:meth:`~pyoomph.generic.problem.Problem.define_problem`. We make sure that the initial condition is perfectly accurate, i.e. that the derivatives at the beginning are approximated correctly by passing the full time-dependent analytical solution :math:`y(t)=A\cos(\omega t+\phi)` as initial condition. This condition and its temporal derivatives will be evaluated at :math:`t=0` and the corresponding discretized history values are set appropriately. If ``"BDF1"`` or ``"BDF2"`` are selected, we also have to explicitly add an initial condition for :math:`z`, i.e. for ``dot_y``, here.

In a next step, we want to monitor the total energy :math:`E=1/2\:(\partial_t y)^2+1/2\:\omega^2y^2`. This is an observable which depends on the unknowns, but it is not an unknown itself. Therefore, we use the class :py:class:`~pyoomph.equations.generic.ODEObservables`, which allows us to add exactly these kind of observables to the output. After running, there will be an additional column in the output file containing the values of ``"Etot"``.

Finally, we let our script successively create a problem for each of the time stepping methods, set an individual output directory with :py:meth:`~pyoomph.generic.problem.Problem.set_output_directory` to prevent overwriting of the previous results and run the simulations:

.. code:: python

   if __name__=="__main__":
   	for scheme in {"Newmark2","BDF1","BDF2"}:
   		with HarmonicOscillatorProblem(scheme=scheme) as problem:
   			problem.set_output_directory("osci_timestepping_"+scheme)
   			problem.run(endtime=100,numouts=1000)
   			

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <oscillator_fully_implicit_schemes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`     			
