.. _secODEfloquet:

Stability of orbits
~~~~~~~~~~~~~~~~~~~

Like stationary solutions, periodic orbits can exhibit different kinds of stability.
In the discussed Lorenz system, we only found unstable orbits, but how do we prove that they are indeed unstable?
As usual, stability can be defined by the linearized dynamics around the state of interest, here the orbit.

Recalling :math:numref:`orbitdefM`, a linearization around the orbit :math:`\vec{x}(t)` corresponds to the solution

.. math:: :label: orbitdeflin

	\begin{aligned}
	\mathbf{M}(\vec{x})\partial_t\vec{v}(t)+\mathbf{J}_0(\vec{x})\vec{v}&=0 \\
	\vec{v}(t+T)&=\lambda\vec{v}(t)
	\end{aligned}
	
Here, :math:`\mathbf{J}_0(\vec{x})` is the Jacobian of the time-independent residual :math:`\vec{R}_0(\vec{x})` and :math:`\vec{v}(t)` is the linear perturbation of the orbit.
After one period, we do not necessarily end up at the very same point, i.e. in general :math:`\vec{v}(t+T)\neq \vec{v}(t)`. Instead, we generically end up at another point, which is expressed by the so-called *Floquet multiplier* :math:`\lambda`. *Floquet theory* tells us that such solutions :math:`\vec{v}(t)` with corresponding values of :math:`\lambda` exist, which resemble the conventional eigenvectors and eigenvalues used for the stability of stationary solutions, although there are some differences as detailed in the following. Due to linearity, a second turn along the orbit will end up at :math:`\vec{v}(t+2T)=\lambda^2 \vec{v}(t)` and (as long as the perturbation remains small) a multiplier of :math:`\lambda^n` after :math:`n` periods. Obviously, an orbit is stable if all Floquet multipliers (which are in general complex-valued) satisfy :math:`|\lambda|<1`. However, there is always a Floquet multiplier :math:`\lambda=1` present in the system, which corresponds to the shift invariance in time. The corresponding perturbation is just :math:`\vec{v}=\partial_t\vec{x}`, i.e. if we just start our perturbation somewhere else in time, we will end up shifted as well after a period of :math:`T`.

Pyoomph can calculate Floquet multipliers, which is demonstrated in the following on the basis of the Langford ODE system. This one reads with suitable parameters :cite:`Gesla2024`:

.. math:: :label: langfordODE

	\begin{aligned}	 
         \partial_t x&=(\mu-3)x-\frac{1}{4}y+x\left(z+\frac{1}{5}(1-z^2)\right) \\
         \partial_t y&=(\mu-3)y+\frac{1}{4}x+y\left(z+\frac{1}{5}(1-z^2)\right) \\
         \partial_t z&=\mu z-\left(x^2+y^2+z^2\right)
	\end{aligned}

For :math:`\mu>1.683`, perfectly circular orbits can be found which change the stability from stable to unstable at :math:`\mu=2` :cite:`Gesla2024`.

Implementing the equation and setting up the problem is again trivial:

.. code:: python

	#See https://arxiv.org/abs/2407.18230v1
	class LangfordSystem(ODEEquations):
	     def __init__(self,mu): 
		     super(LangfordSystem,self).__init__()
		     self.mu=mu

	     def define_fields(self):
		     self.define_ode_variable("x","y","z")

	     def define_residuals(self):
		     x,y,z=var(["x","y","z"])             
		     xrhs=(self.mu-3)*x-0.25*y+x*(z+0.2*(1-z**2))
		     yrhs=(self.mu-3)*y+0.25*x+y*(z+0.2*(1-z**2))
		     zrhs=self.mu*z-(x**2+y**2+z**2)
		     residual=(partial_t(x)-xrhs)*testfunction(x)
		     residual+=(partial_t(y)-yrhs)*testfunction(y)
		     residual+=(partial_t(z)-zrhs)*testfunction(z)
		     self.add_residual(residual)
		        
		        
	class LangfordProblem(Problem):
	    def __init__(self):
		 super().__init__()
		 self.mu=self.define_global_parameter(mu=1.6)
		 
	    def define_problem(self):
		eqs=LangfordSystem(self.mu) 
		eqs+=ODEFileOutput()
		self.add_equations(eqs@"langford")

	    def get_analytical_nontrivial_floquet_multiplier(self):
		# Calculate the analytical nontrivial Floquet multiplier
		muv=self.mu.value
		z=2.5*(1-numpy.sqrt(0.8*muv-1.24))
		r=numpy.sqrt(z*(muv-z))
		exponent=(muv-2*z+numpy.emath.sqrt((muv-2*z)**2-8*r*(r-0.4*r*z)))/2
		T=4*2*numpy.pi
		multiplier=numpy.exp(exponent*T)
		if numpy.imag(multiplier)<0:
		    multiplier=numpy.conjugate(multiplier) # We always consider the one with positive imaginary part
		return multiplier
                   
Note how we also provide a function to calculate the analytical nontrivial Floquet multiplier :cite:`Gesla2024`, i.e. a complex Floquet multiplier which is not the trivial one :math:`\lambda=1`. 
		        
We will then again find the Hopf bifurcation, switch to the orbit and continue in the parameter :math:`\mu`. But at each continuation step, we also calculate the Floquet multipliers and write the non-trivial one (along with the corresponding analytical solution) to the output:

.. code:: python

     with LangfordProblem() as problem:
        # Use again an analytic Hessian for the determination of the first Lyapunov coefficient
        problem.setup_for_stability_analysis(analytic_hessian=True)        
        # We also need the SLEPc eigensolver here
        problem.set_eigensolver("slepc").use_mumps() 
        
        problem+=InitialCondition(x=0.01,z=1.1)@"langford"  # Some non-trivial initial position        
        
        # Find the Hopf bifurcation as usual
        problem.solve()
        problem.solve_eigenproblem(3)
        problem.activate_bifurcation_tracking("mu")
        problem.solve()
        
        # Output file to compare the numerical and analytical Floquet multipliers
        floquet_output=problem.create_text_file_output("floquet.txt",header=["mu","num_real","num_imag","ana_real","ana_imag"])        
        
        # Switch again to the orbits originating from the Hopf bifurcation
        with problem.switch_to_hopf_orbit(NT=50,order=3) as orbit:          
                ds=orbit.get_init_ds()       
                maxds=ds*100 # Limit the maximum step size
                while problem.mu.value<2.05:
                        ds=problem.arclength_continuation("mu",ds,max_ds=maxds)                      
                        F=orbit.get_floquet_multipliers(n=3,shift=3) # Calculate some Floquet multipliers 
                        # However, not always three multipliers are found. We have to consider the cases                                                             
                        if len(F)==3:
                                # Three multipliers found: The trivial one and two complex conjugate ones
                                F=numpy.delete(F,numpy.argmin(numpy.abs(F-1)))
                                nontrivial_floquet=F[0] # Take one of the complex conjugate multipliers
                        elif len(F)==2:
                                # Only two multipliers found: The trivial one and one real one
                                F=numpy.delete(F,numpy.argmin(numpy.abs(F-1)))
                                nontrivial_floquet=F[0] # Take the remaining multiplier                         
                        else:
                                # Only one multiplier found: The trivial one
                                nontrivial_floquet=0 # The others are then very close to 0
                                
                        if numpy.imag(nontrivial_floquet)<0:
                                # conjugate a multiplier with negative imaginary part
                                nontrivial_floquet=numpy.conjugate(nontrivial_floquet)
                                
                        # Output the orbit
                        odir="orbit_{:.3f}".format(problem.mu.value)
                        orbit.output_orbit(odir)
                        
                        # Write to output for comparison
                        floq_ana=problem.get_analytical_nontrivial_floquet_multiplier()
                        floquet_output.add_row(problem.mu, nontrivial_floquet.real,nontrivial_floquet.imag,floq_ana.real,floq_ana.imag)

Floquet multipliers can be calculated via the method :py:meth:`~pyoomph.generic.problem.PeriodicOrbit.get_floquet_multipliers` of the :py:class:`~pyoomph.generic.problem.PeriodicOrbit` class. The internals work analogously to the way proposed in Ref. :cite:`Fairgrieve1991`. However, multipliers close to zero will be discarded. They usually do not give any information on the stability anyway. We carefully have to select the interesting Floquet multiplier and write it to the output. As depicted in :numref:`figfloquetslangford`, the results agree well with the analytical Floquet multiplier.

..  figure:: floquets.*
    :name: figfloquetslangford
    :align: center
    :alt: Floquet multipliers of the Langford ODE system
    :class: with-shadow
    :width: 50%
    
    Floquet multipliers of the Langford ODE system

.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <langford_floquet.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
      

Since the Floquet multipliers at :math:`\mu=2` cross the stability condition :math:`|\lambda|=1` by a complex-conjugated pair, this corresponds to a Neimark-Sacker bifurcation. The orbit becomes unstable to a torus. We can check this by performing time integration. The moment we leave the ``with`` statement of the ``orbit``, pyoomph will initialize the degrees of freedom to the starting point of the orbit. A trivial :py:meth:`~pyoomph.generic.problem.Problem.run` statement will then perform a time integration along the orbit. However, if we start at :math:`\mu>2` (here e.g. :math:`\mu=2.005`), it will be unstable and we can see the torus developing. We just have to replace the orbit loop (i.e. the code after solving for the Hopf bifurcation) by:

.. code:: python

	with problem.switch_to_hopf_orbit(NT=50,order=3) as orbit:          
                ds=orbit.get_init_ds()       
                maxds=ds*100 # Limit the maximum step size
                problem.go_to_param(mu=2.005,startstep=ds,max_step=maxds,call_after_step= lambda ds: orbit.output_orbit("orbit_at_mu_"+str(problem.mu.value)))                
                T=orbit.get_T() # Get the period
                NT=orbit.get_num_time_steps() # Get the number of time steps
                dt=T/NT # And calculate a good time step
                
        # Running a transient integration starting on the orbit
        problem.run(40*T,outstep=dt/4)


..  figure:: torus_unstable.*
    :name: figlangfordtorus
    :align: center
    :alt: Stable orbits and time integration at unstable dynamics building a torus
    :class: with-shadow
    :width: 90%
    
    Stable orbits (color-coded by :math:`\mu`) and time integration at :math:`\mu=2.005` (black) showing the unstable dynamics building a torus. Also, the path of the Floquet multipliers as a function of :math:`\mu` is shown.

    
.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <langford_time_integration.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
