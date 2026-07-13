Solving the heat equation on a domain by two simulations
---------------------------------------------------------

To show how the preCICE adapter in pyoomph works, we cover the preCICE tutorial example `Partitioned heat conduction <https://precice.org/tutorials-partitioned-heat-conduction.html>`__.

A rectangular domain of size :math:`2 \times 1` is separated into a left (Dirichlet) and right (Neumann) participant, each of size :math:`1 \times 1`.
On the full domain, we solve a heat conduction equation, i.e. we also solve it in both participants. preCICE will take the lead, i.e. controls the time stepping in both running pyoomph simulations and transfers the data at the mutual coupling interface from one participant to the other.

Via preCICE, the Dirichlet participant (left half) will receive the values of the temperature at the coupling boundary, impose these values as a Dirichlet condition, solve the system and feed back the heat flux to the Neumann participant (right half). The latter will solve the system with the received heat flux as a Neumann condition and feed the temperature Dirichlet values again to the Dirichlet participant.

More details can be found in the `preCICE tutorial <https://precice.org/tutorials-partitioned-heat-conduction.html>`__.


To use preCICE in pyoomph, you can just import the module :py:mod:`pyoomph.solvers.precice_adapter`. As mentioned on the previous page, you must have installed preCICE and the preCICE python bindings to import it.

We want to formulate the problem in a way that it can be either run monolithically in a single simulation, which calculates the full domain. Alternatively, we can run either the Dirichlet or the Neumann participant. If both are running simultaneously, they will interact via preCICE.

pyoomph's :py:class:`~pyoomph.generic.problem.Problem` has the attributes :py:attr:`~pyoomph.generic.problem.Problem.precice_participant` and :py:attr:`~pyoomph.generic.problem.Problem.precice_config_file`. When using preCICE, you must specify the preCICE config file with the latter and the participant name with the former.
Here, the config file :download:`precice-config.xml` of this example defines the participants ``"Dirichlet"`` and ``"Neumann"``.

As usual, we start by importing the required modules and defining the heat equation, i.e. besides importing pyoomph's preCICE adapter, nothing spectacular is happening here:

.. code:: python

	from pyoomph import *
	from pyoomph.expressions import *

	# Import the preCICE adapter of pyoomph
	from pyoomph.solvers.precice_adapter import *

	# Heat conduction equation with a source term
	class HeatEquation(Equations):
	    def __init__(self,f):
		super().__init__()
		self.f=f
		
	    def define_fields(self):
		self.define_scalar_field("u","C2")
		
	    def define_residuals(self):
		u,v=var_and_test("u")
		self.add_weak(partial_t(u),v).add_weak(grad(u),grad(v)).add_weak(-self.f,v)


The magic happens in the definition of the problem, where we use several classes from the :py:mod:`~pyoomph.solvers.precice_adapter` module:

.. code:: python

	# Generic heat conduction problem. Can be run without preCICE on the full domain or as Dirichlet or Neumann participant
	class HeatConductionProblem(Problem):
	    def __init__(self):
		super().__init__()
		self.alpha=3 # Parameters
		self.beta=1.2    
		# Config file
		self.precice_config_file="precice-config.xml"   
		
	    def get_f(self):
		# Source term
		return self.beta-2-2*self.alpha
	    
	    def get_u_analyical(self):
		# Analytical solution
		return 1+var("coordinate_x")**2+self.alpha*var("coordinate_y")**2+self.beta*var("time")
		
	    def define_problem(self):
		# Depending on the participant, set up the coupling equations
		
		# First of all, we must provide the mesh at the coupling interface to preCICE
		# If we run without preCICE, this equation part is not used, so it will be just discarded
		coupling_eqs=PreciceProvideMesh(self.precice_participant+"-Mesh")
		if self.precice_participant=="Dirichlet":
		    # Dirichlet participant: We take the left part and use the right boundary as coupling boundary
		    x_offset,box_length=0,1
		    coupling_boundary="right"
		    # Here we write the heat flux and read the temperature
		    # Note that we cannot use PreciceWriteData(Heat-Flux=partial_x(var("u",domain=".."))) 
		    # because "Heat-Flux" is not a valid keyword argument. Therefore, we use the **{...} syntax
		    # It will calculate the gradient of u in x-direction and write it to the preCICE data "Heat-Flux"
		    coupling_eqs+=PreciceWriteData(**{"Heat-Flux":partial_x(var("u",domain=".."))})
		    # It reads the preCICE field "Temperature" and stores it in the field "uD"
		    coupling_eqs+=PreciceReadData(uD="Temperature")            
		    # We cannot set a Dirichlet boundary condition depending on a variable, so we use an enforced boundary condition
		    # We adjust u so that u-uD=0 at the coupling boundary. This is equivalent to setting u=uD
		    coupling_eqs+=EnforcedBC(u=var("u")-var("uD"))
		elif self.precice_participant=="Neumann":
		    # The Neuamnn participant is on the right side and uses the left boundary as coupling boundary
		    x_offset,box_length=1,1
		    coupling_boundary="left"
		    # It writes the preCICE field "Temperature" by evaluating the variable u
		    coupling_eqs+=PreciceWriteData(Temperature=var("u"))
		    # It reads the preCICE field "Heat-Flux" and stores it in the field "flux"
		    coupling_eqs+=PreciceReadData(flux="Heat-Flux")            
		    # This flux is used as Neumann boundary condition. 
		    coupling_eqs+=NeumannBC(u=var("flux"))
		elif self.precice_participant=="":
		    # If we run without preCICE, we use the full domain and have no coupling boundary
		    x_offset=0
		    box_length=2
		    coupling_boundary=None
		else:
		    raise Exception("Unknown participant. Choose 'Dirichlet', 'Neumann' or an empty string")
		
		# Create the corresponding mesh
		N0=11
		self+=RectangularQuadMesh(size=[box_length,1],N=[box_length*N0,N0],lower_left=[x_offset,0])
		
		# Assemble the base equations
		eqs=MeshFileOutput()
		eqs+=HeatEquation(self.get_f())
		eqs+=InitialCondition(u=self.get_u_analyical())
		
		# Add the coupling equations and the Dirichlet boundary conditions
		# All potential Dirichlet boundaries        
		dirichlet_bounds=set(["bottom","top","left","right"])
		if coupling_boundary:
		    # Of course, the coupling boundary must not be set as Dirichlet boundary
		    dirichlet_bounds.remove(coupling_boundary)        
		    eqs+=coupling_eqs@coupling_boundary
		eqs+=DirichletBC(u=self.get_u_analyical())@dirichlet_bounds            
		
		# Calculate the error
		eqs+=IntegralObservables(error=(var("u")-self.get_u_analyical())**2)
		eqs+=IntegralObservableOutput()
		                    
		self+=eqs@"domain"
		
		
If we use preCICE, we only define half of the domain. The mesh of the ``"Neumann"`` participant is furthermore shifted to the right. Depending on the side we solve, the ``coupling_boundary`` is either the ``"left"`` or ``"right"`` boundary of the domain. When we set ``precice_participant=""`` (default value), we just solve the full problem and do not add any coupling. 

If we select one of the participants, however, we have to set up the coupling. This happens in multiple steps. First of all, we must export the interface mesh at the coupling boundary to preCICE, which is done by the class :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh`. You must supply a mesh name agreeing with the ``provide-mesh`` definition in the config file :download:`precice-config.xml`. This will tell preCICE where the nodes are located, so that it can be connected to the other participant. 

Then, both participants have to exchange data. For writing data from the current participant to the other, you can use the class :py:class:`~pyoomph.solvers.precice_adapter.PreciceWriteData`. It takes arguments of the form ``PRECICE_NAME = PYOOMPH_EXPRESSION``, where ``PRECICE_NAME`` must coincide with the name of a ``data`` declaration in the preCICE config file. Since ``Heat-Flux`` cannot be used as a keyword argument (due to the dash), we instead supply it via a ``dict`` using the ``**{}`` syntax in the Dirichlet participant. In pyoomph, we calculate the normal gradient and send it to the ``"Heat-Flux"`` data of preCICE. In the Neumann participant, we just write the nodal values of ``var("u")`` to the ``"Temperature"`` data of preCICE. 

For the opposite direction, we can use :py:class:`~pyoomph.solvers.precice_adapter.PreciceReadData`. It takes arguments like ``PYOOMPH_NAME = PRECICE_NAME`` and defines a pyoomph variable given by ``PYOOMPH_NAME``, which will hold the values of the preCICE data given by ``PRECICE_NAME``. Again, all used ``PRECICE_NAME`` must be declared in the config file to be readable from the mesh.

Thereby, the transfer of data is complete, but you still have to use the data read from the other participant in the current participant. In the Dirichlet case, we use :py:class:`~pyoomph.meshes.bcs.EnforcedBC`, since a :py:class:`~pyoomph.meshes.bcs.DirichletBC` may only depend on the time, Lagrangian and Eulerian (for a static mesh only) coordinates. However, in fact the :py:class:`~pyoomph.meshes.bcs.EnforcedBC` does exactly the same as a :py:class:`~pyoomph.meshes.bcs.DirichletBC` here. The Neumann part just imposes the read ``var("flux")`` as :py:class:`~pyoomph.meshes.bcs.NeumannBC`. Here, one has to be careful with the signs. From the weak form of the heat equation, the Neumann term would require a minus sign, but since the normal is pointing in the negative x-direction on the Neumann side of the interface, it cancels out.

Eventually, we just have to attach all coupling equations to the corresponding boundary and make sure to not apply the Dirichlet boundary conditions of the far field here. If we do not use preCICE, we just discard the coupling equations.

The coupling is complete, but for running a coupled simulation, preCICE must take the lead for the time stepping. Typical time steps and the maximum simulation time are given in the config file. Therefore, pyoomph's :py:meth:`~pyoomph.generic.problems.Problem.run` method cannot be used. Instead, the method :py:meth:`~pyoomph.generic.problems.Problem.precice_run` has to be used:

.. code:: python

	if __name__=="__main__":
	    problem=HeatConductionProblem()
	    problem.initialise() # After this, precice_participant could have been set via command line, e.g. by  -P precice_participant=Neumann
	    if problem.precice_participant=="":
		# Just run it manually without preCICE
		problem.run(1,outstep=0.1)
	    else:
		# Run it with preCICE. Time stepping is taken from the config file
		problem.precice_run()
		

This completes the simulation. For running with preCICE, you have to run the script two times, passing the participant name as a command line parameter (see :numref:`secodecmdline`).

.. code:: bash

      python partitioned_heat_conduction.py --outdir Dirichlet -P precice_participant=Dirichlet &
      python partitioned_heat_conduction.py --outdir Neumann -P precice_participant=Neumann
      
Of course, you must place the config file :download:`precice-config.xml` in the same directory.
If you run the scripts without setting :py:attr:`~pyoomph.generic.problem.Problem.precice_participant`, it will just run the monolithic case without preCICE.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <partitioned_heat_conduction.py>`
		
		:download:`Download all examples <../tutorial_example_scripts.zip>`   	
		    
