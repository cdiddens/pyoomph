.. _secpdenavstokes:

Navier-Stokes equation
----------------------

The Navier-Stokes equation is the same as the Stokes equation in :numref:`secspatialstokes`, but with the addition of the inertia term :math:`\partial_t \vec{u}+\vec{u}\cdot\nabla \vec{u}`. Again, we have to make sure to meet the inf-sup criterion, i.e. we select a Taylor-Hood formulation:

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *


   class NavierStokesEquations(Equations):
   	def __init__(self,rho,mu):
   		super(NavierStokesEquations, self).__init__()
   		self.rho, self.mu= rho,mu # Store viscosity and density

   	def define_fields(self):
   		self.define_vector_field("velocity","C2") # Taylor-Hood pair
   		self.define_scalar_field("pressure","C1") 
   		
   	def define_residuals(self):
   		u,v=var_and_test("velocity") # get the fields and the corresponding test functions
   		p,q=var_and_test("pressure")
   		stress=-p*identity_matrix()+2*self.mu*sym(grad(u)) # see Stokes equation
   		inertia=self.rho*material_derivative(u,u) # lhs of the Navier-Stokes eq. rho*Du/dt
   		self.add_residual(weak(inertia,v)+ weak(stress,grad(v)) + weak(div(u),q)) 

The inertia term is obtained by using the function :py:func:`~pyoomph.expressions.generic.material_derivative`. Calling this function with any scalar, vectorial or tensorial expressions :math:`F` and the velocity :math:`\mathbf{u}` will return :math:`\partial_t F+\operatorname{grad}(F)\cdot\mathbf{u}`.

.. warning::

   The trivial implementation of the Navier-Stokes equation does not allow for high Reynolds numbers. The reason is similar to the problem discussed with the advection-diffusion equation in the previous example. There are improved schemes using SUPG and pressure stabilization to accurately account for the term :math:`\vec{u}\cdot\nabla\vec{u}`.


We discuss a few examples of the Navier-Stokes equation in the following sections.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   navier/womersley.rst
   navier/transientstokes.rst   
   navier/rtinstab.rst
   navier/marainstab.rst         
   navier/viscoelastic.rst
