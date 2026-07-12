Lubrication equation
--------------------

The *lubrication approximation* is commonly used for low-Reynolds number flow with a shallow aspect ratio. Instead of solving the (Navier-)Stokes equations with a free interface (which will be done later in :numref:`secALEfreesurfNS`), one instead solves the equation for the fluid height :math:`z=h(\vec{x},t)` as function of the lateral coordinate :math:`\vec{x}` and time :math:`t`. The height of the fluid changes due to pressure gradients, while the viscosity :math:`\mu` and the no-slip boundary condition at :math:`z=0` hamper the flow velocity in relaxing these pressure gradients. The corresponding equations, where the Laplace pressure :math:`p` due to the surface tension :math:`\sigma` and potentially the Marangoni effect are driving the flow, read

.. math:: :label: eqpdelubricationstrong

   \begin{aligned}
   \partial_t h+\nabla \cdot\left(-\frac{h^3}{3\mu}\nabla p+\frac{h^2}{2\mu}\nabla \sigma\right)&=0 \\
   p+\nabla\cdot\left(\sigma\nabla h\right)-\Pi&=0
   \end{aligned}

The pressure is obviously proportional to the approximated curvature in *Monge form*, i.e. approximating the Laplace pressure, and an optional contribution of a *Derjaguin pressure*/*disjoining pressure* :math:`\Pi`. The total volume, i.e. the spatial integral over the height :math:`h`, is obviously conserved over time.

A weak form with test functions :math:`\eta` and :math:`q` for :math:`h` and :math:`p`, respectively, reads

.. math::

   \begin{aligned}
   &\left(\partial_t h,\eta\right)+\left(\frac{h^3}{3\mu}\nabla p-\frac{h^2}{2\mu}\nabla \sigma,\nabla \eta\right)+\left(p-\Pi,q\right)-\left(\sigma\nabla h,\nabla q\right) \\
   -&\left\langle\vec{n}\cdot\left(\frac{h^3}{3\mu}\nabla p-\frac{h^2}{2\mu}\nabla \sigma\right),\eta\right\rangle+\left\langle\vec{n}\cdot\left(\sigma\nabla h\right),q\right\rangle=0
   \end{aligned}

and the corresponding implementation could look like this:

.. code:: python

   from pyoomph import *
   from pyoomph.expressions import *

   class LubricationEquations(Equations):
   	def __init__(self,sigma=1,mu=1,disjoining_pressure=0):
   		super(LubricationEquations, self).__init__()
   		self.sigma=sigma
   		self.mu=mu
   		self.disjoining_pressure=disjoining_pressure
   		
   	def define_fields(self):
   		self.define_scalar_field("h","C2")
   		self.define_scalar_field("p","C2")		
   		
   	def define_residuals(self):
   		h,eta=var_and_test("h")
   		p,q=var_and_test("p")		
   		self.add_residual(weak(partial_t(h),eta)+weak(1/self.mu*(h**3/3*grad(p)-h**2/2*grad(self.sigma)),grad(eta)))
   		self.add_residual(weak(p-self.disjoining_pressure,q)-weak(self.sigma*grad(h),grad(q)))


.. note::

   It can be beneficial to solve the height evolution equation on the test function of the pressure field and the pressure definition on the test functions of the height field, i.e. swap ``eta`` and ``q`` in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.add_residual` calls. This approach respects the fact that we then choose the test function according to the field with the highest spatial derivative. It is also beneficial if e.g. a Dirichlet condition for the height is imposed somewhere, since it does not require any appropriate Neumann term for the pressure.  


With this equation class, we will discuss a few examples in the following:

.. toctree::
	:maxdepth: 5
	:hidden:
	
	lubric/relax.rst
	lubric/spread.rst
	lubric/coalesce.rst    
