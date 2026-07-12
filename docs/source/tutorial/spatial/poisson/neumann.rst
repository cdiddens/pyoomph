Poisson equation with a Neumann boundary condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weak formulation :math:numref:`eqspatialpoissonweak` has a boundary integral term stemming from the Neumann boundary conditions, which we have not accounted for yet. One cannot add this into the ``PoissonEquation`` class from the previous code :download:`poisson.py` directly, since the :py:func:`~pyoomph.expressions.generic.weak`-terms added are always integrated over the entire domain, where the equation is defined, i.e. ``"domain"`` (:math:`=\Omega=[-1,1]`) in the previous two examples. Furthermore, to keep the final assembly of the equation system in the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method as flexible as possible, it is better to create a specific class to account for the surface integral term in :math:numref:`eqspatialpoissonweak`:

.. code:: python

   from pyoomph import *
   from poisson import PoissonEquation,weak  # Load the Poisson equation for the previous class


   # We create a new class, which adds the boundary integration term -<j,v>
   class PoissonNeumannCondition(InterfaceEquations):
       # Makes sure that we can only use it as boundaries for a Poisson equation
       required_parent_type = PoissonEquation

       def __init__(self,name,flux):
           super(PoissonNeumannCondition, self).__init__()
           self.name=name # store the variable name and the flux
           self.flux=flux

       def define_residuals(self):
           u,v=var_and_test(self.name) # Get the test function by the name
           self.add_residual(-weak(self.flux,v)) # and add it to the residual
           # weak(j,v) is now <j,v>, i.e. a boundary integral

   class NeumannPoissonProblem(Problem):
       def define_problem(self):
           mesh = LineMesh(minimum=-1, size=2, N=100)
           self.add_mesh(mesh)

           # Alternative way to assemble the system by restricting directly
           # Poisson equation and output on bulk domain
           equations = (PoissonEquation(source=1)+TextFileOutput()) @ "domain"
           equations += DirichletBC(u=0) @ "domain/left" # Dirichlet BC on domain/left
           equations += PoissonNeumannCondition("u",-1.5) @ "domain/right" # Neumann BC on domain/right
           self.add_equations(equations)


   if __name__ == "__main__":
       with NeumannPoissonProblem() as problem:
           problem.solve()  # Solve the problem
           problem.output()  # Write output

Besides loading again the ``PoissonEquation`` from the previous code, we define a new class ``PoissonNeumannCondition`` inherited from the :py:class:`~pyoomph.generic.codegen.InterfaceEquations` type. :py:class:`~pyoomph.generic.codegen.InterfaceEquations` are a subclass of the :py:class:`~pyoomph.generic.codegen.Equations` class, but provide some additional features which are only relevant at interfaces, i.e. at boundaries. For instance by adding ``required_parent_type=PoissonEquation``, we can make sure that we are only allowed to attach this boundary condition to domains, where the ``PoissonEquation`` is solved. The constructor takes the name of the variable from the bulk as argument and the desired flux :math:`j` to impose. We do not need to :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields`, since the field has been already defined in the bulk and the :py:class:`~pyoomph.generic.codegen.InterfaceEquations` can access these fields and the corresponding test functions. Therefore, we can access the test function :math:`v` of the unknown field :math:`u` directly in the :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals` method. Again :py:func:`~pyoomph.expressions.generic.weak` is used to express the integral contribution :math:`-\langle j_\text{N},v\rangle`. As before in the bulk contribution, it will be expanded to a spatial integral, however, since the ``PoissonNeumannCondition`` will be restricted to the boundary later on, the integral will be not carried out over the domain :math:`\Omega`, but just over the right boundary at :math:`x=1`. Of course, since :math:`\Omega` is just an interval, the boundary integral just comprises the point at :math:`x=1`, so that the integral will just evaluate to identity.

In the :py:meth:`~pyoomph.generic.problem.Problem.define_problem` method of the :py:class:`~pyoomph.generic.problem.Problem` class, there is another way shown how to restrict equations. Instead of restricting any boundary condition, stored in a local variable ``bc`` in the following, twice, i.e. ``(bc @ "right") @ "domain"``, we can also restrict equivalently via a slash, i.e. by ``bc @ "domain/right"`` to apply it on the ``"right"`` boundary of the domain ``"domain"``.


..  figure:: poisson_neumann1.*
	:name: figspatialpoissonneumann1
	:align: center
	:alt: Poisson equation with Dirichlet and Neumann conditions.
	:class: with-shadow
	:width: 50%
	
	One-dimensional Poisson equation with Dirichlet boundary at the left and Neumann boundary at the right.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <poisson_neumann.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    

It is not necessary to implement a Neumann condition for each equation by hand as done with the ``PoissonNeumannCondition`` class. Instead, pyoomph offers the class :py:class:`~pyoomph.meshes.bcs.NeumannBC`, which allows to add the weak contribution :math:`\langle j_\text{N},v \rangle` directly. However, one has to consider the sign: In the weak form of the Poisson equation :math:numref:`eqspatialpoissonweak`, the :math:`\langle \ldots \rangle` term has a negative sign, which is accounted for in the implementation of the ``PoissonNeumannCondition``. The generic class :py:class:`~pyoomph.meshes.bcs.NeumannBC` uses a positive sign instead. One hence has to use ``NeumannBC(u=1.5)`` to get the same effect as ``PoissonNeumannCondition("u",-1.5)``. Note that the keyword argument in ``NeumannBC(u=1.5)`` should be read as *impose the Neumann flux* :math:`1.5` *for* :math:`u`, **not** *set* :math:`u=1.5`, which would be Dirichlet-like.

Note that we have used one Dirichlet and one Neumann condition in the above example. Two Neumann conditions require special treatment, as discussed in the following example.

.. warning::

   If you do not add any boundary condition on a boundary, i.e. neither a :py:class:`~pyoomph.meshes.bcs.DirichletBC` nor any additional boundary integral term, there is no additional contribution to the residuals. This is equivalent to setting the Neumann flux to zero, i.e. a no-flux boundary condition.
