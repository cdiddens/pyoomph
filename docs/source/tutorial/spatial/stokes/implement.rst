.. _secspatialinfsup:

Implementation of the Stokes equations and the inf-sup condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us now implement the aforementioned weak form of the Stokes equation. As usual, this requires only a few lines in pyoomph. However, there is some important requirement on the selection of the basis functions for the velocity and the pressure fields. To try out which choices of the discretizations work, we let user pass the spaces, e.g. ``"C1"`` or ``"C2"``, for the velocity and the pressure.

.. literalinclude:: stokes.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.add_residual(weak(stress,grad(v)) + weak(div(u),q)) # weak form of Stokes flow

We pass the viscosity as ``mu`` and the two spaces ``uspace`` and ``pspace`` to the constructor of the ``StokesEquations`` and store them internally. A vector field can be defined with :py:meth:`~pyoomph.generic.codegen.Equations.define_vector_field` within the specific implementation of :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_fields`. This will automatically create a one-dimensional vector field on a mesh with one spatial dimension and correspondingly two-dimensional and three-dimensional vector fields on two-dimensional and three-dimensional meshes, respectively.

As a test problem, we implement the typical parabolic *Poiseuille flow*. This is achieved by considering a rectangular domain and imposing zero velocity at the top and at the bottom, a parabolic inflow in :math:`x`-direction at :math:`x=0` and an open outflow with :math:`u_y=0`:

.. literalinclude:: stokes.py
   :language: python
   :start-at: eqs=StokesEquations(self.mu,self.uspace,self.pspace) # Stokes equation using the viscosity and the spaces
   :end-at: self.add_equations(eqs@"domain")

Note how the components of the vectorial field :math:`u` can be accessed by ``u_x`` and ``u_y`` in two dimensions for setting e.g. Dirichlet boundary conditions. Furthermore, we take the viscosity and the discretization spaces for the velocity and the pressure as arguments for the problem constructor and pass them to the ``StokesEquations``.

By that, we can now easily try out different combinations for the finite element spaces by constructing a corresponding problem and solve it:

.. literalinclude:: stokes.py
   :language: python
   :start-at: if __name__ == "__main__":
   :end-at: problem.output()

It is trivial to try out other pairs of spaces, but it turns out that for the choice of the continuous spaces ``"C1"`` and ``"C2"``, the only combination that leads to convergence and reasonable solutions is ``"C2"`` for the velocity and ``"C1"`` for the pressure. This combination is called *Taylor-Hood element*. Mathematically, the convergence or divergence/oscillations can be proven and the corresponding theorem is the so-called *inf-sup criterion* or *Ladyzhenskaya-Babuška-Brezzi condition*. In simple words, we have to be aware of the fact that the pressure acts as a field of Lagrange multipliers that enforces the incompressibility, i.e. :math:`\nabla\cdot \mathbf{u}=0`. The pressure field adjusts hence that way and couples back to the momentum equation so that this condition is fulfilled. However, if we choose equal spaces for both velocity and pressure, we are actually over-constraining the requirement of :math:`\nabla\cdot \mathbf{u}=0`, i.e. we enforce it at too many localizations so that a solution cannot be found or suffers from checkerboard-like oscillations.

.. warning::

   Since the pressure acts as Lagrange multiplier field for the incompressibility, one should not be tempted to impose a Dirichlet boundary condition for the pressure, except for a single degree of freedom in a special case discussed later on in :numref:`secspatialstokesdim`.

..  figure:: stokes.*
	:name: figspatialstokes
	:align: center
	:alt: Stokes flow
	:class: with-shadow
	:width: 70%

	Stokes flow example with a Newtonian fluid (left) and a shear-thickening fluid (right, see next page).


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <stokes.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		    
