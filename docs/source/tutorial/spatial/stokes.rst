.. _secspatialstokes:

The Stokes equations
--------------------

The Stokes equations generalizes the simple example of the Poisson equation on various levels. First of all, the Stokes equations are a system of two partial differential equations, one for the velocity field and one for the pressure field. Secondly, the Stokes equations involves the velocity as vectorial unknown. Additionally, particular care has to be taken on the choice of the used finite element spaces for the velocity-pressure discretization.

Therefore, the Stokes equations constitute a perfect example to progress from the simple Poisson equation to more complex problems. 

.. toctree::
   :maxdepth: 5
   :hidden:

   stokes/weakform.rst
   stokes/implement.rst
   stokes/nonnewton.rst
   stokes/puredirichlet.rst
   stokes/dimtraction.rst
   stokes/nonormalflow.rst   
   stokes/stokeslaw.rst   
   stokes/cr.rst   
   stokes/inverse_problem.rst
