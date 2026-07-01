.. _secALE:

Moving Mesh (ALE) Methods
=========================

Until now, we have always considered a static mesh. However, when e.g. a droplet evaporates, it will shrink over time. Of course, one can always take a static domain and solve e.g. a phase field along with the flow, i.e. with a Navier-Stokes-Cahn-Hilliard model (cf. e.g. :cite:t:`Demont2022`). While this has the benefit that the position of the interface is not required to be known at all and topological changes (coalescence, pinch-off) are possible, it is more complicated to impose e.g. Marangoni forces or to add surfactants directly on the interface. As an alternative, pyoomph allows to formulate equations for the mesh coordinates, which are solve along with all other equations. Thereby, one can e.g. define an equation system that solves the governing equations on a moving mesh and couple the mesh motion with the underlying dynamics, i.e. via a kinematic boundary condition. With this approach, it is possible to account for an evaporating droplet with a sharp and well resolved interface, where Marangoni stresses and surfactants can be added easily. Since the mesh is neither fixed in space (Eulerian), nor necessarily co-moving with the fluid velocity (Lagrangian), the approach discussed here is also called *Arbitrary Lagrangian-Eulerian*, abbreviated *ALE*.

We will develop this method in the following, address the necessary correction of the temporal derivative and mesh reconstruction at larger deformations. Afterwards, we provide a few examples:

.. toctree::
   :maxdepth: 5
   :hidden:
   
   ale/lagrange.rst
   ale/laplsmooth.rst   
   ale/aledt.rst      
   ale/remesh.rst
   ale/freesurf.rst         
   ale/gmshfields.rst
   ale/spread.rst            
   ale/solid.rst      


