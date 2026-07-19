.. _secspatialpoisson:

The Poisson equation
--------------------

The Poisson equation is the most common example to illustrate the finite element method. It is the most trivial equation which directly allows discussing the central ideas of the finite element method. Based on the Poisson equation, we will discuss the derivation of the weak formulation, how to incorporate boundary conditions, and how to solve the equation in different dimensions and coordinate systems. Furthermore, adaptive mesh refinement will be discussed. All these topics are illustrated by simple example codes which can be used as a starting point for more complex simulations.


.. toctree::
   :maxdepth: 5
   :hidden:

   poisson/weakform.rst
   poisson/onedimdbc.rst
   poisson/onedimcoupled.rst   
   poisson/neumann.rst   
   poisson/pureneumann.rst   
   poisson/robin.rst   
   poisson/cauchy.rst   
   poisson/twodim.rst
   poisson/adapt.rst         
   poisson/coordsys.rst      

