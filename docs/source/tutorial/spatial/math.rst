.. _secspatialbasisfuncs:


Mathematical details on the solution procedure
----------------------------------------------

So far, we have discussed the weak form and the implementation in pyoomph. This section is devoted to provide some insight into the mathematical details of the solution procedure, i.e. what is going on under the hood in pyoomph.
Pyoomph provides a high level python front-end based on oomph-lib, i.e. most of the details of assembling matrices and residuals are hidden from the user. If you want to go to the very bottom of the solution procedure, it is advised to get familiar with oomph-lib directly. 

.. toctree::
   :maxdepth: 5
   :hidden:

   math/shape.rst
   math/internals.rst   
