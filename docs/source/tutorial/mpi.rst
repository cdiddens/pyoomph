.. _secmpi:

MPI parallelization
===================

Pyoomph can run a simulation on several processes with MPI. This chapter covers what is required to do
so, the two rather different parallel modes pyoomph offers, and what the choice of the linear solver
means for each of them.

Before reaching for more cores, however, it is worth asking whether the problem needs all the degrees of
freedom it has: a system that is half the size is solved considerably faster on the machine you already
have, and unlike parallelization it costs nothing in communication. The last section of this chapter
therefore discusses two ways of removing degrees of freedom without changing the discretization.

.. toctree::
   :maxdepth: 5
   :hidden:

   mpi/modes.rst
   mpi/dofreduction.rst
