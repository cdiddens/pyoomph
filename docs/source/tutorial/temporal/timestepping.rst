.. _secodetimestepping:

Time stepping
-------------

Until now, we just used :py:func:`~pyoomph.expressions.generic.partial_t` to express time derivatives. However, the particular time stepping method in numerical simulations is of utmost importance for accurate results, which is even more the case if e.g. energy should be conserved. Therefore, this section is dedicated to the time stepping within pyoomph. Second order time derivatives are always calculated with the *Newmark-beta method* of second order (called ``"Newmark2"`` in the following), which is exactly the same time stepping which is used in oomph-lib. First order derivatives can be calculated with the *implicit (or backward) Euler method*, which is first order accurate. Since this method coincides with the *backward differentiation formula* (BDF) of first order, it will be called ``"BDF1"`` in the following. The BDF of second order for first derivatives is also implemented, which is called ``"BDF2"``. In fact ``"BDF2"`` is the default method for first order time derivatives in pyoomph.



.. toctree::
   :maxdepth: 5
   :hidden:
   
   timestepping/implicit.rst
   timestepping/midpoint.rst
   timestepping/timescheme.rst
   timestepping/higherorder.rst
   timestepping/adaptivity.rst

