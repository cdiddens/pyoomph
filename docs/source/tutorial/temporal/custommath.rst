.. _sectemporalcustommath:

Adding custom math functions in Python
--------------------------------------

pyoomph comes with the basic math functions implemented, i.e. :py:func:`~pyoomph.expressions.exp`, :py:func:`~pyoomph.expressions.sin`, :py:func:`~pyoomph.expressions.cos`, etc. are implemented in the package :py:mod:`pyoomph.expressions`. However, sometimes, one needs additional custom mathematical functions. This can be done with the class :py:class:`~pyoomph.expressions.cb.CustomMathExpression`. In the following, we will discuss one non-dimensional case and one case with physical dimensions.

.. toctree::
   :maxdepth: 5
   :hidden:

   custommath/drivenho.rst
   custommath/tennis.rst
   custommath/multiret.rst
