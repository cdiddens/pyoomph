.. _secODEhigheroderdt:

Time derivatives of third or higher order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we have seen so far, we can easily calculate first and second order time derivatives by :py:func:`~pyoomph.expressions.generic.partial_t` and - if wrapped in :py:func:`~pyoomph.expressions.generic.time_scheme` - we can select a bunch of time stepping schemes for the first order derivatives. However, if you try to calculate a third or higher order time derivative, e.g. by ``partial_t(...,3)`` or ``partial_t(partial_t(partial_t(...)))``, pyoomph will not be able to handle this and raises an error. This can only be circumvented by reducing it to a system with time derivatives of first or second order as we have done it in the case of the anharmonic oscillator in the previous section, where the second order ODE was split into two first order ODEs. This procedure can be generalized to an arbitrary degree. E.g. the *jerky dynamics* of a system :math:`\dddot{x}=J(x,\dot{x},\ddot{x})` can be conveniently written as :math:`\dot{z}=J(x,y,z)` with :math:`\dot{y}=z` and :math:`\dot{x}=y`.

