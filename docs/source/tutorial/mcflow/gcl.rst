.. _secgcl:

Evaporation from a capillary & the geometric conservation law (GCL)
-------------------------------------------------------------------

We want to use the multi-component flow equation to the evaporation of a mixture from a capillary tube, analogously to what has been done in Ref. :cite:`Raju2024`.
As in this reference, we consider the tube, filled with a mixture of glycerol and water, as a 1d system. At the bottom of the tube, water is evaporating, leaving the non-volatile glycerol behind. This implies that the total mass of glycerol inside the liquid must be conserved.

However, for simple ALE methods which just use ``partial_t(...,ALE=True)``, i.e. the ALE-corrected time derivative :py:func:`~pyoomph.expressions.generic.partial_t`, we will see that it does not perfectly conserve the total mass of glycerol. Instead, we get a time-step-dependent conservation, which is a typical result when using ALE without respecting the *geometric conservation law*.

If we want to have conservation up to machine-precision, float truncation and the Newton solver threshold, we must 
