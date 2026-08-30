.. _secpdeadapt:

Spatial error estimation and adaptivity
---------------------------------------

Every example so far has either used a fixed mesh or refined it with a tolerance: you set
:py:attr:`~pyoomph.generic.problem.Problem.max_permitted_error`, add a
:py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` and let pyoomph refine wherever the
estimated error is too large. That works well when the solution has a few localised features and you
know roughly how accurate you want to be.

This section is about the cases where it does not, and about what you can say to the estimator
instead. Three things are worth understanding before trusting an adapted mesh:

- **What the estimator actually measures.** pyoomph uses a Zienkiewicz--Zhu flux-recovery estimator:
  it reconstructs a smooth flux from the discontinuous finite-element one and calls the difference
  the error. The *flux* is the expression you hand to
  :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator`, so it is a modelling choice, not
  something the framework decides for you. Choose it badly and the mesh will faithfully resolve
  something you do not care about.
- **What the numbers mean.** By default the elemental errors are divided by the recovered flux norm
  of the whole mesh, i.e. they are *relative*: dimensionless, but blind to how well resolved the mesh
  is overall, and a common factor on the flux cancels out exactly. ``normalize_relative=0`` gives the
  absolute error instead, which does shrink under refinement and is comparable between meshes and
  between adaptation steps.
- **When to give up on tolerances entirely.** Some solutions have structure on every scale, so no
  tolerance is ever reached and refinement never stops. There,
  :py:attr:`~pyoomph.generic.problem.Problem.desired_ndof` is the well-posed control: instead of
  asking for an accuracy, you state a budget and let pyoomph spend it on the elements with the
  largest errors.

The first example takes a problem that breaks the tolerance-based approach outright, and uses it to
show all three points at once. The second one has several fields at once and asks a different
question: not how much mesh to spend, but on which field to spend it.

.. toctree::
   :maxdepth: 5
   :hidden:

   adapt/moffatt.rst
   adapt/cylinder.rst
