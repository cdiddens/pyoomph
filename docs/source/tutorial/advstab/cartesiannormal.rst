.. _cartesiannormalstabana:

Cartesian normal mode stability analysis
----------------------------------------

Similar to the azimuthal stability analysis, it is possible to expand a :math:`N`-dimensional Cartesian problem to an :math:`N+1`-dimensional problem for stability analysis. Instead of the azimuthal angle :math:`\phi`, now a wavenumber :math:`k` in the :math:`N+1`-th direction is introduced. Again, we have a stationary solution of our unknown vector :math:`\vec{U}^{(0)}(x_1,\ldots, x_N)`, which is independent of :math:`x_{N+1}`. We then investigate perturbations like :math:`\epsilon\vec{U}^{(k)}\exp(ik x_{N+1}+\lambda_k t)`. 

Analogous to the azimuthal component in azimuthal stability analysis, all vector fields :math:`\vec{v}` will get an additional component, i.e. a contribution :math:`v_{N+1}\vec{e}_{N+1}`.
Main differences are that :math:`k` is real-valued (opposed to the integer-valued azimuthal mode :math:`m`) and that there are no specific boundary conditions for the eigenfunction at an axis of symmetry. However, global degrees of freedom, e.g. a global pressure enforcing a volume, will be deactivated by default when :math:`k\neq 0`.

.. toctree::
   :maxdepth: 5
   :hidden:
   
   cartesiannormal/turingdispersion.rst
   cartesiannormal/rivulet.rst   

