Jumping on bifurcations
~~~~~~~~~~~~~~~~~~~~~~~

If one is just interested in the critical parameter value where the bifurcation happens, e.g. :math:`r=0` in all previously discussed cases, there is another possibility to spot the bifurcation than performing arclength continuation and the calculation of eigenvalues. In complicated systems, one neither knows the critical stationary solution :math:`\vec{x}_\text{c}` (here a vector comprising e.g. multiple coupled ODEs) nor the critical parameter :math:`r_\text{c}`. However, one knows that the following has to hold at e.g. a fold bifurcation:

.. math::

   \begin{aligned}
   \vec{F}(\vec{x}_\text{c},r_\text{c})&=0 \quad \text{i.e. }\vec{x}_\text{c}\text{ is a stationary solution at the parameter }r_\text{c}\\
   \mathbf{J}(\vec{x}_\text{c},r_\text{c})\vec{v}&=0  \quad \text{i.e. there is an eigenvector }\vec{v}\text{ with eigenvalue zero}
   \end{aligned}

For a system with :math:`N` degrees of freedom (:math:`N=\operatorname{dim}(\vec{x})`), we have now :math:`2N+1` unknowns, namely the :math:`N` unknowns of :math:`\vec{x}_\text{c}`, the :math:`N` components of the unknown eigenvector :math:`\vec{v}` and the critical parameter :math:`r_\text{c}`, for which an eigenvalue becomes zero. There is obviously one equation missing, which relates to the magnitude of :math:`\vec{v}`, since without any further equation all parameters :math:`r` with corresponding stationary solutions will solve it for the trivial choice :math:`\vec{v}=0`. One could demand that e.g. :math:`\|\vec{v}\|=1`, but since a reasonable initial guess :math:`\vec{v}_\text{g}` for the eigenvector :math:`\vec{v}` is usually required anyhow, we just demand :math:`\vec{v}\cdot\vec{v}_\text{g}=1`. Thereby, this :math:`(2N+1)^\text{th}` equation is linear.

For the fold normal form :math:numref:`eqodefoldnf`, one gets the system

.. math::

   \begin{aligned}
   r_\text{c}-x^2_\text{c}=0, \qquad -2x_\text{c}v=0, \qquad vv_\text{g}=1
   \end{aligned}

which yields for all :math:`v_\text{g}\neq 0` the known solution :math:`r_\text{c}=0` and :math:`x_\text{c}=0`. For more complicated systems, however, this has to be solved numerically, which can be done in pyoomph by the :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking` method (again after importing the problem and equation classes from :download:`bifurcation_fold_param_change.py`):

.. code:: python

   from bifurcation_fold_param_change import *

   if __name__=="__main__":
       with FoldProblem() as problem:

           # Find any start solution, which must be close to the bifurcation
           problem.r.value=1
           problem.get_ode("fold").set_value(x=1)
           problem.solve()

           # Find a guess for the normalization constraint
           problem.solve_eigenproblem(0)
           vguess=problem.get_last_eigenvectors()[0] # use the eigenvector as guess

           # Activate fold bifurcation tracking in parameter r and solve the augmented system
           problem.activate_bifurcation_tracking(problem.r,"fold",eigenvector=vguess)
           problem.solve()

           print(f"Critical at r_c={problem.r.value} and x_c={problem.get_ode('fold').get_value('x')}")


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_fold_tracking.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		

The same works also for a pitchfork bifurcation, but these are subject to a symmetry, which is broken by the bifurcation. If we apply the fold tracking method to the pitchfork normal form :math:numref:`eqodepitchforknf`, we would get the augmented system

.. math::

   \begin{aligned}
   r_\text{c}x_\text{c}\pm x^3_\text{c}=0, \qquad \left(r\pm 3x^2_\text{c}\right)v=0, \qquad vv_\text{g}=1\,.
   \end{aligned}

While it indeed gives the correct solution :math:`x_\text{c}=0` at :math:`r_\text{c}=0`, the root has a multiplicity of three. Numerically, this hampers the Newton solver from converging.

To reduce the multiplicity and account for the symmetry in the pitchfork bifurcation, a symmetry vector :math:`\vec{\psi}` is considered and the following :math:`2N+2`-system is solved:

.. math::

   \begin{aligned}
   \vec{F}(\vec{x}_\text{c},r_\text{c})+\sigma\vec{\psi}&=0 \quad \text{i.e. }\vec{x}_\text{c}\text{ is a stationary solution at the parameter }r_\text{c}\\
   \mathbf{J}(\vec{x}_\text{c},r_\text{c})\vec{v}&=0 \quad \text{i.e. there is an eigenvector }\vec{v}\text{ with eigenvalue zero} \\
   \vec{v}_\text{g}\cdot\vec{v}&=1 \quad \text{i.e. the eigenvector }\vec{v}\text{ in non-trivial (for reasonables guesses) }\vec{v}_\text{g}\\
   \vec{\psi}\cdot\vec{x}_\text{c}&=0 \quad \text{i.e. the solution is symmetric with respect to the symmetry vector }\vec{\psi}
   \end{aligned}

Note that the slack variable :math:`\sigma` enforcing the symmetry will be zero at the bifurcation. For the pitchfork normal form with the simple scalar equivalents :math:`\psi=v_\text{g}=1`, we obtain indeed :math:`x_\text{c}=0,\,r_\text{c}=0,\,v=1,\,\sigma=0` and the root of the system has a multiplicity of one, i.e. the Jacobian of the augmented system at the solution is invertible. Thereby, the Newton solver converges well. To find a pitchfork bifurcation, one just has to pass ``"pitchfork"`` instead of ``"fold"`` as the second argument for the :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking` method. Again, one can pass an ``eigenvector`` argument which will be used as eigenvector normalization vector :math:`\vec{v}_\text{g}` and as symmetry vector :math:`\psi`. Please refer to the supplied code :download:`bifurcation_pitchfork_tracking.py` (dependent on :download:`bifurcation_pitchfork_arclength_eigen.py`) for an example.

.. warning::

   When solving with :py:meth:`~pyoomph.generic.problem.Problem.activate_bifurcation_tracking`, you must deactivate it via :py:meth:`~pyoomph.generic.problem.Problem.deactivate_bifurcation_tracking` after the solve to solve the normal system again (i.e. the system without the augmentation). Also the calculation of eigenvalues and -vectors does not work as usual while bifurcation tracking is active. See the next section on how to obtain the critical eigenvector.

.. warning::

   Tracking pitchfork bifurcations in spatio-temporal problems (cf. :numref:`secpde`) requires that the mesh is conforming with the symmetry vector, i.e. the mesh should be also symmetric along the symmetry that is broken by the bifurcation.


.. note::

   Pyoomph can improve the convergence of bifurcation tracking by calling the method :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis`. This will generate symbolical C code for the required Hessian terms in the augmented systems for bifurcation tracking. Without calling this method, finite differences are used to calculate the Hessian terms, which can be less accurate and slower. For more details, we refer to :numref:`secdropletdetach` and our article :cite:`Diddens2024`.