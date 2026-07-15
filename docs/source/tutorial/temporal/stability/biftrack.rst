.. _sectemporalbiftrack:

Bifurcation tracking
~~~~~~~~~~~~~~~~~~~~

Finally, when a bifurcation point is a function of multiple parameters, one can jump on the bifurcation by adjusting one parameter and find the position of the bifurcation as a function of another parameter by arc length continuation. To that end, let us consider the Lorenz system :math:numref:`eqodelorenz` from :numref:`secODEtemporaladapt`. It is known that the Lorenz system has one trivial fixed point for :math:`\rho<1`, after which a supercritical pitchfork bifurcation can be found at :math:`\rho=1`, which eventually loses its stability in a subcritical *Hopf bifurcation*, whose location and presence are dependent on the parameters :math:`\sigma` and :math:`\beta`. Beyond the Hopf bifurcation, chaotic behavior can be expected.

First of all, let us find the pitchfork bifurcation at :math:`\rho=1` with pyoomph. To that end, the equations are loaded from the code from :numref:`secODEtemporaladapt` and a problem class is defined where all three parameters can be adjusted at run time, i.e. are bound by :py:meth:`~pyoomph.generic.problem.Problem.define_global_parameter` and can hence be changed after the C code generation and can be used for arc length continuation and bifurcation tracking. We reuse the code of the file :download:`adaptive_lorenz_attractor.py <../timestepping/adaptive_lorenz_attractor.py>` from :numref:`secODEtemporaladapt` here:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: from adaptive_lorenz_attractor import * # import the Lorenz problem
   :end-at: self.add_equations(ode@"lorenz")

First, find the pitchfork bifurcation as described in the previous section, including the analytically derived Hessian terms using :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis`. We must be somewhat close to the pitchfork bifurcation, so we start at :math:`\rho=0.5`, solve for the stationary solution and find the critical eigenvector as guess for :math:`\vec{v}_\text{g}`:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: problem.solve_eigenproblem(0) # get an eigenvector as guess

Now we find the pitchfork and indeed confirm that it is at :math:`\rho=1`:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: # Find the pitchfork in terms of rho
   :end-at: print(f"Pitchfork starts at rho={problem.rho.value}, x,y,z={x,y,z}")

At a pitchfork bifurcation, we cannot easily continue in :math:`\rho` since it is not clear which branch to take. Therefore, we obtain the critical eigenvector by :py:meth:`~pyoomph.generic.problem.Problem.get_last_eigenvectors`. As long as bifurcation tracking is active and it has been solved, it is not necessary (and not possible) to use :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` for that. Instead, :py:meth:`~pyoomph.generic.problem.Problem.get_last_eigenvectors` gives the critical eigenvector at the bifurcation. We therefore first store this eigenvector and then deactivate the bifurcation tracking to be ready to solve the normal Lorenz system (i.e. without the augmentation for the bifurcation tracking):

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: # this will be now the critical eigenvector at the bifurcation
   :end-at: problem.deactivate_bifurcation_tracking()

To jump on the stable branch of the pitchfork bifurcation, we can add this eigenvector to the degrees of freedom using the :py:meth:`~pyoomph.generic.problem.Problem.perturb_dofs` method, increase :math:`\rho` a bit beyond :math:`\rho>1` and perform a few transient solves to move towards the stable branch, before the stationary solve jumps on it:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: problem.perturb_dofs(perturb) # Go in the direction of the critical eigenvector
   :end-at: eigvals, eigvects=problem.solve_eigenproblem(0) # get the initial eigenvalues

Then, we gradually increase :math:`\rho` by arc length continuation, solve for the eigenvalues and monitor whether the largest real part of the eigenvalues crosses zero:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: # Scan rho to the Hopf bifurcation
   :end-at: print(f"On pitchfork branch rho={problem.rho.value}, x,y,z={x, y, z}, eigenvalue={eigvals[0]}")

The eigenvalue will have a non-zero imaginary value, which indicates a Hopf bifurcation. This means the critical eigenvalue will not be zero, but in fact a pair of imaginary values :math:`\pm i\omega`. For the same reason, the eigenvector :math:`\vec{v}` will be complex (and a complex conjugate counter-pair), i.e. :math:`\vec{v}=\vec{\phi}+i\vec{\psi}` with real valued :math:`\vec{\phi}` and :math:`\vec{\psi}`. The bifurcation tracking of a Hopf bifurcation with respect to the parameter :math:`r` (:math:`=\rho` here) is internally again handled by augmenting the system as follows:

.. math::

   \begin{aligned}
   \vec{F}(\vec{x}_\text{c},r_\text{c})&=0 \quad \text{,i.e. }\vec{x}_\text{c}\text{ is a stationary solution at the parameter }r_\text{c}\\
   \mathbf{J}(\vec{x}_\text{c},r_\text{c})\vec{\phi}+\mathbf{M}(\vec{x}_\text{c},r_\text{c})\vec{\psi}&=0 \quad \text{,i.e. the generalized eigenproblem is solved}\\
   \mathbf{J}(\vec{x}_\text{c},r_\text{c})\vec{\psi}-\mathbf{M}(\vec{x}_\text{c},r_\text{c})\vec{\phi}&=0 \quad \text{     for a pure imaginary eigenvalue }i\omega \\
   \vec{v}_\text{g}\cdot\vec{\phi}&=1 \quad \text{,i.e. the eigenvector }\vec{v}\text{ is non-trivial }\\
   \vec{v}_\text{g}\cdot\vec{\psi}&=0 \quad \text{ and }\vec{\psi}\text{ does not contribute to the real part of }\vec{v}_\text{g}
   \end{aligned}

Besides the complex eigenvector :math:`\vec{v}=\vec{\phi}+i\vec{\psi}`, which can be obtained after bifurcation tracking by the two eigenvectors returned from :py:meth:`~pyoomph.generic.problem.Problem.get_last_eigenvectors`, one also gets the critical parameter, where the bifurcation happens, and the frequency :math:`\omega`, which can be obtained by the imaginary part of ``get_last_eigenvalues[0]``.

Now, this is utilized to find the critical :math:`\rho_\text{c}` where the Hopf bifurcation is located:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: # Jump on the Hopf bifurcation
   :end-at: print(f"On Hopf branch rho={problem.rho.value}, x,y,z={x, y, z}, omega={numpy.imag(problem.get_last_eigenvalues()[0])}")

Since we do not :py:meth:`~pyoomph.generic.problem.Problem.deactivate_bifurcation_tracking`, it is still active. We can now perform an arc length continuation in another parameter, e.g. in :math:`\sigma`, to obtain the curve :math:`\rho_\text{c}(\sigma)`:

.. literalinclude:: bifurcation_hopf_tracking_lorenz.py
   :language: python
   :start-at: # Go down with sigma but staying on the Hopf bifurcation (i.e. do not call deactivate_bifurcation_tracking)
   :end-at: print(f"On Hopf branch rho,sigma={problem.rho.value,problem.sigma.value}, x,y,z={x, y, z}, omega={numpy.imag(problem.get_last_eigenvalues()[0])}")

..  figure:: lorenzbifurc.*
	:name: figodelorenzbifurc
	:align: center
	:alt: Hopf bifurcation of the Lorenz system
	:class: with-shadow
	:width: 70%
	
	Position of the Hopf bifurcation of the Lorenz system as a function of the parameters :math:`\rho` and :math:`\sigma` at :math:`\beta=8/3`. The analytical solution :math:`\rho_\text{c}=\sigma(\sigma+\beta+3)/(\sigma-\beta-1)` agrees perfectly.



Thereby, one can directly generate phase diagrams as e.g. depicted in :numref:`figodelorenzbifurc`.

The very same methods also work for spatio-temporal differential equations. An example will be discussed in :numref:`secpdekse`.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <bifurcation_hopf_tracking_lorenz.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		
               
