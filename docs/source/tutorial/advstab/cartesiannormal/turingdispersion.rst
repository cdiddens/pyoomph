Numerically obtaining the dispersion relation of a Turing instability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The additional Cartesian mode can be even used if :math:`N=0` holds, i.e. we can expand a single point to an interval of length :math:`[0:2\pi/k)` (with periodic boundary conditions) and determine the eigenvalues and eigenfunctions here. This allows us to quickly scan over :math:`k` to numerically extract a dispersion relation. We will discuss this here for a simple Turing instability (see :cite:`Turing1952,Gierer1972,Schnorr2023` for more details).

We begin by the equation class, which takes the diffusivity ratio and the reaction terms of the activator and inhibitor as arguments:

.. literalinclude:: turing_dispersion.py
   :language: python
   :start-at: class TuringEquations(Equations):
   :end-at: self.add_weak(partial_t(v)-self.g,vtest).add_weak(self.d*grad(v),grad(vtest))

The problem class now just defines the reaction terms according to the Gierer-Meinhardt model :cite:`Gierer1972,Schnorr2023`. We then add a :py:class:`~pyoomph.meshes.simplemeshes.PointMesh`, which is a :math:`0`-dimensional mesh consisting only of a single point and define the just developed equations on this mesh. Of course, on such a mesh, we could only get the eigendynamics of the ODEs :math:`\dot u=f, \dot v=g`, arising in absence of the diffusion terms. However, we will soon use the additional Cartesian normal mode to add a single :math:`x`-coordinate to quickly and exactly (i.e. without any spatial discretization) calculate the eigenvalues of the corresponding one-dimensional system, i.e. with diffusion terms.

.. literalinclude:: turing_dispersion.py
   :language: python
   :start-at: class TuringProblem(Problem):
   :end-at: self+=eqs@"domain"

In the driver code, we call :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis` with  ``additional_cartesian_mode=True`` to add the additional Cartesian normal mode. Supplying ``normal_mode_k=k`` during the calls of :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` will select the wavenumber :math:`k` for such a one-dimensional case.
We can thereby quickly scan the full dispersion relation:

.. literalinclude:: turing_dispersion.py
   :language: python
   :start-at: with TuringProblem() as problem:
   :end-at: output.add_row(k,numpy.real(evs[0]),numpy.real(evs[1]),numpy.imag(evs[0]),numpy.imag(evs[1]))

Of course, the results depicted in :numref:`figturingdispersion` can also be calculated analytically. However, it is already rather complicated to find the stationary base solution. Also, it is only a few lines of code and we have already implemented our equation class to be used in arbitrary dimensions and coordinate systems.

..  figure:: turingdispersion.*
	:name: figturingdispersion
	:align: center
	:alt: Dispersion relation of the Turing instability
	:class: with-shadow
	:width: 80%

	Numerically calculated dispersion relation of the Turing instability.



In particular, we can quickly get a good guess for the dominant wavenumber, here around :math:`k=0.46`. This can be used to estimate a reasonable domain size for e.g. a two-dimensional transient simulation, where we can reuse the existing equation and problem class to obtain Turing patterns as depicted in :numref:`figturingtransient`. The corresponding script can be found here: :download:`turing_transient.py`.

..  figure:: turing_transient.*
	:name: figturingtransient
	:align: center
	:alt: Turing pattern
	:class: with-shadow
	:width: 50%

	After finding the dominant wave number, we can run transient 2d simulations with a suitable domain size.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <turing_dispersion.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
