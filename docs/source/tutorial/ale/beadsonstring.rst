.. _secalebeadsonstring:

Beads on a string: a free surface driven by a polymer stress
-------------------------------------------------------------

The Rayleigh-Plateau filament of :numref:`secalegmshfields` was Newtonian, and it did what a Newtonian filament does: the neck thins ever faster and pinches off. If the liquid is a polymer solution instead, the outcome changes completely. The thread between two forming drops is a uniaxial extensional flow, which is exactly the flow in which the Oldroyd-B chains of :numref:`secpdeviscoelastic` stretch without bound, and the axial stress they build up arrests the pinch-off. What is left is a *beads-on-a-string* structure: nearly spherical drops joined by threads that keep thinning exponentially instead of breaking :cite:`Clasen2006`.

This section combines the two ingredients -- the moving mesh with reconstruction-based remeshing from the previous section, and the log-conformation viscoelasticity of :numref:`secpdeviscoelastic` -- to reproduce the corresponding verification model of the COMSOL Polymer Flow Module.

Lengths are scaled by the unperturbed filament radius :math:`R_0`, stresses by :math:`\sigma/R_0` and time by the inertio-capillary time :math:`\tau=\sqrt{\rho R_0^3/\sigma}`, which leaves three dimensionless numbers,

.. math::

   \mathrm{Oh}=\frac{\eta_0}{\sqrt{\rho\sigma R_0}},\qquad
   \mathrm{De}=\frac{\lambda}{\tau},\qquad
   \beta=\frac{\eta_\mathrm{s}}{\eta_0}

i.e. the Ohnesorge number, the Deborah number and the solvent fraction of the total viscosity :math:`\eta_0=\eta_\mathrm{s}+\eta_\mathrm{p}`. The dimensionless solvent and polymer viscosities are then :math:`\beta\,\mathrm{Oh}` and :math:`(1-\beta)\mathrm{Oh}`, and the dimensionless relaxation time is numerically :math:`\mathrm{De}`. The reference case is

.. literalinclude:: beads_on_string.py
   :language: python
   :start-at: def __init__(self):
   :end-at: self.tend, self.outstep, self.maxstep = 300.0, 5.0, 1.0

on a domain of two wavelengths, :math:`0\le z\le 8\pi`, starting from :math:`r(z,0)=1+0.05\cos(z/2)` at rest. Note that these are plain attributes of the problem class, so each of them can be overridden on the command line with ``-P``, which is used for the second parameter set at the end of this section.

The mesh class is the one of the previous section almost verbatim -- gmsh size fields that follow the local radius on the interface and the distance to the interface on the axis, and a spline through :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates` whenever we remesh. The only change is that the initial interface is not a straight line but the perturbed cosine profile. That size control is not a refinement here but a necessity: the thread ends up twenty-five times thinner than the beads that feed it, and a mesh which is uniform on the scale of a bead has no elements across the thread at all.

The equations are where the two chapters meet:

.. literalinclude:: beads_on_string.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_equations(eqs @ "liquid")

Three points about this are worth making.

First, the viscosity handed to the :py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` is the *solvent* viscosity only. The polymer enters exclusively through the :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticEquations`, which add both the evolution equation for :math:`\log\mathbf{C}` and the polymer stress in the momentum equation.

Second, and this is the point of the example, the free surface needs no polymer term of its own. :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface` imposes :math:`\vec{n}\cdot\boldsymbol{\sigma}=-\sigma\kappa\vec{n}` weakly, and the momentum residual it completes is the weak form assembled in the bulk. The polymer contributes to that bulk residual through :math:`\langle\boldsymbol{\tau}_\mathrm{p},\nabla\vec{v}\rangle` exactly as the solvent does, so the traction the interface sees is already the total one. Adding the polymer traction explicitly would count it twice.

Third, the two ends are treated as symmetry planes, while the reference uses a periodic flow condition to mimic an infinite filament. The initial perturbation has a maximum at both :math:`z=0` and :math:`z=8\pi`, so the periodic solution is mirror-symmetric there anyway, and free slip together with a mesh that may only slide radially imposes the same thing at a fraction of the cost. The interfacial :math:`\zeta` coordinates are set exactly as in the previous section, since the remeshing machinery is identical.

The run itself just steps through the output times, letting the adaptive time stepping choose the steps in between, and records the minimum radius through the :py:class:`~pyoomph.equations.generic.ExtremumObservables` as before:

.. literalinclude:: beads_on_string.py
   :language: python
   :start-at: if __name__ == "__main__":
   :end-at: minimum_out.add_row(problem.get_current_time(), *problem.minimum_radius_and_position())

:py:class:`~pyoomph.equations.generic.RemeshWhen` rebuilds the mesh whenever the elements have deformed too much, which over these 300 time units happens six times.

..  figure:: beadsonstring_shapes.*
	:name: figalebeadsonstringshapes
	:align: center
	:alt: Beads-on-string filament profiles
	:class: with-shadow
	:width: 70%

	Filament profiles at :math:`t=0,20,30,100` and :math:`300`, to be compared with Fig. 1 of the reference model. The instability grows slowly, collapses into the beads-on-string structure between :math:`t\approx20` and :math:`t\approx30`, and from then on merely drains the threads into the beads, which become spherical.

The quantity the reference reports is the minimum radius. It drops by a factor of four in the first thirty time units and then crosses over to a straight line in :math:`\log r`. That is the elasto-capillary regime: surface tension pulls on the thread, the polymer stress balances it, and the balance can only hold if the thread thins at the rate at which the polymer relaxes,

.. math::

   r\;=\;r_0\exp\left(-\frac{t}{3\lambda}\right)

with the factor three coming from the uniaxial kinematics :cite:`EntovHinch1997`.

..  figure:: beadsonstring_radius.*
	:name: figalebeadsonstringradius
	:align: center
	:alt: Minimum filament radius against time
	:class: with-shadow
	:width: 70%

	The minimum radius, to be compared with Fig. 2 of the reference model, whose curve is the dashed line. Both leave the fast stage at :math:`\log_{10}r\approx-0.65` around :math:`t=30` and follow the asymptotic slope from there.

Fitting the computed curve for :math:`t>100` gives a decay rate of :math:`3.65\cdot10^{-3}` against the predicted :math:`1/(3\lambda)=3.51\cdot10^{-3}`, i.e. the thread thins :math:`4\,\%` faster than the slender asymptote -- unsurprisingly, since that asymptote neglects both the solvent viscosity and the drainage into the beads. The minimum radius ends at :math:`\log_{10}r=-1.085` at :math:`t=300`, against :math:`-1.06` read off the reference figure. This run has about 28 000 degrees of freedom and takes some ten minutes.

Lowering the viscosity and the elasticity to :math:`\mathrm{Oh}=0.4`, :math:`\mathrm{De}=0.8` changes the picture: elasticity no longer holds the thread together on its own and inertia has time to act on it, so each thread grows a *satellite* drop before it breaks. Since every parameter is an attribute of the problem class, this second case needs no second script:

.. code:: bash

   python beads_on_string.py -P Oh=0.4 De=0.8 tend=19.75 outstep=0.25 maxstep=0.25 \
                                elements_per_radius=5 max_elements_per_radius=8 thinnest_thread=0.005

The finer mesh is not optional -- the reference notes the same requirement -- and it is what makes this the expensive one of the two: the size field keeps refining as the thread thins, so the problem grows from 67 000 to nearly 200 000 degrees of freedom on the way to :math:`t=19.75`, and the run takes about a quarter of an hour.

..  figure:: beadsonstring_satellite.*
	:name: figalebeadsonstringsatellite
	:align: center
	:alt: Filament shapes with and without satellite drops
	:class: with-shadow
	:width: 45%

	Compare Fig. 3 of the reference model: the elasticity-dominated case at :math:`t=300` (left) keeps a smooth thread, while the low-Ohnesorge case (right) grows a satellite drop on each of the two threads.


.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <beads_on_string.py>`

		:download:`Download all examples <../tutorial_example_scripts.zip>`
