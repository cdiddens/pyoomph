.. _secalepinchoff:

Pinch-off: letting the mesh change its topology
------------------------------------------------

The Rayleigh-Plateau filament of :numref:`secalegmshfields` was stopped just before the interesting moment: the run ends when the minimum radius falls below a threshold, because at that point the neck has become thinner than the mesh can carry. A sharp-interface method has to be told what to do at a break-up, unlike a phase field or a volume of fluid method, which handle these changes intrinsically. In this section we do tell it, and the same filament runs straight through its own break-up into three separate drops.



.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: class RayleighPlateauPinchOffProblem(Problem):
   :end-at: self.post_pinch_steps = 6

The one genuinely new parameter is ``rmin``: the interface radius at which we declare the column broken. It must be tied to what the mesh can resolve -- here it is two elements wide, ``2*hmin``, and it must be well below the radius of the drops it separates. As a rule of thumb: ``rmin`` at or below a tenth of the drop radius works, considerably coarser values fail.

The imports and the mesh class differ from the previous section in exactly one line, the base class:

.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: from pyoomph import *
   :end-at: self.gmsh_options["General.NumThreads"] = 1

A :py:class:`~pyoomph.equations.topological_changes.TopologicalChangesGmshTemplate` is an ordinary :py:class:`~pyoomph.meshes.gmsh.GmshTemplate` -- it remeshes by recreation, i.e. by calling :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.define_geometry` again -- with one addition: it can be handed a *surgery plan* and then report the reconnected geometry instead of the current one. The plan is computed with `shapely <https://shapely.readthedocs.io>`__, which is an optional dependency of pyoomph (``pip install pyoomph[topology]``); the script imports it at the top so that a machine without it says so immediately rather than in the middle of the transient. The feature is currently restricted to axisymmetric two-dimensional problems.

The gmsh options are those of the previous section, plus ``General.NumThreads = 1``. That is not a performance setting: gmsh parallelises its 2d meshing and the resulting node positions differ in the last bits from run to run, and on a transient that ends in a capillary singularity those bits separate within a few dozen steps. One thread makes successive runs reproducible.

The initial geometry is the perturbed cosine profile, built directly as a spline so that the mesh size fields already see the neck:

.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: if self.is_first_time():
   :end-at: "axisymm", self.point(0, pr.L), "top", pts[-1])[1]]

The remeshing branch is where the topological change is delivered, and the important point about it is that there is only **one** code path:

.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: else:
   :end-at: name="bottom" if y < 0.5 * pr.L else "top")

:py:meth:`~pyoomph.equations.topological_changes.TopologicalChangesGmshTemplate.get_reconnected_boundaries` replaces the call to :py:meth:`~pyoomph.meshes.mesh.MeshedMeshTemplate.get_boundary_coordinates` of the previous section, and it is *not* told whether a pinch-off is pending. If one is, the returned :py:class:`~pyoomph.equations.topological_changes.ReconnectedBoundaries` describes the geometry *after* the surgery; if none is, it wraps the current geometry into exactly the same structure. So the branch that carries the event is the branch that runs on every single quality remesh, and is therefore exercised constantly instead of once per simulation.

What that branch has to cope with is a *variable number of fragments*, which is why it is written as loops:

* :py:attr:`~pyoomph.equations.topological_changes.ReconnectedBoundaries.interface_chains` holds one :py:class:`~pyoomph.equations.topological_changes.ReconnectedChain` per connected fluid fragment, sorted by ascending :math:`z`. :py:meth:`~pyoomph.equations.topological_changes.TopologicalChangesGmshTemplate.spline_from_chain` turns one of them into a named spline, using the local element sizes the chain suggests. (Its ``points`` and ``suggested_sizes`` are public, so dropping down to :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.point` and :py:meth:`~pyoomph.meshes.gmsh.GmshTemplate.spline` for full control over the resolution is always possible.)
* :py:attr:`~pyoomph.equations.topological_changes.ReconnectedBoundaries.axis_segments` holds the pieces of the symmetry axis that belong to this phase -- one per fragment, with gaps where the fragments have separated. :py:meth:`~pyoomph.equations.topological_changes.TopologicalChangesGmshTemplate.lines_from_axis_segments` makes them into named lines.
* :py:attr:`~pyoomph.equations.topological_changes.ReconnectedChain.end_types` tells the two ends of a chain apart: ``"fixed"`` means it ends on a boundary that was already there, i.e. on one of our two symmetry planes, and ``"axis"`` means it closes itself off with a cap on the axis. Initially the single chain has two fixed ends. After the pinch, the lower and the upper fragment have one fixed end each and one fresh cap, and the satellite in between has none at all -- which is why the symmetry planes at :math:`z=0` and :math:`z=L` are created in a loop over the chain ends rather than written out twice.

Everything after that -- the ``plane_surface`` and the gmsh mesh size fields -- is the previous section verbatim, except that the fields now act on a *list* of interface curves and a list of axis curves. Gmsh assembles the named curves into as many closed loops as they form, so nothing has to be said about how many fragments there are.


On the equation side, a single line switches the topological changes on:

.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: def define_problem(self):
   :end-at: self.add_equations(eqs @ "liquid")

The :py:class:`~pyoomph.equations.topological_changes.AxisymmetricReconnection` monitors the interface and, when the minimal interface radius drops below ``rmin``, hands the mesh template a plan and asks for a remesh. It does not modify anything itself: the event is applied by the ordinary remeshing-by-recreation path, i.e. by the ``else:`` branch above. Note also that the :py:class:`~pyoomph.equations.generic.RemeshWhen` is still there and is still doing its usual job -- remeshing after a pinch is not optional, since the two fresh caps retract fast enough to degenerate the mesh within a handful of steps.

The :math:`\zeta` coordinates of the previous section are unchanged and keep working across the event: the reconnection writes the plan's own chart onto the old and the new interface, so the points of a fragment that the surgery did not touch keep exactly the parametrization they had before, and the fields are carried across the break-up rather than re-interpolated from scratch.

The :py:class:`~pyoomph.equations.generic.IntegralObservables` are our accuracy check, discussed below.


The run loop is the generator of the previous section with two additions:

.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: def run_until_broken(self
   :end-at: return

The first is the step control after the break-up. Before it, the step is a fixed fraction of the minimum radius, exactly as before: the inertial collapse follows :math:`r\sim(t_0-t)^{2/3}`, so a step proportional to :math:`r` can never jump over the event. Afterwards the minimum radius is the tip of one of the fresh caps, i.e. exactly zero, and no longer the scale that has to be resolved, so we switch to a fixed modest step.

The second is the one obligation this feature places on the user: A node that the surgery has created has no time history of its own; the interpolation from the old mesh gives it whatever the old mesh held at that position, which for a fresh cap is the middle of a neck that was collapsing at the largest velocity anywhere in the domain. That is what the second-order scheme extrapolates through on the first step past the event, and at a step size that was perfectly comfortable before it the Newton solve simply diverges. 

We therefore call ``problem.timestepper.set_num_unsteady_steps_done(0)``, to that the time stepper assumes we are at the first step of a new transient, and the BDF2 becomes a BDF1 for that step.


.. literalinclude:: rayleigh_plateau_pinchoff.py
   :language: python
   :start-at: if __name__ == "__main__":

The break-up happens at :math:`t\approx2.57`, and the result is shown in figure :numref:`figalepinchoffsatellite`.

..  figure:: pinchoff_satellite.*
	:name: figalepinchoffsatellite
	:align: center
	:alt: Rayleigh-Plateau pinch-off with a satellite drop
	:class: with-shadow
	:width: 100%

	(left) The interface over one wavelength, mirrored about the axis, before and after the break-up. The column does not pinch at one point but at both ends of the thin filament between the two growing drops, so a single event carries two simultaneous pinches and leaves three fragments: two drops and a satellite. (right) The relative change of the liquid volume in each accepted time step, on a logarithmic scale. The pinch-off step is no larger an error than the ordinary steps and quality remeshes around it.


The volume is the quantitative payoff. The surgery is volume-conserving by construction: the plan splits the parent volume at the waist plane and moves only the freshly created points, along their normals, until each child fragment carries its share exactly. What is left is the :math:`\mathcal{O}(h^2)` deviation between the plan's polyline and the Catmull-Rom spline that ``define_geometry`` draws through it -- which is precisely the error an ordinary quality remesh makes as well. Measured on this run (relative change of the liquid volume in one accepted time step):

.. list-table::
   :widths: 60 40

   * - worst ordinary step or quality remesh
     - :math:`9.3\times10^{-6}`
   * - the pinch-off step
     - :math:`-2.8\times10^{-6}`
   * - the whole run (71 steps)
     - :math:`-1.3\times10^{-4}`

i.e. the break-up costs about what remeshing the same mesh costs anyway, and both are far below the drift the free-surface kinematic condition accumulates over the run. :py:attr:`~pyoomph.equations.topological_changes.ReconnectedBoundaries.fragment_volumes` exposes the target volume of each fragment for exactly that purpose.

.. note::

   The reverse event, *coalescence*, is available from the same equation: pass ``distmin`` instead of (or in addition to) ``rmin`` and two fragments whose axial tip-to-tip gap falls below it will merge, with the same one-branch ``define_geometry`` handling the merged geometry. For a two-phase setting, :py:meth:`~pyoomph.equations.topological_changes.TopologicalChangesGmshTemplate.get_reconnected_boundaries` additionally takes an ``opposite_axis_name``, and fills :py:attr:`~pyoomph.equations.topological_changes.ReconnectedBoundaries.opposite_axis_segments` with the complementary axis pieces the surrounding phase has to cover.


.. only:: html

	.. container:: downloadbutton

		Full code available in the

		:download:`pyoomph example bundle <../tutorial_example_scripts.zip>`

		``Moving_Mesh/rayleigh_plateau_pinchoff.py``
