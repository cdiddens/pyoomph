.. _advstabbifguiexample:

A worked example
~~~~~~~~~~~~~~~~

A rather complicated bifurcation diagram showing versatile bifurcation and periodic orbits can be found in Ref. :cite:`Voss2024`.
This paper deals thin-film hydrodynamics with two species of reactive surfactants. The corresponding equations for the film height :math:`h` and the surfactants :math:`\Gamma_1` and :math:`\Gamma_2` reads:

.. math::
	\begin{aligned}
		\partial_t h = & \nabla\cdot \left[\frac{h^3}{3} \vec{J}_p +\frac{h^2}{2}\nabla \left(\Gamma_1+\Gamma_2\right)  \right] \\
		\partial_t \Gamma_1 =& \nabla \cdot \left[ \frac{h^2\Gamma_1}{2}\vec{J}_p+\left(h\Gamma_1+D_1\right)\nabla\Gamma_1+h\Gamma_1\nabla\Gamma_2 \right] +j_r-\beta_1 \ln(\Gamma_1\delta)-\beta\mu  \\
		\partial_t \Gamma_2 = & \nabla \cdot \left[ \frac{h^2\Gamma_2}{2}\vec{J}_p+h\Gamma_2\nabla\Gamma_1+\left(h\Gamma_2+D_2\right)\nabla\Gamma_2 \right] -j_r-\beta_2 \ln(\Gamma_2\delta^{-1})+\beta\mu \\
		\text{with} \quad & \vec{J}_p=\nabla\left[W\left(\frac{1}{h^3}-\frac{1}{h^6}\right)-\nabla^2 h\right] \\
		& j_r=r(\delta\Gamma_1^2\Gamma_2-\delta^3\Gamma_1^3)
	\end{aligned}
	
As in the paper, we keep all parameters fixed and vary the chemostat driving strength :math:`\beta\mu` (to be read as single parameter).
The implementation of the equations is straightforward, one just has to introduce an auxiallary variable for :math:`\nabla^2h` to incorporate the fourth order derivative, as usual:

.. literalinclude:: thin_film_bifurcation_gui.py
   :language: python
   :start-at: class LubricationEquations(Equations):
   :end-at: .add_weak(jr,Gamma2test).add_weak(self.beta2*log(Gamma2/self.delta)-self.beta_mu,Gamma2test)
   
Also the problem itself is business as usual. The only constraint we have to consider is the translational invariance, which we fix by a Lagrange multiplier :math:`U`, which fixes the droplet in the center. Any translational motion is then captured by a nonzero :math:`U`, which of course has to enter back as advection to the PDEs:

.. literalinclude:: thin_film_bifurcation_gui.py
   :language: python
   :start-at: class LubricationProblem(Problem):
   :end-at: self+=eqs@"domain"
   
As explained on the previous page, we just have to start the GUI on this problem:

.. literalinclude:: thin_film_bifurcation_gui.py
   :language: python
   :start-at: if __name__=="__main__":
   :end-at: gui.start(0.002)
   
Once the GUI has opened, we can first use **Step** to perform an arclength continuation in the given parameter :math:`\beta\mu` with an initial step of :math:`0.002` as given by the code. The diagram itself shows the norm of the height plotted against :math:`\beta\mu`.


.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidbifgui1"><video autoplay="True" muted="" playsinline="" controls="" preload="auto" width="90%"><source src="../../../_static/bifurcationGUI1.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Creating bifurcation diagrams with the bifurcation GUI</span> </span></p></figcaption></figure>

Before starting, we set change the comments in the problem constructor to get the fine diagram. The coarse settings will be used for the expensive orbit tracking later.

First, we scan the trivial branch, just a symmetric equilibrium configuration of the droplet. We can either do multiple **Step** (:kbd:`Space`) or do a **Multi step**  (:vidtime:`vidbifgui1#0:00`). The latter will continue the branch until it hits the ranges of the plot. These can be adjusted by selecting the zoom or shift tool in the plot. Make sure to deactivate it afterwards. **Multi step** can be aborted by the **Abort** button at the top right at any time. You can *select* any point by clicking on it. When you press :kbd:`Enter` afterwards, the current state (*active point*) will jump to the *selected point*. You can then continue from there to add more points on the branch by adjusting or flipping the arclength step.

Solid lines represent stable branches, dashed lines unstable ones (at least one :math:`\mathrm{Re}(\lambda)>0`). Segments with unknown stability are plotted in dotted lines.

We then approach the first bifurcation by using the arclength tools to go close to it  (:vidtime:`vidbifgui1#0:02`). A look in the **Points** tab on the right will show the computed eigenvalues. You can select any of these and click on the button beside to try to find the corresponding bifurcation. This works of course only reasonably well in the vicinity of the bifurcation, i.e. when :math:`\mathrm{Re}(\lambda)\approx 0`. When selecting the correct eigenvalue for that, the bifurcation will be correctly identified and a new dot with a **P** (pitchfork) will appear in the diagram (:vidtime:`vidbifgui1#0:04`). Note that this particular bifurcation is quite intricate since there are further eigenvalues with :math:`\mathrm{Re}(\lambda)\approx 0` in the vicinity. Pressing **Find bif** at the top will try to find find the bifurcation corresponding to the eigenvalue with :math:`\min|\mathrm{Re}(\lambda)|`.

Whenever the *active point* is a bifurcation, we can switch the branch by pressing the **Switch** button in the top bar. The switching step is also controlled by the selected arclength, so decrease it in case if fails. In case of success, the former branch will become grey and one ends up on the new branch, which then can be continued as well (:vidtime:`vidbifgui1#0:05`). Clicking on any point on a grayed out branch will switch the active branch.

Afterwards, we continue these branches and identify further bifurcations along (:vidtime:`vidbifgui1#0:06`). In particular, the next pitchfork gives a symmetric branch with translating motion (:vidtime:`vidbifgui1#0:26`). This one cannot be nicely plotted against the currently selected norm of :math:`h`, since it cannot be disentangled. Instead, we can select :math:`U` as y-axis, which shows the data against this observable (:vidtime:`vidbifgui1#0:40`). However, the branch is entirely wrongly ordered in this plot. This is due to the fact that points are inserted into the current branch based on the closest segment in the current plot. So we select **Move point** instead of **Arclength** in the **Continuation** tab. We then select points, click on **Grap selected point** (it will make it blue) and press :kbd:`PageUp` / :kbd:`PageDown` to move the points within the branch until they are disentangled. Alternatively, you can click **Points>Disentagle Branch** in the top menu to try it automatically. Once the branch is disentangled, continuation can be used to fill additional points (change from **Move point** to **Arclength** again before, :vidtime:`vidbifgui1#0:53`). Once the branch is smooth enough, we can click **Interpolated Splines** in the **Continuation tab** to show smoothed curves instead of straight segments (:vidtime:`vidbifgui1#1:06`).

Once the translational motion branch is disentangled and fully captured, we can again switch to show the :math:`h`-norm on the :math:`y`-axis and continue to fill the remaining branches. In particular we can find a Hopf bifurcation (:vidtime:`vidbifgui1#1:12`), which we will investigate in a minute.

.. warning::

	In the following, we will continue periodic orbits and calculate Floquet multipliers. Even for this simple 1d problem, it gets quite expensive at the default resolution. Therefore, make sure to first backup your fine diagram (just rename the output folder), then reduce the resolution (change the commented lines in the problem constructor again) and create the entire diagram again on the coarse scale. 
	

.. only:: html

	.. raw:: html 

		<figure class="align-center" id="vidbifgui2"><video autoplay="True" muted="" playsinline="" controls="" preload="auto" width="90%"><source src="../../../_static/bifurcationGUI2.mp4" type="video/mp4"></video><figcaption><p><span class="caption-text">Tracking periodic orbits in the bifurcation GUI</span> </span></p></figcaption></figure>
		
For tracking of periodic orbits, the PETSc/MUMPS linear solver backend (real-valued) has been proven to be more stable. So after taking a backup of the diagram on the fine mesh and adjusting it for the coarse settings, start the script again, but this time with ``--petsc_mumps``, provided you have set up your ``PYTHONPATH`` accordingly (see :numref:`petscslepc`).

At a Hopf bifurcation, you can switch to the emergent orbit, but for this particular case, it is suitable to go to the **Orbit** tab and set the **Mode** to **bspline**. The periodic orbit will be then calculated based on a *B-spline interpolation* in time, which is usually more stable during continuation. However, for Floquet multipliers, it will implicitly switch back to a *collocation* represatation of the orbit implicitly.

After selecting a Hopf point, press **Switch** to jump on the orbit (:vidtime:`vidbifgui2#0:00`). Again, the chosen arclength step will determine the initial step, so you might have to reduce it in case of failure.
Then you can continue the orbit and a shaded region will indicate the range of the selected observable (:vidtime:`vidbifgui2#0:03`).

You can also switch again the :math:`y`-axis to plot e.g. the orbit time (:vidtime:`vidbifgui2#0:17`).

Once you are done with the diagram, you can mark several interesting points by pressing :kbd:`1`-:kbd:`9`. In the **File** menu, you can then **Export curves** and/or **Output the tagged** points. This allows to either use the state files of tagged points in be used with :py:meth:`~pyoomph.generic.problem.Problem.load_state` in another script (e.g. for manual continuation, replotting, further transient integration, etc.) or plot the bifurcation curves with a plotting backend of your choice.
			

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <thin_film_bifurcation_gui.py>`

		:download:`Download all examples <../../tutorial_example_scripts.zip>`
	
