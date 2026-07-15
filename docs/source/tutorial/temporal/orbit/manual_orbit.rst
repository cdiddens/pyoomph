Constructing orbits manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, we have constructed orbits in the Lorenz system by the emergence at the Hopf bifurcation. While this is a convenient way, you sometimes note dynamics which might be periodic (or at least quasi-periodic/chaotic), but you do not know any Hopf bifurcation in the vicinity. This in particular happens for the Lorenz system beyond the Hopf bifurcations, i.e. for values of :math:`\rho` higher than the Hopf bifurcation point. In the last example, we got unstable orbits for :math:`\rho` below the Hopf bifurcation, since it is a subcritical bifurcation. Beyond the bifurcation, such orbits do not exist. Instead, as we have seen in :numref:`secODElyapunovExponents`, we have chaotic dynamics there. However, inside this chaos, there are actually unstable orbits. Since they are unstable, they are never approached by the dynamics, but yet they exist. However, we cannot reach them from the Hopf bifurcation.

For such cases, one has to manually construct a guess for the periodic orbit. We can do so easily by first performing time integration in the chaotic region. After some transient time integration :math:`[0,t_0)`, we will consider a representative time interval :math:`[t_0,t_0+T)`. Obviously, since we are on a chaotic attractor, the periodicity condition :math:`\vec{x}(t_0)=\vec{x}(t_0+T)` will not be fulfilled. Due to the violated periodicity condition, huge jumps would occur if we just use this representative time dynamics as an initial guess for the orbit solver. This will lead to severe convergence issues when solving for the actual orbit.

Therefore, we will apply a low pass filter to match the start and end point nicely. Huge jumps correspond to high frequencies which then will be filtered out. The evolution after the application of the low pass filter will not fulfil the Lorenz system anymore, but it still constitutes a reasonable guess for the Newton solver.

Then, we manually start orbit tracking to solve for an initial orbit and subsequently continue it in :math:`\rho`. The corresponding code just reads:

.. literalinclude:: manual_orbit.py
   :language: python
   :start-at: # Load the previous code
                   
		        
		        
As described above, we first start in chaos, then run some initial steps followed by a representative period where we write output. This output is loaded, smoothed by a low-pass filter and subsequently used as a guess for the orbit. To that end, we first use :py:meth:`~pyoomph.generic.problem.Problem.set_current_dofs` to set the starting point of the orbit guess and ship the remaining history values to :py:meth:`~pyoomph.generic.problem.Problem.activate_periodic_orbit_handler`. Afterwards, the continuation is analogous to the previous example.



..  figure:: orbits_in_chaos.*
    :name: figmanualorbitslorenz
    :align: center
    :alt: Unstable orbits in the chaotic region of the Lorenz system
    :class: with-shadow
    :width: 100%
    
    Unstable orbits in the chaotic region of the Lorenz system. The color-code indicates the value of :math:`\rho`. Visualized is the pitchfork branch, time integrated chaos in black and the found periodic orbit branch.
    
    
.. warning::

	Here, we could just load the text file and calculate the smoothed history dofs from there. For e.g. PDEs, it might be more involved, since the order of the degrees of freedom is not necessarily the same as in the written output. Therefore, instead of loading the output from the file, it is more suitable to make a loop over :py:meth:`~pyoomph.generic.problem.Problem.solve` command with a suitable ``timestep`` argument and afterwards append the current degrees of freedom to an array. The current degrees of freedom can be obtained by the first return value of :py:meth:`~pyoomph.generic.problem.Problem.get_current_dofs`. After this loop, you have an array of history dofs, exactly as required here, which then can be filtered and used as an orbit guess as above.
    
.. only:: html

   .. container:: downloadbutton

      :download:`Download this example <manual_orbit.py>`
      
      :download:`Download all examples <../../tutorial_example_scripts.zip>`
