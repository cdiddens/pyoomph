.. _secODEtimescheme:

Easily selecting time stepping methods for first order time derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest method to switch between all implemented time stepping schemes is the usage of the function :py:func:`~pyoomph.expressions.generic.time_scheme`. This function can be applied to any mathematical expression and will expand the expression to the time scheme selected via the string argument ``scheme``. This is done in the following with the anharmonic oscillator

.. math::

   \begin{aligned}
   \partial_t y+y^3&=0\\
   \partial_t z-y=0\,.
   \end{aligned}

So let us implement an equation class ``AnharmonicOscillator`` for this as follows

.. code:: python

   # Anharmonic oscillator by first order system with different time stepping schemes
   class AnharmonicOscillator(ODEEquations):
       def __init__(self, scheme):
           super(AnharmonicOscillator, self).__init__()
           self.scheme = scheme

       def define_fields(self):
           self.define_ode_variable("y")
           self.define_ode_variable("dot_y")

       def define_residuals(self):
           y = var("y")
           dot_y = var("dot_y")
           residual = (partial_t(dot_y) + y ** 3) * testfunction(dot_y)
           residual += (partial_t(y) - dot_y) * testfunction(y)
           # Here, we evaluate the chosen scheme just by applying time_scheme(scheme,...)
           self.add_residual(time_scheme(self.scheme, residual))

The constructor takes an argument ``scheme``, which is - as usual - stored in a member variable. At the end of :py:meth:`~pyoomph.generic.codegen.BaseEquations.define_residuals`, just before adding the residuals to the problem, :py:func:`~pyoomph.expressions.generic.time_scheme` with the passed ``scheme`` is applied to the residual. The problem class and the corresponding code to run all simulations in different output directories reads

.. code:: python

   class AnharmonicOscillatorProblem(Problem):
       def __init__(self, scheme):  # Passing scheme here
           super(AnharmonicOscillatorProblem, self).__init__()
           self.scheme = scheme

       def define_problem(self):
           eqs = AnharmonicOscillator(scheme=self.scheme)

           t = var("time")  # Time variable
           eqs += InitialCondition(y=1,dot_y=0)

           # Calculate the total energy. We also use time_scheme here, e.g. the energy is evaluated by the same scheme as the time stepping
           y = var("y")
           total_energy = time_scheme(self.scheme, 1/2 * partial_t(y) ** 2 + 1/4 * y ** 4)
           eqs += ODEObservables(Etot=total_energy)  # Add the total energy as observable

           eqs += ODEFileOutput()
           self.add_equations(eqs @ "anharmonic_oscillator")


   if __name__ == "__main__":
       for scheme in {"BDF1", "BDF2", "Newmark2", "MPT", "TPZ", "Simpson", "Boole"}:
           with AnharmonicOscillatorProblem(scheme) as problem:
               problem.set_output_directory("osci_timestepping_scheme_" + scheme)
               problem.run(endtime=100, numouts=200)

Before comparing all time stepping schemes here, let us briefly discuss what the function :py:func:`~pyoomph.expressions.generic.time_scheme` actually does. Let us consider any generic expression :math:`G(\partial_t \vec{U},\vec{U},t)`, or in Python some expression ``G`` which may contain ``partial_t(var("U"))``, ``var("U")`` and ``var("time")``. To understand what the application of :py:func:`~pyoomph.expressions.generic.time_scheme` on ``G``, i.e. ``time_scheme(scheme,G)``, actually does, let us first introduce the shorthand notation

.. math:: G^{(n-k)}=G\left(\partial_t\vec{U},\vec{U}^{(n-k)},t^{(n-k)}\right)

i.e. the evaluation of the expression ``G`` at the :math:`k^\text{th}` history time step, with :math:`k=0` meaning the step we are currently solving for and :math:`k=1` meaning the values at the last successfully taken step. For a fractional :math:`k`, the arguments are interpolated linearly, e.g. for :math:`k=\frac{1}{4}` we get

.. math:: 
    :label: eqodefrackhistory

     G^{(n-\frac{1}{4})}=G\left(\partial_t\vec{U},\frac{3}{4}\vec{U}^{(n)}+\frac{1}{4}\vec{U}^{(n-1)},\frac{3}{4}t^{(n)}+\frac{1}{4}t^{(n-1)}\right)\,. 

The function :py:func:`~pyoomph.expressions.generic.time_scheme` does two things: depending on the selected ``scheme``, it replaces all occurrences of :math:`\partial_t\vec{U}`, i.e. all :py:func:`~pyoomph.expressions.generic.partial_t` calls, by an approximation (cf. :math:numref:`eqodetsteppweight`) suitable for the particular ``scheme``. Then, the expression ``G`` is expanded by a linear combination of the current and previous values of :math:`\vec{U}` and :math:`t`, i.e.

.. math:: \operatorname{time\_scheme}(\operatorname{scheme},G)=\sum_i g_i G^{(n-k_i)}

where :math:`g_i` are the weights of the contributions and :math:`k_i` are the corresponding history offsets, which might also be fractional. In the latter case, :math:numref:`eqodefrackhistory` is used. The possible time stepping methods with their approximation of :math:`\partial_t\vec{U}` and the used pairs :math:`(g_i,k_i)` are listed in :numref:`tableodetstepmeths`.


.. table:: Time stepping methods for systems of first order ODEs that can be selected via the call of :py:func:`~pyoomph.expressions.generic.time_scheme`. (\*) For non-equidistant :math:`\Delta t`, this approximation is more complicated. (\*\*) The Newmark2 scheme has additional history fields which are not elaborated here.
    :name: tableodetstepmeths

    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | scheme     | :math:`\partial_t \vec{U}` replacement                                                                                 | :math:`(g_i,k_i)`                                                                                                                                                |
    +============+========================================================================================================================+==================================================================================================================================================================+
    | "BDF1"     | :math:`\frac{1}{\Delta t^{(n)}}\left(\vec{U}^{(n)}-\vec{U}^{(n-1)}\right)`                                             | :math:`(1,0)`                                                                                                                                                    |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "BDF2"     | :math:`\frac{1}{\Delta t^{(n)}}\left(\frac{3}{2}\vec{U}^{(n)}-2\vec{U}^{(n-1)}+\frac{1}{2}\vec{U}^{(n-2)}\right)` (\*) | :math:`(1,0)`                                                                                                                                                    |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "Newmark2" | (\*\*)                                                                                                                 | :math:`(1,0)`                                                                                                                                                    |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "TPZ"      | cf. "BDF1"                                                                                                             | :math:`(\frac{1}{2},0)`, :math:`(\frac{1}{2},1)`                                                                                                                 |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "MPT"      | cf. "BDF1"                                                                                                             | :math:`(1,\frac{1}{2})`                                                                                                                                          |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "Simpson"  | cf. "BDF1"                                                                                                             | :math:`(\frac{1}{6},0)`, :math:`(\frac{2}{3},\frac{1}{2})`, :math:`(\frac{1}{6},1)`                                                                              |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | "Boole"    | cf. "BDF1"                                                                                                             | :math:`(\frac{7}{90},0)`, :math:`(\frac{16}{45},\frac{1}{4})`, :math:`(\frac{2}{15},\frac{1}{2})`, :math:`(\frac{16}{45},\frac{3}{4})`, :math:`(\frac{7}{90},1)` |
    +------------+------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Note that the scheme for ``partial_t(...,2)``-terms within ``G`` will not be adjusted by the application of :py:func:`~pyoomph.expressions.generic.time_scheme`. These are always calculated via the Newmark-beta method.

..  figure:: plot_anharm_osci.*
	:name: figodetstepcmpn
	:align: center
	:alt: Comparison of time-stepping schemes for the energy conservation of an anharmonic oscillator
	:class: with-shadow
	:width: 80%
	
	Total energy conservation of an anharmonic oscillator with different time stepping schemes. To visualize the impact a rather larger time step of :math:`\Delta t=0.5` was taken.


In :numref:`figodetstepcmpn`, the total energy of the simulations of the anharmonic oscillator with the different time steps is depicted. The time stepping is quite coarse, so that the differences are easily visible: First of all ``"BDF1"`` fails to conserve the energy dramatically and also ``"BDF2"`` is not suitable for this coarse time step. ``"Newmark2"`` conserves the energy quite acceptably over a long time; however, it has considerable problems in the first time steps. The reason is that ``"Newmark2"`` (and also ``"BDF2"``) require two history values to be set. However, the initial condition specified with :py:class:`~pyoomph.equations.generic.InitialCondition` in the code is independent of the time, i.e. there is no variable :math:`t` (:math:`=`\ ``var("time")``) occurring in the expressions we set for the initial condition. In this case, the first time step of ``"Newmark2"`` and ``"BDF2"`` is evaluated by ``"BDF1"``, which does not require any history values except for the initial values at :math:`t=0`. Alternatively, one can also supplement the :py:class:`~pyoomph.equations.generic.InitialCondition` object with the keyword argument ``degraded_start=False`` to fill all history values with the passed values, i.e. with ``y=1`` and ``dot_y=0`` here. In that case, or if the initial condition explicitly depends on the time, the first time step is not degraded to ``"BDF1"``. One can best circumvent this problem if the analytical solution is known: In that case, one can simply set the initial condition based on the analytical solution, as it was done in :numref:`secodetimesteppingsimple`.

All other methods do not have these problems: they (as also ``"BDF1"``) have no requirements of further history values. Furthermore, by explicitly considering the evaluation of :math:`\vec{U}` at the last successful time step, these methods are quite accurate and energy-conserving, given the large time step. However, in particular the method ``"Boole"`` is quite expensive since it involves a lot of evaluations at different sub-steps. If one reduces the time step, all methods increase in accuracy, but for e.g. ``"BDF1"`` it has to be reduced drastically to yield acceptable results. Moreover, if there is substantial dissipation in the system, i.e. conservation is not required, ``"BDF2"`` can already give quite good results. However, it is always best to check the time stepping scheme for your particular problem with an analytical solution if possible. If an analytical solution is not at hand, one should at least test whether a halved and a doubled time step influence the results. If so, one should take a smaller time step and repeat this procedure.

As a final note, other well-established methods, as e.g. the *Runge-Kutta method* of fourth order, are currently not possible in pyoomph. The reason is that it requires the storage of the results of the sub-stages and multiple solves for each time step. This is not implemented yet in pyoomph.

.. only:: html
    
    .. container:: downloadbutton

        :download:`Download this example <time_stepping_schemes.py>`
        
        :download:`Download all examples <../../tutorial_example_scripts.zip>`   
