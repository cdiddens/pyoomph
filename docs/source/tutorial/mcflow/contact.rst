Contact line models
-------------------

In the previous example, we had a static interface and free flow at the side walls ``"bottom"``/``"top"`` towards the liquid-gas interface. For e.g. an evaporating droplet, as already discussed in :numref:`secmultidomdropevap`, the consideration of contact line dynamics is required. Contact lines can either be pinned or moving and of course this also can change during the drying of the droplet, causing a stick-slip behavior of the contact line. Moreover, when the surface tensions of the three interfaces in contact are changing at the contact line, usually also the equilibrium contact angle in case of a moving contact angle will change.

For multi-component flow, pyoomph has a single equation class, the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations`, which can be added to the contact line domain to handle all considerable cases. The particular dynamics is implemented in versatile contact angle models, which are passed to the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations`.

Both the versatile contact line models and the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations` are defined in the file :py:mod:`pyoomph.equations.contact_angle`.

PinnedContactLine
~~~~~~~~~~~~~~~~~

As we have seen in :numref:`secmultidomdropevap`, a pinned contact line can be realized by enforcing ``partial_t(var("mesh"))``\ :math:`=0` on the test space :math:`\vec{v}` of the velocity. Hence, if an instance of the :py:class:`~pyoomph.equations.contact_angle.PinnedContactLine` model is passed to the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations`, it will add the following weak contribution

.. math:: \left[\partial_t \vec{X}\cdot\vec{t}_\text{w},\mu \right]+\left[\lambda,\vec{v}\cdot\vec{t}_\text{w} \right]

with the position-enforcing Lagrange multiplier :math:`\lambda` with test function :math:`\mu`. :math:`\vec{X}` is the position of the contact line and :math:`\vec{t}_\text{w}` is here the tangent of the wall (i.e. the substrate in the case of an evaporating droplet). The wall tangent can be set by the ``wall_tangent`` keyword argument of the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations` class. It should be tangential to the wall, having a magnitude of unity and pointing inward (i.e. towards the droplet domain in the evaporating droplet case). Since we must have the possibility to add a contribution to the velocity test space in the direction of the wall tangent, it is necessary to allow for slip by using e.g. a slip length boundary condition on the substrate.

UnpinnedContactLine
~~~~~~~~~~~~~~~~~~~

When the contact angle freely moves, we have seen how to impose an equilibrium contact angle :math:`\theta_\text{eq}` in :numref:`secALEdropspread`. More general, when the contact angle is measured with respect to the wall tangent :math:`\vec{t}_\text{w}`, we add

.. math:: :label: eqmcflowcleqneumann

   \left[\sigma\left(\sin(\theta_\text{eq}) \vec{n}_\text{w} + \cos(\theta_\text{eq})\vec{t}_\text{w}\right),\vec{v} \right]\,,

where :math:`\vec{n}_\text{w}` is the wall normal, pointing into the droplet domain. With these vectors, the contact angle :math:`\theta` approaches zero when the droplet is very flat, :math:`\theta=90\:\mathrm{^\circ}` holds for a hemi-spherical droplet and for droplets on hydrophobic substrates, :math:`\theta>90\:\mathrm{^\circ}` can be observed.

However, as also discussed in :numref:`secALEdropspread`, the speed of the contact line motion by which :math:`\theta` approaches the equilibrium :math:`\theta_\text{eq}` is just controlled by the slip length. This is hard to control and to adjust with e.g. experimental observations. Instead, the :py:class:`~pyoomph.equations.contact_angle.UnpinnedContactLine` lets you control the motion velocity by a relation

.. math:: :label: eqmcflowclmovespeed

   U_{\text{cl}}=\partial_t \vec{X}\cdot\vec{t}_\text{w}=-U_\text{cl}^0\left(\theta-\theta_\text{eq}\right)

with some typical velocity scale :math:`U_\text{cl}^0`. In the spirit of the :py:class:`~pyoomph.equations.contact_angle.PinnedContactLine`, which is just given by :math:`U_\text{cl}^0=0`, the :py:class:`~pyoomph.equations.contact_angle.UnpinnedContactLine` class therefore adds the weak contribution

.. math:: :label: eqmcflowclunpinned

   \left[\sigma\left(\sin(\theta_\text{eq}) \vec{n}_\text{w} + \cos(\theta_\text{eq})\vec{t}_\text{w}\right),\vec{v} \right]+\left[\partial_t \vec{X}\cdot\vec{t}_\text{w}-U_{\text{cl}},\mu \right]+\left[\lambda,\vec{v}\cdot\vec{t}_\text{w} \right]\,,

The wall normal :math:`\vec{n}_\text{w}` can be passed by the ``wall_normal`` argument to the :py:class:`~pyoomph.equations.contact_angle.DynamicContactLineEquations` class, whereas the speed scale :math:`U_\text{cl}^0` and the equilibrium contact angle can be passed via ``cl_speed_scale`` (defaults to :math:`10^{-5}\:\mathrm{m}/\mathrm{s}`) and ``theta_eq`` (defaults to the initial contact angle) of the :py:class:`~pyoomph.equations.contact_angle.UnpinnedContactLine`. If you want to modify the relation :math:numref:`eqmcflowclmovespeed`, you can do so by inheriting a custom contact line model from the :py:class:`~pyoomph.equations.contact_angle.UnpinnedContactLine` and overriding the method :py:func:`~pyoomph.equations.contact_angle.UnpinnedContactLine.get_unpinned_motion_velocity_expression` according to your demands.

To adjust the contact angle at infinite speed, i.e. instantaneously to the equilibrium contact angle, you can set ``cl_speed_scale=None``. Thereby, the droplet will always be at :math:`\theta=\theta_\text{eq}`. This can be used e.g. to impose exactly some experimental dynamics, if you set ``theta_eq`` to a function of time that gives the experimental data.

StickSlipContactLine
~~~~~~~~~~~~~~~~~~~~

We have seen similarities between the pinned and the unpinned contact line, where the former is just the latter, but with :math:`U_{\text{cl}}=0`. The additional Neumann term :math:numref:`eqmcflowcleqneumann` imposing the equilibrium contact angle in :math:numref:`eqmcflowclunpinned` will be compensated anyhow by the Lagrange multiplier :math:`\lambda` in the :py:class:`~pyoomph.equations.contact_angle.PinnedContactLine`. Therefore, a stick-slip motion can be realized by toggling between both contact angle modes with an indicator function :math:`\psi`, which is :math:`0` if the contact line is pinned and :math:`1` if the contact line moves according to :math:numref:`eqmcflowclmovespeed`. Hence, the relaxation velocity is just augmented by the factor :math:`\psi` in the weak form:

.. math:: :label: eqmcflowclstickslip

   \left[\sigma\left(\sin(\theta_\text{eq}) \vec{n}_\text{w} + \cos(\theta_\text{eq})\vec{t}_\text{w}\right),\vec{v} \right]+\left[\partial_t \vec{X}\cdot\vec{t}_\text{w}-\psi U_{\text{cl}},\mu \right]+\left[\lambda,\vec{v}\cdot\vec{t}_\text{w} \right]\,,

The factor :math:`\psi` can be controlled by the :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` and :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.unpin` methods of the :py:class:`~pyoomph.equations.contact_angle.StickSlipContactLine`. Initially, it is :math:`1`, i.e. the contact line can freely move, unless you call :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` before running the simulation. The speed of the contact line and the equilibrium contact angle can be set analogously to the :py:class:`~pyoomph.equations.contact_angle.UnpinnedContactLine`.

However, it is cumbersome to switch the contact line dynamics by hand during the simulation. In reality, the contact line of an evaporating droplet starts usually pinned and switches to an unpinned motion once the contact angle :math:`\theta` falls below some receding unpinning contact angle :math:`\theta^\text{rec}_\text{unpin}`. It can then re-pin, when the contact line recedes so that the contact angle has risen above some value :math:`\theta^\text{rec}_\text{pin}`, where :math:`\theta^\text{rec}_\text{unpin}<\theta^\text{rec}_\text{pin}<\theta_\text{eq}` must hold for an alternating stick-slip motion during evaporation. If the droplet grows, e.g. due to condensation, we can have similar effects. Here, the contact line will unpin once the contact angle :math:`\theta` rises above :math:`\theta^\text{adv}_\text{unpin}` and pins again once the contact angle has fallen again below :math:`\theta^\text{adv}_\text{pin}` due to the outward motion of the contact line. For stick-slip behavior, therefore :math:`\theta_\text{eq}<\theta^\text{adv}_\text{pin}<\theta^\text{adv}_\text{unpin}` must hold. To set these contact angles, you can use the following methods:

.. container:: center

   +--------------------------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
   | **Desired event**                          | **relevant quantity**                  | **method to use**                                                                                |
   +============================================+========================================+==================================================================================================+
   | unpin if below and contact angle decreases | :math:`\theta^\text{rec}_\text{unpin}` | :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.set_receding_unpin_below_angle`  |
   +--------------------------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
   | pin if above and contact angle increases   | :math:`\theta^\text{rec}_\text{pin}`   | :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.set_receding_pin_above_angle`    |
   +--------------------------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
   | pin if above and contact angle decreases   | :math:`\theta^\text{adv}_\text{pin}`   | :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.set_advancing_pin_below_angle`   |
   +--------------------------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
   | unpin if above and contact angle increases | :math:`\theta^\text{adv}_\text{unpin}` | :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.set_advancing_unpin_above_angle` |
   +--------------------------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------+

The first argument ``angle`` passed to these methods is the contact angle threshold, where this transition should happen. Note that the angles must be in increasing order following the table from top to bottom and the equilibrium contact angle must be in between the receding angles and the advancing angles. If one passes ``by_factor=True``, the first argument ``angle`` is not interpreted as an angle, but as a factor and the resulting threshold angle is this factor times the equilibrium angle. This means that e.g. ``set_receding_unpin_below_angle(0.9,by_factor=True)`` sets :math:`\theta^\text{rec}_\text{unpin}=0.9\theta_\text{eq}`. Pass ``angle=None`` to remove a previously set angle. Further arguments are ``only_if_decaying``/``only_if_growing`` (default ``True``) to indeed check whether the actual contact angle grows or shrinks when crossing the threshold. ``explicit`` (defaults to ``True``) tells pyoomph to evaluate the actual contact angle :math:`\theta` from the previous time step. If set to ``False``, the contact line dynamics are fully implicitly considered. However, since the transitions from pinned and unpinned are discontinuous, it is likely not converging well in the Newton solver. One can improve it a bit by smearing out the transition angles with the ``heaviside_smoothing`` argument. Thereby, the transitions, which are implemented by heaviside step functions :math:`\Theta`, e.g. :math:`\Theta(\theta^\text{rec}_\text{unpin}-\theta)`, will be smoothed out by :math:`\operatorname{atan}((\theta^\text{rec}_\text{unpin}-\theta)/S)/\pi+1/2` with :math:`S` given by ``heaviside_smoothing``. If ``explicit=True``, ``heaviside_smoothing`` does not improve the convergence.

Once such contact angle thresholds are set, the methods :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` and :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.unpin` will not fix or freely move the contact line, unless it is allowed by the contact angle thresholds (i.e. in the hysteretic regions, e.g. if :math:`\theta` is between :math:`\theta^\text{rec}_\text{unpin}` and :math:`\theta^\text{rec}_\text{pin}`). If we are outside these ranges, :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.unpin` and :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` will have no effect. If we want to override this, i.e. force the contact line to be either pinned or unpinned, irrespectively of the set threshold angles, we must pass ``forced=True`` to :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` or :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.unpin`. The contact line will then remain in this state until one calls :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.pin` or :py:meth:`~pyoomph.equations.contact_angle.StickSlipContactLine.unpin` again. If this time the ``forced`` argument is ``False`` or omitted, the contact line dynamics based on the set angles will take over again.

YoungDupreContactLine
~~~~~~~~~~~~~~~~~~~~~

So far, we had to set the equilibrium contact angle :math:`\theta_\text{eq}` by hand or it will default to the initial contact angle. However, when the surface tension of the liquid-gas interface changes, e.g. due to preferential evaporation of a multi-component droplet, the equilibrium contact angle will change as well. According to the Young-Dupré equation, the equilibrium contact angle is given by

.. math:: :label: eqmcflowclyoungdupre

   \cos\left(\theta_\text{eq}\right)=\frac{\sigma_\text{sg}-\sigma_\text{sl}}{\sigma}

where :math:`\sigma`, :math:`\sigma_\text{sg}` and :math:`\sigma_\text{sl}` are the surface tensions of the liquid-gas, solid-gas and liquid-solid interfaces at the contact line. The :py:class:`~pyoomph.equations.contact_angle.YoungDupreContactLine` inherits from the :py:class:`~pyoomph.equations.contact_angle.StickSlipContactLine`, i.e. all the stick-slip dynamics works as well. In the constructor, one can pass (besides the ``cl_speed_scale``) the surface tensions ``sigma_sg`` and ``sigma_sl``. If one does not pass these, their difference in the numerator of :math:numref:`eqmcflowclyoungdupre` will be determined by the initial contact angle and initial surface tension, so that the contact angle is initially at equilibrium.

KwokNeumannContactLine
~~~~~~~~~~~~~~~~~~~~~~

The Kwok-Neumann contact line model was developed semi-empirically by measuring the contact angle of versatile liquid-substrate combinations :cite:`Kwok2000`. The equilibrium contact angle reads

.. math:: \cos\left(\theta_\text{eq}\right)=-1+2\sqrt{\frac{\sigma_\text{sg}^0}{\sigma}}\exp\left(-\beta\left(\sigma_\text{sg}^0-\sigma\right)^2 \right)

where the coefficient :math:`\beta` (set via ``beta`` in the constructor) defaults to the value of Kwok and Neumann, :math:`\beta=124.7 \:\mathrm{m^4}/\mathrm{J^2}`. The solid-gas surface tension :math:`\sigma_\text{sg}^0` can be set by ``sigma_sg_0`` via the constructor. If not set, it will be calculated so that the initial contact angle is in equilibrium given the initial surface tension. Since the :py:class:`~pyoomph.equations.contact_angle.KwokNeumannContactLine` inherits from the :py:class:`~pyoomph.equations.contact_angle.StickSlipContactLine`, also the ``cl_speed_scale`` can be set and the stick-slip methods are available.
