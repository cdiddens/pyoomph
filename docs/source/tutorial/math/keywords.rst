.. _seckeywordvars:

Keyword variables
-----------------

The following keyword variables are defined and should not be used for any other field, i.e. not defined in an :py:class:`~pyoomph.generic.codegen.Equations` class via e.g. :py:meth:`~pyoomph.generic.codegen.Equations.define_scalar_field`. You can use these keyword variables as usually with :py:func:`~pyoomph.expressions.generic.var` or :py:func:`~pyoomph.expressions.generic.nondim` for the dimensional or nondimensionalized quantity, respectively.

Note that the time-derivative of the ``mesh`` variable only gives the mesh velocity if you use it with ``ALE=False``, i.e. ``partial_t(var("mesh"),ALE=False)``. Therefore, better use :py:func:`~pyoomph.expressions.generic.mesh_velocity` instead and e.g. ``mesh_velocity()[0]`` for the velocity in x-direction.



.. table:: Defined keyword variables to be used with either :py:func:`~pyoomph.expressions.generic.var` or :py:func:`~pyoomph.expressions.generic.nondim`. All mentioned time derivatives must be understood with the ``ALE=False`` setting.

   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"time"``                              | Current time                                                                                                                                                        |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"coordinate"``                        | independent coordinate vector. Its time derivative is zero with ``ALE=False``, but :math:`-\dot{\vec{X}}`, i.e. minus the mesh velocity, with the default           |
   |                                         | ``ALE="auto"`` on a moving mesh                                                                                                                                     |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"coordinate_x"``                      | independent x-coordinate. Time derivative is zero with ``ALE=False``, but minus the mesh x-velocity with the default ``ALE="auto"`` on a moving mesh                |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"coordinate_y"``                      | independent y-coordinate. Time derivative is zero with ``ALE=False``, but minus the mesh y-velocity with the default ``ALE="auto"`` on a moving mesh                |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"coordinate_z"``                      | independent z-coordinate. Time derivative is zero with ``ALE=False``, but minus the mesh z-velocity with the default ``ALE="auto"`` on a moving mesh                |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"mesh"``                              | mesh coordinate vector, similar to ``"coordinate"``. Its time derivative gives the mesh velocity with ``ALE=False``, but zero with the default ``ALE="auto"`` on a  |
   |                                         | moving mesh                                                                                                                                                         |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"mesh_x"``                            | mesh x-coordinate, similar to ``"coordinate_x"``. Its time derivative gives the mesh x-velocity with ``ALE=False``, but zero with the default ``ALE="auto"`` on a   |
   |                                         | moving mesh                                                                                                                                                         |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"mesh_y"``                            | mesh y-coordinate, similar to ``"coordinate_y"``. Its time derivative gives the mesh y-velocity with ``ALE=False``, but zero with the default ``ALE="auto"`` on a   |
   |                                         | moving mesh                                                                                                                                                         |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"mesh_z"``                            | mesh z-coordinate, similar to ``"coordinate_z"``. Its time derivative gives the mesh z-velocity with ``ALE=False``, but zero with the default ``ALE="auto"`` on a   |
   |                                         | moving mesh                                                                                                                                                         |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"lagrangian"``                        | Lagrangian coordinate vector. By default, initialized with the initial Eulerian ``"coordinate"``                                                                    |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"lagrangian_x"``                      | Lagrangian x-coordinate                                                                                                                                             |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"lagrangian_y"``                      | Lagrangian y-coordinate                                                                                                                                             |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"lagrangian_z"``                      | Lagrangian z-coordinate                                                                                                                                             |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"normal"``                            | Normal vector. To be used at elements with co-dimension, i.e. interface elements                                                                                    |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"normal_x"``                          | x-component of the normal                                                                                                                                           |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"normal_y"``                          | y-component of the normal                                                                                                                                           |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"normal_z"``                          | z-component of the normal                                                                                                                                           |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"dx"``                                | Can be used like in FEniCS to express ``weak(a,b)`` as ``a*b*var("dx")``. It does not respect the functional determinant of the coordinate system, though.          |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"dX"``                                | Same as ``"dx"``, but for Lagrangian integrals                                                                                                                      |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"element_size_Eulerian"``             | Eulerian integration of the volume/area/length of the current element. Uses the coordinate system of the element, i.e. considers e.g. :math:`2\pi r` in axisymmetry |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"cartesian_element_size_Eulerian"``   | Same as above, but does not consider the coordinate system                                                                                                          |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"element_size_Lagrangian"``           | Same as ``"element_size_Eulerian"``, but by Lagrangian integration                                                                                                  |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"cartesian_element_size_Lagrangian"`` | Same as ``"cartesian_element_size_Eulerian"``, but by Lagrangian integration                                                                                        |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"element_length_h"``                  | Typical length scale of the element, calculated by taking ``"element_size_Eulerian"`` to the power of one over the element dimension                                |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | ``"cartesian_element_length_h"``        | Typical length scale of the element, but calculated in a cartesian coordinate system                                                                                |
   +-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+

