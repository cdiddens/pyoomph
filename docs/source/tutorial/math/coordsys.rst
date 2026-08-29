.. _seccoordinatesystems:

Coordinate systems
------------------

All differential operators of :numref:`secdiffops` and the integration measure :math:`\mathrm{d}x` of the weak form are evaluated in the **coordinate system** of the equations (default  Cartesian). Globally, it is changed with :py:meth:`~pyoomph.generic.problem.Problem.set_coordinate_system`, which setting all equations on all domains inherit, directly. Changing it on a particular domain and its boundaries can be done by adding a :py:class:`~pyoomph.equations.additional.SetCoordinateSystem` to the equation tree. Individual operators support ``coordsys``, by what it can be changed only for a particular operator, e.g. for :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div`.

This section collects what each of the shipped coordinate systems actually means mathematically. Throughout, :math:`x_1,x_2,\ldots` are the mesh coordinates, i.e. what ``var("coordinate_x")``, ``var("coordinate_y")`` and ``var("coordinate_z")`` return, and :math:`d` is the nodal dimension of the mesh. Two conventions hold in every system:

*   Components are the **physical** ones, i.e. taken with respect to a local *orthonormal* frame - :math:`u_r,u_z,u_\varphi` and not the co- or contravariant components of a general metric. 
*   The index order is :math:`(\nabla \vec{u})_{ij}=\partial u_i/\partial x_j` and :math:`(\nabla\cdot \mathbf{T})_i=\partial_j T_{ij}`, as stated in :numref:`secdiffops`.

Note that a coordinate system may give the operators and the fields **more components than the mesh has dimensions**: an axisymmetric mesh is two-dimensional, but :math:`\nabla f` has three entries, of which the azimuthal one is zero. The unused slots are hard zeros, which contributions hence disappear entirely in the generated code.

Cartesian
^^^^^^^^^

Selected by ``set_coordinate_system("cartesian")`` or an instance of :py:class:`~pyoomph.expressions.coordsys.CartesianCoordinateSystem`. This is the default and works for :math:`d=1,2,3`. It is the trivial case, i.e. with :math:`\vec{x}=(x_1,\ldots,x_d)`

.. math::

   (\nabla f)_i=\partial_i f,\qquad
   (\nabla \vec{u})_{ij}=\partial_j u_i,\qquad
   \nabla\cdot\vec{u}=\sum_{i=1}^{d}\partial_i u_i,\qquad
   (\nabla\cdot\mathbf{T})_i=\sum_{j=1}^{d}\partial_j T_{ij}

and the directional derivative :math:`\left((\vec{d}\cdot\nabla)\mathbf{T}\right)_{ij}=\sum_k d_k\partial_k T_{ij}`. The integration measure is just :math:`\mathrm{d}x=\mathrm{d}^{d}x`, and the geometric Jacobian, i.e. the factor by which the coordinate system inflates the element size, is :math:`1`.

The sums run over the coordinates the *mesh* actually has, not over the three padded slots of a vector: on a two-dimensional mesh ``div(vector(a,b,c))`` does not attempt :math:`\partial_z c`. Note that :math:`\mathbf{T}` is nevertheless always a :math:`3\times 3` object, so the out-of-plane row of :math:`\nabla\cdot\mathbf{T}` is returned, just without any :math:`z`-derivative in it.

The constructor takes ``x_rel_scale``, ``y_rel_scale`` and ``z_rel_scale``, which multiply the corresponding coordinate before differentiating, all defaulting ot :math:`1`.

.. _seccoordsysaxisymm:

Axisymmetric
^^^^^^^^^^^^

Selected by ``set_coordinate_system("axisymmetric")`` or :py:class:`~pyoomph.expressions.coordsys.AxisymmetricCoordinateSystem`. It solves a three-dimensional problem on a mesh of reduced dimension by assuming :math:`\partial_\varphi\equiv 0` and no swirl, i.e. :math:`u_\varphi=0`. The mesh must not have negative :math:`x`-coordinates.

**On a two-dimensional mesh**, :math:`r=x_1` is the radius and :math:`z=x_2` the axis of symmetry, and the component order is :math:`(r,z,\varphi)`. With :math:`\mathrm{d}x=2\pi r\,\mathrm{d}r\,\mathrm{d}z`, i.e. a geometric Jacobian of :math:`2\pi r`:

.. math::

   \nabla f=\begin{pmatrix}\partial_r f\\ \partial_z f\\ 0\end{pmatrix},\qquad
   \nabla \vec{u}=\begin{pmatrix}
   \partial_r u_r & \partial_z u_r & 0\\
   \partial_r u_z & \partial_z u_z & 0\\
   0 & 0 & u_r/r\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\partial_r u_r+\partial_z u_z+\frac{u_r}{r}

.. math::

   \nabla\cdot\mathbf{T}=\begin{pmatrix}
   \partial_r T_{rr}+\partial_z T_{rz}+\dfrac{T_{rr}-T_{\varphi\varphi}}{r}\\[2ex]
   \partial_r T_{zr}+\partial_z T_{zz}+\dfrac{T_{zr}}{r}\\[2ex]
   \partial_r T_{\varphi r}+\partial_z T_{\varphi z}+\dfrac{T_{r\varphi}+T_{\varphi r}}{r}
   \end{pmatrix}

The azimuthal row is unreachable from any field this coordinate system can *define*: :py:meth:`~pyoomph.expressions.coordsys.AxisymmetricCoordinateSystem.define_vector_field` has no :math:`u_\varphi` and :py:meth:`~pyoomph.expressions.coordsys.AxisymmetricCoordinateSystem.define_tensor_field` puts the azimuthal component on the diagonal only (the ``_aa`` entry). It is reachable from a hand-assembled tensor, e.g. ``dyadic(vector(h,0,0),vector(0,0,k))``, and from the azimuthally symmetry-broken system below.

The directional derivative of a tensor picks up the rotation of the frame itself, since :math:`\partial_\varphi\vec{e}_r=\vec{e}_\varphi` and :math:`\partial_\varphi\vec{e}_\varphi=-\vec{e}_r`:

.. math::

   \left((\vec{d}\cdot\nabla)\mathbf{T}\right)_{ij}=d_r\partial_r T_{ij}+d_z\partial_z T_{ij}+\frac{d_\varphi}{r}C_{ij},
   \qquad \mathbf{C}=\mathbf{R}\mathbf{T}+\mathbf{T}\mathbf{R}^\mathrm{T}

with the generator :math:`R_{\varphi r}=1`, :math:`R_{r\varphi}=-1` and all other entries zero. Contracting the second index of this expression folds the two rotations of :math:`\mathbf{C}` together, which is why the connection terms of :math:`\nabla\cdot\mathbf{T}` above look different.

**On a one-dimensional mesh**, the same system means *polar* coordinates in a plane, not cylindrical ones: there is no axial direction at all, the component order is :math:`(r,\varphi)` and the measure is :math:`\mathrm{d}x=2\pi r\,\mathrm{d}r`, i.e. the area of an annulus. Hence

.. math::

   \nabla f=\begin{pmatrix}\partial_r f\\ 0\\ 0\end{pmatrix},\qquad
   \nabla \vec{u}=\begin{pmatrix}\partial_r u_r & 0 & 0\\ 0 & u_r/r & 0\\ 0&0&0\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\partial_r u_r+\frac{u_r}{r}=\frac{1}{r}\partial_r(r u_r)

and, with the azimuthal slot at index :math:`1`,

.. math::

   \nabla\cdot\mathbf{T}=\begin{pmatrix}
   \partial_r T_{rr}+\dfrac{T_{rr}-T_{\varphi\varphi}}{r}\\[2ex]
   \partial_r T_{\varphi r}+\dfrac{T_{r\varphi}+T_{\varphi r}}{r}\\[1ex] 0\end{pmatrix}

Alternatively, on 2d meshes, you can flip the axis with ``set_coordinate_system("axisymmetric_flipped")`` or likewise ``AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True)``, is the same system with the roles of the two mesh coordinates exchanged: :math:`z=x_1` is the axis and :math:`r=x_2` the radius, so that the component order is :math:`(z,r,\varphi)` and the measure is :math:`\mathrm{d}x=2\pi x_2\,\mathrm{d}x_1\,\mathrm{d}x_2`. It cannot be combined with the azimuthal mode expansion below.

Radially symmetric
^^^^^^^^^^^^^^^^^^

``set_coordinate_system("radialsymmetric")`` or :py:class:`~pyoomph.expressions.coordsys.RadialSymmetricCoordinateSystem` assumes the full spherical symmetry, i.e. it solves a three-dimensional problem on a one-dimensional mesh with :math:`r=x_1` and :math:`\partial_\vartheta=\partial_\varphi\equiv 0`. The constructor takes an offset ``Rcenter``, so that the radius is really :math:`r=x_1-R_\text{center}`. The measure is :math:`\mathrm{d}x=4\pi r^2\,\mathrm{d}r` and the component order is :math:`(r,\vartheta,\varphi)`:

.. math::

   \nabla f=\begin{pmatrix}\partial_r f\\ 0\\ 0\end{pmatrix},\qquad
   \nabla \vec{u}=\begin{pmatrix}\partial_r u_r&0&0\\ 0& u_r/r&0\\ 0&0&u_r/r\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\partial_r u_r+\frac{2u_r}{r}=\frac{1}{r^2}\partial_r\left(r^2u_r\right)

Tensor fields, the tensor divergence and the directional tensor derivative are not implemented here and raise an error.

.. _seccoordsysazimuthal:

Azimuthal symmetry breaking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~pyoomph.expressions.coordsys.AxisymmetryBreakingCoordinateSystem` is the axisymmetric system with the azimuthal direction *restored* as a single Fourier mode. It is what an azimuthal stability analysis (:numref:`azimuthalstabana`) runs in. Every field and every test function is expanded as

.. math::

   u(r,z,\varphi,t)=u_0(r,z,t)+\varepsilon\,\hat{u}(r,z,t)\,\mathrm{e}^{\mathrm{i}m\varphi},\qquad
   v(r,z,\varphi)=\hat{v}(r,z)\,\mathrm{e}^{-\mathrm{i}m\varphi}

so that the residual at order :math:`\varepsilon^0` gives the axisymmetric base state and the one at order :math:`\varepsilon^1` the eigenproblem for the mode :math:`m`. Vector fields now genuinely have an azimuthal component, named ``<name>_phi``, and the operators are the full cylindrical ones with :math:`\partial_\varphi\to\mathrm{i}m` acting on the perturbation:

.. math::

   \nabla f=\begin{pmatrix}\partial_r f\\ \partial_z f\\ \frac{1}{r}\partial_\varphi f\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\partial_r u_r+\frac{u_r}{r}+\partial_z u_z+\frac{1}{r}\partial_\varphi u_\varphi

.. math::

   \nabla \vec{u}=\begin{pmatrix}
   \partial_r u_r & \partial_z u_r & \frac{1}{r}\partial_\varphi u_r-\frac{u_\varphi}{r}\\[1ex]
   \partial_r u_z & \partial_z u_z & \frac{1}{r}\partial_\varphi u_z\\[1ex]
   \partial_r u_\varphi & \partial_z u_\varphi & \frac{1}{r}\partial_\varphi u_\varphi+\frac{u_r}{r}
   \end{pmatrix}

The tensor divergence has the same connection terms as in :numref:`seccoordsysaxisymm`, but now with the azimuthal derivatives kept:

.. math::

   \nabla\cdot\mathbf{T}=\begin{pmatrix}
   \partial_r T_{rr}+\partial_z T_{rz}+\frac{1}{r}\partial_\varphi T_{r\varphi}+\frac{T_{rr}-T_{\varphi\varphi}}{r}\\[1ex]
   \partial_r T_{zr}+\partial_z T_{zz}+\frac{1}{r}\partial_\varphi T_{z\varphi}+\frac{T_{zr}}{r}\\[1ex]
   \partial_r T_{\varphi r}+\partial_z T_{\varphi z}+\frac{1}{r}\partial_\varphi T_{\varphi\varphi}+\frac{T_{r\varphi}+T_{\varphi r}}{r}
   \end{pmatrix}

On a moving mesh, the mesh coordinates are themselves perturbed, :math:`r=x_1+\varepsilon\hat{x}_1\mathrm{e}^{\mathrm{i}m\varphi}` and :math:`z=x_2+\varepsilon\hat{x}_2\mathrm{e}^{\mathrm{i}m\varphi}`, while :math:`\varphi` is not. The orthonormal frame therefore does not move at all. The entire effect is a first-order change of what the operators mean once written in mesh coordinates. For any quantity :math:`q`,

.. math::

   \partial_r\big|_{z,\varphi}q=\partial_{1}q-\left(\partial_1\hat{x}_1\,\partial_1 q+\partial_1\hat{x}_2\,\partial_2 q\right),\qquad
   \partial_z\big|_{r,\varphi}q=\partial_{2}q-\left(\partial_2\hat{x}_1\,\partial_1 q+\partial_2\hat{x}_2\,\partial_2 q\right)

.. math::

   \partial_\varphi\big|_{r,z}q=\partial_\varphi q-\mathrm{i}m\left(\hat{x}_1\partial_1 q+\hat{x}_2\partial_2 q\right),\qquad
   \frac{1}{r}=\frac{1}{x_1}-\frac{\hat{x}_1}{x_1^{2}}

all to be understood to first order in :math:`\varepsilon` and only where the mesh is actually a set of unknowns. The measure picks up the corresponding :math:`\mathrm{d}x=2\pi\left(x_1+\varepsilon(\ldots)\right)\mathrm{d}x_1\mathrm{d}x_2` term as well. Substituting these four rules into the fixed-mesh expressions above is precisely how the moving-mesh forms are generated.

On a one-dimensional mesh the same relabelling as in :numref:`seccoordsysaxisymm` applies: the component order is :math:`(r,\varphi)`, the azimuthal slot is index :math:`1`, and there is no axial direction.

Cartesian with an additional normal mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~pyoomph.expressions.coordsys.CartesianCoordinateSystemWithAdditionalNormalMode` does for a Cartesian mesh what the previous system does for an axisymmetric one: it adds **one extra Cartesian direction** :math:`s`, translationally invariant in the base state, and expands

.. math::

   u=u_0(\vec{x},t)+\varepsilon\,\hat{u}(\vec{x},t)\,\mathrm{e}^{\mathrm{i}ks}

This is what the Cartesian normal mode stability analysis (:numref:`cartesiannormalstabana`) uses, e.g. to perturb a two-dimensional base flow by a spanwise wave. Vector fields gain a component named ``<name>_normal``, so that a :math:`d`-dimensional mesh carries :math:`d+1` components, and the operators are the plain Cartesian ones extended by that direction, with :math:`\partial_s\to\mathrm{i}k` on the perturbation:

.. math::

   \nabla f=\begin{pmatrix}\partial_1 f\\ \vdots\\ \partial_d f\\ \partial_s f\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\sum_{i=1}^{d}\partial_i u_i+\partial_s u_s,\qquad
   (\nabla \vec{u})_{ij}=\partial_j u_i,\quad i,j\in\{1,\ldots,d,s\}

There are no connection terms, since the frame is Cartesian and does not turn; the measure is :math:`\mathrm{d}x=\mathrm{d}^dx`, i.e. per unit length in the extra direction. On a moving mesh, the analogous first-order corrections to the derivative operators apply, with :math:`\hat{\vec{x}}` the perturbation of the mesh position, but without any :math:`1/r`-type term. Lagrangian derivatives never see the extra direction, i.e. :math:`\partial_s\to 0` there, because the Lagrangian coordinates are not expanded.

Rectangular-to-polar mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~pyoomph.utils.rectangular_polar_mapping.RectangularToPolarMappingCoordinateSystem` (available as ``rectangular_to_polar``), requires to import ``pyoomph.utils.rectangular_polar_mapping``, lets a *rectangular* two-dimensional mesh represent an annular or disc-shaped domain, by reading :math:`r=x_1` and :math:`\varphi=x_2` as full polar coordinates - as opposed to the axisymmetric system, which drops the azimuthal direction. It is the natural choice when the geometry is periodic in :math:`\varphi` and one wants a structured mesh in :math:`(r,\varphi)`. With :math:`\mathrm{d}x=r\,\mathrm{d}r\,\mathrm{d}\varphi`, i.e. a geometric Jacobian of :math:`r`:

.. math::

   \nabla f=\begin{pmatrix}\partial_r f\\ \frac{1}{r}\partial_\varphi f\end{pmatrix},\qquad
   \nabla \vec{u}=\begin{pmatrix}
   \partial_r u_r & \frac{1}{r}\partial_\varphi u_r-\frac{u_\varphi}{r} & 0\\[1ex]
   \partial_r u_\varphi & \frac{1}{r}\partial_\varphi u_\varphi+\frac{u_r}{r} & 0\\[1ex]
   0&0&0\end{pmatrix},\qquad
   \nabla\cdot\vec{u}=\frac{1}{r}\partial_r\left(r u_r\right)+\frac{1}{r}\partial_\varphi u_\varphi

Tensor fields are not supported here. For output, the companion operator :py:class:`~pyoomph.utils.rectangular_polar_mapping.MeshDataPolarToCartesian` maps the mesh and the vector fields back to the :math:`(x,y)` plane. For plotting, you can use the transformation :py:class`~pyoomph.utils.rectangular_polar_mapping.PlotTransformPolarToCartesian` for the same purpose.

ODEs
^^^^

Domains without any spatial extent, i.e. equations inherited from :py:class:`~pyoomph.generic.codegen.ODEEquations` added to an ODE domain, use :py:class:`~pyoomph.expressions.coordsys.ODECoordinateSystem`. There are no coordinates at all, so :math:`\nabla f=0` by definition and ``var("coordinate")`` expands to zero. There is a single integration point of unit weight, so that ``weak(residual,testfunction)`` on an ODE domain is the residual itself.

Operators on interfaces
^^^^^^^^^^^^^^^^^^^^^^^

On a domain with a co-dimension, :py:func:`~pyoomph.expressions.generic.grad` and :py:func:`~pyoomph.expressions.div` are the *surface* operators, see :numref:`secdiffops`. By default they are given by the same expressions as above, with the derivatives being the surface-projected ones, i.e. :math:`\nabla_S=(\mathbf{1}-\vec{n}\otimes\vec{n})\cdot\nabla` in the Cartesian case. The surface divergence of a velocity in axisymmetry, e.g., keeps its :math:`u_r/r`, and hence

.. math::

   \nabla_S\cdot\vec{u}=\partial_r^S u_r+\partial_z^S u_z+\frac{u_r}{r}

is the surface dilation rate that appears in the surfactant transport equation and in :math:`\nabla_S\cdot\vec{n}=\kappa`, the curvature of an axisymmetric interface, which contains the second principal curvature :math:`n_r/r` this way.
