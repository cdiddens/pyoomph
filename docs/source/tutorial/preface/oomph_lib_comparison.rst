.. _secprefaceoomphcomparison:

Comparison with oomph-lib
-------------------------

pyoomph is built *on top of* `oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/html/>`_ :cite:`Heil2006`: the C++ core of pyoomph links against a (slightly modified) subset of oomph-lib, which provides the underlying finite-element data structures, the Newton solver, the spatial and temporal adaptivity machinery and the arc-length continuation and bifurcation-tracking framework. pyoomph is therefore **not** a competitor of oomph-lib, but a high-level, symbolic Python frontend that trades some of oomph-lib's breadth and low-level control for a drastically reduced implementation effort.

This page gives a systematic, feature-by-feature comparison. It complements the more narrative :ref:`motivation<tutorial/preface/motivation:Motivation>` and :ref:`when to use pyoomph<tutorial/preface/whentouse:When to use pyoomph, and when better use something else>` sections. The comparison was made against the full upstream oomph-lib library, i.e. the *native* library, not the stripped-down and modified partial copy bundled inside pyoomph's ``src/thirdparty/oomph-lib``.

.. note::

   Because pyoomph is fundamentally a *symbolic* framework, many features that oomph-lib ships as hand-written, pre-compiled C++ element classes can instead be assembled by the user in a handful of lines of Python. Throughout this page, a feature "missing" from pyoomph therefore usually means *not pre-packaged* rather than *impossible* — the weak form can often just be typed out directly. Conversely, a few oomph-lib features (e.g. :math:`C^1`-continuous Hermite elements, hp-refinement) rely on machinery that pyoomph currently does not expose at all.


Two development philosophies
============================

The essential difference is *how a new equation gets into the solver*.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Aspect
     - oomph-lib
     - pyoomph
   * - Language / interface
     - C++ library; problems written and compiled as C++ driver codes
     - Python frontend; problems written as Python scripts
   * - Defining a new equation
     - Derive a new ``Element`` class and hand-code the residual assembly loop in C++
     - Add the residual of the weak form symbolically (``add_residual(weak(...))``) in Python
   * - Jacobian matrix
     - Typically hand-coded for best performance/convergence (often hundreds of lines per equation), or a slower finite-difference fallback
     - Derived symbolically and exactly by GiNaC, including derivatives with respect to moving-mesh coordinates
   * - Parameter derivatives / Hessian
     - Hand-coded where needed (used by continuation / bifurcation tracking)
     - Generated symbolically and exactly
   * - Performance path
     - Compiled C++
     - Symbolically generated C code, JIT-compiled (TinyC by default, or gcc/clang/MSVC) and linked back into the running Python process — on par with hand-coded C/C++
   * - Combining multi-physics
     - Make a combined ``Element`` class, adjust nodal storage, implement off-diagonal Jacobian terms
     - Just add ``equationA + equationB``, cross-coupling automatically
   * - Non-dimensionalisation
     - Done by hand before implementation
     - Optional: equations can be written with physical units and are non-dimensionalised automatically

With oomph-lib, you will definitely learn more on the low level of the finite element method, whereas pyoomph usually requires less lines of code.

Physics / equation modules
==========================

oomph-lib ships an extensive catalogue of pre-coded physics modules (one ``src/`` subdirectory each). pyoomph provides a smaller but rapidly growing set of built-in equation classes, and covers a number of oomph-lib's *coordinate-system variants* with a single equation combined with a swappable coordinate system (see :ref:`Coordinate systems <secprefaceoomphcomparisoncoordsys>` below).

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Physics
     - oomph-lib
     - pyoomph
   * - Incompressible Navier–Stokes
     - Yes (Taylor–Hood & Crouzeix–Raviart)
     - Yes (``NavierStokesEquations``; TH / CR / Scott–Vogelius / MINI)
   * - Stokes flow
     - Yes
     - Yes (``StokesEquations``)
   * - Axisymmetric / polar / spherical NS
     - Separate modules (``axisym_``, ``polar_``, ``spherical_navier_stokes``)
     - Same equation + coordinate system
   * - Generalised-Newtonian (shear-thinning etc.)
     - Dedicated modules
     - Set a symbolic, shear-rate-dependent ``dynamic_viscosity`` — no new module needed
   * - Viscoelastic flow
     - Not a dedicated module
     - Assemble symbolically; log-conformation tensor tools provided
   * - Poisson / advection–diffusion / (unsteady) heat
     - Yes (many variants incl. reaction-diffusion)
     - Yes (``PoissonEquation``, ``AdvectionDiffusionEquations``, diffusion)
   * - Helmholtz / acoustics
     - Yes, incl. PML & Fourier-decomposed variants
     - Yes (``HelmholtzEquation``, complex); PML as tutorial
   * - Solid mechanics (large-displacement, hyperelastic)
     - Yes (``solid`` + ``constitutive``)
     - Yes (``DeformableSolidEquations``; Hookean / incompressible constitutive laws)
   * - Linear / time-harmonic elasticity
     - Yes (incl. axisym, Fourier-decomposed, PML)
     - Linear elasticity yes; frequency-domain not packaged
   * - Darcy / porous media
     - Yes (``darcy``, ``poroelasticity``)
     - Darcy yes + porous↔NS coupling; full Biot poroelasticity not packaged
   * - Phase field (Cahn–Hilliard / NSCH)
     - Via ``multi_physics``
     - Yes (``CahnHilliardEquation``, ``NSCH``, low-order NSCH)
   * - Lubrication / thin-film
     - —
     - Yes (``LubricationEquations``)
   * - Potential flow / stream function
     - ``young_laplace`` (capillary)
     - Yes (``PotentialFlow``, ``stokes_stream_func``)
   * - Beams, shells, Föppl–von Kármán plates, biharmonic
     - Yes (``beam``, ``shell``, ``foeppl_von_karman``, ``biharmonic``)
     - Not available (need :math:`C^1`-continuous elements, see below)
   * - Womersley / impedance outflow, linear wave, flux-transport (Euler)
     - Yes
     - Not packaged
   * - Space-time formulations
     - Yes (``space_time``)
     - Not packaged
   * - Multi-component / multi-phase flow with mass transfer
     - Partly, via ``multi_physics``
     - Extensive dedicated support (see below) — a headline pyoomph strength


.. _secprefaceoomphcomparisoncoordsys:

Coordinate systems
==================

Where oomph-lib provides separate Cartesian, axisymmetric, polar and spherical *element* families, pyoomph writes each equation once and evaluates ``grad``/``div`` through a pluggable coordinate system: Cartesian, axisymmetric, radially symmetric, an axisymmetric system with an additional (non-axisymmetric) normal mode, and an azimuthal-symmetry-breaking system (:math:`m`-mode). The same weak form can thus be reused across coordinate systems and combined with other equations.


Elements, spaces and spatial discretisation
============================================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - oomph-lib
     - pyoomph
   * - Dimensions
     - 1D / 2D / 3D
     - 1D / 2D / 3D (+ point elements with normal mode)
   * - Element geometries
     - Lines, triangles, quads, tets, bricks
     - Lines, triangles, quads, tets, bricks, plus wedges/prisms & pyramids
   * - Continuous Lagrange spaces
     - Linear → cubic (and beyond)
     - Linear → quadratic 
   * - Bubble-enriched spaces
     - Crouzeix–Raviart velocity
     - ``C1TB`` / ``C2TB`` (triangle/tet bubble)
   * - Discontinuous Galerkin
     - Flux-transport / discontinuous-pressure elements
     - All continuous spaces are also in DG with ``jump``/``avg`` operators and weak Dirichlet BCs
   * - :math:`C^1`-continuous (Hermite) elements
     - Yes (for beams, shells, biharmonic)
     - Not available
   * - Spectral elements
     - Yes
     - Not available
   * - p- / hp-refinement
     - Yes (``hp_refineable_elements``)
     - Not available (h-refinement only), but manual p-restriction
   * - h-refinement (spatial adaptivity)
     - Yes, Z2 error-estimator driven; quad/oct/binary trees
     - Yes, Z2 error-estimator driven; also on mixed meshes


Meshing
=======

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - oomph-lib
     - pyoomph
   * - Structured mesh templates
     - ~50 built-in templates
     - Line/rectangle/circle/brick/cylinder/octant + droplet meshes
   * - Unstructured generation
     - Triangle, TetGen, Gmsh, Geompack, VMTK, xfig/xda
     - Gmsh integration (via pygmsh/gmsh)
   * - Moving meshes / ALE
     - Spine method + pseudo-solid node update
     - Symbolic monolithic pseudo-elastic / Laplace- / hyperelastic-smoothed ALE
   * - Remeshing
     - Adaptive unstructured remeshing
     - Gmsh-based remeshers, remeshing during continuation
   * - Multiple / multi-domain meshes
     - Yes, including mortar methods
     - Yes, with fields and bulk gradients thereof available on both sides


Solvers, eigensolvers and time integration
===========================================

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Category
     - oomph-lib
     - pyoomph
   * - Direct linear solvers
     - SuperLU (serial/dist), MUMPS, HSL frontal (MA42)
     - Pardiso (MKL), PETSc, MUMPS, SuperLU/UMFPACK (scipy), Apple Accelerate
   * - Iterative solvers
     - CG, GMRES, BiCGStab, GS (+ Trilinos AztecOO)
     - Via PETSc (incl. field-split / block)
   * - Preconditioners
     - Rich framework: block, Navier–Stokes LSC, Lagrange-enforced-flow, Hypre BoomerAMG, Trilinos ML/IFPACK, geometric multigrid
     - PETSc field-split / block; relies on the comprehensive PETSc's preconditioner ecosystem
   * - Nonlinear solver
     - Newton (+ black-box, FD by default)
     - Newton (with exact symbolic Jacobian)
   * - Eigensolvers
     - LAPACK QZ, ARPACK, Trilinos Anasazi
     - SLEPc (+MUMPS), scipy/ARPACK, Pardiso-ARPACK, Accelerate-ARPACK
   * - Time steppers
     - BDF1/2, Newmark, TR/Crank–Nicolson, IMR, explicit (Euler, Runge–Kutta), adaptive time-stepping
     - BDF1/2, Newmark2, TPZ, midpoint, Simpson/Boole/Milne, adaptive time-stepping

pyoomph exposes a broader set of *external direct and eigen-solver bridges* (notably PETSc/SLEPc and Intel MKL Pardiso), which are not wired into this oomph-lib checkout. oomph-lib, in turn, has a considerably more developed specific *preconditioner and multigrid* ecosystem for large-scale iterative solves.


Continuation, bifurcation and stability analysis
=================================================

Both libraries inherit the same conceptual toolbox (pyoomph builds directly on oomph-lib's augmented-system approach), but pyoomph extends it substantially.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Feature
     - oomph-lib
     - pyoomph
   * - Pseudo-arclength continuation
     - Yes
     - Yes
   * - Fold / pitchfork / Hopf tracking
     - Yes
     - Yes
   * - Periodic-orbit tracking / Floquet
     - Yes (periodic-orbit handler)
     - Yes (collocation / Floquet / BDF2 / B-spline representations)
   * - Bifurcation tracking on **moving meshes**
     - Requires manually coded complicated Hessian for good convergence
     - Yes (exact symbolic Hessian incl. mesh-coordinate derivatives)
   * - **Azimuthal symmetry-breaking** stability / tracking
     - Not available
     - Yes — the distinctive feature of :cite:`Diddens2024`
   * - Cartesian normal-mode stability
     - Not available
     - Yes
   * - Adjoint / parameter sensitivity, Hessian-vector products
     - Yes (used internally)
     - Yes (exposed symbolically)
   * - Lyapunov exponents / normal-form / periodic-driving response
     - —
     - Yes (utility modules)


Multi-physics, multi-domain and free surfaces
=============================================

Both frameworks are designed for monolithically coupled multi-physics on multiple domains. oomph-lib provides a very mature fluid–structure-interaction (FSI) infrastructure with an extensive demo suite (driven-cavity FSI, collapsible channel, Turek flag, VMTK physiological flows, acoustic FSI, …). pyoomph provides FSI through ``FSIConnection`` and focuses its multi-physics strength on **multi-component, multi-phase flow with mass transfer**:

* Sharp-interface monolithic ALE free surfaces with mass transfer, the geometric conservation lawsurface tension and Marangoni stresses.
* Soluble/insoluble surfactant transport with several isotherms (Henry, Langmuir, Volmer, Frumkin, van der Waals).
* Contact-angle models (Young–Dupré, Kwok–Neumann, Wenzel, Cassie–Baxter, stick-slip, dynamic).
* A materials database with thermodynamic activity models (original UNIFAC, modified UNIFAC Dortmund, AIOMFAC) and evaporation/mass-transfer models (Hertz–Knudsen–Schrage, LLE).

oomph-lib additionally offers immersed rigid-body dynamics and a hijacking mechanism for fine-grained residual control that pyoomph does not expose directly.


Parallel computing
===================

This is one of oomph-lib's clear strengths: fully distributed meshes with halo/haloed nodes, METIS/ParMETIS partitioning, and parallel direct solvers and preconditioners (SuperLU_DIST, MUMPS/ScaLAPACK, Trilinos, Hypre). pyoomph supports MPI in source builds, but its distributed-solver ecosystem is less mature, and the pre-built wheels ship without MPI. 


What pyoomph adds on top of oomph-lib
=====================================

* Symbolic and coordinate-system agnostic weak-form entry directly in Python, with **exact, automatically generated Jacobians, parameter derivatives and Hessians** — including derivatives with respect to moving-mesh coordinates.
* JIT C code generation and compilation (TinyC / gcc / clang / MSVC), giving hand-coded performance without hand-coding.
* A full physical **units / non-dimensionalisation** system tracked symbolically through code generation.
* A **materials** database with multi-component mixtures, thermodynamic activity models (UNIFAC/AIOMFAC) and mass-transfer models.
* **Azimuthal symmetry-breaking** stability analysis and native bifurcation tracking on moving meshes.
* A rich set of external linear/eigen-solver bridges (PETSc, SLEPc, MKL Pardiso, …).
* Built-in matplotlib / VTK / ParaView output and a preCICE coupling adapter.


Features of oomph-lib not (yet) available in pyoomph
====================================================

* The large catalogue of pre-coded physics (beams, shells, Föppl–von Kármán plates, biharmonic, Womersley, flux-transport/Euler, space-time, time-harmonic elasticity, full poroelasticity).
* :math:`C^1`-continuous Hermite elements, spectral elements, and p-/hp-refinement.
* The mature specific block-preconditioners, algebraic/geometric multigrid ecosystem for large iterative solves.
* Highly scalable distributed-memory (MPI) meshes and parallel solvers.
* Immersed rigid bodies and the residual-hijacking mechanism.
* A wider selection of unstructured mesh generators (TetGen, VMTK, Geompack, …).

Keep in mind that, thanks to pyoomph's symbolic nature, several of these "missing" equations can nevertheless be implemented by the user in a few lines of Python. The genuine limitations are those requiring machinery pyoomph does not expose — most notably :math:`C^1`-continuous elements, hp-refinement, and large-scale parallelism.


Summary
=======

Use pyoomph when you value rapid, symbolic problem setup, black-box symbolic differentiation, physical units, multi-component flow with mass transfer, or (azimuthal) bifurcation tracking on moving meshes — and when your problems fit within continuous Lagrange elements and moderate parallelism. Use native oomph-lib directly when you need its full breadth of pre-coded physics, :math:`C^1`/spectral/hp elements, its advanced preconditioners, or heavily parallelised large-scale (3D) computations. See :ref:`When to use pyoomph<tutorial/preface/whentouse:When to use pyoomph, and when better use something else>` for a shorter decision guide.
