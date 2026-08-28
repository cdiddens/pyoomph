.. pyoomph documentation master file, created by
   sphinx-quickstart on Tue Apr 16 16:36:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. include:: <isonum.txt>



Welcome to pyoomph's documentation!
===================================

.. only:: html
	
   .. image:: _static/pyoomph_logo_full.png
      :width: 300
      :alt: Pyoomph Logo
      :align: center


pyoomph is a python multi-physics finite element framework based on `oomph-lib <http://www.oomph-lib.org>`__ and `GiNaC <http://www.ginac.de>`__.

pyoomph lets you assemble quite arbitrary multi-physics and multi-domain problems directly in python, including custom equations, constraints and moving meshes.
By generating symbolically derived C code, the performance of pyoomph is on par with hand-coded finite element implementations, but with a considerably lower workload to set up complicated problems.

You can either browse the tutorial online or download a `PDF version <https://pyoomph.readthedocs.io/_/downloads/en/latest/pdf/>`__.

The source code of pyoomph is hosted on `Github <https://github.com/pyoomph/pyoomph>`__ and you can find its homepage at `pyoomph.github.io <https://pyoomph.github.io>`__.


Tutorial
========

.. toctree::
   :maxdepth: 1      

   tutorial.rst


.. |tuto_lorenz| thumbnail:: tutorial/temporal/timestepping/plot_lorenz_thumb.png
  :width: 270px
  :align: center  
  :class: framed  
  :title: Adaptive timestepping for the Lorenz attractor
  
    
.. |tuto_odebif| thumbnail:: tutorial/temporal/stability/bifurcs_thumb.png
  :width: 270px
  :align: center
  :class: framed    
  :title: Stability analysis and bifurcation tracking of ODEs

.. |tuto_hopfswitch| thumbnail:: tutorial/temporal/orbit/switch_hopf_thumb.png
  :width: 200px
  :align: center  
  :class: framed  
  :title: Hopf branch switching to periodic orbits

.. |tuto_floquet| thumbnail:: tutorial/temporal/orbit/torus_unstable_thumb.png
  :width: 270px
  :align: center  
  :class: framed  
  :title: Stability of periodic orbits via Floquet multipliers
  
.. |tuto_adaptpoisson| thumbnail:: tutorial/spatial/poisson/poisson2d_adapt_thumb.png
  :width: 140px
  :align: center
  :class: framed
  :title: Spatial adaptivity for a 2d Poisson equation
  
.. |tuto_stokeslaw| thumbnail:: tutorial/spatial/stokes/stokes_law_thumb.png
  :width: 290px
  :align: center 
  :class: framed    
  :title: Flow around a sphere (Stokes law)
  
.. |tuto_stokesdrop| thumbnail:: tutorial/spatial/stokes/stokes_noflux_thumb.png
  :width: 270px
  :align: center 
  :class: framed    
  :title: No normal flow through a curved interface
  
.. |tuto_fishmesh| thumbnail:: tutorial/spatial/mesh/fishgmsh_eye_thumb.png
  :width: 180px
  :align: center  
  :class: framed   
  :title: Generation of custom meshes
  
.. |tuto_waveeq| thumbnail:: tutorial/pde/wave/waveeqdoubleslit_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: Wave equation
  
.. |tuto_rtinstab| thumbnail:: tutorial/pde/navier/rayleigh_taylor_thumb.png
  :width: 200px
  :align: center 
  :class: framed    
  :title: Rayleigh-Taylor instability
  
.. |tuto_lubric| thumbnail:: tutorial/pde/lubric/lubric_coalescence_thumb.png
  :width: 200px
  :align: center 
  :class: framed    
  :title: Coalescence of droplets via lubrication theory
  
.. |tuto_pattern| thumbnail:: tutorial/pde/patterns/kse_ics_thumb.png
  :width: 250px
  :align: center 
  :class: framed    
  :title: Pattern formation and stability analysis

.. |tuto_nsfreesurf| thumbnail:: tutorial/ale/free_surface_ns_thumb.png
  :width: 300px
  :align: center  
  :class: framed      
  :title: Transient simulation of a free surface
  
.. |tuto_drop3d| thumbnail:: tutorial/ale/spread/threedim_spread_single_thumb.png
  :width: 150px
  :align: center  
  :class: framed   
  :title: Three-dimensional droplet with varying wettability
  
.. |tuto_remeshing| thumbnail:: tutorial/ale/remeshing_thumb.png
  :width: 200px
  :align: center  
  :class: framed      
  :title: Moving meshes and mesh reconstruction

  
.. |tuto_dropmara| thumbnail:: tutorial/ale/spread/droplet_mara_grav_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: Droplet with Marangoni flow and gravity
  
.. |tuto_icecyl| thumbnail:: tutorial/multidom/icecylinder_thumb.png
  :width: 270px
  :align: center 
  :class: framed    
  :title: Melting of an ice cylinder
  
.. |tuto_stokessurf| thumbnail:: tutorial/multidom/falling_droplet_thumb.png
  :width: 260px
  :align: center 
  :class: framed    
  :title: Flow around a sphere with insoluble surfactants
  
.. |tuto_evap| thumbnail:: tutorial/multidom/dropevap_thumb.png
  :width: 270px
  :align: center 
  :class: framed    
  :title: Evaporation of a water droplet
  
.. |tuto_heleshaw| thumbnail:: tutorial/mcflow/heleshaw_half_thumb.png
  :width: 210px
  :align: center 
  :class: framed    
  :title: Mixture evaporation from a Hele-Shaw cell
  
.. |tuto_solid| thumbnail:: tutorial/ale/solid/torsion_thumb.png
  :width: 230px
  :align: center 
  :class: framed    
  :title: Nonlinear elasticity

.. |tuto_fsi| thumbnail:: tutorial/multidom/fsi_thumb.png
  :width: 270px
  :align: center 
  :class: framed    
  :title: Nonlinear elasticity
  
.. |tuto_dg| thumbnail:: tutorial/dg/dg_fvm_d2_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: Discontinuous Galerkin methods
  
.. |tuto_linresp| thumbnail:: tutorial/advstab/response/pdrdrum_thumb.png
  :width: 260px
  :align: center 
  :class: framed    
  :title: Linear response to periodic driving
  
.. |tuto_dropdetach| thumbnail:: tutorial/advstab/movmesh/dropstab_thumb.png
  :width: 270px
  :align: center 
  :class: framed    
  :title: Bifurcation tracking of a detaching droplet
  
.. |tuto_azimuthal| thumbnail:: tutorial/advstab/azimuthal/rb_cyl_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: Azimuthal stability analysis

.. |tuto_cartnormal| thumbnail:: tutorial/advstab/cartesiannormal/rivuletplots_thumb.png
  :width: 300px
  :align: center  
  :class: framed   
  :title: Cartesian normal mode stability analysis

.. |tuto_viscoelastic| thumbnail:: tutorial/ale/beadsonstring_viscoelastic_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: Viscoelastic flow

.. |tuto_bifgui| thumbnail:: tutorial/advstab/bifgui_thumb.png
  :width: 300px
  :align: center 
  :class: framed    
  :title: The interactive bifurcation GUI

.. |tuto_precice| thumbnail:: tutorial/precice/heat_circle_thumb.png
  :width: 170px
  :align: center 
  :class: framed    
  :title: Coupling multiple simulations with preCICE


.. list-table:: Some representative tutorial cases
	:widths: 50 50
	:align: center
	:class: exampletable
	
	* - 	|tuto_lorenz|  
		
		:ref:`Adaptive timestepping for the Lorenz attractor<secODEtemporaladapt>`
	  - 	|tuto_odebif|
	  
		:ref:`Stability analysis and bifurcation tracking of ODEs<secODEarclength>`
		
	* - 	|tuto_hopfswitch|  
		
		:ref:`Hopf branch switching to periodic orbits<secODEhopfswitch>`
	  - 	|tuto_floquet|
	  
		:ref:`Stability of periodic orbits via Floquet multipliers<secODEfloquet>`	  			  
		
	* -	|tuto_adaptpoisson|
	
		:ref:`Spatial adaptivity for a 2d Poisson equation<secspatialadapt>`
		
	  - 	|tuto_stokeslaw|
	  
		:ref:`Flow around a sphere (Stokes law)<secspatialstokes_law>`
		
	* -	|tuto_stokesdrop|
	
		:ref:`No normal flow through a curved interface<secspatialzeroflowenforcing>`
		
	  - 	|tuto_fishmesh|
	  
		:ref:`Generation of custom meshes<secspatialfishmesheye>`
		
	* -	|tuto_waveeq|
	
		:ref:`Wave equation<secpdedoubleslit>`
		
	  - 	|tuto_rtinstab|
	  
		:ref:`Rayleigh-Taylor instability<secpdertinstab>`
		
	* -	|tuto_lubric|
	
		:ref:`Coalescence of droplets via lubrication theory<secpdelubric_coalescence>`
		
	  - 	|tuto_pattern|
	  
		:ref:`Pattern formation and stability analysis<secpdepattformeigs>`

		
	* - 	|tuto_nsfreesurf|
	  
		:ref:`Transient simulation of a free surface<secALEfreesurfNS>`

	  -	|tuto_drop3d|
	
		:ref:`Three-dimensional droplet with varying wettability<threedimdroplet>`

	* -	|tuto_remeshing|
	
		:ref:`Moving meshes and mesh reconstruction<secaleremeshing>`		
		
	  - 	|tuto_dropmara|
	  
		:ref:`Droplet with Marangoni flow and gravity<secALEstatdroplet>`
				
	* -	|tuto_icecyl|
	
		:ref:`Melting of an ice cylinder<secmultidomicecyl>`		
		
	  - 	|tuto_stokessurf|
	  
		:ref:`Flow around a sphere with insoluble surfactants<secmultidomstokessurfact>`						

	* -	|tuto_evap|
	
		:ref:`Evaporation of a water droplet<secmultidomdropevap>`		
		
	  - 	|tuto_heleshaw|
	  
		:ref:`Mixture evaporation from a Hele-Shaw cell<secRicardo>`
		
	* -	|tuto_solid|
	
		:ref:`Nonlinear elasticity<solidoscillations>`		
		
	  - 	|tuto_fsi|
	  
		:ref:`Fluid-structure interaction<simplefsi>`		
		
	* -	|tuto_dg|
	
		:ref:`Discontinuous Galerkin methods<secdgweakdbc>`		
		
	  - 	|tuto_linresp|
	  
		:ref:`Linear response to periodic driving<secadvstabdrumresponse>`
		
	* -	|tuto_dropdetach|
	
		:ref:`Bifurcation tracking of a detaching droplet<secdropletdetach>`		
		
	  - 	|tuto_azimuthal|
	  
		:ref:`Azimuthal stability analysis<secadvstabrbconv>`														
		
	* -	|tuto_cartnormal|
	
		:ref:`Cartesian normal mode stability analysis<secadvstabrivulets>`		
		
	  - 	|tuto_precice|
	  
		:ref:`Coupling multiple simulations with preCICE<secprecicenonmatch>`																
		
	* -	|tuto_viscoelastic|
	
		:ref:`Viscoelastic flow<secalebeadsonstring>`		
		
	  - 	|tuto_bifgui|
	  
		:ref:`The interactive bifurcation GUI<advstabbifgui>`																
	
	
	

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
