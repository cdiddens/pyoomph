.. _secode:

Temporal Ordinary Differential Equations
========================================

We start our tutorial with the simplest form of equations pyoomph can handle, namely (systems of) ordinary differential equations (ODEs) as function of time. When reading this section, one might ask: *can't this be done easier, e.g. with MATLAB?* - And the answer is probably *yes*. And for simple ODEs, MATLAB will be even faster than pyoomph. However, all steps performed and explained in this section are relevant for the understanding of the assembly of complicated coupled multi-physics problems later on. In fact, when a system of spatio-temporal differential equations is spatially discretized, one ends up with a large system of ODEs. Furthermore, ODEs offer the possibility to discuss a few steps analytically and thereby show what pyoomph actually does.

This section is devoted to (a set of) ordinary differential equations in the sense of an initial value problem. In physical problems, the independent variable is usually the time. If you want to solve a boundary value problem for (a set of) ordinary differential equations, please refer to :numref:`secspatial`. For such purposes, the independent variable is sampled on a one-dimensional mesh, i.e. you will use a spatial coordinate instead of the time as independent variable.


.. toctree::
   :maxdepth: 5
   :hidden:

   temporal/nondimho.rst
   temporal/ownho.rst
   temporal/testfunc.rst
   temporal/coupled.rst                     
   temporal/nonlinho.rst          
   temporal/timestepping.rst
   temporal/lagrange.rst
   temporal/units.rst
   temporal/changeparams.rst
   temporal/custommath.rst                                             
   temporal/stability.rst       
   temporal/orbit.rst          
   
