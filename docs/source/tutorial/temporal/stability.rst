Stability analysis
------------------

So far, we have only considered time-dependent ODEs. However, it is also of interest how ODEs, in particular nonlinear ones, behave by performing a stability analysis. This means that we are not interested in the total time behavior, but just whether a stationary solution of a system of ODEs will be stable under a small perturbation around this stationary solution or not. In particular, the stability of a stationary solution usually depends on a parameter, i.e. a stationary solution may switch from unstable to stable or vice versa at a critical value of this parameter, which is called a *bifurcation*. Different from the previous section, we do not want to start a bunch of simulations to scan this parameter, but ideally, one wants to find the critical value of the parameter automatically.

Obviously, in order to investigate the stability of stationary solutions of ODE systems, we first need to address two things, namely how to find a stationary solution and how to introduce parameters that can be changed at run time.

.. toctree::
   :maxdepth: 5
   :hidden:

   stability/globalparams.rst
   stability/stationary.rst   
   stability/eigenvals.rst   
   stability/constraints.rst
   stability/arclength.rst         
   stability/jumponbif.rst   
   stability/biftrack.rst
   stability/deflation.rst   
   stability/lyapunov.rst                                 

