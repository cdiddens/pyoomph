When to use pyoomph, and when better use something else
-------------------------------------------------------

As every numerical framework, pyoomph has some strong points but of course also plenty of limitations.

**You can consider using pyoomph**

* when you want to have a simple python interface, but still want to have high computational speed
* when you want to quickly set up a multi-physics problem
* when you are too lazy to nondimensionalize your equations by hand before implementation
* when you want to write equations only once and reuse them in different coordinate systems, potentially in combination with other equations
* for problems involving multi-component & multi-phase flow, including Marangoni flow, mass transfer and surfactants
* when you don't want to code a lot of matrix filling routines by hand
* when you want to track (azimuthal) bifurcations
* when you want to use a monolithic sharp-interface moving mesh method


**You should consider using something else**

* when you want to use all features of `oomph-lib <https://oomph-lib.github.io/oomph-lib/doc/html/>`_.
* when you want to operate on a lower level for more flexibility
* when you need highly parallelized computational power
* for computationally expensive three-dimensional problems
* for high Reynolds numbers (go for e.g. advanced finite differences as in `AFiD <https://stevensrjam.github.io/Website/afid.html>`_)
* for topological changes (the sharp-interface method is not well suited for this, go for VoF instead, e.g. `Basilisk <http://basilisk.fr/>`_)
* if you need more fancy finite-element spaces (go for `FEniCS <https://fenicsproject.org/>`_ or `NGSolve <https://ngsolve.org/>`_)
* if you need spline basis functions (go for `nutils <https://nutils.org/>`_)
