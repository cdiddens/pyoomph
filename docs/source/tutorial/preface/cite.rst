How to cite
===========

At the moment, just cite the following paper for `pyoomph`:

* Christian Diddens and Duarte Rocha, *Bifurcation tracking on moving meshes and with consideration of azimuthal symmetry breaking instabilities*, J. Comput. Phys. **518**, 113306, (2024), doi:`10.1016/j.jcp.2024.113306 <https://dx.doi.org/10.1016/j.jcp.2024.113306>`__.

Please mention that `pyoomph` is based on `oomph-lib` and `GiNaC`, i.e. **also cite at least**:

* M. Heil, A. L. Hazel, *oomph-lib - An Object-oriented multi-physics finite-element library*, Lect. Notes Comput. Sci. Eng. **53**, 19-49, (2006), `doi:10.1007/3-540-34596-5_2 <https://dx.doi.org/10.1007/3-540-34596-5_2>`__.

* C. Bauer, A. Frink, R. Kreckel, *Introduction to the GiNaC framework for symbolic computation within the C++ programming language*, J. Symb. Comput. **33** (1), 1-12, (2002), doi:`10.1006/jsco.2001.0494 <https://dx.doi.org/10.1006/jsco.2001.0494>`__.



**Citing of material properties and activity models**

When using UNIFAC-like group contribution methods (cf. :numref:`secmcflowunifac`), please cite the following:

* The published parameters of the original and modified (Dortmund) UNIFAC model were taken with kind permission from the *DDBST* website https://www.ddbst.com. Please cite the papers listed at https://www.ddbst.com/published-parameters-unifac.html for original UNIFAC and https://www.ddbst.com/PublishedParametersUNIFACDO.html for modified (Dortmund) UNIFAC.

* Also note that the *UNIFAC Consortium* provides *updated and revised parameters*, which will increase the accuracy of the predicted activity coefficients. Please refer to the `https://unifac.ddbst.com/unifac_.html <https://unifac.ddbst.com/unifac_.html>`_ for more information. These updated parameters are not included in pyoomph.
   
* When using the AIOMFAC model, please cite the papers listed here https://aiomfac.lab.mcgill.ca/citation.html.

* When using the material properties from :py:mod:`pyoomph.materials.default_materials`, please have a look at the comments in this file to cite the correct papers.
