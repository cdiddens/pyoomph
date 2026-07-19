pyoomph.\_pyoomph\_core module
===============================

This is the nanobind-based coupling between the compiled C++ core and the Python frontend of pyoomph.

Several classes here are only the low-level nanobind base; pyoomph wraps them in a Python
subclass of the *same name* elsewhere (e.g. :class:`pyoomph.meshes.mesh.ODEStorageMesh`),
which is the one actually used from Python and documented on its own page. Those shadowed
base classes are excluded below to avoid duplicate, ambiguous documentation.

Module contents
----------------

.. automodule:: pyoomph._pyoomph_core
   :members:
   :show-inheritance:
   :exclude-members: GeneralSolverCallback, Problem, CustomMathExpression, CustomMultiReturnExpression, MeshTemplate, InterfaceMesh, ODEStorageMesh, LaTeXPrinter
