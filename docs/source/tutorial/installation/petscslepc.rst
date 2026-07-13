.. _petscslepc:

Optional installation of PETSc/SLEPc
------------------------------------

If you want to solve for eigenvalue problems, pyoomph by default will invoke `scipy's eigensolver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`__ based on `ARPACK <https://github.com/opencollab/arpack-ng>`__.
However, for unsymmetric matrices which usually arise in complicated problems, `SLEPc <https://slepc.upv.es/>`__ provides a much more stable alternative.
So whenever you want to investigate linear stability, you should consider performing the following steps. They are all optional, but usually give better and more stable eigenvalue results. In any case, it is advised to occasionally check your eigenvalues by adding ``report_accuracy=True`` to calls of :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`. 
We unfortunately do not really know how to install SLEPc on Windows, so you have to find your own way of installing it (and let us know the steps. A good start can be found `here <https://petsc.org/release/install/windows/>`__).

SLEPc depends on `PETSc <https://petsc.org>`__, so it is advisable to install both of these packages together. Also, since we often will obtain matrices with a zero on a diagonal (mainly due to Lagrange multipliers, incompressibility constraints, etc.) we need a suitable linear solver backend in PETSc which can perform pivoting. We usually use `MUMPS <https://mumps-solver.org/>`__ for that.
Also, if you want to solve normal mode eigenvalue problems (cf. :numref:`azimuthalstabana` and :numref:`cartesiannormalstabana`), we must have support for complex-valued eigenvalue problems, which requires compiling PETSc/SLEPc and MUMPS with complex values.

All three packages can be downloaded and installed together. On Mac with an M1 (arm64) chip, this must be again done in a Rosetta 2 terminal, at least if you want to use MKL Pardiso as linear solver (see previous pages). 

We start by downloading PETSc in a folder of our choice (replace ``A_FOLDER_OF_YOUR_CHOICE`` in the following accordingly). If you have installed pyoomph in a python environment, it is advisable to also activate this environment now.

.. code:: bash
	
	cd A_FOLDER_OF_YOUR_CHOICE
	git clone -b release https://gitlab.com/petsc/petsc.git petsc
	cd petsc
	
We now have to export some environment variables:

.. code:: bash

	export PETSC_DIR=$(pwd)
	export PETSC_ARCH=pyoomph_petsc_arch
	
Note that the choice of the name ``pyoomph_petsc_arch`` can be changed arbitrarily.

We then have to make sure that we have `flex <https://github.com/westes/flex>`__ and `Bison <https://www.gnu.org/software/bison/>`__. On Ubuntu (and other Linux types analogously), you can install it system-wide via 

.. code:: bash

	sudo apt install flex bison

Alternatively, you can let PETSc download it as well by adding ``--download-bison`` at the end of the following configuration command. Note that we download and install further solver packages here, which are usually not needed, but likely will be used in future.

.. code:: bash

	./configure --with-mpi  --with-petsc4py --download-mumps=yes --download-hypre=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=complex 
	
You can also add optimization or OpenMP support, e.g. ``--with-debugging=0``, ``-with-openmp``,  ``--with-openmp-kernels``. For all details, please call ``./configure --help``.

.. note::
	If you should have issues with `cmake` on Ubuntu (and potentially other distros), try
		#. Install cmake (updated version, see https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line) 
		#. add flag ``--download-fblasapack=1`` when configuring
	

At the end of the configuration process, a ``make`` command will be written, which you have to execute as a next step.

Afterwards, PETSc/SLEPc is installed to the folder ``A_FOLDER_OF_YOUR_CHOICE/petsc/pyoomph_petsc_arch``.
At the end, it will also show a test command, by which you can test the basic functionality of your installation.

To use it within pyoomph, you have to make sure that you always 

.. code:: bash

	export PETSC_DIR=A_FOLDER_OF_YOUR_CHOICE/petsc
	export PETSC_ARCH=pyoomph_petsc_arch
	export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/$PETSC_ARCH/lib
	
It is advised to copy these statements into your ``.bashrc`` or ``.zshrc`` (depending on the terminal you use). Alternatively, if you use a python environment for pyoomph, you can also put these in the ``activate`` script of the environment. Note, however, that these won't be unset automatically if you deactivate the environment then, only if you close the terminal.

To use SLEPc with MUMPS as eigensolver, either set it in python during your driver code, e.g.

.. code:: python

	problem.set_eigensolver("slepc").use_mumps()
	
or supply the flag ``--slepc_mumps`` when calling your driver code:

.. code:: bash

	python my_eigenvalue_simulation.py --slepc_mumps
