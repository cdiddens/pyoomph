.. _petscslepc:

Optional installation of PETSc/SLEPc
------------------------------------

If you want to solve for eigenvalue problems, pyoomph by default will invoke `scipy's eigensolver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`__ based on `ARPACK <https://github.com/opencollab/arpack-ng>`__.
However, for unsymmetric matrices which usually arise in complicated problems, `SLEPc <https://slepc.upv.es/>`__ provides a much more stable alternative, since it also supports nonsymmetric mass matrices.
So whenever you want to investigate linear stability, you should consider performing the following steps. In any case, it is advised to occasionally check your eigenvalues by adding ``report_accuracy=True`` to calls of :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`. 

Also on Mac arm64, we recommend using PETSc with MUMPS as linear solver backend for the linear solves during Newton's method, since the default MKL Pardiso backend is not yet fully supported on arm64.

We unfortunately do not really know how to install PETSc/SLEPc natively on Windows, so you have to find your own way of installing it (and let us know the steps. A good start can be found `here <https://petsc.org/release/install/windows/>`__). 
If you are on Windows, the easiest way to get PETSc/SLEPc working is to install pyoomph inside the Windows Subsystem for Linux (WSL) and follow the normal Linux instructions below -- see :numref:`installwsl` for step-by-step instructions.

SLEPc depends on `PETSc <https://petsc.org>`__, so it is advisable to install both of these packages together. Also, since we often will obtain matrices with a zero on a diagonal (mainly due to Lagrange multipliers, incompressibility constraints, etc.) we need a suitable linear solver backend in PETSc which can perform pivoting. We usually use `MUMPS <https://mumps-solver.org/>`__ for that.

PETSc/SLEPc are compiled for exactly one scalar type -- either real or complex -- and pyoomph picks up whichever ``petsc4py``/``slepc4py`` happens to be first on ``PYTHONPATH`` when the process starts. Which one you need depends on what you do with it:

* The **real** build is used for the linear solves during Newton's method (with PETSc/MUMPS as linear solver backend) and for conventional, non-normal-mode eigenvalue problems (with SLEPc).
* The **complex** build is mandatory if you want to solve normal mode eigenvalue problems (cf. :numref:`azimuthalstabana` and :numref:`cartesiannormalstabana`), since these introduce an explicit imaginary prefactor into the equations. If you try to solve such a problem with only a real build available, pyoomph will raise a ``RuntimeError`` telling you to switch to a complex installation.

Since most users will want both at some point, we recommend building both a real and a complex arch of PETSc/SLEPc side by side in the same source tree (they can share one ``PETSC_DIR``, just use two different ``PETSC_ARCH`` names), and then select the one you currently need via ``PETSC_ARCH``/``PYTHONPATH`` before running a particular script. If you only ever need conventional (real) eigenvalue problems, you can of course skip the complex build and just do the real one.

All packages can be downloaded and installed together. If you have opted for the Rosetta 2 installation route on a Mac arm64 (see previous pages) to also support MKL Pardiso, the following steps must be again done in a Rosetta 2 terminal.

We start by downloading PETSc in a folder of our choice (replace ``A_FOLDER_OF_YOUR_CHOICE`` in the following accordingly). If you have installed pyoomph in a python environment, it is advisable to also activate this environment now.

.. code:: bash

	cd A_FOLDER_OF_YOUR_CHOICE
	git clone -b release https://gitlab.com/petsc/petsc.git petsc
	cd petsc

We now have to export some environment variables. ``PETSC_DIR`` is shared between both arches, while ``PETSC_ARCH`` selects which one we are currently building:

.. code:: bash

	export PETSC_DIR=$(pwd)
	export PETSC_ARCH=pyoomph_petsc_arch_real

Note that the choice of the name ``pyoomph_petsc_arch_real`` can be changed arbitrarily.

We then have to make sure that we have `flex <https://github.com/westes/flex>`__ and `Bison <https://www.gnu.org/software/bison/>`__. On Ubuntu (and other Linux types analogously), you can install it system-wide via

.. code:: bash

	sudo apt install flex bison

On Mac, you can install it via `Homebrew <https://brew.sh/>`__:

.. code:: bash

	brew install flex bison

Alternatively, you can let PETSc download it as well by adding ``--download-bison`` at the end of the following configuration command. Note that we download and install further solver packages here, which are usually not needed, but likely will be used in future.

.. code:: bash

	./configure --with-mpi  --with-petsc4py --with-slepc4py --download-mumps=yes --download-hypre=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=real

You can also add optimization or OpenMP support, e.g. ``--with-debugging=0``, ``-with-openmp``,  ``--with-openmp-kernels``. For all details, please call ``./configure --help``.

.. note::
	If you should have issues with `cmake` on Ubuntu (and potentially other distros), try
		#. Install cmake (updated version, see https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)
		#. add flag ``--download-fblasapack=1`` when configuring


At the end of the configuration process, a ``make`` command will be written, which you have to execute as a next step.

Afterwards, PETSc/SLEPc is installed to the folder ``A_FOLDER_OF_YOUR_CHOICE/petsc/pyoomph_petsc_arch_real``.
At the end, it will also show a test command, by which you can test the basic functionality of your installation.

If you also need normal mode eigenvalue problems, repeat the same steps for a second, complex arch. Since ``PETSC_DIR`` is already set from above, only ``PETSC_ARCH`` and the ``--with-scalar-type`` flag change:

.. code:: bash

	export PETSC_ARCH=pyoomph_petsc_arch_complex
	./configure --with-mpi  --with-petsc4py --with-slepc4py --download-mumps=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=complex

Again, execute the ``make`` command that ``configure`` prints at the end. This installs the complex build alongside the real one, to ``A_FOLDER_OF_YOUR_CHOICE/petsc/pyoomph_petsc_arch_complex``.

Note that ``--download-hypre`` is left out here: `HYPRE <https://github.com/hypre-space/hypre>`__ (and thus its BoomerAMG preconditioner) only supports real-valued matrices, and PETSc's ``configure`` will fail if you request it together with ``--with-scalar-type=complex``. This is not a problem in practice, since MUMPS is used as the actual factorization backend anyway (via ``use_mumps()``) -- HYPRE in the real build above is only downloaded for potential future use as an alternative preconditioner.

To use one of the two builds within pyoomph, ``PETSC_DIR``, ``PETSC_ARCH`` and ``PYTHONPATH`` must be set *before* you start your Python driver script, e.g. for the real build:

.. code:: bash

	export PETSC_DIR=A_FOLDER_OF_YOUR_CHOICE/petsc
	export PETSC_ARCH=pyoomph_petsc_arch_real
	export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/$PETSC_ARCH/lib

or, analogously, with ``pyoomph_petsc_arch_complex`` for the complex build. Since only one of the two can be active in a given terminal/process, do **not** put both blocks unconditionally into your ``.bashrc``/``.zshrc`` -- the second one would just shadow the first on ``PYTHONPATH``. Instead, either export the variables for the arch you currently need right before running your script, or wrap each block in a shell function (e.g. ``pyoomph_petsc_real()`` / ``pyoomph_petsc_complex()``) that you call as needed. If you only installed one arch, it is fine to export it unconditionally in your shell startup file as before.

To use SLEPc with MUMPS as eigensolver, either set it in python during your driver code, e.g.

.. code:: python

	problem.set_eigensolver("slepc").use_mumps()

or supply the flag ``--slepc_mumps`` when calling your driver code:

.. code:: bash

	python my_eigenvalue_simulation.py --slepc_mumps
