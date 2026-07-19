.. _installwsl:

Installing pyoomph (with PETSc/SLEPc) on Windows via WSL
----------------------------------------------------------

PETSc/SLEPc cannot be built natively on Windows (see :numref:`petscslepc`). If you are on Windows and want a good, stable eigensolver, the recommended route is to install pyoomph inside the `Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/windows/wsl/>`__. WSL2 runs a real Linux kernel, so once you are inside it, pyoomph and PETSc/SLEPc are installed exactly as on native Linux -- the restrictions that apply to native Windows builds simply do not apply anymore.

Install WSL
~~~~~~~~~~~

From an elevated Windows PowerShell, run

.. code:: powershell

	wsl --install -d Ubuntu

Reboot if prompted, then open the "Ubuntu" app it creates and finish the first-run setup (choose a username/password). Make sure you end up with WSL2, not WSL1:

.. code:: powershell

	wsl -l -v

If a distribution is listed as version 1, upgrade it with

.. code:: powershell

	wsl --set-version Ubuntu 2

Install pyoomph inside WSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the Ubuntu terminal and treat it exactly like a normal Linux installation (cf. :numref:`installonlinux`):

.. code:: bash

	sudo apt update
	sudo apt install gcc libopenmpi-dev flex bison
	python3 -m pip install --upgrade pyoomph

Or, for an editable install from source:

.. code:: bash

	git clone https://www.github.com/pyoomph/pyoomph.git
	cd pyoomph
	bash ./build_for_develop.sh

Check that the installation works:

.. code:: bash

	python -m pyoomph check all

.. note::

	Keep the pyoomph (and later PETSc) source trees on the Linux filesystem, e.g. under ``~/...``, rather than under ``/mnt/c/...``. Cross-filesystem I/O between Windows and WSL is much slower and can cause issues with some build tooling.

Build PETSc/SLEPc inside WSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since WSL is a genuine Linux environment, the normal PETSc/SLEPc build recipe from :numref:`petscslepc` applies without any changes -- including the option to build both a real and a complex arch side by side if you also need normal mode eigenvalue problems (cf. :numref:`azimuthalstabana` and :numref:`cartesiannormalstabana`):

.. code:: bash

	cd A_FOLDER_OF_YOUR_CHOICE
	git clone -b release https://gitlab.com/petsc/petsc.git petsc
	cd petsc
	export PETSC_DIR=$(pwd)
	export PETSC_ARCH=pyoomph_petsc_arch_real

	./configure --with-mpi  --with-petsc4py --with-slepc4py --download-mumps=yes --download-hypre=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=real

At the end, ``configure`` prints the exact ``make`` command to run next; execute it. If you also need normal mode eigenvalue problems, repeat with a complex arch:

.. code:: bash

	export PETSC_ARCH=pyoomph_petsc_arch_complex
	./configure --with-mpi  --with-petsc4py --with-slepc4py --download-mumps=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=complex

and execute the ``make`` command it prints, as before. Note that ``--download-hypre`` is dropped for the complex arch, since HYPRE only supports real-valued matrices (see :numref:`petscslepc` for details); MUMPS remains the factorization backend used via ``use_mumps()`` either way.

Afterwards, add the following to ``~/.bashrc`` inside WSL so the variables are set in every new terminal (use the ``pyoomph_petsc_arch_complex`` arch here instead if you mainly need normal mode eigenvalue problems; only one arch can be active per process, see :numref:`petscslepc` for how to switch between them):

.. code:: bash

	export PETSC_DIR=A_FOLDER_OF_YOUR_CHOICE/petsc
	export PETSC_ARCH=pyoomph_petsc_arch_real
	export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/$PETSC_ARCH/lib

You can then select SLEPc with MUMPS as eigensolver, either in your driver script

.. code:: python

	problem.set_eigensolver("slepc").use_mumps()

or via the command line flag

.. code:: bash

	python my_eigenvalue_simulation.py --slepc_mumps

Notes specific to WSL
~~~~~~~~~~~~~~~~~~~~~

- **Memory**: building PETSc with MUMPS, SuperLU_dist and ScaLAPACK is memory-hungry. If ``make`` gets killed (out of memory), increase the memory available to WSL2 by creating/editing ``C:\Users\<you>\.wslconfig`` on the Windows side:

  .. code:: ini

  	[wsl2]
  	memory=8GB

  and then restart WSL (``wsl --shutdown`` from PowerShell, then reopen the Ubuntu terminal).

- **Editor integration**: if you use VS Code, install the "WSL" remote extension and open the project from within WSL (``code .`` from the Ubuntu terminal). This makes sure the Python interpreter and terminal used by the editor are the WSL ones, so the ``PETSC_DIR``/``PETSC_ARCH``/``PYTHONPATH`` variables set above are actually visible to the process running your script. Running the script from a plain Windows terminal or Windows Python installation will not see them.

- **Everything else** -- MKL, ``gcc``, output via ParaView on the Windows side (WSL2 can access the Windows filesystem under ``/mnt/c/...`` if you want to write output there directly), etc. -- behaves as described for native Linux in :numref:`installonlinux` and :numref:`secinstallationmsbuild`.
