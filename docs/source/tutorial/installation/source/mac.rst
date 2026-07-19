.. _installonmac:

On Mac
------

.. warning::

   If you are using a recent Mac with an Apple silicon (arm64 architecture) processor, you might encounter some problems, since not all required python packages are present in the pip repository yet. Therefore, in order to use the fast MKL Pardiso solver, you can use Rosetta 2 to emulate the x86_64 architecture. You must execute the following commands in a Rosetta terminal. At https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/ you can find instructions on how to create such a Rosetta terminal. On more recent systems, please refer to https://developer.apple.com/forums/thread/718666 to setup a corresponding terminal.
      

To clone the git repository, you require git, but this comes along with the Xcode developer tools, which is required anyhow. The latter can be installed via

.. code:: bash

      xcode-select --install

for a terminal. After that, you should have git so that you can clone the repository:

.. code:: bash

      git clone https://www.github.com/pyoomph/pyoomph.git 
      
The current development version (**might be unstable**) is hosted at

.. code:: bash

      git clone https://www.github.com/ciddens/pyoomph.git 
   

Before building it, a bunch of additional software has to be installed. For Mac, there is e.g. homebrew (https://brew.sh), which easily manages these additional packages. Hence, install homebrew by pasting the installation command from https://brew.sh.

Afterwards, you can install some required tools, by

.. code:: bash

      brew install openmpi cmake pkg-config

You might have to close and reopen the (Rosetta) terminal now.

Before building pyoomph, we first have to make sure that additional python packages are installed. This can be done e.g. by

.. code:: bash

      python3 -m pip install pybind11 gmsh mpi4py matplotlib numpy pygmsh scipy meshio pybind11-stubgen setuptools scikit_build_core

The ``python3`` command might be also ``python``, depending on the system. Be sure to use the more recent version of python.

When using the Rosetta 2 terminal approach, make sure you have not upgraded your mkl package to the recent version, which actually crashes on Mac:

.. code:: bash

      python3 -m pip install mkl==2021.4.0


Afterwards, you should be able to build pyoomph by:


For an editable local install (with MPI), run

.. code:: bash

      bash ./build_for_develop.sh
      
Alternatively, you can run

.. code:: bash

      python -m pip install .


Finally, check whether it works:

.. code:: bash

      python -m pyoomph check all
