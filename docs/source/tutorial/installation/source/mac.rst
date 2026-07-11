.. _installonmac:

On Mac
------

.. warning::

   If you are using a recent Mac with an Apple silicon (arm64 architecture) processor, you might encounter some problems, since not all required python packages are present in the pip repository yet. Therefore, you must use Rosetta 2 to emulate the x86 64 architecture. You must execute the following commands in a Rosetta terminal. At https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/ you can find instructions how to create such a Rosetta terminal. On more recent systems, please refer to https://developer.apple.com/forums/thread/718666 to setup a corresponding terminal.
   
   
.. note::

   If you do not want to use the `Rosetta 2 terminal` detour, you can also install it directly on arm64 systems. `Vatsal Sanjay <https://github.com/VatsalSy>`__ from `CoMPhy Lab <https://comphy-lab.org/>`__ has contributed an installation script
   
   .. code:: bash

	bash installOnArm.sh

   to perform this installation. The only downside is that the fast `MKL Pardiso` solver is not supported on native arm64 systems. For optimal performance, in particular for larger problems, one therefore should consider the `Rosetta 2 terminal` way for the time being. Also, you might have to use a less recent version of python. See https://github.com/pyoomph/pyoomph/blob/main/Mac_arm64_with_Pardiso.md for details.


To clone the git repository, you require git, but this comes along with the Xcode developer tools, which is required anyhow. The latter can be installed via

.. code:: bash

      xcode-select --install

for a terminal. After that, you should have git so that you can clone the repository:

.. code:: bash

      git clone https://www.github.com/pyoomph/pyoomph.git 



Before building it, a bunch of additional software has to be installed. For Mac, there is e.g. homebrew (https://brew.sh), which easily manage these additional packages. Hence, install homebrew by pasting the installation command from https://brew.sh.

Afterwards, you can install some required tools, by

.. code:: bash

      brew install openmpi ccache ginac

You might have to close and reopen the (Rosetta) terminal now. Afterwards, you first have to build the stripped and slightly modified version of oomph-lib which is shipped along with pyoomph:

.. code:: bash

      cd <PYOOMPH_DIR>
      bash ./prebuild.sh

where ``<PYOOMPH_DIR>`` is the directory of your local pyoomph repository.

Before building pyoomph, we first have to make sure that additional python packages are installed. This can be done e.g. by

.. code:: bash

      python3 -m pip install pybind11 gmsh mpi4py matplotlib numpy pygmsh scipy meshio pybind11-stubgen setuptools scikit_build_core

The ``python3`` command might be also ``python``, depending on the system. Be sure to use the more recent version of python.

On Mac, make sure you have not upgraded your mkl package to the recent version, which actually crashes on Mac:

.. code:: bash

      python3 -m pip install mkl==2021.4.0

For older Macs with Intel processor, you can install the recent version

.. code:: bash

      python3 -m pip install mkl

Afterwards, you should be able to build pyoomph by:

.. code:: bash

      cd <PYOOMPH_DIR>
      bash ./build_for_develop.sh

In the worst case, try to execute the last command up to three times. If it still does not work, please contact me at c.diddens@utwente.nl. Finally, check whether everything works well via:

.. code:: bash

      python3 -m pyoomph check all

