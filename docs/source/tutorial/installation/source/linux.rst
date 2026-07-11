On Linux
--------

To obtain the code, clone the GitHub repository

.. code:: bash

      git clone https://www.github.com/pyoomph/pyoomph.git 


Once you have cloned the repository with git, you first have to install a few packages. On a Debian/Ubuntu distribution, you habe to do e.g.

.. code:: bash

      sudo apt-get install libopenmpi-dev pybind11-dev libginac-dev libcln-dev libgmp-dev ccache libmkl-rt 

There are additional python packages required. You can either install these with ``python3 -m pip install ...`` (note that you might have to use ``python`` or ``python3`` as command) or find the corresponding Linux packages. If you do not install them now, they should be installed during the first build of pyoomph via ``pip``. The required ``python`` libraries are

.. code:: bash

      gmsh mkl mpi4py matplotlib numpy petsc4py pybind11 pygmsh scipy meshio pybind11-stubgen setuptools wheel scikit_build_core

Make sure you have recent versions, e.g. when using ``pip``, you could do 

.. code:: bash

      python -m pip install --upgrade gmsh mkl mpi4py matplotlib numpy petsc4py pybind11 pygmsh scipy meshio pybind11-stubgen setuptools wheel scikit_build_core

Afterwards, you first have to build the stripped and slightly modified version of oomph-lib which is shipped along with pyoomph. 

.. code:: bash

      cd <PYOOMPH_DIR>
      bash ./prebuild.sh

where ``<PYOOMPH_DIR>`` is the directory of your local pyoomph repository.

Finally, build pyoomph and install it, and check whether it works:

.. code:: bash

      cd <PYOOMPH_DIR>
      bash ./build_for_develop.sh
      python -m pyoomph check all
      

.. note::

      If you encounter segmentation faults during solving, you likely have a bugged version of the MKL package installed. In that case, please downgrade to an older version, e.g. via *python -m pip install mkl==2024.1.0*.
            

