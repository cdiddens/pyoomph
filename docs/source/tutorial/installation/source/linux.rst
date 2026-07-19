.. _installonlinux:

On Linux
--------

To obtain the code, clone the GitHub repository

.. code:: bash

      git clone https://www.github.com/pyoomph/pyoomph.git 
      
The current development version (**might be unstable**) is hosted at

.. code:: bash

      git clone https://www.github.com/ciddens/pyoomph.git 


Once you have cloned the repository with git, you first have to install a few packages. On a Debian/Ubuntu distribution, you have to do e.g.

.. code:: bash

      sudo apt-get install libopenmpi-dev pybind11-dev 

There are additional python packages required. You can either install these with ``python3 -m pip install ...`` (note that you might have to use ``python`` or ``python3`` as command) or find the corresponding Linux packages. If you do not install them now, they should be installed during the first build of pyoomph via ``pip``. The required ``python`` libraries are

.. code:: bash

      gmsh mkl mpi4py matplotlib numpy pybind11 pygmsh scipy meshio pybind11-stubgen setuptools scikit_build_core

Make sure you have recent versions, e.g. when using ``pip``, you could do 

.. code:: bash

      python -m pip install --upgrade gmsh mkl mpi4py matplotlib numpy pybind11 pygmsh scipy meshio pybind11-stubgen setuptools scikit_build_core

For an editable local install (with MPI), run

.. code:: bash

      bash ./build_for_develop.sh
      
Alternatively, you can run

.. code:: bash

      python -m pip install .


Finally, check whether it works:

.. code:: bash

      python -m pyoomph check all
      

.. note::

      If you encounter segmentation faults during solving, you likely have a bugged version of the MKL package installed. In that case, please downgrade to an older version, e.g. via *python -m pip install mkl==2024.1.0*.
            

