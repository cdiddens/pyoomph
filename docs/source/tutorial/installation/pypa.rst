Installing the precompiled wheels
---------------------------------

The easiest method to get pyoomph installed on your system is using ``pip``, i.e. just enter the following in a terminal (works for CPython on an x86_64 system):

.. code:: bash

      python -m pip install pyoomph

This will install pyoomph on your system. However, please also read the system-specific steps below.

If you get errors, let us know (c.diddens@utwente.nl), and we see whether we can provide a suitable wheel for your system.

.. warning::

   If you are using a recent Mac with the Apple silicon (arm64 architecture) processor, you must execute this command in a Rosetta terminal. At https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/ you can find instructions how to create such a Rosetta terminal (**note**: recent systems must be handled differently, see e.g. here: https://developer.apple.com/forums/thread/718666). Also, please see below regarding the `mkl` module.
   
   You can install pyoomph from source directly on arm64, but unfortunately without support for the fast MKL Pardiso solver. See :numref:`installonmac` for details. Also, you might have to use a less recent version of python. See https://github.com/pyoomph/pyoomph/blob/main/Mac_arm64_with_Pardiso.md for details.


Depending on your system, you have to do additional steps to obtain the full performance:


.. _secinstallationmsbuild:

Windows
~~~~~~~

pyoomph can use a minimal C compiler, namely the TinyC compiler (TCC), wrapped by the ``tccbox`` package. This compiler generates machine code very quickly, but the generated code is usually not well optimized so that the execution is slow compared to more sophisticated compilers. On Windows, Microsoft offers the MS Build Tools, a free compiler suite, which can be utilized by pyoomph to generate faster machine code. It is best to download the Build Tools for Visual Studio 2019, since this is the version Python is usually referring to. The installer can be found at https://aka.ms/vs/16/release/vs_buildtools.exe. It is important to install at least the following components:

..  figure:: msbuild.*
    :alt: Required packages from MS Build Tools
    :class: with-shadow
    :width: 100%
    :align: center
    
    Required packages to install from MS Build Tools
    
If you do not want to install MS Build Tools for any reason, you always can use the internal TinyC compiler. To do so, call the method ``set_c_compiler("tcc")`` of the :py:class:`~pyoomph.generic.problem.Problem` class so select the internal compiler. This has to be done for each problem and before any calls of the methods :py:meth:`~pyoomph.generic.problem.Problem.initialise`, :py:meth:`~pyoomph.generic.problem.Problem.output`, :py:meth:`~pyoomph.generic.problem.Problem.solve` or :py:meth:`~pyoomph.generic.problem.Problem.run`. Alternatively, you can add the command line arguments *--tcc*, e.g. run a your simulation script ``my_simulation.py`` as follows:


.. code:: bash

      python my_simulation.py --tcc


.. note::

      If you encounter segmentation faults during solving, you likely have a bugged version of the MKL package installed. In that case, please downgrade to an older version, e.g. via *python -m pip install mkl==2024.1.0*.
      
Mac
~~~

On Mac, ``clang`` will be used as high performance compiler. To get ``clang``, install the developer tools via

.. code:: bash

      xcode-select --install
      

.. warning::

   If you are using a recent Mac with an Apple silicon processor (arm64 architecture), make sure to not upgrade the package ``mkl``. Also on Macs with an Intel processor, more recent versions can cause a crash. If you by accident upgrade your mkl package, reset it by entering (in a Rosetta 2 terminal for arm64 chips):
   
   .. code:: bash
   
   	python -m pip install mkl==2021.4.0
   	   

      
Linux
~~~~~

On Linux, make sure that you have the ``gcc`` compiler installed to get optimal performance, e.g. on Ubuntu by

.. code:: bash

      sudo apt install gcc
      
Other Linux distributions, you might have to use ``yum``, ``pacman``, etc., instead.

.. note::

      If you encounter segmentation faults during solving, you likely have a bugged version of the MKL package installed. In that case, please downgrade to an older version, e.g. via *python -m pip install mkl==2024.1.0*.
      
      

Trying whether pyoomph works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To check whether pyoomph has been installed and the compilers and solvers can be detected, try it with

.. code:: bash

      python -m pyoomph check all


Updating pyoomph
~~~~~~~~~~~~~~~~

Pyoomph is under continuous development and the wheels are regularly updated. To update pyoomph to the recent version, just do a

.. code:: bash

      python -m pip install --upgrade pyoomph

