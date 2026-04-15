# Installation

When you have python 3.9 to 3.14 (CPython, 64-bit, x86_64) installed, 

> python -m pip install pyoomph

should install the basic framework. On Mac silicon processor systems (arm64, e.g. M1-M4), please execute it in a Rosetta2 terminal (see below).
For the maximum performance and system-specific information, please refer to the sections below. 

If you cannot manage to install it, refer to our [tutorial](https://pyoomph.readthedocs.io/). If this cannot help, you can ask for help c.diddens@utwente.nl

## On Windows

For maximum performance, also install [Microsoft Build Tools](https://docs.microsoft.com/visualstudio/msbuild/msbuild), available for download [here](https://aka.ms/vs/16/release/vs_buildtools.exe). 

Verify whether everything runs fine by 

> python -m pyoomph check all

If you encounter segmentation faults during solving, please try to downgrade your MKL package, e.g. via *python -m pip install mkl==2024.1.0*.

## On Linux

If you have installed via pip (see above), just make sure that you have the `gcc` compiler installed and check via

> python -m pyoomph check all

If you encounter segmentation faults during solving, please try to downgrade your MKL package, e.g. via *python -m pip install mkl==2024.1.0*.

## On Mac

**If you have a recent Mac with an Apple silicon (arm64) chip**, you must run all commands  in a `Rosetta 2 terminal`, see [here](https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/) how to set it up (**note**: recent systems must be handled differently, see [here](https://developer.apple.com/forums/thread/718666)). Also, please downgrade `mkl` by

> python3 -m pip install mkl==2021.4.0

**Since the development of Intel MKL (containing the fast MKL Pardiso solver) was discontinued for Mac, using this solver might require installing a less recent version of python. See [`Mac_arm64_with_Pardiso.md`](https://github.com/pyoomph/pyoomph/blob/main/Mac_arm64_with_Pardiso.md) for more details.**

Make sure to have the `XCode` developer tools, e.g. by installing them via

> xcode-select --install

and test pyoomph via

> python -m pyoomph check all

Alternatively, if you do not want to use the `Rosetta 2 terminal` detour, you can also install it directly on arm64 systems. [Vatsal Sanjay](https://github.com/VatsalSy) from [CoMPhy Lab](https://comphy-lab.org/) has contributed an installation script

> bash installOnArm.sh

to perform this installation. The only downside is that the fast `MKL Pardiso` solver is not supported on native arm64 systems. For optimal performance, in particular for larger problems, one therefore should consider the `Rosetta 2 terminal` way for the time being.

## Compilation from source


### Linux

First, you have to make sure to have installed all dependencies, including some development files (headers). On e.g. Ubuntu, you can do the following

> sudo apt-get install libopenmpi-dev pybind11-dev libginac-dev libcln-dev libgmp-dev ccache libmkl-rt 

On other Linux distributions, other package manager like `yum` or `pacman` can be used to install the same libraries and headers.
Afterwards, compile the required parts of oomph-lib by running 

> bash ./prebuild.sh

in your pyoomph directory. 

Install required and optional python modules via

> python -m pip gmsh mkl mpi4py matplotlib numpy petsc4py pybind11 pygmsh scipy meshio pybind11-stubgen setuptools

If you want to install pyoomph **for development**, it is best to install it via 

> bash ./build_for_develop.sh

In that case, changes you make in the python files of pyoomph will be used automatically.
**Alternatively**, for a **system- or user-wide installation** by installing pyoomph, do

> bash ./install.sh

as last step instead.

Verify whether everything runs fine by 

> python -m pyoomph check all


### Mac

Besides XCode, you must install a few third-party tools. This can be done by e.g. [Homebrew](https://brew.sh):

> brew install openmpi ccache ginac

**Since the development of Intel MKL (containing the fast MKL Pardiso solver) was discontinued for Mac, using this solver might require installing a less recent version of python. See [`Mac_arm64_with_Pardiso.md`](https://github.com/pyoomph/pyoomph/blob/main/Mac_arm64_with_Pardiso.md) for more details.**

Restart your (Rosatta) terminal afterwards.
Install required python modules via

> python3 -m pip install pybind11 gmsh commonmark six pyparsing pygments pillow numpy mpi4py kiwisolver fonttools cycler scipy rich python-dateutils packaging meshio matplotlib pygmsh pybind11-stubgen

**If you have a recent Mac with an Apple silicon (arm64) chip**, install 

> python3 -m pip install mkl==2021.4.0

**otherwise**, you can install

> python3 -m pip install mkl

Then, compile the required parts of oomph-lib:

> bash ./prebuild.sh

in your pyoomph directory. 

If you want to install pyoomph **for development**, it is best to install it via 

> bash ./build_for_develop.sh

In that case, changes you make in the python files of pyoomph will be used automatically.
**Alternatively**, for a **system- or user-wide installation** by installing pyoomph, do

> bash ./install.sh

as last step instead.

Verify whether everything runs fine by 

> python -m pyoomph check all

