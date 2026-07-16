# Installation

When you have python 3.10 to 3.15 (CPython, 64-bit) installed, 

> python -m pip install pyoomph

should install the basic framework. For the maximum performance and system-specific information, please refer to the sections below. 

If you cannot manage to install it, refer to our [tutorial](https://pyoomph.readthedocs.io/). If this cannot help, you can ask for help at c.diddens@utwente.nl

## On Windows

For maximum performance, also install [Microsoft Build Tools](https://docs.microsoft.com/visualstudio/msbuild/msbuild), available for download [here](https://aka.ms/vs/17/release/vs_buildtools.exe). 

Verify whether everything runs fine by 

> python -m pyoomph check all

If you encounter segmentation faults during solving, please try to downgrade your MKL package, e.g. via *python -m pip install mkl==2024.1.0*.

## On Linux

If you have installed via pip (see above), just make sure that you have the `gcc` compiler installed and check via

> python -m pyoomph check all

If you encounter segmentation faults during solving, please try to downgrade your MKL package, e.g. via *python -m pip install mkl==2024.1.0*.

## On Mac

The fast `MKL Pardiso` solver from `mkl` is not available on `arm64` Macs. If you want to use it, install pyoomph in a `Rosetta 2 terminal`, see [here](https://www.courier.com/blog/tips-and-tricks-to-setup-your-apple-m1-for-development/) how to set it up (**note**: recent systems must be handled differently, see [here](https://developer.apple.com/forums/thread/718666)).
Also, please downgrade `mkl` by

> python3 -m pip install mkl==2021.4.0

Make sure to have the `Xcode` developer tools, e.g. by installing them via

> xcode-select --install

and test pyoomph via

> python -m pyoomph check all

Alternatively, if you do not want to use the `Rosetta 2 terminal` detour, you can also install it directly on arm64 systems. This will use the `Accelerate Framework` as default solver.


## Compilation from source (including MPI)

### Linux

First, you have to make sure to have installed all dependencies, including some development files (headers). On e.g. Ubuntu, you can do the following

> sudo apt-get install libopenmpi-dev build-essential cmake python3-dev pkg-config wget bzip2 patch

On other Linux distributions, other package manager like `yum` or `pacman` can be used to install the same libraries and headers.
Also install `scikit_build_core`, `pybind11` and `pybind11-stubgen`, either via the system's package manager or via `pip`

To obtain an editable build with MPI, do

> bash ./build_for_develop.sh

Otherwise, for a non-editable install you can use

> python -m pip install .


### Mac

Besides Xcode, you must install a few third-party tools. This can be done by e.g. [Homebrew](https://brew.sh):

> brew install openmpi pkg-config wget bzip2 patch

If you run in Rosetta (see above), restart your (Rosatta) terminal afterwards.
Install required python modules via

> python3 -m pip install pybind11 pybind11-stubgen scikit_build_core

To obtain an editable build with MPI, do

> bash ./build_for_develop.sh

Otherwise, for a non-editable install you can use

> python -m pip install .

Verify whether everything runs fine by 

> python -m pyoomph check all

