# Installation

When you have python 3.10 to 3.14 (CPython, 64-bit) installed, 

> python -m pip install pyoomph

should install the basic framework. For the maximum performance and system-specific information, please refer to the sections below. 

If you cannot manage to install it, refer to our [tutorial](https://pyoomph.readthedocs.io/). If this cannot help, you can ask for help at c.diddens@utwente.nl

## On Windows

For maximum performance, also install [Microsoft Build Tools](https://docs.microsoft.com/visualstudio/msbuild/msbuild), available for download [here](https://aka.ms/vs/17/release/vs_buildtools.exe). 

Verify whether everything runs fine by 

> python -m pyoomph check all

If you encounter segmentation faults during solving, please try to downgrade your MKL package, e.g. via *python -m pip install mkl==2024.1.0*.

### PETSc/SLEPc via WSL

PETSc/SLEPc (used for a much more stable eigensolver, see the [tutorial](https://pyoomph.readthedocs.io/)) cannot be built natively on Windows. If you need it, install pyoomph inside the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/windows/wsl/) instead, where it can be built exactly as on native Linux.

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, open the "Ubuntu" app, then inside it install pyoomph as on Linux (see below), keeping the source trees on the Linux filesystem (e.g. `~/...`, not `/mnt/c/...`):

```bash
sudo apt update
sudo apt install gcc libopenmpi-dev flex bison
python3 -m pip install --upgrade pyoomph
```

Then build PETSc/SLEPc as on Linux:

```bash
cd A_FOLDER_OF_YOUR_CHOICE
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc
export PETSC_DIR=$(pwd)
export PETSC_ARCH=pyoomph_petsc_arch
./configure --with-mpi --with-petsc4py --download-mumps=yes --download-hypre=yes --download-parmetis=yes --download-ptscotch=yes --download-slepc=yes --download-superlu=yes --download-superlu_dist=yes --download-suitesparse=yes --download-metis=yes --download-scalapack --with-scalar-type=complex
```

`configure` prints the exact `make` command to run next. Afterwards, add to `~/.bashrc`:

```bash
export PETSC_DIR=A_FOLDER_OF_YOUR_CHOICE/petsc
export PETSC_ARCH=pyoomph_petsc_arch
export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/$PETSC_ARCH/lib
```

If `make` is killed for running out of memory, raise the memory available to WSL2 via `C:\Users\<you>\.wslconfig` (`[wsl2]` section, `memory=8GB`), then `wsl --shutdown` and reopen. If you use VS Code, install the "WSL" remote extension and open the project from within WSL so the editor picks up the environment variables above. Full details, including how to select the SLEPc/MUMPS eigensolver, are in the [tutorial](https://pyoomph.readthedocs.io/en/latest/tutorial/installation/wsl.html).

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
Also install `scikit_build_core` and `nanobind`, either via the system's package manager or via `pip`

To obtain an editable build with MPI, do

> bash ./build_for_develop.sh

Otherwise, for a non-editable install you can use

> python -m pip install .


### Mac

Besides Xcode, you must install a few third-party tools. This can be done by e.g. [Homebrew](https://brew.sh):

> brew install openmpi pkg-config wget bzip2 patch

If you run in Rosetta (see above), restart your (Rosatta) terminal afterwards.
Install required python modules via

> python3 -m pip install nanobind scikit_build_core

To obtain an editable build with MPI, do

> bash ./build_for_develop.sh

Otherwise, for a non-editable install you can use

> python -m pip install .

Verify whether everything runs fine by 

> python -m pyoomph check all

