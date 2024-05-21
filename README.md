# Description

pyoomph is a object-oriented multi-physics finite element framework.
It is mainly a custom high level python frontend for the main functionalities (but by far not all) of the powerful C++ library [`oomph-lib`](http://www.oomph-lib.org).

For performance reasons, pyoomph uses [`GiNaC`](https://www.ginac.de/) and [`CLN`](https://www.ginac.de/CLN) to automatically generate C code for the equations you have entered in python. It automatically generates C code for symbolically derived Jacobian matrices, parameter derivatives and Hessians. These even include the complicated derivatives with respect to the moving mesh coordinates on a symbolical level. The generated code is compiled and linked back to the running python script, either with the [`TinyC`](https://bellard.org/tcc/) compiler (invoked by [`tccbox`](https://github.com/metab0t/tccbox)) or, when installed, with a more performant alternative like [`gcc`](https://gcc.gnu.org/), [`LLVM/clang`](https://clang.llvm.org/) or [`MSBuild`](https://docs.microsoft.com/visualstudio/msbuild/msbuild).

If you want to use the full flexibility of oomph-lib, you are probably better suited using [`oomph-lib`](http://www.oomph-lib.org) directly. If your want to use python to solve equations on a single static mesh, you might want to check out [`FEniCS`](https://fenicsproject.org/) instead. Also, have a look at [`NGSolve`](https://ngsolve.org/) or [`nutils`](https://nutils.org/) which have similar and complementary features.
If you are looking for a python framework for multi-physics problems formulated on (potentially multiple) moving meshes, including the possibility of (azimuthal) bifurcation tracking, pyoomph might be the right choice for you.

**pyoomph is still in an early stage of development:** While most features work nicely, it is neither feature-complete, nor free of bugs.

## Installation

Please consult the file [`INSTALL.md`](https://github.com/pyoomph/pyoomph/blob/main/INSTALL.md) in the git repository for installation instructions.
Alternatively, follow the [instructions in our tutorial](https://pyoomph.readthedocs.io/en/latest/tutorial/installation.html).

## Documentation and Examples

Documentation of the API and tons of examples can be found at [pyoomph.readthedocs.io](https://pyoomph.readthedocs.io/en/latest/tutorial.html). 
A [PDF version](https://pyoomph.readthedocs.io/_/downloads/en/latest/pdf/) of the tutorial is also available.

Some more examples can be found in our repository [pyoomph_examples](https://www.github.com/pyoomph/pyoomph_examples).

## License

pyoomph itself is distributed as combined work under the GPL v3 license. However, mind the third-party licences stated below, in particular when distributing derived work or redistributing pyoomph. The full license file can be found in [COPYING](COPYING).

## Third-Party-Licenses 

The distribution of pyoomph **contains code** taken from **other authors/projects**:

- In [`src/thirdparty/oomph-lib/include`](https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/oomph-lib/include), you find the necessary main files of [`oomph-lib`](http://www.oomph-lib.org), ([LGPL v2.1 or later license](https://github.com/oomph-lib/oomph-lib/blob/main/LICENCE)). Minor **modifications**, as mentioned in [src/thirdparty/INFO_oomph-lib](https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/INFO_oomph-lib), had to be made. Furthermore, code parts of these oomph-lib files had been copied to corresponding derived classes of pyoomph.
- A copy of the header-only library [`nanoflann`](https://github.com/jlblancoc/nanoflann) is located in [`src/thirdparty/nanoflann.hpp`](https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/nanoflann.hpp), ([BSD license](https://github.com/jlblancoc/nanoflann/blob/master/COPYING)).
- A copy of the header-only library [`delaunator-cpp`](https://github.com/delfrrr/delaunator-cpp) is located in [`src/thirdparty/delaunator.hpp`](https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/delaunator.hpp), ([MIT license](https://github.com/delfrrr/delaunator-cpp/blob/master/LICENSE)).
- The file [src/pyginacstruct.hpp](https://github.com/pyoomph/pyoomph/blob/main/src/pyginacstruct.hpp) is strongly based on the file [structure.h](https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=ginac/structure.h;hb=HEAD) of [GiNaC](https://www.ginac.de/) ([GPL v2 or later license](https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=COPYING;hb=HEAD)).
- A copy of the library [Project Nayuki/smallest enslosing circle](https://www.nayuki.io/page/smallest-enclosing-circle) ([LGPL v3 license or later](https://github.com/nayuki/Nayuki-web-published-code/blob/master/smallest-enclosing-circle/COPYING.LESSER.txt)) is added (after adding type specifications) to [pyoomph/utils/smallest_circle.py](https://github.com/pyoomph/pyoomph/blob/main/pyoomph/utils/smallest_circle.py).
- When using materials or the thermodynamic activity models AIOMFAC, original UNIFAC or modified UNIFAC (Dortmund), please [see below](#cite).

The third-party licenses/acknowledgement files can be found in [src/thirdparty](https://github.com/pyoomph/pyoomph/tree/main/src/thirdparty). In the provided python wheels, these requirements are statically included.

During compilation, pyoomph includes/links against or makes use of the following libraries:

- [`GiNaC`](https://www.ginac.de/) ([GPL v2 or later license](https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=COPYING;hb=HEAD)), GiNaC is statically included in the distribution as wheels.
- [`CLN`](https://www.ginac.de/CLN) ([GPL v2 or later license](https://www.ginac.de/CLN/cln.git/?p=cln.git;a=blob_plain;f=COPYING;hb=HEAD)), CLN is statically included in the distribution as wheels.
- [`python3.8 or higher`](https://www.python.org/), ([PSF license](https://docs.python.org/3/license.html)).
- [`pybind11`](https://github.com/pybind/pybind11), ([BSD-style license](https://github.com/sizmailov/pybind11-stubgen/blob/master/LICENSE)).
- [`pybind11-stubgen`](https://github.com/sizmailov/pybind11-stubgen), ([BSD-style license](https://github.com/sizmailov/pybind11-stubgen/blob/master/LICENSE)).
- [`pip`](https://github.com/pypa/pip), ([MIT license](https://github.com/pypa/pip/blob/main/LICENSE.txt)).

Beyond that, pyoomph makes use of the following libraries at runtime. During installation with `pip`, many (but not all) of these libraries are automatically fetched as requirements.

- [`python core libraries`](https://www.python.org/), ([PSF license](https://docs.python.org/3/license.html)).
- [`numpy`](https://numpy.org/), ([BSD license](https://numpy.org/doc/stable/license.html)).
- [`pygmsh`](https://github.com/nschloe/pygmsh), ([GPL v3 license](https://github.com/nschloe/pygmsh/blob/main/LICENSE.txt)).
- [`gmsh`](https://gmsh.info/), ([GPL v2 or later license](https://gmsh.info/LICENSE.txt)).
- [`meshio`](https://github.com/nschloe/meshio), ([MIT license](https://github.com/nschloe/meshio/blob/main/LICENSE.txt)).
- [`mpi4py`](https://github.com/mpi4py/mpi4py/), ([BSD 2-Clause "Simplified" license](https://github.com/erdc/mpi4py/blob/master/LICENSE.txt)).
- [`scipy`](https://github.com/scipy/scipy), ([BSD-3-Clause license](https://github.com/scipy/scipy/blob/main/LICENSES_bundled.txt)).
- [`matplotlib`](https://github.com/matplotlib/matplotlib), ([PSF-based license](https://matplotlib.org/stable/users/project/license.html)).
- [`mkl`](https://pypi.org/project/mkl/), ([Intel Simplified Software license](https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license.pdf)).
- [`petsc`](https://petsc.org/release/) and [`petsc4py`](https://petsc.org/release/petsc4py/), ([BSD 2-Clause license](https://petsc.org/release/install/license/)).
- [`slepc`](https://slepc.upv.es/) and [`slepc4py`](https://gitlab.com/slepc/slepc), ([BSD 2-Clause license](https://slepc.upv.es/contact/copy.htm)).
- [`vtk`](https://vtk.org/), ([BSD 3-clause license](https://vtk.org/about/)).
- [`paraview`](https://www.paraview.org/), ([BSD 3-clause license](https://www.paraview.org/license/)). Only used for the included [`Paraview filter for visualizing azimuthal perturbations`](https://github.com/pyoomph/pyoomph/blob/main/pyoomph/paraview/pyoomph_eigen_extrusion_filter.py).
- [`setuptools`](https://github.com/pypa/setuptools) ([MIT license](https://github.com/pypa/setuptools?tab=MIT-1-ov-file#readme)) is used for installation, wheel generation and to invoke the system's C compiler
- [`cibuildwheel`](https://cibuildwheel.pypa.io), ([BSD 2-Clause license](https://github.com/pypa/cibuildwheel?tab=License-1-ov-file#readme)) is used to compile the provided wheels
- [`pybind11-stubgen`](https://github.com/sizmailov/pybind11-stubgen), ([BSD 3-Clause license](https://github.com/sizmailov/pybind11-stubgen?tab=License-1-ov-file#readme)) is used to generate python stubs from the C++ core
- [`tccbox`](https://github.com/metab0t/tccbox) used as wrapper for the [`TinyC`](https://bellard.org/tcc/) compiler, ([LPGL 2 or later license](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html))

Be aware that some of these libraries can have further dependencies.

## Authors and acknowledgements

pyoomph was founded in 2021 by [**Christian Diddens**](https://github.com/cdiddens). Later, [**Duarte Rocha**](https://github.com/duarterocha) joined the team.

The authors gratefully acknowledge financial support by the Industrial Partnership Programme `Fundamental Fluid Dynamics Challenges in Inkjet Printing` of the Netherlands Organisation for Scientific Research (NWO) & High Tech Systems and Materials (HTSM), co-financed by Canon Production Printing Netherlands B.V., IamFluidics B.V., TNO Holst Centre, University of Twente, Eindhoven University of Technology and Utrecht University.

## Contributing

If you want to contribute by e.g. adding new equations, meshes, problems, materials or additional features, get in contact with us or send us a pull request.
If you encounter a bug, please also let us know at c.diddens@utwente.nl or d.rocha@utwente.nl.

## How to cite
At the moment, just cite the following preprint for pyoomph:

> Christian Diddens and Duarte Rocha, `Bifurcation tracking on moving meshes and with consideration of azimuthal symmetry breaking instabilities`, [arXiv:2312.11416](https://arxiv.org/abs/2312.11416), (2023).

Please mention that pyoomph is based on [`oomph-lib`](http://www.oomph-lib.org) and [`GiNaC`](https://www.ginac.de/), i.e. also cite at least:

> **oomph-lib**: M. Heil, A. L. Hazel, `oomph-lib - An Object-oriented multi-physics finite-element library`, Lect. Notes Comput. Sci. Eng. **53**, 19-49, (2006), [doi:10.1007/3-540-34596-5_2](https://dx.doi.org/10.1007/3-540-34596-5_2).

> **GiNaC**: C. Bauer, A. Frink, R. Kreckel, `Introduction to the GiNaC framework for symbolic computation within the C++ programming language`, J.
Symb. Comput. **33**(1), 1-12, (2002), [doi:10.1006/jsco.2001.0494](https://dx.doi.org/10.1006/jsco.2001.0494).


### <a id="cite"></a>Citing of material properties and activity models
- pyoomph includes some parameters of the [AIOMFAC](http://www.aiomfac.caltech.edu/) activity model in [pyoomph/materials/UNIFAC/aiomfac.py](https://github.com/pyoomph/pyoomph/blob/main/pyoomph/materials/UNIFAC/aiomfac.py). These are based on the source code of [AIOMFAC](https://github.com/andizuend/AIOMFAC) ([GPL v3 license](https://github.com/andizuend/AIOMFAC/blob/master/LICENSE)). Cite the [relevant scientific publications](https://aiomfac.lab.mcgill.ca/citation.html) when publishing results based on the AIOMFAC activity model.
- Alternatively, you can choose the [original UNIFAC model](https://www.ddbst.com/published-parameters-unifac.html) or the [modified UNIFAC model (Dortmund)](https://www.ddbst.com/PublishedParametersUNIFACDO.html). In that case, cite the publications listed under these links if you use these models. 
- For more accurate results, it is advised to obtain the updated parameters for the [UNIFAC Consortium](https://unifac.ddbst.com/unifac_.html). Such updated parameter sets can be implemented by hand following the templates for the free parameters in [pyoomph/materials/UNIFAC/](https://github.com/pyoomph/pyoomph/tree/main/pyoomph/materials/UNIFAC). The usage of such updated parameters of the UNIFAC Consortium are **restricted to the terms of use** of the UNIFAC Consortium.
- When using the material properties from [pyoomph/materials/default_materials.py](https://github.com/pyoomph/pyoomph/blob/main/pyoomph/materials/default_materials.py), please have a look at the comments in this file to cite the correct papers.
