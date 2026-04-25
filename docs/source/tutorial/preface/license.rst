License and acknowledgements
============================

For pyoomph, the conditions of the `GNU General Public License 3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`__ apply:

.. container:: licensebox

   .. code-block:: text

      pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
      Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>.

A `copy of the license <https://github.com/pyoomph/pyoomph/blob/main/COPYING>`__ can be found in distribution. Mind also the following licenses

#. pyoomph contains code taken from other authors/projects:

   -  In `src/thirdparty/oomph-lib/include <https://github.com/pyoomph/pyoomph/tree/main/src/thirdparty/oomph-lib/include>`__, you find the necessary main files of `oomph-lib <http://www.oomph-lib.org>`__, `[LGPL v2.1 or later license] <https://github.com/oomph-lib/oomph-lib/blob/main/LICENCE>`__. Minor modifications as mentioned in `src/thirdparty/INFO_oomph-lib <https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/INFO_oomph-lib>`__ had to be made. Furthermore, code parts of these oomph-lib files had been copied to corresponding derived classes of pyoomph.

   -  A copy of the header-only library `nanoflann <https://github.com/jlblancoc/nanoflann>`__ is located in `src/thirdparty/nanoflann.hpp <https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/nanoflann.hpp>`__, `[BSD license] <https://github.com/jlblancoc/nanoflann/blob/master/COPYING>`__:

      .. container:: licensebox

         .. code-block:: text

            Software License Agreement (BSD License)

            Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
            Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
            Copyright 2011 Jose L. Blanco (joseluisblancoc@gmail.com). All rights reserved.

            THE BSD LICENSE

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions
            are met:

            1. Redistributions of source code must retain the above copyright
               notice, this list of conditions and the following disclaimer.
            2. Redistributions in binary form must reproduce the above copyright
               notice, this list of conditions and the following disclaimer in the
               documentation and/or other materials provided with the distribution.

            THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
            IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
            OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
            IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
            INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
            NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
            DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
            THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
            THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   -  A copy of the header-only library `delaunator-cpp <https://github.com/delfrrr/delaunator-cpp>`__ is located in `src/thirdparty/delaunator.hpp <https://github.com/pyoomph/pyoomph/blob/main/src/thirdparty/delaunator.hpp>`__, `[MIT license] <https://github.com/delfrrr/delaunator-cpp/blob/master/LICENSE>`__:

      .. container:: licensebox

         .. code-block:: text

            MIT License

            Copyright (c) 2018 Volodymyr Bilonenko

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.

   -  The file `src/pyginacstruct.hpp <https://github.com/pyoomph/pyoomph/blob/main/src/pyginacstruct.hpp>`__ is strongly based on the file `structure.h <https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=ginac/structure.h;hb=HEAD>`__ of `GiNaC <https://www.ginac.de/>`__, `[GPL v2 or later license] <https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=COPYING;hb=HEAD>`__.

   -  A copy of the library `Project Nayuki/smallest enslosing circle <https://www.nayuki.io/page/smallest-enclosing-circle>`__, `[LGPL v3 or later license] <https://github.com/nayuki/Nayuki-web-published-code/blob/master/smallest-enclosing-circle/COPYING.LESSER.txt>`__ is added (after adding type specifications) to `pyoomph/utils/smallest_circle.py <https://github.com/pyoomph/pyoomph/blob/main/pyoomph/utils/smallest_circle.py>`__.

   -  Also, when using materials or the thermodynamic activity models AIOMFAC, original UNIFAC or modified UNIFAC (Dortmund), :ref:`please cite the relevant publications <secboxunifacinfo>`.

   The third-party licenses/acknowledgement files can be also found in `src/thirdparty <https://github.com/pyoomph/pyoomph/tree/main/src/thirdparty>`__.

#. During compilation, pyoomph includes/links against or makes use of the following libraries:

   -  `GiNaC <https://www.ginac.de/>`__, `[GPL v2 or later license] <https://www.ginac.de/ginac.git/?p=ginac.git;a=blob_plain;f=COPYING;hb=HEAD>`__, also statically linked in the distribution as python wheels

   -  `CLN <https://www.ginac.de/CLN>`__, `[GPL v2 or later license] <https://www.ginac.de/CLN/cln.git/?p=cln.git;a=blob_plain;f=COPYING;hb=HEAD>`__, also statically linked in the distribution as python wheels

   -  MPI, depending on the system e.g. `OpenMPI <https://www.open-mpi.org>`__ `[3-clause BSD license] <https://www.open-mpi.org/community/license.php>`__, `MPICH <https://www.mpich.org/>`__ `[MPICH license] <https://github.com/pmodels/mpich/blob/main/COPYRIGHT>`__, `Microsoft MPI <https://github.com/Microsoft/Microsoft-MPI>`__ `[MIT license] <https://github.com/microsoft/Microsoft-MPI/blob/master/LICENSE.txt>`__, note that MPI support is experimental and deactivated in the python wheels

   -  `python3.8\ + <https://www.python.org/>`__, `[PSF license] <https://docs.python.org/3/license.html>`__, also dynamically linked in the distribution as python wheels

   -  `pybind11 <https://github.com/pybind/pybind11>`__, `[BSD-style license] <https://github.com/sizmailov/pybind11-stubgen/blob/master/LICENSE>`__, also statically linked in the distribution as python wheels

   -  `pybind11-stubgen <https://github.com/sizmailov/pybind11-stubgen>`__, `[BSD-style license] <https://github.com/sizmailov/pybind11-stubgen/blob/master/LICENSE>`__

   -  `pip <https://github.com/pypa/pip>`__, `[MIT license] <https://github.com/pypa/pip/blob/main/LICENSE.txt>`__

#. Beyond that, pyoomph makes use of the following libraries at runtime. During installation with pip, many (but not all) of these libraries are automatically fetched as requirements.

   -  `python core libraries <https://www.python.org/>`__, `[PSF license] <https://docs.python.org/3/license.html>`__

   -  `numpy <https://numpy.org/>`__, `[BSD license] <https://numpy.org/doc/stable/license.html>`__

   -  `pygmsh <https://github.com/nschloe/pygmsh>`__, `[GPL v3 license] <https://github.com/nschloe/pygmsh/blob/main/LICENSE.txt>`__

   -  `gmsh <https://gmsh.info/>`__, `[GPL v2 or later license] <https://gmsh.info/LICENSE.txt>`__

   -  `meshio <https://github.com/nschloe/meshio>`__, `[MIT license] <https://github.com/nschloe/meshio/blob/main/LICENSE.txt>`__

   -  `mpi4py <https://github.com/mpi4py/mpi4py/>`__, `[BSD 2-Clause "Simplified" license] <https://github.com/erdc/mpi4py/blob/master/LICENSE.txt>`__

   -  `scipy <https://github.com/scipy/scipy>`__, `[BSD-3-Clause license] <https://github.com/scipy/scipy/blob/main/LICENSES_bundled.txt>`__

   -  `matplotlib <https://github.com/matplotlib/matplotlib>`__, `[PSF-based license] <https://matplotlib.org/stable/users/project/license.html>`__

   -  `mkl <https://pypi.org/project/mkl/>`__, `[Intel Simplified Software license] <https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license.pdf>`__

   -  `petsc <https://petsc.org/release/>`__ and `petsc4py <https://petsc.org/release/petsc4py/>`__, `[BSD 2-Clause license] <https://petsc.org/release/install/license>`__

   -  `slepc <https://slepc.upv.es/>`__ and `slepc4py <https://gitlab.com/slepc/slepc>`__, `[BSD 2-Clause license] <https://slepc.upv.es/contact/copy.htm>`__

   -  `vtk <https://vtk.org/>`__, `[BSD 3-clause license] <https://vtk.org/about/>`__

   -  `paraview <https://www.paraview.org/>`__, `[BSD 3-clause license] <https://www.paraview.org/license/>`__
   
   -  `setuptools <https://github.com/pypa/setuptools>`__, `[MIT license] <https://github.com/pypa/setuptools?tab=MIT-1-ov-file#readme>`__
   
   -  `pybind11-stubgen <https://github.com/sizmailov/pybind11-stubgen>`__, `[BSD 3-Clause license] <https://github.com/sizmailov/pybind11-stubgen?tab=License-1-ov-file#readme>`__ is used to generate python stubs from the C++ core
   
   -  `cibuildwheel <https://cibuildwheel.pypa.io>`__, `[BSD 2-Clause license] <https://github.com/pypa/cibuildwheel?tab=License-1-ov-file#readme>`__ is used to compile the provided wheels   
   
   -  `tccbox <https://github.com/metab0t/tccbox>`__ used to invoke the `TinyC <https://bellard.org/tcc/>`__ compiler
   
   
   

   Be aware that some of these libraries can have further dependencies.


Acknowledgements
----------------

The authors gratefully acknowledge financial support by the Industrial Partnership Programme Fundamental Fluid Dynamics Challenges in Inkjet Printing of the Netherlands Organisation for Scientific Research (NWO) & High Tech Systems and Materials (HTSM), co-financed by Canon Production Printing Netherlands B.V., IamFluidics B.V., TNO Holst Centre, University of Twente, Eindhoven University of Technology and Utrecht University. This work was supported by an Industrial Partnership Programme, High Tech Systems and Materials (HTSM), of the Netherlands Organisation for Scientific Research (NWO); a funding for public-private partnerships (PPS) of the Netherlands Enterprise Agency (RVO) and the Ministry of Economic Affairs (EZ); Canon Production Printing Netherlands B.V.; and the University of Twente.
