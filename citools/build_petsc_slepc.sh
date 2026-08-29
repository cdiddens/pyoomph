#!/usr/bin/env bash
# Build the two PETSc/SLEPc installations the tutorial harness and the MPI tests want - one with
# real scalars, one with complex - into a single relocatable-by-path tree, and tar it up for
# .github/workflows/prebuild_petsc_slepc.yml.
#
# The layout is dictated by citools/test_all_tutorial_scripts.py, which puts
# $PETSC_DIR/$PETSC_ARCH_{REAL,COMPLEX}/lib on PYTHONPATH and nothing else:
#
#     $PETSC_PREFIX/mpi/         MPICH, built once and shared by both PETSc builds
#     $PETSC_PREFIX/real/lib     libpetsc, libslepc, MUMPS, petsc4py, slepc4py  (real scalars)
#     $PETSC_PREFIX/complex/lib  the same, complex scalars
#     $PETSC_PREFIX/env.sh       exports PETSC_DIR/PETSC_ARCH_REAL/PETSC_ARCH_COMPLEX/PATH/...
#
# so PETSC_DIR=$PETSC_PREFIX, PETSC_ARCH_REAL=real, PETSC_ARCH_COMPLEX=complex.
#
# NOT relocatable: PETSc bakes the prefix into lib/petsc/conf/petscvariables, into the ELF RPATHs
# and (macOS) into the install_names. Unpack the tarball at exactly $PETSC_PREFIX or nothing
# imports. That is why the default is an absolute path under /opt rather than something next to the
# checkout - the CI runners are identical images, so the same path exists on the consumer.
#
# Python-version specific: petsc4py/slepc4py are compiled extension modules, so a tree built for
# 3.12 is useless to 3.13. The artifact name carries the cpXY tag for that reason.
#
# Usage:  PETSC_PREFIX=/opt/pyoomph-petsc ./citools/build_petsc_slepc.sh

set -euo pipefail

PETSC_VERSION="${PETSC_VERSION:-3.25.4}"
SLEPC_VERSION="${SLEPC_VERSION:-3.25.1}"
MPICH_VERSION="${MPICH_VERSION:-4.2.3}"

PETSC_PREFIX="${PETSC_PREFIX:-/opt/pyoomph-petsc}"
WORKDIR="${WORKDIR:-${PWD}/petsc_build}"
PYTHON="${PYTHON:-$(command -v python3)}"
NPROC="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4 )"

# ScaLAPACK 2.2 opens with cmake_minimum_required(VERSION 2.8), and CMake 4 - which Homebrew now
# ships, so the macOS runners have it - removed compatibility below 3.5 outright. This is the
# escape hatch CMake itself names in the error; it only supplies a default for projects that ask
# for less, so it does not affect anything else PETSc downloads.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

echo "==> PETSc ${PETSC_VERSION}, SLEPc ${SLEPC_VERSION}, MPICH ${MPICH_VERSION}"
echo "==> prefix ${PETSC_PREFIX}, python $("${PYTHON}" -c 'import sys;print(sys.version.split()[0],sys.executable)')"

# The prefix has to be writable without sudo for the rest of the script (PETSc's `make install`
# runs as the build user), so take ownership once up front. Only escalate when the plain mkdir
# fails - a local test run points PETSC_PREFIX somewhere in $HOME and has no sudo at all.
if ! mkdir -p "${PETSC_PREFIX}" 2>/dev/null; then
  sudo mkdir -p "${PETSC_PREFIX}"
  sudo chown -R "$(id -u):$(id -g)" "${PETSC_PREFIX}"
fi
[ -w "${PETSC_PREFIX}" ] || { echo "${PETSC_PREFIX} is not writable" >&2; exit 1; }
mkdir -p "${WORKDIR}"

# PETSc's configure refuses --with-petsc4py without numpy and setuptools in the target interpreter,
# and the hosted-tool Pythons on the runners are bare.
#
# Cython is pinned below 3.1 rather than left out: petsc4py always regenerates PETSc.c from the
# .pyx, and its setup_requires fetches a Cython of its own if none is installed, which is how it
# got 3.3.0. petsc4py 3.22's conf/cyautodoc.py calls ExpressionWriter.emit_string, removed in
# Cython 3.1, so that build ends in "Compiler crash in ExpressionWriter". Installing 3.0.x
# satisfies its "Cython >= 3.0.0" and stops it fetching anything. Droppable once PETSc is bumped
# past 3.23, where cyautodoc.py no longer uses that API.
#
# setuptools is pinned below 81 for the same kind of reason: petsc4py's conf/confpetsc.py does
# "from distutils.util import execute" - which on 3.12 is setuptools' shim - and calls it with
# dry_run=. Measured here: 80.9.0 still takes that argument, 81.0.0 does not, so from 81 on the
# build dies with "execute() got an unexpected keyword argument 'dry_run'" after having compiled
# and linked the extension. Droppable together with the Cython pin on a PETSc bump.
#
# Both pins are for petsc4py 3.22 and older only, so they are applied by version rather than
# always: on a newer PETSc they would hold the toolchain back for no reason.
case "${PETSC_VERSION}" in
  3.1?.*|3.2[0-2].*) BUILD_PINS=( "setuptools<81" "cython<3.1" ) ;;
  *)                 BUILD_PINS=( "setuptools" "cython" ) ;;
esac
"${PYTHON}" -m pip install --upgrade --quiet numpy "${BUILD_PINS[@]}"

fetch() { # url sha-unchecked-tarball -> extracted dir name on stdout
  local url="$1" tarball="${WORKDIR}/$(basename "$1")"
  [ -f "${tarball}" ] || curl -fsSL --retry 3 -o "${tarball}" "${url}"
  tar xzf "${tarball}" -C "${WORKDIR}"
}

# ---------------------------------------------------------------------------------------------
# MPI, built once and shared
# ---------------------------------------------------------------------------------------------
# Deliberately not the system MPI (apt openmpi / brew open-mpi): the runner images are refreshed
# monthly, so a tree built against one image's MPI can stop importing on the next one. A pinned
# MPICH inside the tarball makes the artifact self-contained and the two PETSc builds share it,
# which matters because pyoomph launches mpirun from $PETSC_PREFIX/mpi/bin.
MPI_PREFIX="${PETSC_PREFIX}/mpi"
if [ ! -x "${MPI_PREFIX}/bin/mpicc" ]; then
  echo "==> Building MPICH ${MPICH_VERSION}"
  fetch "https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz"
  pushd "${WORKDIR}/mpich-${MPICH_VERSION}" >/dev/null
  # ch3:nemesis rather than the ch4/ofi default: ch4 probes for libfabric providers that are absent
  # on the runners and falls over at MPI_Init inside a container.
  ./configure --prefix="${MPI_PREFIX}" \
      --with-device=ch3:nemesis \
      --disable-dependency-tracking \
      --enable-shared --disable-static \
      FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch
  make -j"${NPROC}"
  make install
  popd >/dev/null
fi
export PATH="${MPI_PREFIX}/bin:${PATH}"

# mpi4py has to exist, built against THIS MPICH, before PETSc ever configures petsc4py: petsc4py's
# setup takes its MPI compiler from mpi4py.get_config() whenever an mpi4py is importable, so a
# system one (Ubuntu ships an OpenMPI build) would decide it for us. Ours goes first on PYTHONPATH
# to remove the choice.
#
# Not PETSc's --download-mpi4py: PETSc 3.22 fetches mpi4py 4.0.1 and drives it with `setup.py
# clean`, which that version no longer implements, so configure dies with "Error cleaning mpi4py".
#
# LDFLAGS repeats what mpicc already passes because distutils puts the interpreter's own
# -L/usr/lib/<triplet> ahead of the wrapper's flags: on a distro that keeps an OpenMPI libmpi.so
# in the default library path, -lmpi then resolves to that one and the module ends up bound to
# libmpi.so.40 while libpetsc uses MPICH's libmpi.so.12. check_links_mpich below is what makes
# that fail here rather than as a segfault inside PetscInitialize much later.
MPI_PY="${MPI_PREFIX}/python"
if [ ! -d "${MPI_PY}/mpi4py" ]; then
  echo "==> Building mpi4py against ${MPI_PREFIX}"
  MPICC="${MPI_PREFIX}/bin/mpicc" \
  LDFLAGS="-L${MPI_PREFIX}/lib -Wl,-rpath,${MPI_PREFIX}/lib ${LDFLAGS:-}" \
    "${PYTHON}" -m pip install --no-binary=mpi4py --no-cache-dir --target "${MPI_PY}" mpi4py
fi
export PYTHONPATH="${MPI_PY}${PYTHONPATH:+:${PYTHONPATH}}"

# Which MPI a module ended up against is invisible until it crashes, so assert it. MPICH is
# libmpi.so.12/libmpi.12.dylib; anything resolving a bare libmpi.so.40 has picked up OpenMPI.
#
# This is not hypothetical: it happened to all three modules during this script's own bring-up.
# Nothing complains until PetscInitialize resolves PMPI_Comm_set_errhandler out of the wrong
# library and segfaults on import, with a traceback that blames petsc4py.
check_links_mpich() { # path-to-extension-module
  local mod="$1" deps
  if [ "$(uname)" = "Darwin" ]; then deps="$(otool -L "${mod}")"; else deps="$(ldd "${mod}")"; fi
  if echo "${deps}" | grep -q "libmpi\\.so\\.40\\|libmpi\\.40\\.dylib\\|openmpi"; then
    echo "${mod} links a foreign MPI, not the MPICH in ${MPI_PREFIX}:" >&2
    echo "${deps}" | grep -i mpi >&2
    exit 1
  fi
}
check_links_mpich "$(echo "${MPI_PY}"/mpi4py/MPI*.so)"

# ---------------------------------------------------------------------------------------------
# PETSc + SLEPc, once per scalar type
# ---------------------------------------------------------------------------------------------
fetch "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz"
fetch "https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz"

build_one() { # arch-name scalar-type
  local arch="$1" scalar="$2" prefix="${PETSC_PREFIX}/$1"
  echo "==> Building PETSc (${scalar} scalars) into ${prefix}"

  # A fresh copy per scalar type: PETSc's in-tree configure state does not survive being
  # reconfigured for a different scalar type, and the two builds run in the same workspace.
  rm -rf "${WORKDIR}/petsc-${arch}"
  cp -a "${WORKDIR}/petsc-${PETSC_VERSION}" "${WORKDIR}/petsc-${arch}"
  pushd "${WORKDIR}/petsc-${arch}" >/dev/null

  # --with-fortran-bindings=0 drops PETSc's own Fortran interface (nothing in pyoomph calls it) but
  # keeps the Fortran compiler, which MUMPS and ScaLAPACK are written in.
  # BLAS: Accelerate on macOS (picked up automatically), apt's libopenblas-dev on Linux - see the
  # consumer step in the workflow, which installs the same runtime package.
  # configure.log holds the actual reason; the terminal summary is a one-liner like "Error cleaning
  # mpi4py". Dumping the tail here is the difference between one CI round-trip and three.
  if ! "${PYTHON}" ./configure \
      --prefix="${prefix}" \
      --with-scalar-type="${scalar}" \
      --with-debugging=0 \
      COPTFLAGS='-O2 -g0' CXXOPTFLAGS='-O2 -g0' FOPTFLAGS='-O2 -g0' \
      --with-shared-libraries=1 \
      --with-fortran-bindings=0 \
      --with-x=0 \
      --with-mpi-dir="${MPI_PREFIX}" \
      --download-scalapack \
      --download-mumps \
      --with-petsc4py=1 \
      --with-64-bit-indices=0
  then
    echo "=== configure.log (last 300 lines) ==============================================" >&2
    tail -300 configure.log >&2
    exit 1
  fi
  # PETSc 3.23+ builds petsc4py during `make install` and writes its output to a log of its own,
  # reporting only "Check .../petsc4py.build.log" on the terminal. Dump it, or the failure is a
  # dead end.
  dump_binding_logs() {
    local log
    for log in "${PETSC_ARCH_DIR}"/lib/petsc/conf/*4py*.log; do
      [ -f "${log}" ] || continue
      echo "=== ${log} (last 200 lines) =====================================================" >&2
      tail -200 "${log}" >&2
    done
  }
  PETSC_ARCH_DIR="$(echo "${PWD}"/arch-*-c-opt)"
  if ! make -j"${NPROC}" all; then dump_binding_logs; exit 1; fi
  if ! make install; then dump_binding_logs; exit 1; fi
  popd >/dev/null

  echo "==> Building SLEPc (${scalar} scalars) into ${prefix}"
  rm -rf "${WORKDIR}/slepc-${arch}"
  cp -a "${WORKDIR}/slepc-${SLEPC_VERSION}" "${WORKDIR}/slepc-${arch}"
  pushd "${WORKDIR}/slepc-${arch}" >/dev/null
  # SLEPc is configured against the *installed* PETSc, hence an empty PETSC_ARCH. SLEPC_DIR has to
  # be passed to `make` as well: SLEPc's makefiles do not derive it from the working directory, and
  # an unset one expands to //lib/slepc/conf/slepcvariables, which does not exist.
  # ${prefix}/lib on PYTHONPATH because --with-slepc4py imports petsc4py during configure, and
  # petsc4py has just been installed there ("ERROR: Cannot import petsc4py" otherwise).
  if ! PETSC_DIR="${prefix}" PETSC_ARCH="" SLEPC_DIR="${PWD}" \
       PYTHONPATH="${prefix}/lib:${MPI_PY}" \
       "${PYTHON}" ./configure --prefix="${prefix}" --with-slepc4py=1; then
    echo "=== SLEPc configure.log (last 300 lines) ========================================" >&2
    tail -300 configure.log >&2
    exit 1
  fi
  PETSC_DIR="${prefix}" PETSC_ARCH="" SLEPC_DIR="${PWD}" PYTHONPATH="${prefix}/lib:${MPI_PY}" make -j"${NPROC}"
  PETSC_DIR="${prefix}" PETSC_ARCH="" SLEPC_DIR="${PWD}" PYTHONPATH="${prefix}/lib:${MPI_PY}" make install
  popd >/dev/null

  # A copy of the one mpi4py next to petsc4py, so that the single PYTHONPATH entry the harness adds
  # ($PETSC_DIR/$ARCH/lib) brings all three modules with it. It does not depend on the scalar type,
  # hence a copy of the shared build rather than a second one.
  cp -a "${MPI_PY}"/mpi4py* "${prefix}/lib/"

  check_links_mpich "$(echo "${prefix}"/lib/petsc4py/lib/PETSc*.so)"
  check_links_mpich "$(echo "${prefix}"/lib/slepc4py/lib/SLEPc*.so)"
}

build_one real    real
build_one complex complex

# ---------------------------------------------------------------------------------------------
# Slim the tree down
# ---------------------------------------------------------------------------------------------
# Measured on a 3.22 opt build: 133 MB raw -> 82 MB -> 36 MB gzipped. Almost all of the win is
# stripping petsc4py's extension module (29 MB of debug symbols by itself) and dropping the static
# archives, which only a build against PETSc would want and nothing here links.
echo "==> Stripping"
find "${PETSC_PREFIX}" -name '*.a' -delete
find "${PETSC_PREFIX}" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "${PETSC_PREFIX}" -path '*/lib/petsc/conf/*.log' -delete
rm -rf "${PETSC_PREFIX}"/*/share/petsc/examples "${PETSC_PREFIX}"/*/share/slepc/examples
if [ "$(uname)" = "Darwin" ]; then
  find "${PETSC_PREFIX}" \( -name '*.so' -o -name '*.dylib' \) -type f -exec strip -x {} + || true
else
  find "${PETSC_PREFIX}" -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true
fi

# ---------------------------------------------------------------------------------------------
# The environment the consumer sources
# ---------------------------------------------------------------------------------------------
cat > "${PETSC_PREFIX}/env.sh" <<ENV_EOF
# Source this (or feed it to \$GITHUB_ENV) before running the pytest suite or the tutorial harness.
# The harness itself picks real vs complex per script from PETSC_ARCH_REAL/PETSC_ARCH_COMPLEX;
# PYTHONPATH below points at the REAL build so that a plain 'python3 script.py' also works.
export PETSC_DIR="${PETSC_PREFIX}"
export PETSC_ARCH_REAL=real
export PETSC_ARCH_COMPLEX=complex
export PATH="${PETSC_PREFIX}/mpi/bin:\${PATH}"
export PYTHONPATH="${PETSC_PREFIX}/real/lib\${PYTHONPATH:+:\${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PETSC_PREFIX}/mpi/lib:${PETSC_PREFIX}/real/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export DYLD_LIBRARY_PATH="${PETSC_PREFIX}/mpi/lib:${PETSC_PREFIX}/real/lib\${DYLD_LIBRARY_PATH:+:\${DYLD_LIBRARY_PATH}}"
ENV_EOF

# ---------------------------------------------------------------------------------------------
# Verify before packing, not after downloading
# ---------------------------------------------------------------------------------------------
for arch in real complex; do
  echo "==> Checking ${arch}"
  PYTHONPATH="${PETSC_PREFIX}/${arch}/lib" "${PYTHON}" - "${arch}" <<'CHECK_EOF'
import sys, numpy
want_complex = sys.argv[1] == "complex"
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
is_complex = PETSc.ScalarType is numpy.complex128
assert is_complex == want_complex, "%s build has ScalarType %s" % (sys.argv[1], PETSc.ScalarType)
# MUMPS is the whole reason for the ScaLAPACK/Fortran half of this build: check it is registered
# rather than trusting configure, because PETSc happily builds without it.
pc = PETSc.PC().create(); pc.setType("lu"); pc.setFactorSolverType("mumps")
print("%-8s ok: PETSc %s, SLEPc %s, mpi4py %s, MUMPS registered" %
      (sys.argv[1], PETSc.Sys.getVersion(), SLEPc.Sys.getVersion(), MPI.Get_version()))
CHECK_EOF
done

# ---------------------------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------------------------
# One tarball, not a directory upload: actions/upload-artifact zips what it is given and loses both
# symlinks (lib/ is full of libfoo.so -> libfoo.so.3.x chains, which would be duplicated) and the
# executable bit on mpiexec.
OUT="${OUT:-${PWD}/pyoomph-petsc.tar.gz}"
echo "==> Packing ${OUT}"
tar czf "${OUT}" -C "$(dirname "${PETSC_PREFIX}")" "$(basename "${PETSC_PREFIX}")"
ls -lh "${OUT}"
