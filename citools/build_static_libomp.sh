#!/usr/bin/env bash

set -euo pipefail

# A static LLVM OpenMP runtime for the macOS wheels.
#
# Why this exists rather than `brew install libomp`:
#
#   * Homebrew bottles are built per macOS release, so the libomp on the macos-14 runner requires
#     macOS 14 and the one on macos-15-intel requires 15. delocate checks that when it bundles a
#     dylib and refuses, and the only way to satisfy it would be to raise the wheel's
#     MACOSX_DEPLOYMENT_TARGET to match - i.e. to drop every user on an older mac in exchange for an
#     opt-in threading flag. Built here, libomp gets the SAME deployment target as the wheel.
#
#   * Static (LIBOMP_ENABLE_SHARED=OFF), so nothing lands in the wheel's .dylibs at all and there is
#     no second libomp.dylib for dyld to reason about. Note what this does NOT solve: libomp
#     registers itself per process through an environment variable, so a statically linked copy is
#     still a "duplicate runtime" if the user also imports something that ships its own libomp
#     (PyTorch does). That is the OMP: Error #15 abort, and pyoomph/__init__.py defuses it on Darwin.
#     It is only reachable at all once --omp N>1 actually starts our runtime.
#
# Linux and Windows do not need any of this: libgomp comes with GCC on both, and auditwheel/
# delvewheel bundle it without complaint.
#
# Usage:  ./citools/build_static_libomp.sh [install-prefix]
# Result: <prefix>/lib/libomp.a and <prefix>/include/omp.h, i.e. a prefix to hand to CMake as
#         -DOpenMP_ROOT=<prefix>.

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build_static_libomp.sh is only for macOS (Linux/Windows use the toolchain's libgomp)" >&2
  exit 1
fi

PREFIX="${1:-${PWD}/libomp_static/install}"
mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"

# Any recent LLVM release will do - the __kmpc_* ABI that clang's -fopenmp emits calls into is
# stable, so the runtime version does not have to match the AppleClang that compiles pyoomph.
LIBOMP_VERSION="${LIBOMP_VERSION:-19.1.7}"

ARCH="$(uname -m)"
# arm64 macOS starts at 11.0; anything lower is silently promoted by the toolchain, and having the
# two disagree would only produce confusing warnings. On x86_64 follow the wheel's own target.
if [[ "${ARCH}" == "arm64" ]]; then
  DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
  if [[ "${DEPLOYMENT_TARGET%%.*}" -lt 11 ]]; then DEPLOYMENT_TARGET="11.0"; fi
else
  DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.13}"
fi

WORK="$(cd "$(dirname "${PREFIX}")" && pwd)/src"
rm -rf "${WORK}"
mkdir -p "${WORK}"
cd "${WORK}"

BASE="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LIBOMP_VERSION}"

download() {
  local url="$1" out="$2"
  curl -fsSL --retry 5 --retry-delay 2 "${url}" -o "${out}"
}

echo "== fetching LLVM ${LIBOMP_VERSION} openmp sources"
download "${BASE}/openmp-${LIBOMP_VERSION}.src.tar.xz" openmp.tar.xz
# Since LLVM 15 the runtimes cannot be configured on their own: they include LLVM's shared CMake
# modules from a sibling directory called "cmake", which ships as its own tarball. Without it the
# configure step fails on a missing include(base-config-ix), which reads like a broken checkout.
download "${BASE}/cmake-${LIBOMP_VERSION}.src.tar.xz" cmake.tar.xz

tar -xf openmp.tar.xz
tar -xf cmake.tar.xz
mv "openmp-${LIBOMP_VERSION}.src" openmp
mv "cmake-${LIBOMP_VERSION}.src" cmake

echo "== building libomp.a for ${ARCH}, deployment target ${DEPLOYMENT_TARGET}"
cmake -S openmp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_OSX_ARCHITECTURES="${ARCH}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DLIBOMP_ENABLE_SHARED=OFF \
  -DLIBOMP_INSTALL_ALIASES=OFF \
  -DLIBOMP_OMPT_SUPPORT=OFF \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DOPENMP_ENABLE_TESTING=OFF

cmake --build build --parallel "$(sysctl -n hw.ncpu)"
cmake --install build

test -f "${PREFIX}/lib/libomp.a" || { echo "no libomp.a in ${PREFIX}/lib" >&2; exit 1; }
test -f "${PREFIX}/include/omp.h" || { echo "no omp.h in ${PREFIX}/include" >&2; exit 1; }
# A dylib here would defeat the point: CMake's find_library prefers .dylib over .a, so a stray one
# would be what the wheel links against, and delocate would then bundle it.
if compgen -G "${PREFIX}/lib/libomp*.dylib" > /dev/null; then
  echo "a shared libomp was installed as well; the wheel would pick that one up" >&2
  exit 1
fi

echo "== libomp ready in ${PREFIX}"
otool -l "${PREFIX}/lib/libomp.a" 2>/dev/null | grep -A3 -m1 LC_BUILD_VERSION || true
