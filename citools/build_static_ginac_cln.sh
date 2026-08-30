#!/usr/bin/env bash

set -euo pipefail

# Dedicated static prebuild for CLN + GiNaC (Unix runners)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_ginac_cln_versions.sh"
PYOOMPH_STATIC_GINAC_DIR="${PYOOMPH_STATIC_GINAC_DIR:-GiNaC_static}"
PYOOMPH_GINAC_CONFIGURE_OPTIONS="${PYOOMPH_GINAC_CONFIGURE_OPTIONS:-}"

mkdir -p "${PYOOMPH_STATIC_GINAC_DIR}"
ROOT_DIR="$(cd "${PYOOMPH_STATIC_GINAC_DIR}" && pwd)"
PREFIX="${ROOT_DIR}/install"

rm -rf "${ROOT_DIR}/cln" "${ROOT_DIR}/ginac" "${PREFIX}"
mkdir -p "${PREFIX}"

NPROC=4
if command -v nproc >/dev/null 2>&1; then
  NPROC="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  NPROC="$(sysctl -n hw.ncpu)"
fi

append_flag_if_missing() {
  local value="$1"
  local flag="$2"
  case " ${value} " in
    *" ${flag} "*)
      printf '%s' "${value}"
      ;;
    *)
      printf '%s %s' "${value}" "${flag}"
      ;;
  esac
}

# Force PIC so the produced static archives can be linked into shared libraries.
export CFLAGS="${CFLAGS:--O2 -g0 -DNDEBUG}"
export CXXFLAGS="${CXXFLAGS:--O2 -g0 -DNDEBUG}"
export CPPFLAGS="${CPPFLAGS:--O2 -g0 -DNDEBUG -DNO_ASM}"
export CFLAGS="$(append_flag_if_missing "${CFLAGS}" "-fPIC")"
export CXXFLAGS="$(append_flag_if_missing "${CXXFLAGS}" "-fPIC")"
export CPPFLAGS="$(append_flag_if_missing "${CPPFLAGS}" "-fPIC")"

CXX_COMPILER="${CXX:-g++}"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 5 --retry-delay 2 "${url}" -o "${out}"
  else
    wget --retry-connrefused --read-timeout=20 --timeout=15 --tries=40 "${url}" -O "${out}"
  fi
}

cd "${ROOT_DIR}"

download "https://www.ginac.de/CLN/cln-${CLN_VERSION}.tar.bz2" "cln-${CLN_VERSION}.tar.bz2"
tar -xjf "cln-${CLN_VERSION}.tar.bz2"
mv "cln-${CLN_VERSION}" cln

download "https://www.ginac.de/ginac-${GINAC_VERSION}.tar.bz2" "ginac-${GINAC_VERSION}.tar.bz2"
tar -xjf "ginac-${GINAC_VERSION}.tar.bz2"
mv "ginac-${GINAC_VERSION}" ginac

echo "BUILDING CLN"
cd "${ROOT_DIR}/cln"
./configure --without-gmp --disable-shared --enable-static --with-pic=yes --prefix "${PREFIX}" ${PYOOMPH_GINAC_CONFIGURE_OPTIONS}
make MAKEINFO=true -j "${NPROC}" install

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "BUILDING GINAC"
cd "${ROOT_DIR}/ginac"
# Deterministic term/hash ordering (see citools/patches/ginac-deterministic-*.patch and
# branch deterministic_codegen) - without this, a prebuilt GiNaC here would make pyoomph's
# JIT code cache (pyoomph/generic/jit_cache.py) unsafe to trust, since it would then have no
# way to tell this static prebuild apart from a genuinely unpatched system GiNaC.
bash "${SCRIPT_DIR}/patches/apply_ginac_patch.sh"
autoreconf -i -f
CLN_CFLAGS="-I${PREFIX}/include" CLN_LIBS="-L${PREFIX}/lib -l:libcln.a" \
  ./configure --disable-shared --enable-static --with-pic=yes --prefix "${PREFIX}" ${PYOOMPH_GINAC_CONFIGURE_OPTIONS}
make MAKEINFO=true -C ginac -j "${NPROC}" install

test -f "${PREFIX}/lib/libcln.a"
test -f "${PREFIX}/lib/libginac.a"

if ls "${PREFIX}/lib"/*.so "${PREFIX}/lib"/*.dylib "${PREFIX}/lib"/*.dll >/dev/null 2>&1; then
  echo "Unexpected shared libraries found in ${PREFIX}/lib"
  exit 1
fi

if command -v pkg-config >/dev/null 2>&1; then
  PKG_LIBS="$(PKG_CONFIG_PATH="${PKG_CONFIG_PATH}" pkg-config --static --libs ginac || true)"
  echo "pkg-config --static --libs ginac: ${PKG_LIBS}"
  if echo "${PKG_LIBS}" | grep -Eqi '(^|[[:space:]])-lgmp([[:space:]]|$)'; then
    echo "Detected gmp linkage in ginac static flags"
    exit 1
  fi
fi

# Validate that the static archives can be consumed while producing a shared lib.
SHARED_EXT="so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  SHARED_EXT="dylib"
fi
cat > "${ROOT_DIR}/ginac_shared_link_test.cpp" <<'EOF'
#include <ginac/ginac.h>

extern "C" int ginac_shared_link_test() {
  GiNaC::symbol x("x");
  GiNaC::ex expr = GiNaC::pow(x + 1, 2);
  return expr.nops();
}
EOF

"${CXX_COMPILER}" -std=c++17 -fPIC -I"${PREFIX}/include" -c "${ROOT_DIR}/ginac_shared_link_test.cpp" -o "${ROOT_DIR}/ginac_shared_link_test.o"
"${CXX_COMPILER}" -std=c++17 -shared -o "${ROOT_DIR}/libginac_shared_link_test.${SHARED_EXT}" \
  "${ROOT_DIR}/ginac_shared_link_test.o" \
  "${PREFIX}/lib/libginac.a" "${PREFIX}/lib/libcln.a"

test -f "${ROOT_DIR}/libginac_shared_link_test.${SHARED_EXT}"

echo "CLN/GiNaC static prebuild completed at ${PREFIX}"