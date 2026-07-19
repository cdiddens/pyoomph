#!/usr/bin/env bash
# Resolves the current CLN/GiNaC release versions by scraping ginac.de's own
# download pages, since that site only ever hosts the *current* release
# tarball - a URL pinned to an older version goes dead as soon as a new
# release is published. Source this file rather than executing it.
#
# Set CLN_VERSION / GINAC_VERSION in the environment beforehand to pin a
# specific (still-live) version instead of auto-detecting.

_pyoomph_fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1"
  else
    wget -qO- --read-timeout=20 --timeout=15 --tries=5 "$1"
  fi
}

_pyoomph_resolve_ginac_de_version() {
  local page="$1" prefix="$2" name="$3" fname
  fname=$(_pyoomph_fetch "$page" | grep -oE "${prefix}-[0-9]+\.[0-9]+\.[0-9]+\.tar\.bz2" | head -1)
  if [[ -z "$fname" ]]; then
    echo "Could not find a $name download link on $page (site layout may have changed) - set ${prefix^^}_VERSION explicitly to pin a version instead" >&2
    exit 1
  fi
  fname="${fname#${prefix}-}"
  echo "${fname%.tar.bz2}"
}

if [[ -z "${CLN_VERSION:-}" ]]; then
  CLN_VERSION="$(_pyoomph_resolve_ginac_de_version "https://www.ginac.de/CLN/" cln CLN)"
fi

if [[ -z "${GINAC_VERSION:-}" ]]; then
  GINAC_VERSION="$(_pyoomph_resolve_ginac_de_version "https://www.ginac.de/Download.html" ginac GiNaC)"
fi

echo "Using CLN ${CLN_VERSION} and GiNaC ${GINAC_VERSION} (current releases on ginac.de)" >&2
