#!/usr/bin/env bash
# Remove the build/ directory, i.e. force a full rebuild of pyoomph's own code.
#
# By default, the downloaded/compiled CLN and GiNaC (build/{cln_build,ginac_build,thirdparty-install})
# are kept, since rebuilding them from source via autotools takes several minutes. Pass
# --with-thirdparty to also wipe those and force CLN/GiNaC to be re-downloaded and rebuilt from scratch.

set -euo pipefail
cd "$(dirname "$0")"

wipe_thirdparty=0
for arg in "$@"; do
	case "$arg" in
	--with-thirdparty)
		wipe_thirdparty=1
		;;
	-h | --help)
		echo "Usage: $0 [--with-thirdparty]"
		echo "  --with-thirdparty  Also remove the downloaded/built CLN and GiNaC. Off by default, since"
		echo "                     rebuilding them from source is slow; use this if they need to be redone"
		echo "                     (e.g. after changing PYOOMPH_CLN_VERSION/PYOOMPH_GINAC_VERSION)."
		exit 0
		;;
	*)
		echo "Unknown argument: $arg" >&2
		echo "Usage: $0 [--with-thirdparty]" >&2
		exit 1
		;;
	esac
done

if [ "$wipe_thirdparty" = 1 ]; then
	rm -rf build/
else
	if [ -d build ]; then
		find build -mindepth 1 -maxdepth 1 \
			! -name cln_build ! -name ginac_build ! -name thirdparty-install \
			-exec rm -rf {} +
	fi
fi
