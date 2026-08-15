#!/usr/bin/env bash

# pip signals a failed compile only through its exit status, and the last lines of its output are
# CMake/ninja noise by which point the first "error:" has long scrolled past - so a piped build reads
# as successful unless you go looking. Hence the trap: it puts a greppable verdict on the very last
# line, which is what survives "build_for_develop.sh | tail". Note that a caller who pipes still has
# to test ${PIPESTATUS[0]} rather than $?; no setting inside this script can fix that for them.
set -euo pipefail

verdict() {
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "build_for_develop: BUILD OK"
  else
    echo "build_for_develop: BUILD FAILED (exit $rc) - search the log above for 'error:'" >&2
  fi
  exit "$rc"
}
trap verdict EXIT

cd "$(dirname "$0")"

PYTHON=python3
$PYTHON -m pip install  --no-build-isolation -e . -v \
    --config-settings=editable.mode=redirect \
    --config-settings=build-dir=build \
    --config-settings=build.verbose=true \
    --config-settings=build.tool-args=-j4 \
    --config-settings=cmake.build-type=RelWithDebInfo \
    --config-settings=cmake.define.PYOOMPH_USE_MPI=ON     

