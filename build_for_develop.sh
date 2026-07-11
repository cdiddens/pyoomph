#!/usr/bin/env bash

cd $(dirname $0)

PYTHON=python3
$PYTHON -m pip install  --no-build-isolation -e . -v \
    --config-settings=editable.mode=redirect \
    --config-settings=build-dir=build \
    --config-settings=build.verbose=true \
    --config-settings=build.tool-args=-j4 \
    --config-settings=cmake.build-type=RelWithDebInfo \
    --config-settings=cmake.define.PYOOMPH_USE_MPI=ON     

