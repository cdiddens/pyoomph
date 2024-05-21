#!/usr/bin/env bash

COLOR_OUTPUT=-fdiagnostics-color=always



#PYVERS=$(python3 -c 'import sys; print(sys.version_info>=(3,10,12));')

PYVERS=$(python3 -c 'import pip; print(int(pip.__version__.split(".")[0])>=24);')


if [[ "$PYVERS" == "True" ]]; then
 EDITABLE_MODE="--config-settings editable_mode=strict"
else
 echo "ERROR: Please upgrade pip to at least version 24.0!"
 echo "pyoomph -m pip install --upgrade pip"
 exit
 EDITABLE_MODE=
fi

if [[ "$PYOOMPH_CONFIG_FILE" == "" ]]; then
cd $(readlink -f $(dirname $0))
echo "Sourcing default config pyoomph_config.env"
source pyoomph_config.env ||  exit 1
PYOOMPH_CONFIG_FILE=pyoomph_config.env
else
echo "Sourcing custom config file $PYOOMPH_CONFIG_FILE"
source "$PYOOMPH_CONFIG_FILE" ||  exit 1
fi


if [[ $CXX == "" ]]; then
CXX=g++
fi


if [[ "$OSTYPE" != "darwin"* ]]; then
cd $(readlink -f $(dirname $0))
if which ccache &>/dev/null; then 
NPY_NUM_BUILD_JOBS=4 CXX="$CXX -pthread -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions  -Wl,-z,relro" CC="ccache $CXX $COLOR_OUTPUT" LDSHARED="$CXX -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro"  python3 -m pip install  -v "$@"  .  $EDITABLE_MODE || exit 1
 else
 NPY_NUM_BUILD_JOBS=4 CXX="$CXX -pthread -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro" CC="$CXX $COLOR_OUTPUT" LDSHARED="$CXX -g3 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro"  python3 -m pip install  -v "$@"  $EDITABLE_MODE || exit 1
 fi
else
 if which ccache &>/dev/null; then 
  NPY_NUM_BUILD_JOBS=4 CXX="$CXX -L/usr/local/lib" CC="ccache $CXX -I/usr/local/include $COLOR_OUTPUT" LD="$CXX"  python3 -m pip install -v  "$@" . $EDITABLE_MODE || exit 1
 else
   NPY_NUM_BUILD_JOBS=4 CXX="$CXX -L/usr/local/lib" CC="$CXX -I/usr/local/include $COLOR_OUTPUT" LD="$CXX"  python3 -m pip install -v  "$@" .  $EDITABLE_MODE || exit 1
 fi
fi

#CC="ccache g++"  python3 -m pip install -e . -v "$@"

#
