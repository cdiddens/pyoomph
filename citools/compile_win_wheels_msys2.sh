#!/usr/bin/env bash

# MUST BE CALLED FROM ROOT DIRECTORY

export PYOOMPH_PACKAGE_NAME=pyoomph ### TODO


export PYOOMPH_MARCH_NATIVE=false
export PYOOMPH_DEBUG_INFOS=false
export PYOOMPH_PARANOID=false
export PYOOMPH_NO_TCC=true
export PYOOMPH_STATIC_GINAC_DIR=./GiNaC_static
export PYOOMPH_GINAC_INCLUDE_DIR=./GiNaC_static/install/include
export PYOOMPH_GINAC_LIB_DIR=./GiNaC_static/install/lib
export PYOOMPH_CLN_INCLUDE_DIR=./GiNaC_static/install/include
export PYOOMPH_CLN_LIB_DIR=./GiNaC_static/install/lib
export PYOOMPH_CONFIG_FILE=$(readlink -f ./citools/pyoomph_config_windows_cross.env)
# if this is set, you must first call the Makefile in src/ to build everything unrelated of python
export PYOOMPH_FAST_MULTI_VERSION_BUILD=true

export CFLAGS="-O2 -g0 -DNDEBUG"
export CXXFLAGS="-O2 -g0 -DNDEBUG"
export CPPFLAGS="-DNO_ASM -g0 -DNDEBUG"

mkdir -p "$PYOOMPH_STATIC_GINAC_DIR" || exit 1

PYOOMPH_STATIC_GINAC_DIR=$(readlink -f "$PYOOMPH_STATIC_GINAC_DIR")
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PYOOMPH_STATIC_GINAC_DIR/install/lib/pkgconfig
PREFIX=$PYOOMPH_STATIC_GINAC_DIR/install

mkdir -p $PREFIX || exit 1


#if false; then ####TODO REMOVE####
# cln
rm -rf "$PYOOMPH_STATIC_GINAC_DIR/cln"  || exit 1
cd $PYOOMPH_STATIC_GINAC_DIR || exit 1
git clone git://www.ginac.de/cln.git || exit 1
cd cln || exit 1
./autogen.sh || exit 1
./configure --without-gmp --disable-shared --enable-static --with-pic=yes --prefix "$PREFIX" $PYOOMPH_GINAC_CONFIGURE_OPTIONS || exit 1
make  MAKEINFO=true install -j 4 || exit


# ginac
rm -rf "$PYOOMPH_STATIC_GINAC_DIR/ginac"  || exit 1
cd $PYOOMPH_STATIC_GINAC_DIR
git clone git://www.ginac.de/ginac.git || exit 1
cd ginac || exit 1
autoreconf -i -f  || exit 1
CLN_CFLAGS="-I$PREFIX/include" CLN_LIBS="-L$PREFIX/lib -l:libcln.a" ./configure --with-pic=yes $DEBUG_CONFIGURE --disable-shared --enable-static --prefix "$PREFIX" $PYOOMPH_GINAC_CONFIGURE_OPTIONS
make  MAKEINFO=true install -C ginac -j 4 || exit

# back to base
cd $PYOOMPH_STATIC_GINAC_DIR/..



## Building the source parts

# First the oomph-lib part
./clean.sh oomph-lib
./prebuild.sh

# Make sure whe use the full pathes
export PYOOMPH_GINAC_INCLUDE_DIR=$(readlink -f $PYOOMPH_GINAC_INCLUDE_DIR)
export PYOOMPH_CLN_INCLUDE_DIR=$(readlink -f $PYOOMPH_CLN_INCLUDE_DIR)
# Now the python independent core
make -C src/ clean
make -C src/

# Clean the build
mkdir -p build
rm -rf ./build/*

wget https://dist.nuget.org/win-x86-commandline/latest/nuget.exe -O build/nuget



# TODO: Loop
# see https://github.com/pypa/cibuildwheel/blob/main/cibuildwheel/resources/build-platforms.toml
for pyversion in "3.8.10" "3.9.13" "3.10.11" "3.11.9" "3.12.3"; do
export PYOOMPH_PYVERSION=$pyversion

export PYOOMPH_SHORTPYVERSION=$(echo $PYOOMPH_PYVERSION | cut -d . -f 1,2 | tr -d . )


#if false; then ####TODO REMOVE####
./build/nuget install python -OutputDirectory build -Version $PYOOMPH_PYVERSION
#fi ####TODO REMOVE####


export TAG=cp${PYOOMPH_SHORTPYVERSION}-win_amd64
export WHEELTAG=cp${PYOOMPH_PYVERSION}-${TAG}
CURRENT_PYTHON=$(readlink -f ./build/python.${PYOOMPH_PYVERSION}/tools/python.exe)
$CURRENT_PYTHON  -m pip install wheel pybind11 setuptools --upgrade

#if false; then ####TODO REMOVE####
make -f citools/MakefileMSYS2
#fi ####TODO REMOVE####


rm -f ./dist/*.whl *.whl
PYOOMPH_SKIP_EXTENSION=true $CURRENT_PYTHON setup.py bdist_wheel
mkdir -p ./unpack/
rm -rf ./unpack/*
$CURRENT_PYTHON -m wheel unpack ./dist/*.whl --dest ./unpack/

cp -r _*-stubs/ ./unpack/*/
cp ./build/${PYOOMPH_SHORTPYVERSION}/*.pyd ./unpack/*/
cp ./src/jitbridge*.h ./unpack/*/pyoomph/jitbridge

$CURRENT_PYTHON -m wheel pack ./unpack/*
NEWNAME=$(ls *.whl | cut -d - -f 1,2)-cp${PYOOMPH_SHORTPYVERSION}-${TAG}.whl
mkdir -p wheelhouse
cp *.whl wheelhouse/${NEWNAME}

done
