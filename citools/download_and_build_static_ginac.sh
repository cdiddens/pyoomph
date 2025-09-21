#!/usr/bin/env bash


(


if [[ "$PYOOMPH_CONFIG_FILE" == "" ]]; then
cd $(readlink -f $(dirname $0))
echo "Sourcing default config pyoomph_config.env"
source pyoomph_config.env ||  exit 1
else
echo "Sourcing custom config file $PYOOMPH_CONFIG_FILE"
source "$PYOOMPH_CONFIG_FILE" ||  exit 1
fi

if [[ "$PYOOMPH_STATIC_GINAC_DIR" == "" ]]; then
PYOOMPH_STATIC_GINAC_DIR=GiNaC_static
fi


mkdir -p "$PYOOMPH_STATIC_GINAC_DIR" || exit 1

PYOOMPH_STATIC_GINAC_DIR=$(readlink -f "$PYOOMPH_STATIC_GINAC_DIR")
AUTOTTOLS_FILES=$(readlink -f "citools/autotools_files")


rm -rf "$PYOOMPH_STATIC_GINAC_DIR/cln"  || exit 1
rm -rf "$PYOOMPH_STATIC_GINAC_DIR/ginac"  || exit 1




cd $PYOOMPH_STATIC_GINAC_DIR || exit 1
git clone git://www.ginac.de/cln.git || exit 1
git clone git://www.ginac.de/ginac.git || exit 1

#cp -r $AUTOTTOLS_FILES/m4 cln/
#cp -r $AUTOTTOLS_FILES/build-aux cln/

mkdir -p $PYOOMPH_STATIC_GINAC_DIR/install || exit 1


PREFIX=$PYOOMPH_STATIC_GINAC_DIR/install

if $PYOOMPH_DEBUG_INFOS; then
CFLAGS="-O2"
CXXFLAGS="-O2"
CPPFLAGS="-DNO_ASM"
DEBUG_CONFIGURE="--with-debug"
else
CFLAGS="-O2 -g0 -DNDEBUG"
CXXFLAGS="-O2 -g0 -DNDEBUG"
CPPFLAGS="-DNO_ASM -g0 -DNDEBUG"
DEBUG_CONFIGURE="--without-debug"
fi

if $PYOOMPH_MARCH_NATIVE; then
 export CFLAGS="$CFLAGS -march=native"
 export CXXFLAGS="$CXXFLAGS -march=native"
 export CPPFLAGS="$CPPFLAGS -march=native"
fi

echo
echo
echo
echo "BUILDING CLN"
echo 
echo
cd cln || exit 1
./autogen.sh || exit 1
echo
echo
echo
echo "BUILDING CLN  - AUTOGEN DONE"
echo 
echo
./configure --without-gmp $DEBUG_CONFIGURE --disable-shared --enable-static --with-pic=yes --prefix "$PREFIX" $PYOOMPH_GINAC_CONFIGURE_OPTIONS || exit 1
echo
echo
echo
echo "BUILDING CLN  - CONFIGURE DONE"
echo 
echo
make  MAKEINFO=true install -j 4 || exit


export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PYOOMPH_STATIC_GINAC_DIR/install/lib/pkgconfig


echo
echo
echo
echo "BUILDING GINAC"
echo 
echo
cd ../ginac || exit 1
autoreconf -i -f  || exit 1
echo
echo
echo
echo "BUILDING GINAC  - AUTORECONF DONE"
echo 
echo
CLN_CFLAGS="-I$PREFIX/include" CLN_LIBS="-L$PREFIX/lib -l:libcln.a" ./configure --with-pic=yes $DEBUG_CONFIGURE --disable-shared --enable-static --prefix "$PREFIX" $PYOOMPH_GINAC_CONFIGURE_OPTIONS


make  MAKEINFO=true install -C ginac -j 4 || exit

echo
echo
echo
echo "BUILDING GINAC DONE"
echo 
echo

)
