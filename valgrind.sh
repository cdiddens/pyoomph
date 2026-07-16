#!/usr/bin/env bash
#SUPPFILE=$(readlink -f $(dirname $0))/valgrind.supp
#--suppressions=$SUPPFILE
PYTHONMALLOC=malloc PYOOMPH_DEBUG=1 valgrind  "$@"
