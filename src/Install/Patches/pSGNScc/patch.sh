#!/bin/bash

# Anchor path to great-grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))));
cd $anchor;

patch -l pSGNScc/makefile Install/Patches/pSGNScc/makefile.patch;
patch -l pSGNScc/install.sh Install/Patches/pSGNScc/install.patch;
patch -l pSGNScc/pSGNScc.cpp Install/Patches/pSGNScc/pSGNScc.patch;

