#!/bin/bash

# Anchor path to grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(realpath $0))));
cd $anchor;

patch -R pSGNScc/makefile Install/Patches/pSGNScc/makefile.patch;
patch -R pSGNScc/install.sh Install/Patches/pSGNScc/install.patch;
patch -R pSGNScc/pSGNScc.cpp Install/Patches/pSGNScc/pSGNScc.patch;

