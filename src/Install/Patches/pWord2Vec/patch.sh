#!/bin/bash

# Anchor path to great-grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))));
cd $anchor;

patch -l pWord2Vec/makefile Install/Patches/pWord2Vec/makefile.patch;
patch -l pWord2Vec/install.sh Install/Patches/pWord2Vec/install.patch;
patch -l pWord2Vec/pWord2Vec.cpp Install/Patches/pWord2Vec/pWord2Vec.patch;

