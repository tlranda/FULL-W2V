#!/bin/bash

# Anchor path to great-grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))));
cd $anchor;

patch -l accSGNS/makefile Install/Patches/accSGNS/makefile.patch;
patch -l accSGNS/word2vec.cu Install/Patches/accSGNS/word2vec.patch;

