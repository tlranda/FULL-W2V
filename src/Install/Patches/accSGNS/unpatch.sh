#!/bin/bash

# Anchor path to grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(realpath $0))));
cd $anchor;

patch -R accSGNS/makefile Install/Patches/accSGNS/makefile.patch;
patch -R accSGNS/word2vec.cu Install/Patches/accSGNS/word2vec.patch;

