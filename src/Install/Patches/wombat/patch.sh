#!/bin/bash

# Anchor path to great-grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))));
cd $anchor;

patch -l wombat/src/console.cpp Install/Patches/wombat/console.patch;
patch -l wombat/src/main-cuda.cpp Install/Patches/wombat/main-cuda.patch;
patch -l wombat/src/sgd_trainers/cuda_kernel.wombat.cu Install/Patches/wombat/cuda_kernel.wombat.patch;

