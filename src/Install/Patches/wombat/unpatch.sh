#!/bin/bash

# Anchor path to grandparent directory based on script location
anchor=$(dirname $(dirname $(dirname $(realpath $0))));
cd $anchor;

patch -R wombat/src/console.cpp Install/Patches/wombat/console.patch;
patch -R wombat/src/main-cuda.cpp Install/Patches/wombat/main-cuda.patch;
patch -R wombat/src/sgd_trainers/cuda_kernel.wombat.cu Install/Patches/wombat/cuda_kernel.wombat.patch;

