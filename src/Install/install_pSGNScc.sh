#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

# Fetch original repository at commit
echo "FULL: Fetch pSGNScc...";
git clone git@github.com:vasupsu/pWord2Vec.git pSGNScc;
cd pSGNScc;
# Verified commit hash
echo "FULL: Checkout commit";
git checkout 5b892594c64139e3fd3dd5bb2bb5a46c9be25d96 2>&1 | head -n 1;

echo "FULL: Clean up data...";
# Remove unneeded data references
rm -rf hyperwords data billion IA3_AE_test_cases;
# Reduce code and sandbox where applicable
rm -f word2vec.c pWord2Vec.cpp pWord2Vec_mpi.cpp sandbox/run_*.sh sandbox/eval.sh;

echo "FULL: Patch...";
# Run patch.sh
../Install/Patches/pSGNScc/./patch.sh;

echo "FULL: Run pSGNScc installer";
./install.sh;

echo "FULL: pSGNScc Done";
