#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

# Fetch original repository at commit
echo "FULL: Fetch pWord2Vec...";
git clone git@github.com:IntelLabs/pWord2Vec.git pWord2Vec;
cd pWord2Vec;
# Verified commit hash
echo "FULL: Checkout commit";
git checkout 8c68606a68b5b5a92fdc604c5d37d11111fa094e 2>&1 | head -n 1;

echo "FULL: Clean up data...";
# Remove unneeded data references
rm -rf data billion;
# Reduce sandbox where applicable
rm -f sandbox/eval.sh sandbox/run_mpi_text8.sh sandbox/run_single_text8.sh;
# NOTE: MPI implementation not compared to, but left in repository

echo "FULL: Patch...";
# Run patch.sh
../Install/Patches/pWord2Vec/./patch.sh;

echo "FULL: Run pWord2Vec installer";
./install.sh;

echo "FULL: pWord2Vec Done";
