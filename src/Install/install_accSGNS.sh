#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

# Fetch original repository at commit
echo "FULL: Fetch accSGNS...";
git clone git@github.com:kinchi22/word2vec-acc-cuda.git accSGNS;
cd accSGNS;
# Verified commit hash
echo "FULL: Checkout commit";
git checkout af0756dddd8c4692471302d0296d3a4a5675d050 2>&1 | head -n 1;

echo "FULL: Clean up data...";
# Remove unneeded data references
rm -f demo-word.sh;

echo "FULL: Patch...";
# Run patch.sh
../Install/Patches/accSGNS/./patch.sh;

echo "FULL: No accSGNS installer to run";

echo "FULL: accSGNS Done";
