#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

# Fetch original repository at commit
echo "FULL: Fetch wombat...";
git clone git@github.com:tmsimont/wombat.git wombat;
cd wombat;
# Verified commit hash
echo "FULL: Checkout commit";
git checkout 411199080b7a13840cc66eeb9e7015a2b656a90e 2>&1 | head -n 1;

echo "FULL: Clean up data...";
# Remove unneeded data references
rm -rf scripts;

echo "FULL: Patch...";
# Run patch.sh
../Install/Patches/wombat/./patch.sh;

echo "FULL: No wombat installer to run";

echo "FULL: wombat Done";
