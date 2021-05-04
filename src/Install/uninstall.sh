#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

rm -rf accSGNS pWord2Vec pSGNScc wombat;

