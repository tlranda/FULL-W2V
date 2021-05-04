#!/bin/bash

# Anchor path to parent directory based on script location
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

Install/install_accSGNS.sh;
Install/install_pWord2Vec.sh;
Install/install_pSGNScc.sh;
Install/install_wombat.sh;

