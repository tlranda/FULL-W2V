#!/bin/bash

# Anchor path to script directory
anchor=$(dirname $(realpath $0));
cd $anchor;
./getText8.sh;
./getBillion.sh;
