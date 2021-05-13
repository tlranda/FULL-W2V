#!/bin/bash

# Anchor path to script directory                                              
anchor=$(dirname $(realpath $0));                                              
cd $anchor;      
echo "Fetch the text8 dataset"

if [[ ! -d "corpus" ||  ! -e "corpus/text8" ]]; then
  mkdir -p corpus;
  mkdir -p vocab;
  wget http://mattmahoney.net/dc/text8.zip -O corpus/text8.gz
  gzip -d corpus/text8.gz -f
fi
