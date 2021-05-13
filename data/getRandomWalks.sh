#!/bin/bash

# Anchor path to script directory                                              
anchor=$(dirname $(realpath $0));                                              
cd $anchor;
mkdir -p graphs;
cd graphs;

wget https://www.dropbox.com/s/5sdqv854ioody6i/blog_catalog_random_walks
wget https://www.dropbox.com/s/omft42jgmun1lan/PPI_random_walks
wget https://www.dropbox.com/s/bhhb0uk5s8d2ot6/wikipedia_random_walks
wget https://www.dropbox.com/s/wn1skq3p3pep7n2/facebook_random_walks
wget https://www.dropbox.com/s/efh8e6wyhu5n3so/ASTRO_PH_random_walks
