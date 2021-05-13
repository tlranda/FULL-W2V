#!/bin/bash
# Anchor path to parent directory of script
anchor=$(dirname $(dirname $(realpath $0)));
cd $anchor;

# ALL PATHS RELATIVE TO PYTHON SCRIPT
make FULL-W2V > make_FULL-W2V_log 2>&1;
if [[ $? -ne 0 ]]; then
  echo "Make failed. Log:";
  cat make_FULL-W2V_log;
  exit 1;
else rm make_FULL-W2V_log;
fi

echo "OK -- Begin text8 test of FULL-W2V";
mkdir -p replication/evaluation/models;
python3 replication/sweep.py --config evaluation/1_text8.txt evaluation/2_all_v100.txt evaluation/3_FULL-W2V_text8.txt\
                             --model-dir evaluation/models/ --repeat 5 --executable ../FULL-W2V_${HOSTNAME} --identifier FULL-W2V_text8;
