#!/bin/bash

# Anchor path to repository base for executable
anchor=$(dirname $(dirname $(dirname $(realpath $0))));

WORDS="${1}"; # "V100" "XP" or "P100"
EXE="${2}"; # "FULL-W2V", "wombat", or "accSGNS"

cd "${anchor}";
make ${EXE};

NSIGHT_EXE= # path to your nv-nsight-cu-cli

OUT="${WORDS}_${EXE}.ncu-rep";
APP="${anchor}/${EXE}_${HOSTNAME}";
if [[ "${EXE}" == "FULL-W2V" ]]; then
  ARGS="-read-vocab ${anchor}/data/text8.vocab -train ${anchor}/data/text8 -iter 3 -size 128 -threads 1 -debug 0 -streams 1 -window 3 -output ${WORDS}.words -kernel-batch-size 10000";
else
  ARGS="-read-vocab ${anchor}/data/text8.vocab -train ${anchor}/data/text8 -iter 3 -size 128 -threads 1 -debug 0 -window 5 -output ${WORDS}.words";
fi

NSIGHT="--target-process all --set full -c 4 -s 1 --summary per-kernel";
echo "OUT: ${OUT}";
echo "NSIGHT: ${NSIGHT_EXE} ${NSIGHT}";
echo "APP: ${APP}";
echo "ARGS: ${ARGS}";

echo "${NSIGHT_EXE} ${NSIGHT} -f -o ${OUT} ${APP} ${ARGS}";
eval "${NSIGHT_EXE} ${NSIGHT} -f -o ${OUT} ${APP} ${ARGS}";

