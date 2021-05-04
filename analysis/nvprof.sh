#!/bin/bash

# Anchor path to repository base for executable
anchor=$(dirname $(dirname $(dirname $(realpath $0))));

WORDS="${1}"; # "V100" "XP" or "P100"
EXE="${2}"; # "FULL-W2V", "wombat", or "accSGNS"

cd "${anchor}";
make ${EXE};

OUT="${WORDS}_${EXE}.csv";
APP="${anchor}/${EXE}_${HOSTNAME}";
if [[ "${EXE}" == "FULL-W2V" ]]; then
  ARGS="-read-vocab ${anchor}/data/text8.vocab -train ${anchor}/data/text8 -iter 3 -size 128 -threads 1 -debug 0 -streams 1 -window 3 -output ${WORDS}.words -kernel-batch-size 10000";
else
  ARGS="-read-vocab ${anchor}/data/text8.vocab -train ${anchor}/data/text8 -iter 3 -size 128 -threads 1 -debug 0 -window 5 -output ${WORDS}.words";
fi

NVPROF="--concurrent-kernels on --events all --profile-child-processes --csv";
OUTPUT="events_${OUT}";
echo "OUTPUT: ${OUTPUT}";
echo "NVPROF: ${NVPROF}";
echo "APP: ${APP}";
echo "ARGS: ${ARGS}";

echo "nvprof ${NVPROF} -f -o ${OUTPUT} ${APP} ${ARGS}";
nvprof ${NVPROF} -f -o ${OUTPUT} ${APP} ${ARGS};

NVPROF="--concurrent-kernels on --metrics all --profile-child-processes --csv";
OUTPUT="metrics_${OUT}";
echo "OUTPUT: ${OUTPUT}";
echo "NVPROF: ${NVPROF}";
echo "APP: ${APP}";
echo "ARGS: ${ARGS}";

echo "nvprof ${NVPROF} -f -o ${OUTPUT} ${APP} ${ARGS}";
nvprof ${NVPROF} -f -o ${OUTPUT} ${APP} ${ARGS};

