#!/bin/bash
# SELECT SCHEDULER NEEDS
#PBS -N p100_w2vsweep
#PBS -l select=1:ncpus=24:mem=125gb:ngpus=1:gpu_model=p100,walltime=72:00:00

#SBATCH --time=1-00:00:00
#SBATCH -w n01 --exclusive
#SBATCH --output=titanxp_%j.out

if [ ! -z ${PBS_JOBNAME+x} ]; then
  # In PPS Queue
  # Load modules necessary for execution, such as compilers and Python3 environment
  module load anaconda3/5.1.0-gcc cuda/11.0.3-gcc/8.3.1 intel;
else if [ ! -z ${SLURM_JOB_NAME+x} ]; then
  # In SLURM Queue
  # Load modules necessary for execution, such as compilers and Python3 environment
  module load cuda/11.2;
else
  # Interactive
  echo "Running interactively";
fi
fi

hostname; date;
# ALL PATHS RELATIVE TO THE PYTHON SCRIPT
MAKE_TARGET=${1}; # Should be the rule to create executable in base repo makefile
FRIENDLY_NAME=${2}; # Friendly implementation name to look up bonus rules
CORPUS=${3}; # Should be '1bw' or 'text8'
SYSTEM='p100'; # Should be 'p100' or 'v100'
# DERIVED
executable='../'${MAKE_TARGET}'_'${HOSTNAME};
OUTPUT_UID=${MAKE_TARGET}'_'${CORPUS}'_'${SYSTEM};
all_spec='evaluation/1_'${CORPUS}'.txt evaluation/2_all_'${SYSTEM}'.txt';
FILE='evaluation/3_'${FRIENDLY_NAME}'_'${CORPUS}'.txt';
if [[ -f "replication/${FILE}" ]]; then
  unique_spec=${FILE};
  echo "Found unique spec ${unique_spec}";
else
  unique_spec='';
  echo "No unique spec for ${FRIENDLY_NAME} and ${CORPUS}";
  echo "Looked for ${FILE}";
fi
# COMPILE
make ${MAKE_TARGET} >make_${MAKE_TARGET}_log 2>&1;
# Check good make
if [[ $? -ne 0 ]]; then
  echo "Make failed, log contents:";
  cat make_${MAKE_TARGET}_log;
  exit 1;
else rm make_${MAKE_TARGET}_log;
fi

# EZ DRY MODE (any arg turns into dry run)
if [[ $# -ge 4 ]]; then
  DRY='--dry-run';
else
  DRY='';
fi

echo "OK -- Begin evaluation of ${MAKE_TARGET} on ${CORPUS} corpus with ${SYSTEM} system ${DRY}";
python3 replication/sweep.py --config ${all_spec} ${unique_spec} --model-dir evaluation/models/ --repeat 5 --executable ${executable} --identifier ${OUTPUT_UID} ${DRY};
date;

