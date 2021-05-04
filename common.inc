# Compilers
GPUCC = nvcc
CPUCC = icpc #~/tools/intel_installation/bin/icpc #icpc
MPICC = mpiicpc
CPU_GCC = gcc

# Determine appropriate compute capablitity
HOSTNAME = $(shell hostname)

# Standard Arguments
C_FLAGS = -std=c++11 -fopenmp -O3 -lm -march=native
C_DEBUG_FLAGS = -std=c++11 -fopenmp -O0 -g -lm
NVIDIA_FLAGS = -Xptxas -dlcm=ca -lineinfo -Xptxas -v -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -O3 -Xcompiler "$(C_FLAGS)" -lm -lcuda -lcudart
NVIDIA_DEBUG_FLAGS = -G -DABORT_ON_CUDA_ERROR -DCUDA_TALK -Xptxas -v -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -Xcompiler "$(C_DEBUG_FLAGS)" -g -G -lm -lcuda -lcudart

