#!/usr/bin/env bash 

## Configure the MKL library (Only required for benchmarks that use Intel MKL)
#### Path to MKL
export MKL_PREFIX=/opt/intel/compilers_and_libraries/mac/mkl/
#### MKL library flag
export MKL_FLAGS="-lcblas"
export MKL_LIB_PATH_SUFFIX=

################################################################
# Most of the following options do not need to be modified.

# Path to the Tiramisu root director
# If this fails for some reason, just specify the path to tiramisu manually.
# For example /Users/b/Documents/src/MIT/tiramisu/
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export TIRAMISU_ROOT=${SCRIPTPATH}/../
echo ${TIRAMISU_ROOT}

# Number of Halide threads used for parallelization
export CORES=1

# Paths to Tiramisu 3rd party libraries
export ISL_INCLUDE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/include/
export ISL_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
export HALIDE_SOURCE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide
export HALIDE_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide/lib

# Lanka configuration
# export TIRAMISU_ROOT=/data/scratch/baghdadi/tiramisu/
# export MKL_FLAGS="-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -ldl -lm"
# export MKL_LIB_PATH_SUFFIX=intel64/
# export LANKA_OPTIONS="-DMKL_ILP64 -m64 -fopenmp"
# export USE_PERF=1
# export MKL_PREFIX=/data/scratch/baghdadi/libs/intel/mkl/
# export CORES=32
