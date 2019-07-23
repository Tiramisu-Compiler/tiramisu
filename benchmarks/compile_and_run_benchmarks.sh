#!/bin/bash

CXX=/data/scratch/ychen306/gcc/g++

#set -x

if [ $# -eq 0 ]; then
      echo "Usage: TIRAMISU_SMALL=1 script.sh <KERNEL_FOLDER> <KERNEL_NAME_WITHOUT_EXTENSION>"
      echo "Example: script.sh level1/axpy axpy"
      exit
fi

# Define data sizes, possible value: -DTIRAMISU_XLARGE, -DTIRAMISU_LARGE, -DTIRAMISU_MEDIUM, -DTIRAMISU_SMALL
DEFINED_SIZE="-DTIRAMISU_LARGE"

KERNEL_FOLDER=$1
KERNEL=$2
source configure_paths.sh

CXXFLAGS="-std=c++11 -O3 -fno-rtti -mavx2"

# Compile options
# - Make ${CXX} dump generated assembly
#   CXXFLAGS: -g -Wa,-alh
# - Get info about ${CXX} vectorization
#   CXXFLAGS -fopt-info-vec
# - Pass options to the llvm compiler
#   HL_LLVM_ARGS="-help" 
# - Set thread number for Halide
#   HL_NUM_THREADS=32
# - Execution env variables
#   OMP_NUM_THREADS=48
#   to set the number of threads to use by OpenMP.
# - Command to run Vtune
#   source /data/scratch/yunming/intel_parallel_studio_cluster/parallel_studio_xe_2017/install/vtune_amplifier_xe/amplxe-vars.sh
#   amplxe-cl -collect hpc-performance -result-dir vtune_results -quiet ./binary
#   Guide: https://software.intel.com/en-us/vtune-amplifier-help-amplxe-cl-command-syntax

INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread ${EXTRA_LIBRARIES}"
LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

echo "Compiling ${KERNEL}"

cd ${KERNEL_FOLDER}

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}_halide.o

# Generate code from Tiramisu
${CXX} ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator
echo "Running ${KERNEL} generator (Tiramisu)"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ./${KERNEL}_generator

if [ $? -ne 0 ]; then
	exit
fi

# echo "Compiling ${KERNEL} wrapper"
${CXX} ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_wrapper.cpp   ${LIBRARIES_DIR} ${LIBRARIES} generated_${KERNEL}.o ${LIBRARIES} -o ${KERNEL}_wrapper
echo "Running ${KERNEL} wrapper"
# To enable profiling:
## Perf:
#PROFILING_COMMAND="perf stat -e cycles,instructions,cache-misses,L1-icache-load-misses,LLC-load-misses,dTLB-load-misses,cpu-migrations,context-switches,bus-cycles,cache-references,minor-faults"
## Vtune:
#VTUNE_METRIC=hpc-performance
#VTUNE_METRIC=memory-access
#PROFILING_COMMAND="amplxe-cl -collect ${VTUNE_METRIC} -result-dir vtune_results -quiet"
#rm -rf vtune_results
RUN_REF=1 RUN_TIRAMISU=1 HL_NUM_THREADS=$CORES LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ${PROFILING_COMMAND} ./${KERNEL}_wrapper

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}.o.h

cd -
