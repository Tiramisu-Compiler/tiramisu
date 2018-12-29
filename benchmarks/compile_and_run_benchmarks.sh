#!/bin/bash

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

CXXFLAGS="-std=c++11 -O3 -fno-rtti"

# Compile options
# - Make g++ dump generated assembly
#   CXXFLAGS: -g -Wa,-alh
# - Get info about g++ vectorization
#   CXXFLAGS -fopt-info-vec
# - Pass options to the llvm compiler
#   HL_LLVM_ARGS="-help" 
# - Set thread number for Halide
#   HL_NUM_THREADS=32
# Execution env variables
#   OMP_NUM_THREADS=48
#   to set the number of threads to use by OpenMP.

INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread"
LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

echo "Compiling ${KERNEL}"

cd ${KERNEL_FOLDER}

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}_halide.o

# Generate code from Tiramisu
g++ ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator
echo "Running ${KERNEL} generator (Tiramisu)"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ./${KERNEL}_generator

if [ $? -ne 0 ]; then
	exit
fi

# To enable profiling use the command
# perf stat -e cycles,instructions,cache-misses,L1-icache-load-misses,LLC-load-misses,dTLB-load-misses,cpu-migrations,context-switches,bus-cycles,cache-references,minor-faults
echo "Compiling ${KERNEL} wrapper"
g++ ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_wrapper.cpp   ${LIBRARIES_DIR} ${LIBRARIES} generated_${KERNEL}.o ${LIBRARIES} -o ${KERNEL}_wrapper
echo "Running ${KERNEL} wrapper"
RUN_REF=1 RUN_TIRAMISU=1 HL_NUM_THREADS=$CORES LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ./${KERNEL}_wrapper

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}.o.h

cd -
