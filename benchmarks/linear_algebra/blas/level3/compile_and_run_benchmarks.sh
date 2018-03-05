#!/bin/bash

#set -x

LANKA=1

if [ $LANKA -eq 0 ]; then
	TIRAMISU_ROOT=/Users/b/Documents/src/MIT/tiramisu/
	MKL_FLAGS="-lcblas"
	MKL_LIB_PATH_SUFFIX=
	LANKA_OPTIONS=
else
	TIRAMISU_ROOT=/data/scratch/baghdadi/tiramisu/
	MKL_FLAGS="-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -ldl -lm"
	MKL_LIB_PATH_SUFFIX=intel64/
	LANKA_OPTIONS="-DMKL_ILP64 -m64" 
fi

if [ $# -eq 0 ]; then
      echo "Usage: TIRAMISU_SMALL=1 script.sh <KERNEL_FOLDER>"
      echo "Example: script.sh axpy"
      exit
fi

if [ "${TIRAMISU_XLARGE}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_XLARGE"
elif [ "${TIRAMISU_LARGE}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_LARGE"
elif [ "${TIRAMISU_MEDIUM}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_MEDIUM"
elif [ "${TIRAMISU_SMALL}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_SMALL"
else
    DEFINED_SIZE="-DTIRAMISU_XLARGE"
fi

KERNEL=$1
source ${TIRAMISU_ROOT}/configure_paths.sh

CXXFLAGS="-std=c++11 -O3"

INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${TIRAMISU_ROOT}/${HALIDE_SOURCE_DIRECTORY}/include/ -I${TIRAMISU_ROOT}/${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread"
LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${TIRAMISU_ROOT}/$HALIDE_LIB_DIRECTORY -L${TIRAMISU_ROOT}/${ISL_LIB_DIRECTORY} -L${TIRAMISU_ROOT}/build/"

echo "Compiling ${KERNEL}"

cd ${KERNEL}

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}_halide.o

#echo "Compiling ${KERNEL} generator (Halide)" >> log
#g++ $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator_halide.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator_halide & >> log
#echo "Running ${KERNEL} generator (Halide)" >> log
#HL_DEBUG_CODEGEN=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./${KERNEL}_generator_halide


g++ ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator
#&>> log
echo "Running ${KERNEL} generator (Tiramisu)"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ./${KERNEL}_generator
#&>> log

if [ $? -ne 0 ]; then
	exit
fi

echo "Compiling ${KERNEL} wrapper"
g++ ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_wrapper.cpp   ${LIBRARIES_DIR} ${LIBRARIES} generated_${KERNEL}.o ${LIBRARIES} -o ${KERNEL}_wrapper &>> log
echo "Running ${KERNEL} wrapper"
for ((i=0; i<1; i++)); do
	RUN_MKL=1 RUN_TIRAMISU=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} perf stat -e instructions,cache-misses,L1-icache-load-misses,LLC-load-misses,dTLB-load-misses,cpu-migrations,context-switches,bus-cycles,cache-references,minor-faults ./${KERNEL}_wrapper
done

cd -
