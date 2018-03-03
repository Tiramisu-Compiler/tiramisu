#set -x

TIRAMISU_ROOT=/Users/b/Documents/src/MIT/tiramisu/

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
    DEFINED_SIZE="-DTIRAMISU_LARGE"
fi

KERNEL=$1
source ${TIRAMISU_ROOT}/configure_paths.sh

CXXFLAGS="-std=c++11 -O3"

INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${TIRAMISU_ROOT}/${HALIDE_SOURCE_DIRECTORY}/include/ -I${TIRAMISU_ROOT}/${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
LIBRARIES="-ltiramisu -lcblas -lhalide -lisl -lz -lpthread"
LIBRARIES_DIR="-L${MKL_PREFIX}/lib/ -L${TIRAMISU_ROOT}/$HALIDE_LIB_DIRECTORY -L${TIRAMISU_ROOT}/${ISL_LIB_DIRECTORY} -L${TIRAMISU_ROOT}/build/"

echo "Compiling ${KERNEL}"

cd ${KERNEL}

rm -rf ${KERNEL}_generator ${KERNEL}_wrapper generated_${KERNEL}.o generated_${KERNEL}_halide.o

#echo "Compiling ${KERNEL} generator (Halide)" >> log
#g++ $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator_halide.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator_halide & >> log
#echo "Running ${KERNEL} generator (Halide)" >> log
#HL_DEBUG_CODEGEN=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./${KERNEL}_generator_halide

echo "Compiling ${KERNEL} generator (Tiramisu)" >> log
g++ $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_generator.cpp ${LIBRARIES_DIR} ${LIBRARIES}                       -o ${KERNEL}_generator
echo "Running ${KERNEL} generator (Tiramisu)" >> log
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./${KERNEL}_generator
#&> log

echo "Compiling ${KERNEL} wrapper" >>log
g++ $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} ${KERNEL}_wrapper.cpp   ${LIBRARIES_DIR} ${LIBRARIES} generated_${KERNEL}.o -o ${KERNEL}_wrapper
echo "Running ${KERNEL} wrapper" >> log
for ((i=0; i<10; i++)); do
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./${KERNEL}_wrapper
done

cd -
