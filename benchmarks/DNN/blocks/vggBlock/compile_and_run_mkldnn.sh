#set -x

source ../../../configure_paths.sh

export HalideSrc=${TIRAMISU_ROOT}/3rdParty/Halide/

export INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
export LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

export HL_DEBUG_CODEGEN=1

echo "MKLDNN VGG"

g++ -std=c++11 -fopenmp -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib vgg_block_generator_mkldnn.cpp -lmkldnn -o vggBlock_mkldnn_result -lcblas
LD_LIBRARY_PATH=${MKL_PREFIX}/build/src/:$LD_LIBRARY_PATH ./vggBlock_mkldnn_result