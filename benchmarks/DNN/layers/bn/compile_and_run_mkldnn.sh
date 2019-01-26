#set -x

source ../../../configure_paths.sh

CXXFLAGS="-std=c++11 -O3 -fno-rtti"

INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread"
LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

echo "MKL-DNN BN"
g++ ${LANKA_OPTIONS} $CXXFLAGS ${INCLUDES} bn_layer_generator_mkldnn.cpp ${LIBRARIES_DIR} ${LIBRARIES} -lmkldnn -o bn_layer_mkldnn_result
LD_LIBRARY_PATH=${MKL_PREFIX}/build/src/:$LD_LIBRARY_PATH ./bn_layer_mkldnn_result
