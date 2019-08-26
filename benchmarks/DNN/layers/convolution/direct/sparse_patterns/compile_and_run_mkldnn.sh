#set -x

source ../../../../../configure_paths.sh

PGM=mkldnn_dense_convolution
BIN=spconv_mkldnn_result

g++ -std=c++11 -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib ${PGM}.cpp -lmkldnn -o $BIN

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/:${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} ./$BIN
