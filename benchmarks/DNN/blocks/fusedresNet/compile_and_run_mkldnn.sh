#set -x

source ../../../configure_paths.sh

echo "MKL-DNN fused_resnet"
g++ -std=c++11 -fopenmp -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib fused_resnet_block_generator_mkldnn.cpp -lmkldnn -o fused_resnetBlock_mkldnn_result
LD_LIBRARY_PATH=${MKL_PREFIX}/build/src/:$LD_LIBRARY_PATH ./fused_resnetBlock_mkldnn_result
