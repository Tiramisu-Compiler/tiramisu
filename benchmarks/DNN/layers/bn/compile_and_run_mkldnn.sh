#set -x

source ../../../configure_paths.sh

echo
echo
echo "MKL-DNN BN "

g++ -std=c++11 -fopenmp -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib bn_layer_generator_mkldnn.cpp -lmkldnn  -o bn_layer_mkldnn_result
LD_LIBRARY_PATH=${MKL_PREFIX}/build/src/:$LD_LIBRARY_PATH ./bn_layer_mkldnn_result
