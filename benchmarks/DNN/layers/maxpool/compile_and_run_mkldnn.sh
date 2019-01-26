#set -x

source ../../../configure_paths.sh

rm -rf _results

echo
echo
echo "MKL maxpool "

g++ -std=c++11 -fopenmp -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib maxpool_layer_generator_mkldnn.cpp -lmkldnn  -o maxpool_layer_mkldnn_result
LD_LIBRARY_PATH=${MKL_PREFIX}/build/src/:$LD_LIBRARY_PATH  ./maxpool_layer_mkldnn_result
