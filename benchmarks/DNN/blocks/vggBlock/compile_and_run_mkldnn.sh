#set -x

source ../../../configure_paths.sh

echo
echo "MKLDNN VGG"

g++ -std=c++11 -I${MKL_PREFIX}/include -L${MKL_PREFIX}/lib vgg_block_generator_mkldnn.cpp -lmkldnn -o vggBlock_mkldnn_result
./vggBlock_mkldnn_result
