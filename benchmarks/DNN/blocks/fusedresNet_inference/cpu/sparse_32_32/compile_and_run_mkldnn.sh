#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib fused_resnet_block_generator_mkldnn.cpp -lmkldnn -o fused_resnet_block_mkldnn_result
./fused_resnet_block_mkldnn_result
