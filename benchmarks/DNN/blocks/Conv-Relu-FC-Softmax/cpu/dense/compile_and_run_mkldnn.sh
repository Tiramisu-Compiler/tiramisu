#set -x

MKLDNNROOT=/usr/local

g++ -g -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib conv_relu_fc_softmax_generator_mkldnn.cpp -lmkldnn -o conv_relu_fc_softmax_mkldnn
./conv_relu_fc_softmax_mkldnn
