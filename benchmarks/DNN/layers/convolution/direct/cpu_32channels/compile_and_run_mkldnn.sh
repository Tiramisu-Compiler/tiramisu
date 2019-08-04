#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib conv_layer_generator_mkldnn.cpp -lmkldnn -o conv_layer_mkldnn
./conv_layer_mkldnn