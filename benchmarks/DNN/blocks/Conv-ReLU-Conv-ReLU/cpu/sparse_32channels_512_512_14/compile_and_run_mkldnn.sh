#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib conv_relu_conv_relu_generator_mkldnn.cpp -lmkldnn -o conv_relu_conv_relu_mkldnn
./conv_relu_conv_relu_mkldnn
