#set -x

MKLDNNROOT=/usr/local

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib transpose_conv_generator_mkldnn.cpp -lmkldnn -o transpose_conv_mkldnn
./transpose_conv_mkldnn
