#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib mkldnn_dense_convolution.cpp -lmkldnn -o spconv_mkldnn_result
./spconv_mkldnn_result
