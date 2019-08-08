#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib densenet_block_generator_mkldnn.cpp -lmkldnn -o densenet_block_mkldnn
./densenet_block_mkldnn
