#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib vgg_block_generator_mkldnn.cpp -lmkldnn -o vgg_block_mkldnn
./vgg_block_mkldnn