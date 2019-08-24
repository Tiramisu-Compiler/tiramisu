#set -x

MKLDNNROOT=/usr/local/

g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib lstm_block_generator_mkldnn.cpp -lmkldnn -o lstm_mkldnn
./lstm_mkldnn