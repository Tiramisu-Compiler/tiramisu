## Set the Tiramisu path
TIRAMISU_PATH=/Users/b/Documents/src/MIT/tiramisu/

# Set to true if you want to run the reference when running Tiramisu code
USE_REFERENCE_WITH_TIRAMISU=1

########################################################################
## General settings
CC=gcc
CXX=g++
CFLAGS=-O3 -mavx2 -fno-rtti
CXXFLAGS=-O3 -std=c++11 -fno-rtti

## No need to modify the rest
LIB_DIR=-L${TIRAMISU_PATH}/build/ -L${TIRAMISU_PATH}/3rdParty/Halide/bin/
INCLUDE_DIR=-I${TIRAMISU_PATH}/include/ -I${TIRAMISU_PATH}/3rdParty/Halide/include/
TIRAMISU_GENERATED_OBJ=../tiramisu_make_local_single_double_block/tiramisu_make_local_single_double_block.o

reference:
	$(CC) $(CFLAGS) qblocks_2pt_scalar.c qblocks_2pt_test.c -o qblocks_2pt_test

tiramisu:
	$(CXX) $(CXXFLAGS) ${INCLUDE_DIR} -DUSE_REFERENCE=0 -c tiramisu_wrapper.cpp -o tiramisu_wrapper
	$(CC) $(CFLAGS) -c -DUSE_REFERENCE=$(USE_REFERENCE_WITH_TIRAMISU) -DUSE_TIRAMISU=1 qblocks_2pt_scalar.c -o qblocks_2pt_scalar.o
	$(CC) $(CFLAGS) -c -DUSE_REFERENCE=$(USE_REFERENCE_WITH_TIRAMISU) -DUSE_TIRAMISU=1 qblocks_2pt_test.c -o qblocks_2pt_test.o
	$(CXX) $(CXXFLAGS) ${INCLUDE_DIR} ${LIB_DIR} tiramisu_wrapper qblocks_2pt_scalar.o qblocks_2pt_test.o ${TIRAMISU_GENERATED_OBJ} -ltiramisu -lHalide -o qblocks_2pt_test

clean:
	rm -rf *.o qblocks_2pt_test tiramisu_wrapper
