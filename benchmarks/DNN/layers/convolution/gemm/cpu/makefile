TIRAMISU_ROOT = ../../../../../..
TIRAMISU_LIBRARY = ${TIRAMISU_ROOT}/build/libtiramisu.dylib
ISL_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/include/
ISL_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
HALIDE_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/include/
HALIDE_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/lib/

CXXFLAGS = -std=c++11 -O3 -Wl,-rpath,${TIRAMISU_ROOT}/build
INCLUDES = -I${TIRAMISU_ROOT}/include/ -I${HALIDE_INCLUDE_DIRECTORY} -I${ISL_INCLUDE_DIRECTORY}
LIBRARIES = -lcblas -lhalide -lisl -lz -lpthread
LIBRARIES_DIR = -L${HALIDE_LIB_DIRECTORY} -L${ISL_LIB_DIRECTORY} -L${TIRAMISU_ROOT}/build

gemm_conv_wrapper: gemm_conv_wrapper.cpp gemm_conv_wrapper.h gemm_conv_fct.o
	g++ ${CXXFLAGS} ${INCLUDES} gemm_conv_wrapper.cpp gemm_conv_fct.o ${TIRAMISU_LIBRARY} ${LIBRARIES_DIR} ${LIBRARIES} -o gemm_conv_wrapper

gemm_conv_fct.o: gemm_conv_generator
	./gemm_conv_generator

gemm_conv_generator: gemm_conv_generator.cpp
	g++ ${CXXFLAGS} ${INCLUDES} gemm_conv_generator.cpp ${LIBRARIES_DIR} ${TIRAMISU_LIBRARY} ${LIBRARIES} -o gemm_conv_generator

clean:
	-rm -f gemm_conv_generator
	-rm -f gemm_conv_fct.o
	-rm -f gemm_conv_fct.o.h
	-rm -f gemm_conv_wrapper
