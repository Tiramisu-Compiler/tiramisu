TIRAMISU_ROOT = ../../../../../..
ISL_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/include/
ISL_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
HALIDE_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/include/
HALIDE_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/lib/

CXXFLAGS = -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -Wl,-rpath,${TIRAMISU_ROOT}/build
INCLUDES = -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build
LIBRARIES = -ltiramisu -lHalide -ldl -lpthread -lz -lm
LIBRARIES_DIR = -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/ -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/

gemm_conv_wrapper: gemm_conv_wrapper.cpp gemm_conv_wrapper.h gemm_conv_fct.o
	g++ ${CXXFLAGS} ${INCLUDES} gemm_conv_wrapper.cpp gemm_conv_fct.o ${LIBRARIES_DIR} ${LIBRARIES} -o gemm_conv_wrapper

gemm_conv_fct.o: gemm_conv_generator
	./gemm_conv_generator

gemm_conv_generator: gemm_conv_generator.cpp
	g++ ${CXXFLAGS} ${INCLUDES} gemm_conv_generator.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o gemm_conv_generator

clean:
	-rm -f gemm_conv_generator
	-rm -f gemm_conv_fct.o
	-rm -f gemm_conv_fct.o.h
	-rm -f gemm_conv_wrapper
