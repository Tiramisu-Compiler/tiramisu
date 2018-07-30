TIRAMISU_ROOT = ../../../../../..
ISL_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/include/
ISL_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
HALIDE_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/include/
HALIDE_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/lib/
CUDA_ROOT = /usr/local/cuda

CXX = g++
CXXFLAGS = -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -Wl,-rpath,${TIRAMISU_ROOT}/build -O3
INCLUDES = -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -I${CUDA_ROOT}/include
LIBRARIES = -ltiramisu -lHalide -ldl -lpthread -lz -lm -lcudart -lcublas -lcuda -lrt
LIBRARIES_DIR = -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/ -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${CUDA_ROOT}/lib64

RUNN ?= 100

# TODO: Setting testN higher values (e.g. 100) makes tiramisu faster!?
run: wrapper
	LD_LIBRARY_PATH=${CUDA_ROOT}/lib64 ./wrapper ${RUNN} 0

nvprof: wrapper
	LD_LIBRARY_PATH=${CUDA_ROOT}/lib64 ${CUDA_ROOT}/bin/nvprof --print-gpu-trace ./wrapper 1 0

correctness: wrapper
	LD_LIBRARY_PATH=${CUDA_ROOT}/lib64 ./wrapper 0 1

wrapper: fct.o wrapper.cpp configuration.h
	${CXX} ${CXXFLAGS} ${INCLUDES} wrapper.cpp ${TIRAMISU_ROOT}/src/tiramisu_cuda_wrappers.cpp fct.o fct.o_cpu.o fct.o_gpu.o ${TIRAMISU_LIBRARY} ${LIBRARIES_DIR} ${LIBRARIES} -o wrapper

generator: generator.cpp configuration.h
	${CXX} ${CXXFLAGS} ${INCLUDES} generator.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o generator

fct.o: generator
	./generator

clean:
	-rm -f fct.o
	-rm -f fct.o.cu
	-rm -f fct.o_cpu.o
	-rm -f fct.o_gpu.o
	-rm -f fct.o.h
	-rm -f generator
	-rm -f wrapper
