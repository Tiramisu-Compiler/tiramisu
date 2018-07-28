TIRAMISU_ROOT = ../../../../../..
ISL_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/include/
ISL_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
HALIDE_INCLUDE_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/include/
HALIDE_LIB_DIRECTORY = ${TIRAMISU_ROOT}/3rdParty/Halide/lib/

CXXFLAGS = -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -Wl,-rpath,${TIRAMISU_ROOT}/build
INCLUDES = -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build
LIBRARIES = -ltiramisu -lHalide -ldl -lpthread -lz -lm
LIBRARIES_DIR = -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/ -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/

wrapper: wrapper.cpp wrapper.h fct.o
	g++ ${CXXFLAGS} ${INCLUDES} wrapper.cpp fct.o ${LIBRARIES_DIR} ${LIBRARIES} -o wrapper

fct.o: generator
	./generator

generator: generator.cpp
	g++ ${CXXFLAGS} ${INCLUDES} generator.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o generator

clean:
	-rm -f generator
	-rm -f fct.o
	-rm -f fct.o.h
	-rm -f wrapper
