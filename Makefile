# Set the following PATHs before compiling COLi
ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/include/
ISL_LIB_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/lib/
HALIDE_SOURCE_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/
HALIDE_LIB_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/bin/

# Examples
#ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/include/
#ISL_LIB_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/lib/
#HALIDE_SOURCE_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/
#HALIDE_LIB_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/bin/

CXX=g++
CXXFLAGS=-g -std=c++11 -O3 -Wall
INCLUDES=-I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools -Iinclude/ -I${ISL_INCLUDE_DIRECTORY}
LIBRARIES=-L${ISL_LIB_DIRECTORY} -lisl -L${HALIDE_LIB_DIRECTORY} -lHalide `libpng-config --cflags --ldflags`
OBJ=build/coli_core.o build/coli_codegen_halide.o build/coli_codegen_c.o build/coli_debug.o


all: builddir tutorial

builddir:
	mkdir -p build

build/coli_%.o: src/coli_%.cpp include/coli/*.h
	$(CXX) -c -fPIC ${CXXFLAGS} ${INCLUDES} $< -o $@

build/coli_codegen_%.o: src/coli_codegen_%.cpp include/coli/*.h
	$(CXX) -c -fPIC ${CXXFLAGS} ${INCLUDES} $< -o $@

tutorial: $(OBJ) examples/tutorial/*.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} ${OBJ} examples/tutorial/coli_tutorial.cpp ${LIBRARIES} -o build/coli_tutorial
	@DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ build/coli_tutorial
	$(CXX) ${CXXFLAGS} ${INCLUDES} examples/tutorial/generated_code_wrapper.cpp LLVM_generated_code.o ${LIBRARIES} -o build/final
	@DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ build/final

doc:
	doxygen Doxyfile

clean:
	rm -rf LLVM_generated_code.o *~ src/*~ include/*~ build/* doc/
