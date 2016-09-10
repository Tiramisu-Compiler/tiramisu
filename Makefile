# Set the following PATHs before compiling COLi
ISL_INCLUDE_DIRECTORY=/usr/local/include
ISL_LIB_DIRECTORY=/usr/local/lib
HALIDE_SOURCE_DIRECTORY=/Users/psuriana/Halide
HALIDE_LIB_DIRECTORY=/Users/psuriana/Halide/lib

# Examples
#ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/include/
#ISL_LIB_DIRECTORY=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/lib/
#HALIDE_SOURCE_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/
#HALIDE_LIB_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/bin/

CXX=g++
CXXFLAGS=-g -std=c++11 -O3 -Wall -Wno-sign-compare
INCLUDES=-Iinclude/ -I${ISL_INCLUDE_DIRECTORY} -I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools
LIBRARIES=-L${ISL_LIB_DIRECTORY} -lisl -lgmp -L${HALIDE_LIB_DIRECTORY} -lHalide -ldl -lpthread -lz
HEADER_FILES=include/coli/core.h include/coli/debug.h
OBJ=build/coli_core.o build/coli_codegen_halide.o build/coli_codegen_c.o build/coli_debug.o
TUTO_GEN=build/tutorial_01_lib_generator build/tutorial_02_lib_generator
TUTO_BIN=build/tutorial_01 build/tutorial_02

all: builddir tutorial

builddir:
	@if [ ! -d "build" ]; then mkdir -p build; fi

# Build the coli library object files.  The list of these files is in $(OBJ).
build/coli_%.o: src/coli_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@
build/coli_codegen_%.o: src/coli_codegen_%.cpp include/coli/*.h
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@

# Build the tutorials.  First Object files need to be build, then the
# library generators need to be build and execute (so that they generate
# the libraries), then the wrapper should be built (wrapper are programs that call the
# library functions).
tutorial: $(OBJ) $(TUTO_GEN) $(TUTO_BIN)
build/tutorial_%_lib_generator: tutorials/tutorial_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/tutorial_%: tutorials/wrapper_tutorial_%.cpp build/generated_lib_tutorial_%.o tutorials/wrapper_tutorial_%.h
	$(CXX) ${CXXFLAGS} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
	@DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@

doc:
	doxygen Doxyfile

clean:
	rm -rf *~ src/*~ include/*~ build/* doc/
