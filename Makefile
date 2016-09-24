# Set the following PATHs before compiling COLi
ISL_INCLUDE_DIRECTORY=/usr/local/include
ISL_LIB_DIRECTORY=/usr/local/lib
HALIDE_SOURCE_DIRECTORY=/Users/psuriana/halide
HALIDE_LIB_DIRECTORY=/Users/psuriana/halide/lib

# Examples
#ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/include/
#ISL_LIB_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/lib/
#HALIDE_SOURCE_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/
#HALIDE_LIB_DIRECTORY=/Users/b/Documents/src/MIT/halide/halide_src/bin/

CXX=g++
CXXFLAGS=-g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden
INCLUDES=-Iinclude/ -I${ISL_INCLUDE_DIRECTORY} -I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools
LIBRARIES=-L${ISL_LIB_DIRECTORY} -lisl -lgmp -L${HALIDE_LIB_DIRECTORY} -lHalide -ldl -lpthread -lz `libpng-config --cflags --ldflags`
HEADER_FILES=include/coli/core.h include/coli/debug.h include/coli/utils.h include/coli/expr.h include/coli/parser.h include/coli/type.h
OBJ=build/coli_core.o build/coli_codegen_halide.o build/coli_codegen_c.o build/coli_debug.o build/coli_utils.o build/coli_codegen_halide_lowering.o build/coli_codegen_from_halide.o
TUTO_GEN=build/tutorial_01_fct_generator build/tutorial_02_fct_generator build/tutorial_03_fct_generator
TUTO_BIN=build/tutorial_01 build/tutorial_02 build/tutorial_03
TEST_GEN=build/test_01_fct_generator build/test_02_fct_generator build/test_03_fct_generator
TEST_BIN=build/test_01 build/test_02 build/test_03


all: builddir ${OBJ}


builddir:
	@if [ ! -d "build" ]; then mkdir -p build; fi


# Build the coli library object files.  The list of these files is in $(OBJ).
build/coli_%.o: src/coli_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@
build/coli_codegen_%.o: src/coli_codegen_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@


# Build the tutorials.  First Object files need to be build, then the
# library generators need to be build and execute (so that they generate
# the libraries), then the wrapper should be built (wrapper are programs that call the
# library functions).
tutorials: $(OBJ) $(TUTO_GEN) $(TUTO_BIN) run_tutorials
build/tutorial_%_fct_generator: tutorials/tutorial_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/tutorial_%: tutorials/wrapper_tutorial_%.cpp build/generated_fct_tutorial_%.o tutorials/wrapper_tutorial_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_tutorials:
	@for tt in ${TUTO_BIN}; do LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done


tests: $(OBJ) $(TEST_GEN) $(TEST_BIN) run_tests
build/test_%_fct_generator: tests/test_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/test_%: tests/wrapper_test_%.cpp build/generated_fct_test_%.o tests/wrapper_test_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_tests:
	@for tt in ${TEST_BIN}; do LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done


doc:
	doxygen Doxyfile

clean:
	rm -rf *~ src/*~ include/*~ build/* doc/
