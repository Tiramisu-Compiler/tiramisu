# Set the following PATHs before compiling COLi
ISL_INCLUDE_DIRECTORY=/usr/local/include
ISL_LIB_DIRECTORY=/usr/local/lib
HALIDE_SOURCE_DIRECTORY=Halide
HALIDE_LIB_DIRECTORY=Halide/lib

# Examples
#ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/include/
#ISL_LIB_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/lib/

CXX=g++
CXXFLAGS=-g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden
INCLUDES=-Iinclude/ -I${ISL_INCLUDE_DIRECTORY} -I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools -Ibuild/
LIBRARIES=-L${ISL_LIB_DIRECTORY} -lisl -lgmp -L${HALIDE_LIB_DIRECTORY} -lHalide -ldl -lpthread -lz `libpng-config --cflags --ldflags`
HEADER_FILES=include/coli/core.h include/coli/debug.h include/coli/utils.h include/coli/expr.h include/coli/parser.h include/coli/type.h
OBJ=build/coli_core.o build/coli_codegen_halide.o build/coli_codegen_c.o build/coli_debug.o build/coli_utils.o build/coli_codegen_halide_lowering.o build/coli_codegen_from_halide.o
TUTO_GEN=build/tutorial_01_fct_generator build/tutorial_02_fct_generator build/tutorial_03_fct_generator build/tutorial_04_fct_generator build/tutorial_05_fct_generator
TUTO_BIN=build/tutorial_01 build/tutorial_02 build/tutorial_03 build/tutorial_04 build/tutorial_05
TEST_GEN=build/test_01_fct_generator build/test_02_fct_generator build/test_03_fct_generator build/test_04_fct_generator build/test_05_fct_generator build/test_06_fct_generator
#build/test_07_fct_generator
TEST_BIN=build/test_01 build/test_02 build/test_03 build/test_04 build/test_05 build/test_06
#build/test_07
BENCH_REF_GEN=build/bench_halide_blurxy_generator
# build/bench_halide_fusion_generator
BENCH_COLI_GEN=build/bench_coli_blurxy_generator build/bench_coli_stencil1_generator
# build/bench_coli_fusion_generator
BENCH_BIN=build/bench_blurxy build/bench_stencils_stencil1
# build/bench_fusion

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
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/tutorial_%: tutorials/wrapper_tutorial_%.cpp build/generated_fct_tutorial_%.o tutorials/wrapper_tutorial_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_tutorials:
	@for tt in ${TUTO_BIN}; do LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done


tests: $(OBJ) $(TEST_GEN) $(TEST_BIN) run_tests
build/test_%_fct_generator: tests/test_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/test_%: tests/wrapper_test_%.cpp build/generated_fct_test_%.o tests/wrapper_test_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_tests:
	@for tt in ${TEST_BIN}; do LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done


benchmarks: $(OBJ) $(BENCH_COLI_GEN) $(BENCH_REF_GEN) $(BENCH_BIN) run_benchmarks
build/bench_coli_%_generator: benchmarks/halide/%_coli.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/bench_coli_%_generator: benchmarks/stencils/%_coli.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/bench_halide_%_generator: benchmarks/halide/%_ref.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/bench_%: benchmarks/halide/wrapper_%.cpp build/generated_fct_%.o build/generated_fct_%_ref.o benchmarks/halide/wrapper_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) $(word 3,$^) -o $@ ${INCLUDES} ${LIBRARIES}
build/bench_stencils_%: benchmarks/halide/stencils/wrapper_%.cpp build/generated_fct_%.o benchmarks/halide/wrapper_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_benchmarks:
	@for tt in ${BENCH_BIN}; do LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done



doc:
	doxygen Doxyfile


clean:
	rm -rf *~ src/*~ include/*~ build/* doc/
