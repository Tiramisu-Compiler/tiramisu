# Set the following PATHs before compiling Tiramisu
ISL_INCLUDE_DIRECTORY=/usr/local/include
ISL_LIB_DIRECTORY=/usr/local/lib
HALIDE_SOURCE_DIRECTORY=Halide
HALIDE_LIB_DIRECTORY=Halide/lib

# Examples
#ISL_INCLUDE_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/include/
#ISL_LIB_DIRECTORY=/Users/b/Documents/src/ISLs/isl_prefix/lib/

CXX = g++
CXXFLAGS = -g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden
INCLUDES = -Iinclude/ -I${ISL_INCLUDE_DIRECTORY} -I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools -Ibuild/
LIBRARIES = -L${ISL_LIB_DIRECTORY} -lisl -lgmp -L${HALIDE_LIB_DIRECTORY} -lHalide -ldl -lpthread -lz `libpng-config --cflags --ldflags` -ljpeg
HEADER_FILES = \
	include/tiramisu/core.h \
	include/tiramisu/debug.h \
	include/tiramisu/utils.h \
	include/tiramisu/expr.h \
	include/tiramisu/type.h
OBJ = \
	build/tiramisu_core.o \
	build/tiramisu_codegen_halide.o \
	build/tiramisu_codegen_c.o \
	build/tiramisu_debug.o \
	build/tiramisu_utils.o \
	build/tiramisu_codegen_halide_lowering.o \
	build/tiramisu_codegen_from_halide.o

TUTO_GEN = \
	build/tutorial_01_fct_generator \
	build/tutorial_02_fct_generator \
	build/tutorial_03_fct_generator \
	build/tutorial_04_fct_generator \
	build/tutorial_05_fct_generator
TUTO_BIN = \
	build/tutorial_01 \
	build/tutorial_02 \
	build/tutorial_03 \
	build/tutorial_04 \
	build/tutorial_05

TEST_GEN = \
	build/test_01_fct_generator \
	build/test_02_fct_generator \
	build/test_03_fct_generator \
	build/test_04_fct_generator \
	build/test_05_fct_generator \
	build/test_06_fct_generator \
	build/test_08_fct_generator \
	build/test_09_fct_generator \
	build/test_10_fct_generator \
	build/test_11_fct_generator \
	build/test_12_fct_generator \
	build/test_13_fct_generator \
	build/test_14_fct_generator \
	build/test_15_fct_generator \
	build/test_16_fct_generator \
	build/test_17_fct_generator \
	build/test_18_fct_generator \
	build/test_19_fct_generator \
	build/test_20_fct_generator \
	build/test_21_fct_generator \
	build/test_22_fct_generator \
	build/test_23_fct_generator build/test_24_fct_generator
#build/test_07_fct_generator

TEST_BIN = \
	build/test_01 \
	build/test_02 \
	build/test_03 \
	build/test_04 \
	build/test_05 \
	build/test_06 \
	build/test_08 \
	build/test_09 \
	build/test_10 \
	build/test_11 \
	build/test_12 \
	build/test_13 \
	build/test_14 \
	build/test_15 \
	build/test_16 \
	build/test_17 \
	build/test_18 \
	build/test_19 \
	build/test_20 \
	build/test_21 \
	build/test_22 \
	build/test_23 \
	build/test_24
#build/test_07

BENCH_REF_GEN = \
	build/bench_halide_recfilter_generator \
	build/bench_halide_divergence2d_generator \
	build/bench_halide_heat2d_generator \
	build/bench_halide_cvtcolor_generator \
	build/bench_halide_filter2D_generator \
	build/bench_halide_blurxy_generator \
	build/bench_halide_gaussian_generator \
	build/bench_halide_fusion_generator
# Not supported yet: build/bench_halide_rgbyuv420_generator
BENCH_TIRAMISU_GEN = \
	build/bench_tiramisu_recfilter_generator \
	build/bench_tiramisu_divergence2d_generator \
	build/bench_tiramisu_heat2d_generator \
	build/bench_tiramisu_cvtcolor_generator \
	build/bench_tiramisu_filter2D_generator \
	build/bench_tiramisu_blurxy_generator \
	build/bench_tiramisu_gaussian_generator \
	build/bench_tiramisu_fusion_generator
# Not supported yet: build/bench_tiramisu_rgbyuv420_generator
BENCH_BIN = \
	build/bench_recfilter \
	build/bench_divergence2d \
	build/bench_heat2d \
	build/bench_cvtcolor \
	build/bench_filter2D \
	build/bench_blurxy \
	build/bench_gaussian \
	build/bench_fusion
# Not supported yet: build/bench_rgbyuv420

all: builddir ${OBJ}

builddir:
	@if [ ! -d "build" ]; then mkdir -p build; fi


# Build the Tiramisu library object files.  The list of these files is in $(OBJ).
build/tiramisu_%.o: src/tiramisu_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@
build/tiramisu_codegen_%.o: src/tiramisu_codegen_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@


# Build the tutorials. First Object files need to be build, then the
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


benchmarks: $(OBJ) $(BENCH_TIRAMISU_GEN) $(BENCH_REF_GEN) $(BENCH_BIN) run_benchmarks
build/bench_tiramisu_%_generator: benchmarks/halide/%_tiramisu.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/bench_tiramisu_%_generator: benchmarks/stencils/%_tiramisu.cpp
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
