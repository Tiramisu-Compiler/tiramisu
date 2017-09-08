include configure_paths.sh

#############################################################@

CXX = g++
CXXFLAGS = -g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden
INCLUDES = -Iinclude/ -I${ISL_INCLUDE_DIRECTORY} -I${HALIDE_SOURCE_DIRECTORY}/include -I${HALIDE_SOURCE_DIRECTORY}/tools -Ibuild/
LIBRARIES = -L${ISL_LIB_DIRECTORY} -lisl -lgmp -L${HALIDE_LIB_DIRECTORY} -lHalide -ldl -lpthread -lz `libpng-config --cflags --ldflags` -ljpeg `${LLVM_CONFIG_BIN}llvm-config --system-libs`
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
	build/tutorial_05_fct_generator \
	build/tutorial_06_fct_generator \
	build/tutorial_08_fct_generator \
	build/tutorial_09_fct_generator \
	build/tutorial_10_fct_generator

TUTO_BIN = \
	build/tutorial_01 \
	build/tutorial_02 \
	build/tutorial_03 \
	build/tutorial_04 \
	build/tutorial_05 \
	build/tutorial_06 \
	build/tutorial_08 \
	build/tutorial_09 \
	build/tutorial_10

TUTO_RUN = \
	run_tutorial_01 \
	run_tutorial_02 \
	run_tutorial_03 \
	run_tutorial_04 \
	run_tutorial_05 \
	run_tutorial_06 \
	run_tutorial_08 \
	run_tutorial_09 \
	run_tutorial_10


#####################################################


TEST_GEN = build/test_01_fct_generator \
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
    build/test_23_fct_generator \
    build/test_24_fct_generator \
    build/test_25_fct_generator \
    build/test_26_fct_generator \
    build/test_27_fct_generator \
    build/test_28_fct_generator \
    build/test_29_fct_generator \
    build/test_30_fct_generator \
    build/test_31_fct_generator \
    build/test_32_fct_generator \
    build/test_33_fct_generator \
    build/test_34_fct_generator \
    build/test_35_fct_generator \
    build/test_36_fct_generator \
    build/test_37_fct_generator \
    build/test_38_fct_generator \
    build/test_39_fct_generator \
    build/test_40_fct_generator \
    build/test_41_fct_generator \
    build/test_42_fct_generator \
    build/test_43_fct_generator \
    build/test_44_fct_generator \
    build/test_45_fct_generator \
    build/test_46_fct_generator \
    build/test_47_fct_generator \
    build/test_48_fct_generator \
    build/test_49_fct_generator \
    build/test_50_fct_generator \
    build/test_51_fct_generator \
    build/test_52_fct_generator \
    build/test_53_fct_generator \
    build/test_54_fct_generator \
    build/test_55_fct_generator \
    build/test_56_fct_generator \
    build/test_58_fct_generator \
    build/test_59_fct_generator \
    build/test_60_fct_generator \
    build/test_61_fct_generator \
    build/test_62_fct_generator \
    build/test_63_fct_generator \
    build/test_64_fct_generator \
    build/test_65_fct_generator \
    build/test_66_fct_generator \
    build/test_67_fct_generator \
    build/test_68_fct_generator \
    build/test_70_fct_generator \
    build/test_71_fct_generator \
    build/test_72_fct_generator



#build/test_07_fct_generator

TEST_BIN = build/test_global \
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
    build/test_24 \
    build/test_25 \
    build/test_26 \
    build/test_27 \
    build/test_28 \
    build/test_29 \
    build/test_30 \
    build/test_31 \
    build/test_32 \
    build/test_33 \
    build/test_34 \
    build/test_35 \
    build/test_36 \
    build/test_37 \
    build/test_38 \
    build/test_39 \
    build/test_40 \
    build/test_41 \
    build/test_42 \
    build/test_43 \
    build/test_44 \
    build/test_45 \
    build/test_46 \
    build/test_47 \
    build/test_48 \
    build/test_49 \
    build/test_50 \
    build/test_51 \
    build/test_52 \
    build/test_53 \
    build/test_54 \
    build/test_55 \
    build/test_56 \
    build/test_58 \
    build/test_59 \
    build/test_60 \
    build/test_61 \
    build/test_62 \
    build/test_63 \
    build/test_64 \
    build/test_65 \
    build/test_66 \
    build/test_67 \
    build/test_68 \
    build/test_70 \
    build/test_71 \
    build/test_72



#build/test_07

TEST_RUN = \
    run_test_01 \
    run_test_02 \
    run_test_03 \
    run_test_04 \
    run_test_05 \
    run_test_06 \
    run_test_08 \
    run_test_09 \
    run_test_10 \
    run_test_11 \
    run_test_12 \
    run_test_13 \
    run_test_14 \
    run_test_15 \
    run_test_16 \
    run_test_17 \
    run_test_18 \
    run_test_19 \
    run_test_20 \
    run_test_21 \
    run_test_22 \
    run_test_23 \
    run_test_24 \
    run_test_25 \
    run_test_26 \
    run_test_27 \
    run_test_28 \
    run_test_29 \
    run_test_30 \
    run_test_31 \
    run_test_32 \
    run_test_33 \
    run_test_34 \
    run_test_35 \
    run_test_36 \
    run_test_37 \
    run_test_38 \
    run_test_39 \
    run_test_40 \
    run_test_41 \
    run_test_42 \
    run_test_43 \
    run_test_44 \
    run_test_45 \
    run_test_46 \
    run_test_47 \
    run_test_48 \
    run_test_49 \
    run_test_50 \
    run_test_51 \
    run_test_52 \
    run_test_53 \
    run_test_54 \
    run_test_55 \
    run_test_56 \
    run_test_58 \
    run_test_59 \
    run_test_60 \
    run_test_61 \
    run_test_62 \
    run_test_63 \
    run_test_64 \
    run_test_65 \
    run_test_66 \
    run_test_67 \
    run_test_68 \
    run_test_70 \
    run_test_71 \
    run_test_72



#################################################


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


###################################################


all: builddir ${OBJ}

builddir:
	@if [ ! -d "build" ]; then mkdir -p build; fi


# Build the Tiramisu library object files.  The list of these files is in $(OBJ).
build/tiramisu_%.o: src/tiramisu_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@
build/tiramisu_codegen_%.o: src/tiramisu_codegen_%.cpp $(HEADER_FILES)
	$(CXX) -fPIC ${CXXFLAGS} ${INCLUDES} -c $< -o $@


####################################################


# Build the tutorials. First Object files need to be build, then the
# library generators need to be build and execute (so that they generate
# the libraries), then the wrapper should be built (wrapper are programs that call the
# library functions).
tutorials: $(OBJ) $(TUTO_GEN) $(TUTO_BIN) $(TUTO_RUN)
build/tutorial_%_fct_generator: tutorials/tutorial_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/generated_fct_tutorial_%.o: build/tutorial_%_fct_generator

build/tutorial_%: tutorials/wrapper_tutorial_%.cpp build/tutorial_%_fct_generator build/generated_fct_tutorial_%.o tutorials/wrapper_tutorial_%.h ${OBJ} ${HEADER_FILES}
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 3,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_tutorial_%: build/tutorial_%
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $< 


###################################################################


tests: $(OBJ) $(TEST_GEN) $(TEST_BIN) $(TEST_RUN) build/test_global
build/test_%_fct_generator: tests/test_%.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} $< -o $@ ${INCLUDES} ${LIBRARIES}
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $@
build/generated_fct_test_%.o: build/test_%_fct_generator
build/test_%: tests/wrapper_test_%.cpp build/test_%_fct_generator build/generated_fct_test_%.o tests/wrapper_test_%.h ${OBJ} ${HEADER_FILES}
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 3,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_test_%: build/test_%
	@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $< 
build/test_global: tests/test_global.cpp
	$(CXX) ${CXXFLAGS} ${OBJ} tests/test_global.cpp -o $@ ${INCLUDES} ${LIBRARIES}


###################################################################


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
build/bench_%: benchmarks/halide/wrapper_%.cpp build/generated_fct_%.o build/generated_fct_%_ref.o benchmarks/halide/wrapper_%.h ${OBJ} ${HEADER_FILES}
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) $(word 3,$^) -o $@ ${INCLUDES} ${LIBRARIES}
build/bench_stencils_%: benchmarks/halide/stencils/wrapper_%.cpp build/generated_fct_%.o benchmarks/halide/wrapper_%.h
	$(CXX) ${CXXFLAGS} ${OBJ} $< $(word 2,$^) -o $@ ${INCLUDES} ${LIBRARIES}
run_benchmarks:
	@for tt in ${BENCH_BIN}; do LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${PWD}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${PWD}/build/ $${tt}; done


doc: ${HEADER_FILES}
	doxygen Doxyfile

clean:
	rm -rf *~ src/*~ include/*~ build/* doc/
