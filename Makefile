ISL_INCLUDE=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/include/
ISL_LIB=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/lib/
HALIDE_INCLUDE_FLAGS=-I ~/Documents/src/MIT/halide/halide_src/include/ -I ~/Documents/src/MIT/halide/halide_src/tools/
HALIDE_LIB_DIR_FLAGS=-L ~/Documents/src/MIT/halide/halide_src/bin/ -lHalide  `libpng-config --cflags --ldflags`
EXTRA_FLAGS=-O3


#all: generate_DEBUG_code_from_halide
all: isl_test


isl_test:
	g++ -g -std=c++11 ${EXTRA_FLAGS} tests/test_isl.cpp src/IR.cpp src/DebugIR.cpp -L${ISL_LIB} -lisl ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} -Iinclude/ -I${ISL_INCLUDE} -o build/isl_test_executable
	@echo; echo;echo; echo;
	@DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/b/Documents/src/MIT/halide/halide_src/bin/ build/isl_test_executable
	g++ -g -std=c++11 ${EXTRA_FLAGS} tests/generated_code_wrapper.cpp LLVM_generated_code.o -L${ISL_LIB} -lisl ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} -Iinclude/ -I${ISL_INCLUDE} -o build/final
	@echo; echo;echo; echo;
	@DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/b/Documents/src/MIT/halide/halide_src/bin/ build/final


clean:
	rm -rf LLVM_generated_code.o src/*~ include/*~ build/*
