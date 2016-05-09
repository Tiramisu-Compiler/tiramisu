ISL_INCLUDE=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/include/
ISL_LIB=/Users/b/Documents/src/MIT/IR/isl_jan_2016_prefix/lib/
HALIDE_INCLUDE_FLAGS=-I ~/Documents/src/MIT/halide/halide_src/include/ -I ~/Documents/src/MIT/halide/halide_src/tools/
HALIDE_LIB_DIR_FLAGS=-L ~/Documents/src/MIT/halide/halide_src/bin/ -lHalide  `libpng-config --cflags --ldflags`
EXTRA_FLAGS=-O3


#all: generate_DEBUG_code_from_halide
all: isl_test


isl_test:
	g++ -std=c++11 ${EXTRA_FLAGS} tests/test_isl.cpp src/IR.cpp src/DebugIR.cpp -L${ISL_LIB} -lisl ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} -Iinclude/ -I${ISL_INCLUDE} -o build/isl_test_executable
	echo; echo;
	DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/b/Documents/src/MIT/halide/halide_src/bin/ build/isl_test_executable


generate_C_code_from_halide:
	g++ -std=c++11 -DGENERATE_C tests/generate_code_from_halide_IR.cpp -I include/ ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} -o build/generate_code_from_halide_IR
	cd build; DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/b/Documents/src/MIT/halide/halide_src/bin/ ./generate_code_from_halide_IR
	g++ -std=c++11 ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} tests/generated_C_pgm_tester.cpp build/generated_C_pgm.cpp -o build/generated_C_mgm_tester 


generate_DEBUG_code_from_halide:
	g++ -std=c++11 -DGENERATE_DEBUGING_CODE tests/generate_code_from_halide_IR.cpp -I include/ ${HALIDE_INCLUDE_FLAGS} ${HALIDE_LIB_DIR_FLAGS} -o build/generate_code_from_halide_IR
	cd build; DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/b/Documents/src/MIT/halide/halide_src/bin/ ./generate_code_from_halide_IR


clean:
	rm -rf src/*~ include/*~ build/*
