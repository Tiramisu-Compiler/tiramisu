LANKA=0

if [ $LANKA -eq 0 ]; then
	TIRAMISU_ROOT=/Users/b/Documents/src/MIT/tiramisu/
	MKL_FLAGS="-lcblas"
	MKL_LIB_PATH_SUFFIX=
	LANKA_OPTIONS=
	USE_PERF=0
	MKL_PREFIX=/opt/intel/compilers_and_libraries/mac/mkl/
	CORES=4
else
	TIRAMISU_ROOT=/data/scratch/baghdadi/tiramisu/
	MKL_FLAGS="-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -ldl -lm"
	MKL_LIB_PATH_SUFFIX=intel64/
	LANKA_OPTIONS="-DMKL_ILP64 -m64 -fopenmp"
	USE_PERF=1
	MKL_PREFIX=/data/scratch/baghdadi/libs/intel/mkl/
	CORES=32
fi

ISL_INCLUDE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/include/
ISL_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
HALIDE_SOURCE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide
HALIDE_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide/lib

