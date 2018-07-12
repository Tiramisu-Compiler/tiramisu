LANKA=0

if [ $LANKA -eq 0 ]; then
	export TIRAMISU_ROOT=/Users/b/Documents/src/MIT/tiramisu/
	export MKL_FLAGS="-lcblas"
	export MKL_LIB_PATH_SUFFIX=
	export LANKA_OPTIONS=
	export USE_PERF=0
	export MKL_PREFIX=/opt/intel/compilers_and_libraries/mac/mkl/
	export CORES=4
else
	export TIRAMISU_ROOT=/data/scratch/baghdadi/tiramisu/
	export MKL_FLAGS="-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -ldl -lm"
	export MKL_LIB_PATH_SUFFIX=intel64/
	export LANKA_OPTIONS="-DMKL_ILP64 -m64 -fopenmp"
	export USE_PERF=1
	export MKL_PREFIX=/data/scratch/baghdadi/libs/intel/mkl/
	export CORES=32
fi

export ISL_INCLUDE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/include/
export ISL_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/isl/build/lib/
export HALIDE_SOURCE_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide
export HALIDE_LIB_DIRECTORY=${TIRAMISU_ROOT}/3rdParty/Halide/lib

