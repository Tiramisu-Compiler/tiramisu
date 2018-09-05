LANKA=1

export TIRAMISU_ROOT=/data/scratch/baghdadi/tiramisu/
export BENCHMARK_ROOT=${TIRAMISU_ROOT}/benchmarks/framework_benchmarking/
export LLVM_PREFIX=${TIRAMISU_ROOT}/3rdParty/llvm/prefix/

if [ $LANKA -eq 0 ]; then
	export CXXFLAGS=""
	export CC=clang-omp
	export OPENMP_LIB=-liomp5
	export OPENMP_DIR=/usr/local/Cellar/libiomp/20150701/
	export OPENBLAS_DIR=/Volumes/ALL/extra/OpenBLAS/
	export OpenBLAS_FLAGS="-lcblas"
else
	export CXXFLAGS="-fopenmp"
	export CC=gcc
	export OPENMP_LIB=""
	export OPENMP_DIR=.
	export OPENBLAS_DIR=.
	export OpenBLAS_FLAGS=""
fi

PPCG=${BENCHMARK_ROOT}/software/ppcg/ppcg
HALIDE_PREFIX=${TIRAMISU_ROOT}/3rdParty/Halide
PLUTO=/Users/b/Documents/src-not-saved/pluto-0.11.4/polycc
PLUTO_OPTS="--tile --parallel"
POLLY="/Volumes/ALL/extra/polly/llvm_build/bin/clang"
POLLY_OPTS="-mllvm -polly -mllvm -polly-vectorizer=stripmine -mllvm -polly-parallel"

RUN_PLUTO=0
RUN_PENCIL=1
RUN_ALPHAZ=0
RUN_OpenBLAS=0
RUN_HALIDE=0
RUN_POLLY=0
