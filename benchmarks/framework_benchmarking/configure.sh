TIRAMISU_ROOT=/Users/b/Documents/src/MIT/tiramisu/
BENCHMARK_ROOT=${TIRAMISU_ROOT}/benchmarks/framework_benchmarking/
LLVM_PREFIX=${TIRAMISU_ROOT}/3rdParty/llvm/prefix/
CC=${LLVM_PREFIX}/bin/clang
OPENMP_LIB=iomp5
OPENMP_DIR=/usr/local/Cellar/libiomp/20150701/
OPENBLAS_DIR=/Volumes/ALL/extra/OpenBLAS/
OpenBLAS_FLAGS="-lcblas"

PPCG=${BENCHMARK_ROOT}/software/ppcg/ppcg
HALIDE_PREFIX=${TIRAMISU_ROOT}/3rdParty/Halide/
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
