#!/bin/bash

set -x

if [ $# -eq 0 ]; then
      echo "Usage: TIRAMISU_SMALL=1 script.sh <KERNEL_FOLDER> <KERNEL_NAME_WITHOUT_EXTENSION>"
      echo "Example: script.sh level1/axpy axpy"
      exit
fi

KERNEL_FOLDER=$1
KERNEL=$2
source ./configure.sh

LLVM_SYS_FLAGS="-lz -lpthread"
#`${TIRAMISU_ROOT}/3rdParty/llvm/build/bin/llvm-config --ignore-libllvm --system-libs`
CXXFLAGS="-O3 $CXXFLAGS"
CFLAGS="-O3 --std=c99"
NVCCFLAGS="-O3 -Wno-deprecated-gpu-targets"
INCLUDES="-I${OPENBLAS_DIR} -I${HALIDE_PREFIX}/include/ -I${TIRAMISU_ROOT}/benchmarks/ -I${TIRAMISU_ROOT}/include/ -I${OPENMP_DIR}/include/libiomp/ -I${BENCHMARK_ROOT}/software/polybench/ -I${BENCHMARK_ROOT}/software/pencil/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/tools/ -I${CUDA_HOME}/include/ -I${PPCG_DIR}/ -I${JPEG}/include/"
LIBRARIES="${OpenBLAS_FLAGS} -ltiramisu -lHalide ${LLVM_SYS_FLAGS} -lpng -ljpeg ${OPENMP_LIB}"
LIBRARIES_DIR="-L${HALIDE_PREFIX}/lib/ -L${OPENBLAS_DIR} -L${TIRAMISU_ROOT}/build/ -L${OPENMP_DIR} -L${CUDA_HOME}/lib64/ -L${JPEG}/lib/"
TILE_TUNING=0

if [ "${TIRAMISU_XLARGE}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_XLARGE"
elif [ "${TIRAMISU_LARGE}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_LARGE"
elif [ "${TIRAMISU_MEDIUM}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_MEDIUM"
elif [ "${TIRAMISU_SMALL}" = "1" ]; then
    DEFINED_SIZE="-DTIRAMISU_SMALL"
else
    DEFINED_SIZE="-DTIRAMISU_LARGE"
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/:${JPEG}/lib/
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/

# Parameters: tile size for dimension 1, tile size for dimension 2 and tile size for dimension 3.
compile_tilable_sgemms()
{
    if [ ${TILE_TUNING} -ne 0 ]; then
        TILE_D1=$1
	TILE_D2=$2
        TILE_D3=$3
    fi

    #################################################################
    #################### Compile PENCIL ########################

    if [ ${RUN_PENCIL} -ne 0 ]; then
	echo "Compiling with PENCIL"
	cd ${KERNEL_FOLDER}

	# Tuned tile sizes for Macbook Pro
	if [ ${TILE_TUNING} -eq 0 ]; then
	    TILE_D1=32
	    TILE_D2=64
	    TILE_D3=128
	fi

	if [ ${COMPILE_WITH_PENCIL} -ne 0 ]; then
		$PPCG ${INCLUDES} --target=c --openmp --tile --tile-size="${TILE_D1},${TILE_D2},${TILE_D3}" --no-isl-schedule-separate-components --isl-schedule-fuse=max $KERNEL.c
	fi
	$CC -c $CFLAGS ${INCLUDES} -fopenmp $KERNEL.ppcg.c -o $KERNEL
	$CC -c $CFLAGS ${INCLUDES} ${BENCHMARK_ROOT}/software/polybench/polybench.c -o polybench
	g++ -fPIC -fno-rtti -std=c++11 $CXXFLAGS ${INCLUDES} $KERNEL polybench wrapper_${KERNEL}.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o wrapper_${KERNEL}

	echo "Running PENCIL-$KERNEL"
	./wrapper_${KERNEL}

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################### Compile PENCIL GPU ########################

    if [ ${RUN_PENCIL_GPU} -ne 0 ]; then
	echo "Compiling with PENCIL GPU"
	cd ${KERNEL_FOLDER}

	# Tuned tile sizes for Macbook Pro
	if [ ${TILE_TUNING} -eq 0 ]; then
	    TILE_D1=32
	    TILE_D2=64
	    TILE_D3=128
	fi

	if [ ${COMPILE_WITH_PENCIL} -ne 0 ]; then
		$PPCG ${INCLUDES} --target=cuda --no-isl-schedule-separate-components --isl-schedule-fuse=max $KERNEL.c
	fi

	${NVCC} --std=c++11 -c ${KERNEL}_host.cu -o ${KERNEL}_host.o
	${NVCC} --std=c++11 -c ${KERNEL}_kernel.cu -o ${KERNEL}_kernel.o
    if [ -z ${PROFILE_CUDA} ]; then
        ${NVCC} -std=c++11 ${INCLUDES} ${KERNEL}_host.o ${KERNEL}_kernel.o wrapper_${KERNEL}.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o wrapper_${KERNEL}
    else
        ${NVCC} -std=c++11 -D__PROFILE_CUDA__ ${INCLUDES} ${KERNEL}_host.o ${KERNEL}_kernel.o wrapper_${KERNEL}.cpp ${LIBRARIES_DIR} ${LIBRARIES} -o wrapper_${KERNEL}
    fi

	echo "Running PENCIL GPU-$KERNEL"
    if [ -z ${PROFILE_CUDA} ]; then
        ./wrapper_${KERNEL}
    else
        ${CUDA_HOME}/bin/nvprof --print-gpu-trace ./wrapper_${KERNEL}
    fi

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi


    #################################################################
    #################################################################
    #################### Compile AlpahZ Gemm ########################

    if [ ${RUN_ALPHAZ} -ne 0 ]; then
	echo "Compiling AlphaZ-gemm"
	cd AlphaZ-gemm/
	./make.sh
	echo "Running AlphaZ-gemm"

	# Matrix size
	p1=1024
	p2=1024
	p3=1024

	# Tuned tile sizes for Macbook Pro
	if [ ${TILE_TUNING} -eq 0 ]; then
	    TILE_D1=128
	    TILE_D2=32
	    TILE_D3=128
	fi

	# parameters for macboopro 1024 1024 1024 32 32 64
	find . -name "gemm"; find . -name "gemm" -exec {} $p1 $p2 $p3 ${TILE_D1} ${TILE_D2} ${TILE_D3} \;

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################################################################
    #################### Compile with Pluto ########################

    if [ ${RUN_PLUTO} -ne 0 ]; then
	echo "Compiling gemm with PLUTO"
	cd ${KERNEL_FOLDER}

	$PLUTO ${PLUTO_OPTS} $KERNEL.c
	$CC $CXXFLAGS ${INCLUDES} wrapper_${KERNEL}.cpp $KERNEL.pluto.c ${BENCHMARK_ROOT}/software/polybench/polybench.c -o $KERNEL
	echo "Running PLUTO generated code"
	./$KERNEL

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################################################################
    #################### Compile with LLVM ##########################

    if [ ${RUN_LLVM} -ne 0 ]; then
	echo "Compiling gemm with LLVM"
	cd ${KERNEL_FOLDER}

	$LLVM -O3 $CXXFLAGS ${INCLUDES} $KERNEL.c ${BENCHMARK_ROOT}/software/polybench/polybench.c -o ${KERNEL}
	echo "Running LLVM generated code"
	./${KERNEL}

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################################################################
    #################### Compile with GCC ###########################

    if [ ${RUN_GCC} -ne 0 ]; then
	echo "Compiling gemm with GCC"
	cd ${KERNEL_FOLDER}

	gcc -O3 $CXXFLAGS ${INCLUDES} $KERNEL.c ${BENCHMARK_ROOT}/software/polybench/polybench.c -o ${KERNEL}
	echo "Running LLVM generated code"
	./${KERNEL}

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################################################################
    #################### Compile with Polly #########################

    if [ ${RUN_POLLY} -ne 0 ]; then
	echo "Compiling gemm with POLLY"
	cd ${KERNEL_FOLDER}

	$LLVM ${POLLY_OPTS} $CXXFLAGS ${INCLUDES} $KERNEL.c ${BENCHMARK_ROOT}/software/polybench/polybench.c -o ${KERNEL}
	echo "Running POLLY generated code"
	./${KERNEL}

	if [ $? -ne 0 ]; then
		exit
	fi

	cd ${BENCHMARK_ROOT}
    fi

    #################################################################
    #################################################################
    #################### Generate Halide Gemm ########################

    if [ ${RUN_HALIDE} -ne 0 ]; then
	cd ${TIRAMISU_ROOT}/3rdParty/Halide/apps/linear_algebra/
	make -j halide_l3_benchmark_sgemm
	cp bin/build/halide_sgemm_notrans.o ${BENCHMARK_ROOT}/OpenBLAS-Halide-gemm/generated_sgemm_halide.o
	cp bin/build/halide_sgemm_notrans.h ${BENCHMARK_ROOT}/OpenBLAS-Halide-gemm/generated_sgemm_halide.h
	cp bin/build/halide_sgemm_notrans.stmt ${BENCHMARK_ROOT}/OpenBLAS-Halide-gemm/generated_sgemm_halide.stmt
	cd -

	cd OpenBLAS-Halide-gemm/

       g++ sgemm.manual.halide.cpp $CXXFLAGS ${INCLUDES} ${LIBRARIES_DIR} ${LIBRARIES} -o sgemm.manual.halide -std=c++11 -fno-rtti 
       HL_DEBUG_CODEGEN=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/ ./sgemm.manual.halide
    fi

    #################################################################
    #################################################################
    #################### Compile OpenBLAS Gemm and Halide wrapper ###

    if [ ${RUN_OpenBLAS} -ne 0 ]; then
        g++ -std=c++11 -fno-rtti $CXXFLAGS ${INCLUDES} ${DEFINED_SIZE} sgemm_wrapper.cpp ${LIBRARIES_DIR} ${LIBRARIES} ${LIBRARIES} -o sgemm_wrapper
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_PREFIX}:${OPENBLAS_DIR}:${TIRAMISU_ROOT}/build/ ./sgemm_wrapper
    fi

    cd ${BENCHMARK_ROOT}
}


if [ ${TILE_TUNING} -ne 0 ]; then
    for D1 in 16 32 64 128
    do
	for D2 in 16 32 64 128
	do
	    for D3 in 16 32 64 128
	    do
		echo "-----------------------------"
		echo "Trying tile size $D1,$D2,$D3"
		compile_tilable_sgemms $D1 $D2 $D3
	    done
	done
    done
else
    compile_tilable_sgemms 64 64 64
fi

