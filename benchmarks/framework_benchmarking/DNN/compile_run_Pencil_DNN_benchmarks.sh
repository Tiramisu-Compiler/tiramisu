#!/bin/bash
source ./configure.sh
if [ "$#" -eq 0 ]; then
	echo "Error: use  compile_run_Pencil_DNN_benchmarks.sh <folder_benchmark_name> <benchmark_name>"
	exit 1
fi
KERNEL_FOLDER=$1
KERNEL=$2

echo "Compiling with PENCIL"
	cd ${KERNEL_FOLDER}
	if [ ${TILE_TUNING} -eq 0 ]; then
	    TILE_D1=32
	    TILE_D2=64
	    TILE_D3=128
	fi

        $PPCG --target=c --openmp --tile --tile-size="${TILE_D1},${TILE_D2},${TILE_D3}" $KERNEL.c 
	gcc $DNNFLAGS $KERNEL.ppcg.c 

echo "Running $KERNEL"
	./a.out


