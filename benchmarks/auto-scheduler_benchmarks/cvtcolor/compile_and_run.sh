#!/bin/bash

TIRAMISU_ROOT=../../../../tiramisu

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./cvtcolor_ref.cpp.o -c ./cvtcolor_ref.cpp

g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./cvtcolor_ref.cpp.o  -o bench_halide_cvtcolor_generator ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./cvtcolor_tiramisu.cpp.o -c ./cvtcolor_tiramisu.cpp

g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./cvtcolor_tiramisu.cpp.o  -o bench_tiramisu_cvtcolor_generator -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 

./bench_tiramisu_cvtcolor_generator

./bench_halide_cvtcolor_generator

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./wrapper_cvtcolor.cpp.o -c ./wrapper_cvtcolor.cpp

g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./wrapper_cvtcolor.cpp.o generated_fct_cvtcolor.o generated_fct_cvtcolor_ref.o  -o bench_cvtcolor -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 

./bench_cvtcolor

rm *.o* bench* 