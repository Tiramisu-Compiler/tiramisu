#!/bin/bash

TIRAMISU_ROOT=../../../../tiramisu

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./rgbyuv420_ref.cpp.o -c ./rgbyuv420_ref.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./rgbyuv420_ref.cpp.o  -o bench_halide_rgbyuv420_generator ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./rgbyuv420_tiramisu.cpp.o -c ./rgbyuv420_tiramisu.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./rgbyuv420_tiramisu.cpp.o  -o bench_tiramisu_rgbyuv420_generator -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
./bench_halide_rgbyuv420_generator
./bench_tiramisu_rgbyuv420_generator
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./wrapper_rgbyuv420.cpp.o -c ./wrapper_rgbyuv420.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./wrapper_rgbyuv420.cpp.o generated_fct_rgbyuv420.o generated_fct_rgbyuv420_ref.o  -o bench_rgbyuv420 -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
./bench_rgbyuv420
 
rm *.o* bench* 