#!/bin/bash

TIRAMISU_ROOT=../../../../tiramisu

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./heat2d_ref.cpp.o -c ./heat2d_ref.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./heat2d_ref.cpp.o  -o bench_halide_heat2d_generator ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./heat2d_tiramisu.cpp.o -c ./heat2d_tiramisu.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./heat2d_tiramisu.cpp.o  -o bench_tiramisu_heat2d_generator -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
./bench_halide_heat2d_generator
./bench_tiramisu_heat2d_generator
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./wrapper_heat2d.cpp.o -c ./wrapper_heat2d.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./wrapper_heat2d.cpp.o generated_fct_heat2d.o generated_fct_heat2d_ref.o  -o bench_heat2d -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
./bench_heat2d
 
rm *.o* bench* 