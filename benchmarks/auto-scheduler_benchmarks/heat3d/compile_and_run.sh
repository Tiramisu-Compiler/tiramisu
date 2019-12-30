#!/bin/bash

TIRAMISU_ROOT=../../../../tiramisu

g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./heat3d_ref.cpp.o -c ./heat3d_ref.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./heat3d_ref.cpp.o  -o bench_halide_heat3d_generator ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./heat3d_tiramisu.cpp.o -c ./heat3d_tiramisu.cpp
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./heat3d_tiramisu.cpp.o  -o bench_tiramisu_heat3d_generator -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
g++   -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/build/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/Halide/tools -I${TIRAMISU_ROOT}/build  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   -I/usr/include/libpng16 -o ./wrapper_heat3d.cpp.o -c ./wrapper_heat3d.cpp
./bench_tiramisu_heat3d_generator
./bench_halide_heat3d_generator
g++  -std=c++11 -Wall -Wno-sign-compare -fno-rtti -g -O0   ./wrapper_heat3d.cpp.o generated_fct_heat3d.o generated_fct_heat3d_ref.o  -o bench_heat3d -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/isl/build/lib ${TIRAMISU_ROOT}/build/libtiramisu.so ${TIRAMISU_ROOT}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU_ROOT}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -ldl -lpthread -lz -lm -lpng16 -ljpeg 
./bench_heat3d

rm *.o* bench* 