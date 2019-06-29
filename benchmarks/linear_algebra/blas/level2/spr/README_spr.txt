//================================== follow STEPS: ===================================

1- put spr folder in /home/b/tiramisu/benchmarks/linear_algebra/blas/level2/


2- set Golbal variables :
export TIRAMISU_ROOT=/home/b/tiramisu
export PROG_FOLDER_PATH=/home/b/tiramisu/benchmarks/linear_algebra/blas/level2/spr

3- 
cd /home/b/tiramisu/build

4-
g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o spr_fct_generator  -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${PROG_FOLDER_PATH}/spr.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm

5-
cd ../

6-
./build/spr_fct_generator

7-
cd build/

8-
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o wrapper_spr  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${PROG_FOLDER_PATH}/wrapper_spr.cpp ./generated_fct_developers_spr.o -ltiramisu -lHalide -ldl -lpthread -lz -lm

9-
./wrapper_spr
