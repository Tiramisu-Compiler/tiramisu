export BEAM_SIZE=1
export DYNAMIC_RUNS=1
export MAX_RUNS=1
export NB_EXEC=1
export AS_VERBOSE=1
export EXPLORE_BY_EXECUTION=1
export SAVE_BEST_SCHED_IN_FILE=0
export EXECUTE_BEST_AND_INITIAL_SCHED=1
export SET_DEFAULT_EVALUATION=0
export PRUNE_SLOW_SCHEDULES=0
export MAX_DEPTH=4
export MIN_RUNS=1
export TIRAMISU_ROOT=/data/kb4083/tiramisu/
mkdir build ; cd build
export BUILD_TARGET="generator"
export FUNCTION_TO_BUILD=$1
cmake ..
make
../generator
g++ -shared -o function.o.so $1.o
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -DTIRAMISU_MINI -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o $1_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ../$1_wrapper.cpp ./function.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm
echo running autoscheduler
export BUILD_TARGET="autoscheduler"
cmake ..
make
../generator