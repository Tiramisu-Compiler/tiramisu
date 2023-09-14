# Script to evaluate the generator cpp file without going through the autoscheduler
# This could be useful for manually testing transformations
# To do this add your transformations directly in the generator file before the code generation
# To call this script specify the function name as a parameter
# Example: if your files are named: function_2mm_LARGE_wrapper and function_2mm_LARGE_wrapper
# The call would be: bash evaluate_generator.sh  function_2mm_LARGE
export NB_EXEC=1 # The number of times to execute the program
export TIRAMISU_ROOT=/path/to/Tiramisu # Absolute path to the Tiramisu source code 
export LD_LIBRARY_PATH=${TIRAMISU_ROOT}/3rdParty/Halide/build/src:${TIRAMISU_ROOT}/3rdParty/llvm/build/lib:$LD_LIBRARY_PATH

# Code generation
mkdir build 
cd build
cmake .. -DBUILD_TARGET="generator"  -DFUNCTION_TO_BUILD=$1
make
../generator
g++ -shared -o function.o.so $1.o
# Generate a wrapper
g++ -std=c++17 -fno-rtti -I${TIRAMISU_ROOT}/include -DTIRAMISU_MINI -I${TIRAMISU_ROOT}/3rdParty/Halide/install/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/install/lib64 -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o $1_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ../$1_wrapper.cpp ./function.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm

if [ -z "$GXX" ]; then
    g++ -shared -o $1.o.so $1.o
else
    ${GXX} -shared -o $1.o.so $1.o
fi
# Save the results in a file
echo "$1" >> measurements.txt
./$1_wrapper >> measurements.txt