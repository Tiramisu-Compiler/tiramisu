export TIRAMISU_ROOT=/home/nabil/tiramisu/
export LD_LIBRARY_PATH=/home/nabil/tiramisu/3rdParty/Halide/lib

cd ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry11/build
rm -dr ./*

echo trying program auto_scheduler
sed -i -E "s/perform_autoscheduling=(false|true)/perform_autoscheduling=false/gi" ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry11/generator.cpp
cmake ..
make
../generator > log1.txt
g++ -shared -o function.o.so function.o
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o wrapper  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ../wrapper.cpp ./function.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm

sed -i -E "s/perform_autoscheduling=(false|true)/perform_autoscheduling=true/gi" ${TIRAMISU_ROOT}/tutorials/tutorial_autoscheduler/retry11/generator.cpp
cmake ..
make
echo "======================step2==================================" >> log1.txt
../generator >> log1.txt


