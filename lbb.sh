export TIRAMISU_ROOT=/home/nassim/Desktop/tiramisu/
#export LD_LIBRARY_PATH=/home/nassim/Desktop/tiramisu/3rdParty/Halide/lib


g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o new_generator  -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build new.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm   



cd ${TIRAMISU_ROOT}
./build/new_generator
  
  This generator generates the file generated_fct_developers_tutorial_01.o
  You can compile the wrapper code (code that uses the generated code) and link it to the generated object file.
  
  cd ${TIRAMISU_ROOT}/build

  g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/build -L${TIRAMISU_ROOT}/3rdParty/isl/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o new_exec  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,. ./wrapper_new.cpp  new.o -ltiramisu -lHalide -ldl -lpthread -lz -lm 
  
  To run the program.
  
  ./wrapper_tutorial_01



#
g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o build/new_generator  -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${TIRAMISU_ROOT}/tutorials/developers/a_test/new.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm   

cd ${TIRAMISU_ROOT}
./new_generator

cd ${TIRAMISU_ROOT}/build
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o exe_new  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${TIRAMISU_ROOT}/tutorials/developers/a_test/wrapper_new.cpp  new.o -ltiramisu -lHalide -ldl -lpthread -lz -lm 

./wrapper_tutorial_01
