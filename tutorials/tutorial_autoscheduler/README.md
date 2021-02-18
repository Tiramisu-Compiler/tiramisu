# Autoscheduler tutorial

You need to build the autoscheduler, and you need PyTorch installed to proceed.

To run this tutorial, first set environment variable ```TIRAMISU_ROOT``` to the root directory of tiramisu, and then execute what follows in the root directory of the tutorial :

1. ```mkdir build ; cd build```

2. ```cmake ..```

3. ```make``` : build ```generator.cpp```.

4. ```../generator``` : execute the generator a first time (with the boolean ```perform_autoscheduling = false```), to generate a first ```function.o``` (needed to compile the wrapper).

5. Execute ```g++ -shared -o function.o.so function.o``` to generate a shared library version of ```function.o``` (needed to compile the wrapper).
   
6. Compile the wrapper with the following command :

```g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o wrapper  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ../wrapper.cpp ./function.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm```

7. Open ```generator.cpp```, set ```perform_autoscheduling``` to ```true```, set ```py_cmd_path``` to where python is located, and set ```py_interface_path``` to ```model/main.py``` (please give absolute paths).

8. Open ```model/main.py``` and set ```model_path``` to ```model/hier_LSTM_fusion_tree_tagLo_transfer_5bl.pkl``` (please give absolute path).

9. In the build directory, do : ```make``` to build the generator.

10. Execute the generator to perform autoscheduling : ```../generator```.

11. At the end of autoscheduling, you will see some information. The generated program is in ```function.o```,
and ```function.o.so``` is the same as ```function.o``` but it's a shared library.

12. You can run the generated program by running the wrapper : ```./wrapper```.
