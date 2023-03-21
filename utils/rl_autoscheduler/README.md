# RL_in_Search_space_exploration

## Running the script
```bash
sbatch run_rllib_slurm.sh

```  

## TODO:  
- Fix Hydra.
- Store explored schedules.

```bash
c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 
${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl
    
g++ -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o ${FUNC_NAME}_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ./${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm'
${GXX} -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O3 -o ${FUNC_NAME}_wrapper ${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl


```