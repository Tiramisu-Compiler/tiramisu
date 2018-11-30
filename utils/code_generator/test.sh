export last_code_num=4
export test_path=`pwd`
cd ../..
export TIRAMISU_ROOT=`pwd`
cd ${TIRAMISU_ROOT}/build
mkdir generated
cd generated
touch exec_times.txt
cp ../libtiramisu.so .
for test_no in $(seq 0 $last_code_num)
do
	echo "running program no : "$test_no
    export TEST_ROOT=$test_path"/samples/function${test_no}"
    cd ${TIRAMISU_ROOT}/build/generated
    g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/ -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o test_fct_${test_no}_generator  -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build/generated ${TEST_ROOT}/function${test_no}_file.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm   
  
    cd ${TIRAMISU_ROOT}
    ./build/generated/test_fct_${test_no}_generator

    cd ${TIRAMISU_ROOT}/build/generated
    g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o wrapper_function${test_no}  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build/generated ${TEST_ROOT}/function${test_no}_wrapper_file.cpp  generated_function${test_no}.o -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm 
  	start=`date +%s%N | cut -b1-13`
    ./wrapper_function${test_no}
	end=`date +%s%N | cut -b1-13`
	runtime=$((end-start))
	echo "runtime : "$runtime" ms" > $TEST_ROOT/stats.txt 
done
