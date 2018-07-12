#set -x

source ../../configure_paths.sh

export HalideSrc=${TIRAMISU_ROOT}/3rdParty/Halide/

export INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
export LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread"
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

export HL_DEBUG_CODEGEN=1


rm -rf ./conv_layer_generator ./conv_layer_generator_2 ./conv_layer_generator_tiramisu ./conv_layer_generator_2_tiramisu ./wrapper_nn ./wrapper_nn_2

echo "Halide convolution generator (conv_layer_generator_2.cpp)"
echo "	.Compiling conv_layer_generator_2.cpp (two halide convolutions)"
./compile_halide_code.sh conv_layer_generator_2.cpp
echo "	.Compiling conv_layer_generator_2_tiramisu.cpp (two tiramisu convolutions)"
./compile_tiramisu_code.sh conv_layer_generator_2_tiramisu.cpp
echo "	.Running conv_layer_generator_2 (generator for two-halide-convolutions)"
./conv_layer_generator_2
echo "	.Running conv_layer_generator_tiramisu_2 (generator for two-tiramisu-convolutions)"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./conv_layer_generator_2_tiramisu

echo
echo "	.Compiling wrapper_nn_2.cpp"
g++ -O3 wrapper_nn_2.cpp generated_conv_2.o generated_conv_2_tiramisu.o -g -I $HalideSrc/include -L $HalideSrc/bin ${INCLUDES} -lHalide -lpthread -ldl -ltiramisu ${LIBRARIES_DIR}   -std=c++11 -o wrapper_nn_2
echo "	.Running wrapper_nn_2"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./wrapper_nn_2


echo
echo
echo "MKLDNN convolution (s_score_sample_2)"
./compile_mkldnn_and_run.sh
