#set -x

source ../../configure_paths.sh

export HalideSrc=${TIRAMISU_ROOT}/3rdParty/Halide/

export INCLUDES="-I${MKL_PREFIX}/include/ -I${TIRAMISU_ROOT}/include/ -I${HALIDE_SOURCE_DIRECTORY}/include/ -I${ISL_INCLUDE_DIRECTORY} -I${TIRAMISU_ROOT}/benchmarks/"
export LIBRARIES="-ltiramisu ${MKL_FLAGS} -lHalide -lisl -lz -lpthread"
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${HALIDE_LIB_DIRECTORY}/ -L${ISL_LIB_DIRECTORY}/ -L${TIRAMISU_ROOT}/build/"

export HL_DEBUG_CODEGEN=1


rm -rf ./conv_layer_generator ./conv_layer_generator_tiramisu ./wrapper_nn

echo "Halide convolution generator (conv_layer_generator.cpp)"
echo "	.Compiling conv_layer_generator.cpp (one halide convolution)"
./compile_halide_code.sh conv_layer_generator.cpp
echo "	.Compiling conv_layer_generator_tiramisu.cpp (one tiramisu convolution)"
./compile_tiramisu_code.sh conv_layer_generator_tiramisu.cpp
echo "	.Running conv_layer_generator (generator for one-halide-convolution)"
./conv_layer_generator
echo "	.Running conv_layer_generator_tiramisu (generator for one-tiramisu-convolution)"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./conv_layer_generator_tiramisu

echo
echo "Halide convolution test code (wrapper_nn.cpp)"
echo "	.Compiling wrapper_nn.cpp"
g++ -O3 wrapper_nn.cpp generated_conv.o generated_conv_tiramisu.o -g -I $HalideSrc/include ${INCLUDES} -L $HalideSrc/bin -lHalide -lpthread -ldl -ltiramisu ${LIBRARIES_DIR} -std=c++11 -o wrapper_nn
echo "	.Running wrapper_nn"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${ISL_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HALIDE_LIB_DIRECTORY}:${TIRAMISU_ROOT}/build/ ./wrapper_nn


echo
echo
echo "MKLDNN convolution (s_score_sample.c)"
./compile_mkldnn_and_run.sh
