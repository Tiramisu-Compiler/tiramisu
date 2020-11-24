current_dir=$(pwd)
echo "========================================================================================="
echo "COMPILING the Conv-ReLU-Conv-ReLU block with FIN=256 FOUT=512 and N=28"
echo "========================================================================================="
cd ../../../../build/benchmarks/DNN/blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "========================================================================================="
echo "COMPILING the Conv-ReLU-Conv-ReLU block with FIN=512 FOUT=512 and N=14"
echo "========================================================================================="
cd ../sparse_32channels_512_512_14
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "========================================================================================="
echo "COMPILING the vggBlock (Conv-ReLU-Conv-ReLU-MaxPool) block with FIN=512 FOUT=512 and N=28"
echo "========================================================================================="
cd ../../../vggBlock/cpu/sparse_32channels_512_512_28
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "========================================================================================="
echo "COMPILING the vggBlock (Conv-ReLU-Conv-ReLU-MaxPool) block with FIN=512 FOUT=512 and N=14"
echo "========================================================================================="
cd ../sparse_32channels_512_512_14
make > /dev/null 2>&1
echo "        Done."
echo ""

echo ""
echo "========================================================================================="
echo "COPYING ALL OF THE OBJECT FILES TO End_To_End_VGG19/tiramisu_functions"
echo "========================================================================================="
cd ../../../../../../../benchmarks/DNN/end_to_end_architectures/End_To_End_VGG19
cp ../../blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28/generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o ./tiramisu_functions
cp ../../blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28/generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o.h ./tiramisu_functions
echo "        Done. 1/4"

cp ../../blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_512_512_14/generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o ./tiramisu_functions
cp ../../blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_512_512_14/generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o.h ./tiramisu_functions
echo "        Done. 2/4"


cp ../../blocks/vggBlock/cpu/sparse_32channels_512_512_28/generated_sparse_vgg_block_512_512_28_tiramisu.o ./tiramisu_functions
cp ../../blocks/vggBlock/cpu/sparse_32channels_512_512_28/generated_sparse_vgg_block_512_512_28_tiramisu.o.h ./tiramisu_functions
echo "        Done. 3/4"

cp ../../blocks/vggBlock/cpu/sparse_32channels_512_512_14/generated_sparse_vgg_block_512_512_14_tiramisu.o ./tiramisu_functions
cp ../../blocks/vggBlock/cpu/sparse_32channels_512_512_14/generated_sparse_vgg_block_512_512_14_tiramisu.o.h ./tiramisu_functions
echo "        Done. 4/4"

echo ""
echo "========================================================================================="
echo "COMPILING THE END TO END VGG19 ARCHITECTURE"
echo "========================================================================================="
cd ../../../../build/benchmarks/DNN/end_to_end_architectures/End_To_End_VGG19/
make > /dev/null 2>&1
echo "        Finished."
cd $current_dir

echo "============================================================================================================================"
echo "The VGG19 Architecture has been successfully built and compiled"
echo "You can now execute the MKL-DNN code using ./compile_and_run_mkldnn.sh in the benchmarks/DNN/blocks/End_To_End_VGG19/ directory"
echo "Then execute ./wrapper_end_to_end_vgg19_tiramisu to compare results"
