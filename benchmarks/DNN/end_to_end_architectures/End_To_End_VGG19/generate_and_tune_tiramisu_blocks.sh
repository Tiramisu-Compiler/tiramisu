current_dir=$(pwd)

echo "==================================================================="
echo "TUNING THE Conv-ReLU-Conv-ReLU block with FIN=256 FOUT=512 and N=28"
echo "==================================================================="
cd ../../blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28
echo "Tuning..."
./tuning_script.sh
echo "Copying object files..."
cp generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
cp generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o.h ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
echo "Done."
echo ""

echo "======================================================================================"
echo "TUNING THE vggBlock (Conv-ReLU-Conv-ReLU-MaxPool) block with FIN=512 FOUT=512 and N=28"
echo "======================================================================================"
cd ../../../vggBlock/cpu/sparse_32channels_512_512_28
echo "Tuning..."
./tuning_script.sh
echo "Copying object files..."
cp generated_sparse_vgg_block_512_512_28_tiramisu.o ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
cp generated_sparse_vgg_block_512_512_28_tiramisu.o.h ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
echo "Done."
echo ""

echo "==================================================================="
echo "TUNING THE Conv-ReLU-Conv-ReLU block with FIN=512 FOUT=512 and N=14"
echo "==================================================================="
cd ../../../Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_512_512_14
echo "Tuning..."
./tuning_script.sh
echo "Copying object files..."
cp generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
cp generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o.h ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
echo "Done."
echo ""

echo "======================================================================================"
echo "TUNING THE vggBlock (Conv-ReLU-Conv-ReLU-MaxPool) block with FIN=512 FOUT=512 and N=14"
echo "======================================================================================"
cd ../../../vggBlock/cpu/sparse_32channels_512_512_14
echo "Tuning..."
./tuning_script.sh
echo "Copying object files..."
cp generated_sparse_vgg_block_512_512_14_tiramisu.o ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
cp generated_sparse_vgg_block_512_512_14_tiramisu.o.h ../../../../end_to_end_architectures/End_To_End_VGG19/tiramisu_functions/
echo "Done."
echo ""

echo "========================================================================================="
echo "COMPILING THE END TO END VGG19 ARCHITECTURE"
echo "========================================================================================="
cd ../../../../../../build/benchmarks/DNN/end_to_end_architectures/End_To_End_VGG19
make > /dev/null 2>&1
echo "        Finished."
cd $current_dir

echo ""
echo "============================================================================================================================"
echo "The blocks have been tuned and succesfully compiled."
echo "The VGG19 Architecture has been successfully built and compiled"
echo "You can now execute the MKL-DNN code using ./compile_and_run_mkldnn.sh in the benchmarks/DNN/blocks/End_To_End_VGG19/ directory"
echo "Then execute ./wrapper_end_to_end_vgg19_tiramisu to compare results"
