current_dir=$(pwd)
echo "========================================================================================="
echo "                              COMPILING Sparse ResNet Blocks"
echo "========================================================================================="
echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 2 ResNet block with FIN=16 FOUT=16 and N=112"
echo "_________________________________________________________________________________________"
cd ../../../../build/benchmarks/DNN/blocks/fusedresNet_inference/cpu/sparse_16_16_stride2
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 1 ResNet block with FIN=16 FOUT=16 and N=56"
echo "_________________________________________________________________________________________"
cd ../sparse_16_16
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 2 ResNet block with FIN=16 FOUT=32 and N=56"
echo "_________________________________________________________________________________________"
cd ../sparse_16_32_stride2
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 1 ResNet block with FIN=32 FOUT=32 and N=28"
echo "_________________________________________________________________________________________"
cd ../sparse_32_32
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 2 ResNet block with FIN=32 FOUT=64 and N=28"
echo "_________________________________________________________________________________________"
cd ../sparse_32_64_stride2
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 1 ResNet block with FIN=64 FOUT=64 and N=14"
echo "_________________________________________________________________________________________"
cd ../sparse_64_64
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "========================================================================================="
echo "                              COMPILING SpConv-ReLU-MaxPool Block"
echo "========================================================================================="
echo "_________________________________________________________________________________________"
echo "COMPILING the Stride 1 SpConv-ReLU-MaxPool block with FIN=3 FOUT=16 and N=224"
echo "_________________________________________________________________________________________"
cd ../../../Conv-ReLU-MaxPool/cpu/sparse_3_16_224
make > /dev/null 2>&1
echo "        Done."
echo ""

echo "========================================================================================="
echo "                              COMPILING Add-ReLU Blocks"
echo "========================================================================================="
cd ../../../Add-ReLU/cpu_32_16_58
make > /dev/null 2>&1
echo "        Done."
echo ""

cd ../cpu_32_32_30
make > /dev/null 2>&1
echo "        Done."
echo ""

cd ../cpu_32_64_16
make > /dev/null 2>&1
echo "        Done."
echo ""

echo ""
echo "========================================================================================="
echo "COPYING ALL OF THE OBJECT FILES TO End_To_End_ResNet/tiramisu_functions"
echo "========================================================================================="
cd $current_dir
cd ../../blocks/Conv-ReLU-MaxPool/cpu/
cp ./sparse_3_16_224/generated_spconv_relu_maxpool.o $current_dir/tiramisu_functions
cp ./sparse_3_16_224/generated_spconv_relu_maxpool.o.h $current_dir/tiramisu_functions
crm_dir=$(pwd)/sparse_3_16_224/
echo "        Done. 1/10"

cd $current_dir
cd ../../blocks/fusedresNet_inference/cpu/
cp ./sparse_16_16_stride2/generated_fused_sparse_resnet_block_16_16_stride2.o $current_dir/tiramisu_functions
cp ./sparse_16_16_stride2/generated_fused_sparse_resnet_block_16_16_stride2.o.h $current_dir/tiramisu_functions
rb1_dir=$(pwd)/sparse_16_16_stride2/
echo "        Done. 2/10"

cp ./sparse_16_16/generated_fused_sparse_resnet_block16_16.o $current_dir/tiramisu_functions
cp ./sparse_16_16/generated_fused_sparse_resnet_block16_16.o.h $current_dir/tiramisu_functions
rb2_dir=$(pwd)/sparse_16_16/
echo "        Done. 3/10"


cp ./sparse_16_32_stride2/generated_fused_sparse_resnet_block_16_32_stride2.o $current_dir/tiramisu_functions
cp ./sparse_16_32_stride2/generated_fused_sparse_resnet_block_16_32_stride2.o.h $current_dir/tiramisu_functions
rb3_dir=$(pwd)/sparse_16_32_stride2
echo "        Done. 4/10"

cp ./sparse_32_32/generated_fused_sparse_resnet_block32_32.o $current_dir/tiramisu_functions
cp ./sparse_32_32/generated_fused_sparse_resnet_block32_32.o.h $current_dir/tiramisu_functions
rb4_dir=$(pwd)/sparse_32_32
echo "        Done. 5/10"

cp ./sparse_32_64_stride2/generated_fused_sparse_resnet_block_32_64_stride2.o $current_dir/tiramisu_functions
cp ./sparse_32_64_stride2/generated_fused_sparse_resnet_block_32_64_stride2.o.h $current_dir/tiramisu_functions
rb5_dir=$(pwd)/sparse_32_64_stride2
echo "        Done. 6/10"

cp ./sparse_64_64/generated_fused_sparse_resnet_block64_64.o $current_dir/tiramisu_functions
cp ./sparse_64_64/generated_fused_sparse_resnet_block64_64.o.h $current_dir/tiramisu_functions
rb6_dir=$(pwd)/sparse_64_64
echo "        Done. 7/10"

cd ../../Add-ReLU
cp ./cpu_32_16_58/add_relu_inplace_32_16_58_tiramisu.o $current_dir/tiramisu_functions/add_relu
cp ./cpu_32_16_58/add_relu_inplace_32_16_58_tiramisu.o.h $current_dir/tiramisu_functions/add_relu
add_relu1_dir=$(pwd)/cpu_32_16_58
echo "        Done. 8/10"

cp ./cpu_32_32_30/add_relu_inplace_32_32_30_tiramisu.o $current_dir/tiramisu_functions/add_relu
cp ./cpu_32_32_30/add_relu_inplace_32_32_30_tiramisu.o.h $current_dir/tiramisu_functions/add_relu
add_relu2_dir=$(pwd)/cpu_32_32_30
echo "        Done. 9/10"

cp ./cpu_32_64_16/add_relu_inplace_32_64_16_tiramisu.o $current_dir/tiramisu_functions/add_relu
cp ./cpu_32_64_16/add_relu_inplace_32_64_16_tiramisu.o.h $current_dir/tiramisu_functions/add_relu
add_relu3_dir=$(pwd)/cpu_32_64_16
echo "        Done. 10/10"

echo "========================================================================================="
echo "CLEANING BLOCKS' FOLDERS"
echo "========================================================================================="
echo "        Cleaning Conv-ReLU-MaxPool folder..."
cd $crm_dir
./clean.sh > /dev/null 2>&1
echo "             Done."

echo "        Cleaning all ResNet Block folders..."
cd $rb1_dir
./clean.sh > /dev/null 2>&1
cd $rb2_dir
./clean.sh > /dev/null 2>&1
cd $rb3_dir
./clean.sh > /dev/null 2>&1
cd $rb4_dir
./clean.sh > /dev/null 2>&1
cd $rb5_dir
./clean.sh > /dev/null 2>&1
cd $rb6_dir
./clean.sh > /dev/null 2>&1
echo "             Done."

echo "        Cleaning all Add-ReLU Block folders..."
cd $add_relu1_dir
./clean.sh > /dev/null 2>&1
cd $add_relu2_dir
./clean.sh > /dev/null 2>&1
cd $add_relu3_dir
./clean.sh > /dev/null 2>&1
echo "             Done."

echo ""
echo "========================================================================================="
echo "COMPILING THE END TO END ResNet ARCHITECTURE"
echo "========================================================================================="
cd $current_dir
cd ../../../../build/benchmarks/DNN/end_to_end_architectures/End_To_End_ResNet
make > /dev/null 2>&1
echo "        Finished."
cd $current_dir

echo "============================================================================================================================"
echo "The ResNet Architecture has been successfully built and compiled"
echo "You can now execute the MKL-DNN code using ./compile_and_run_mkldnn.sh in the benchmarks/DNN/blocks/End_To_End_ResNet/ directory"
echo "Then execute ./wrapper_end_to_end_resnet_tiramisu to compare results"
