This benchmarks adds two output feature maps then applies a ReLU, it is used in the ResNet Architecture

To run this benchmark:

    At the directory build/benchmarks/DNN/blocks/Add-ReLU/cpu_32_32_30 execute
	    make

    wrapper_nn_block_add_relu_inplace_32_32_30 executable will be created in the current directory.

    To compare the result of tiramisu with MKL execute :
        ./compile_and_run_mkl.sh
    then
        ./wrapper_nn_block_add_relu_inplace_32_32_30

    execution results could be found in the text files :
        mkl_result.txt (same for Intel MKL)
        tiramisu_result.txt
