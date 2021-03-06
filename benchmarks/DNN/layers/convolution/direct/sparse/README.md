The files in this folder are organized as follows:

    General
        clean.sh : remove some useless files.
        compile_and_run_mkldnn.sh : compile MKL-DNN code and run it.
        compile_and_run_mkl.sh : compile MKL code and run it.
        configure.h: define some configuration constants.

    Tiramisu
        spconv_generator.cpp: Tiramisu code generator.

    Wrapper
        spconv_wrapper.cpp: wrapper file that calls the code generated by Tiramisu for sparse weights convolution.

    Intel MKL-DNN
        mkldnn_dense_convolution.cpp : code that calls Intel MKL-DNN's dense convolution.

    Intel MKL
        mkl_dense_convolution.c: code that calls Intel MKL's dense convolution.

To run this benchmark:

    At the directory build/benchmarks/DNN/layers/convolution/direct/sparse execute
	    make

    spconv_wrapper executable will be created in the current directory.

    To compare the result of tiramisu with MKL-DNN execute :
        ./compile_and_run_mkldnn.sh
    then
        ./spconv_wrapper

    To compare the result of tiramisu with MKL execute :
        ./compile_and_run_mkl.sh
    then
        ./spconv_wrapper

    execution results could be found in the text files :
        mkl_result.txt (same for Intel MKL and Intel MKL-DNN)
        tiramisu_result.txt
