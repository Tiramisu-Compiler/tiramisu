## Configuration

First you need to edit the benchmarks/framework_benchmarking/DNN/configure.sh file to set the right paths.

## Compiling a benchmark

./compile_run_Pencil_DNN_benchmarks.sh <BENCHMARK_FOLDER_NAME> <BENCHMARK_NAME_WITHOUT_EXTENSION>  

**Example**

to compile and run the convolution_Pencil benchmark which is in 

benchmarks/framework_benchmarking/DNN/convolution_layer 

run:  

      cd benchmarks/framework_benchmarking/DNN/

    ./compile_run_Pencil_DNN_benchmarks.sh convolution_layer convolution_Pencil
