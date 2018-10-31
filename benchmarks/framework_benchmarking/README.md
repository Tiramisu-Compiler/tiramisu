# Configuration

First you need to edit the benchmarks/framework_benchmarking/configure.sh file to set the right paths.

# Compiling a benchmark

      ./compile_and_run_benchmarks.sh <KERNEL_FOLDER> <KERNEL_NAME_WITHOUT_EXTENSION>

Example: to compile and run the convolution benchmark which is in benchmarks/framework_benchmarking, run:

      cd benchmarks/framework_benchmarking
      ./compile_and_run_benchmarks.sh image_processing/convolution convolution
