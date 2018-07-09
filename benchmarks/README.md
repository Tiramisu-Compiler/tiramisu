## Setup

#### Prerequisites
1) [Libpng](http://www.libpng.org/pub/png/libpng.html): to run Halide benchmarks.

        # On Ubuntu
        sudo apt-get install libpng-dev
        
        # On MacOS
        sudo brew install libpng

2) [Intel MKL](https://software.intel.com/mkl): to run BLAS and DNN benchmarks.

        # Download and install Intel MKL from:
        https://software.intel.com/mkl


#### Configuration
Edit the file `configure.cmake` to set the variable USE_LIBPNG to TRUE and variable MKL_PREFIX to point to the Intel MKL library.  An example of how the MKL_PREFIX variable should be set is available in `configure.cmake`.

## Run Benchmarks

To run all the benchmarks, assuming you are in the build/ directory

    make benchmarks

To run only one benchmark (cvtcolor for example)

    make run_benchmark_cvtcolor

If you want to force the rebuild of a given benchmark, add -B option.

    make -B run_benchmark_cvtcolor

This will rebuild tiramisu, rebuild all the stage of code generation and run
the benchmark.

To add a given benchmark to the build system, add its name in the file
`benchmarks/benchmark_list.txt`.

