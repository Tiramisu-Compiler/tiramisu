## Setup

#### Prerequisites
1) [Libpng](http://www.libpng.org/pub/png/libpng.html): to run Halide benchmarks.

        # On Ubuntu
        sudo apt-get install libpng-dev
        
        # On MacOS
        sudo brew install libpng

2) [Libjpeg](http://libjpeg.sourceforge.net/): to run Halide benchmarks.

        # On Ubuntu
        sudo apt-get install libjpeg-dev
        
        # On MacOS
        sudo brew install libjpeg

3) [Intel MKL](https://software.intel.com/mkl): to run BLAS and DNN benchmarks.

        # Download and install Intel MKL from:
        https://software.intel.com/mkl


#### Configuration
Edit the file `configure.cmake` to set the variables USE_LIBPNG and USE_JPEG to TRUE and to set the variable MKL_PREFIX to point to the Intel MKL library.  An example of how the MKL_PREFIX variable should be set is available in `configure.cmake`.

In order to use the JPEG library, first you need to recompile Halide with JPEG support. In order to do that, edit the file utils/scripts/install_submodules.sh to set the variable USE_LIBJPEG to 1 and then, clean the Halide installation (cd 3rdParty/Halide/; make clean) and then rerun the Tiramisu installation script (utils/scripts/install_submodules.sh). This will recompile Halide with JPEG support.

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

