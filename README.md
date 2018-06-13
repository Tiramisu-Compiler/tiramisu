## Overview
Tiramisu is a compiler for expressing fast, portable and composable data parallel computations. The user can express algorithms using a simple C++ API and can automatically generate highly optimized code. Tiramisu can be used in areas such as linear and tensor algebra, deep learning, image processing, stencil computations and machine learning.

The Tiramisu compiler is based on the polyhedral model thus it can express a large set of loop optimizations and data layout transformations. It can also target (1) multicore X86 CPUs, (2) ARM CPUs, (3) Nvidia GPUs, (4) Xilinx FPGAs (Vivado HLS) and (5) distributed machines (using MPI) and is designed to enable easy integration of code generators for new architectures.

## Example

The user can write `Tiramisu expressions` within a C++ code as follows.

```cpp
// C++ code with a Tiramisu expression.
#include "tiramisu.h"

void foo(int N, int array_a[N], int array_b[N], int array_c[N])
{
    tiramisu::init();

    tiramisu::in A(int32_t, {N}, array_a), B(int32_t, {N}, array_b);
    tiramisu::out C(int32_t, {N}, array_c);
    
    tiramisu::var i;
    C(i) = A(i) + B(i);
    
    tiramisu::eval("CPU");
}
```

## Building Tiramisu

This section provides a short description of how to build Tiramisu.  A more detailed description is provided in [INSTALL](INSTALL.md).  The installation instructions below have been tested on Linux Ubuntu (14.04) and MacOS (10.12) but should work on other Linux and MacOS versions.

#### Prerequisites
###### Required
1) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).
2) [CMake](https://cmake.org/): version 3.5 or greater.
  
###### Optional
1) [OpenMPI](https://www.open-mpi.org/) and [OpenSSh](https://www.openssh.com/): to run the generated distributed code (MPI).
2) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): to run the generated CUDA code.
3) [Doxygen](http://www.stack.nl/~dimitri/doxygen/): to generate documentation.
4) [Libpng](http://www.libpng.org/pub/png/libpng.html) and [libjpeg](http://libjpeg.sourceforge.net/): to run Halide benchmarks.
5) [Intel MKL](https://software.intel.com/mkl): to run BLAS and DNN benchmarks.


#### Building
1) Get Tiramisu

        git clone https://github.com/Tiramisu-Compiler/tiramisu.git
        cd tiramisu

2) Get and install Tiramisu submodules (ISL, LLVM and Halide)

        ./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_DIR>

3) Optional: configure the tiramisu build by editing `configure.cmake`.  Needed only if you want to generate MPI or GPU code, or if you want to run the BLAS benchmarks.  A description of what each variable is and how it should be set is provided in comments in `configure.cmake`.

    - To use the GPU backend, set `USE_GPU` to `true`.  If the CUDA library is not found automatically while building Tiramisu, the user will be prompt to provide the path to the CUDA library.
    - To use the distributed backend, set `USE_MPI` to `true`.  If the MPI library is not found automatically, set the following variables: MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS.
    - Set MKL_PREFIX to run the BLAS benchmarks.

4) Build the main Tiramisu library

        mkdir build
        cd build
        cmake ..
        make -j tiramisu


## Getting Started
- Read the [Tutorials](tutorials/README.md).
- Read the [Tiramisu Paper](https://arxiv.org/abs/1804.10694).
- Subscribe to Tiramisu [mailing list](https://lists.csail.mit.edu/mailman/listinfo/tiramisu).
- Compiler internal [documentation](TODO).



## Tutorials, Tests and Documentation
#### Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial (tutorial_01 for example)

    make run_tutorial_01
    
This will compile and run the code generator and then the wrapper.

#### Run Tests

To run all the tests, assuming you are in the build/ directory

    make test

or

    ctest
    
To run only one test (test_01 for example)

    ctest -R 01

This will compile and run the code generator and then the wrapper.

To view the output of a test pass the `--verbose` option to `ctest`.

To add a new test, add two files in `tests/`.  Assuming `XX` is the number
of the test that you want to add, `test_XX.cpp` should contain
the actual Tiramisu generator code while `wrapper_test_XX.cpp` should contain
wrapper code.  Wrapper code initializes the input data, calls the generated function,
and compares its output with a reference output.  You should then add the
test number `XX` in the file `tests/test_list.txt`.

#### Run Benchmarks

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

#### Build Documentation

To build documentation (doxygen required)

    make doc
