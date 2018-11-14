[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Build Status](https://travis-ci.org/Tiramisu-Compiler/tiramisu.svg?branch=master)](https://travis-ci.org/Tiramisu-Compiler/tiramisu)

## Overview

Tiramisu is a compiler for expressing fast and portable data parallel computations.  It provides a simple C++ API for expressing algorithms (`Tiramisu expressions`) and how these algorithms should be optimized by the compiler.  Tiramisu can be used in areas such as linear and tensor algebra, deep learning, image processing, stencil computations and machine learning.

The Tiramisu compiler is based on the polyhedral model thus it can express a large set of loop optimizations and data layout transformations.  Currently it targets (1) multicore X86 CPUs, (2) Nvidia GPUs, (3) Xilinx FPGAs (Vivado HLS) and (4) distributed machines (using MPI).  It is designed to enable easy integration of code generators for new architectures.

### Example

The following is an example of a Tiramisu program specified using the C++ API.

```cpp
// C++ code with a Tiramisu expression.
#include "tiramisu/tiramisu.h"
using namespace tiramisu;

void generate_code()
{
    // Specify the name of the function that you want to create.
    tiramisu::init("foo");

    // Declare two iterator variables (i and j) such that 0<=i<100 and 0<=j<100.
    var i("i", 0, 100), j("j", 0, 100);

    // Declare a Tiramisu expression (algorithm) that is equivalent to the following C code
    // for (i=0; i<100; i++)
    //   for (j=0; j<100; j++)
    //     C(i,j) = 0;
    computation C({i,j}, 0);
    
    // Specify optimizations
    C.parallelize(i);
    C.vectorize(j, 4);
    
    buffer b_C("b_C", {100, 100}, p_int32, a_output);
    C.store_in(&b_C);

    // Generate code
    C.codegen({&b_C}, "generated_code.o");
}
```

## Building Tiramisu from Sources

This section provides a short description of how to build Tiramisu.  A more detailed description is provided in [INSTALL](INSTALL.md).  The installation instructions below have been tested on Linux Ubuntu (14.04 and 18.04) and MacOS (10.12) but should work on other Linux and MacOS versions.

#### Prerequisites
###### Required
1) [CMake](https://cmake.org/): version 3.5 or greater.

2) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).
  
###### Optional
1) [OpenMPI](https://www.open-mpi.org/) and [OpenSSh](https://www.openssh.com/): if you want to generate and run distributed code (MPI).
2) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): if you want to generate and run CUDA code.


#### Building
1) Get Tiramisu

        git clone https://github.com/Tiramisu-Compiler/tiramisu.git
        cd tiramisu

2) Get and install Tiramisu submodules (ISL, LLVM and Halide).  This step may take between few minutes to few hours (downloading and compiling LLVM is time consuming).

        ./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_DIR>

3) Optional: configure the tiramisu build by editing `configure.cmake`.  Needed only if you want to generate MPI or GPU code, or if you want to run the BLAS benchmarks.  A description of what each variable is and how it should be set is provided in comments in `configure.cmake`.

    - To use the GPU backend, set `USE_GPU` to `TRUE`. If the CUDA library is not found automatically while building Tiramisu, the user will be prompt to provide the path to the CUDA library.
    - To use the distributed backend, set `USE_MPI` to `TRUE`. If the MPI library is not found automatically, set the following variables: MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS.

4) Build the main Tiramisu library

        mkdir build
        cd build
        cmake ..
        make -j tiramisu

## Tiramisu on a Virtual Machine
Users can use the Tiramisu [virtual machine disk image](http://groups.csail.mit.edu/commit/software/TiramisuVM.zip).  The image is created using virtual box (5.2.12) and has Tiramisu already pre-compiled and ready for use. It was compiled using the same instructions in this README file.

Once you download the image, unzip it and use virtual box to open the file 'TiramisuVM.vbox'.

Once the virtual machine has started, open a terminal, then go to the Tiramisu directory

    cd /home/b/tiramisu/
    
If asked for a username/password

    Username:b
    Password:b


## Getting Started
- Build [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu/).
- Read the [Tutorials](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/tutorials/README.md).
- Read the [Tiramisu Paper](https://arxiv.org/abs/1804.10694).
- Subscribe to Tiramisu [mailing list](https://lists.csail.mit.edu/mailman/listinfo/tiramisu).
- Read the compiler [internal documentation](https://tiramisu-compiler.github.io/doc/) (if you want to contribute to the compiler).


## Run Tests

To run all the tests, assuming you are in the build/ directory

    make test

or

    ctest
    
To run only one test (test_01 for example)

    ctest -R 01

This will compile and run the code generator and then the wrapper.

To view the output of a test pass the `--verbose` option to `ctest`.
