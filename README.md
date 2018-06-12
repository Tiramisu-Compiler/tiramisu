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


How To Use Tiramisu
----------------------
Tiramisu provides few classes to enable users to represent their program:
- The `tiramisu::computation` class: a computation is composed of an expression and an iteration space but is not associated with any memory location.
- The `tiramisu::function` class: a function is composed of multiple computations and a vector of arguments (functions arguments).
- The `tiramisu::buffer`: a class to represent memory buffers.

In general, in order to use Tiramisu to optimize, all what a user needs to do is the following:
- Represent the program that needs to be optimized
    - Instantiate a `tiramisu::function`,
    - Instantiate a set of `tiramisu::computation` objects for each function,
- Provide the list of optimizations (memory mapping and schedule)
    - Provide the mapping of each `tiramisu::computation` to memory (i.e. where each computation should be stored in memory),
    - Provide the schedule of each `tiramisu::computation` (a list of loop nest transformations and optimizations such as tiling, parallelization, vectorization, fusion, ...),
- Generate code
    - Generate an AST (Abstract Syntax Tree),
    - Generate target code (an object file),

Ressources
------------
#### Learning Tiramisu
- The page [Tutorials](tutorials/README.md) has two tutorials: one for Tiramisu users and the other one for new Tiramisu compiler developers (useful for adding new features, optimizations and backends).
