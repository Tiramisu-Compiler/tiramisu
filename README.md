## Building Tiramisu
#### Prerequisites
###### Required
1) Autoconf and libtool.
2) CMake: version 3.5 or greater. [Installation instructions](https://cmake.org/install/).
  
###### Optional
1) Libpng and libjpeg: to run Halide benchmarks/tests.
2) Intel MKL BLAS: to run BLAS benchmarks.
3) Doxygen: to generate documentation.


#### Building
1) Get Tiramisu

        git clone https://github.com/Tiramisu-Compiler/tiramisu.git
        cd tiramisu

2) Get and install submodules (ISL, LLVM and Halide)

        ./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_DIR>

3) Optional: configure the tiramisu build by editing `configure.cmake`.  Needed only if you want to generate MPI or GPU code, or if you want to run the BLAS benchmarks.

    - To use the GPU backend, set `USE_GPU` to `true`.  If the CUDA library is not found automatically, set the following variables: .... (TODO Malek).
    - To use the distributed backend, set `USE_MPI` to `true`.  If the MPI library is not found automatically, set the following variables: MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS.
    - Set MKL_PREFIX to run the BLAS benchmarks.

4) Build the main tiramisu library

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



## Building Tiramisu (Long Version)
#### Prerequisites
###### Required Packages

1) autoconf and libtool.
        
        # On Ubuntu
        sudo apt-get install autoconf libtool
        
        # On MacOS
        sudo brew install autoconf libtool

2) CMake: version 3.5 or greater.
  
        # On Ubuntu
        sudo apt-get install cmake

        # On MacOS
        sudo brew install cmake



###### Optional Packages
1) libpng and libjpeg: to run Halide benchmarks/tests.
2) Intel MKL BLAS: to run BLAS benchmarks.
3) Doxygen: to generate documentation.

The following requirements are supposed to be installed automatically by calling the script `./utils/scripts/install_submodules.sh`. If that script does not work for any reason, you can install the library manually.

3) LLVM-5.0 or greater (required by the [Halide](https://github.com/halide/Halide) framework,
  check the section "Acquiring LLVM" in the Halide [README](https://github.com/halide/Halide/blob/master/README.md) for details on how to get LLVM and install it).



##### Compiling ISL

Install the [ISL](http://repo.or.cz/w/isl.git) Library.  Check the ISL [README](http://repo.or.cz/isl.git/blob/HEAD:/README) for details.  Make sure that autoconf and libtool are installed on your system before building ISL.

To install ISL

        cd 3rdParty/isl
        git submodule update --init --remote --recursive
        mkdir build/
        ./autogen.sh
        ./configure --prefix=$PWD/build/ --with-int=imath
        make -j
        make install

After installing ISL, you need to update the following paths in the configure.cmake to point to the ISL prefix (include and lib directories)

    ISL_INCLUDE_DIRECTORY: path to the ISL include directory
    ISL_LIB_DIRECTORY: path to the ISL library (lib/)

##### Compiling Halide

You need first to set the variable LLVM_CONFIG_BIN to point to the folder that contains llvm-config in the installed LLVM.  You can set this variable in the file configure_paths.sh. If you installed LLVM from source, this path is usually set as follows

    LLVM_CONFIG_BIN=<path to llvm>/build/bin/
    
If you installed LLVM from the distribution packages, you need to find where it was installed and make LLVM_CONFIG_BIN point to the folder that contains llvm-config

To get the Halide submodule and compile it run the following commands (in the Tiramisu root directory)

    git submodule update --init --remote
    cd Halide
    git checkout tiramisu
    make -j

Otherwise, you can use the script retrieve_and_compile_halide.sh to retrieve and compile Halide.

You may get an access rights error from git when running trying to retrieve Halide. To solve this error, be sure to have your machine's ssh key added to your github account, the steps to do so could be found [HERE](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).

##### Building Tiramisu

To build Tiramisu

    cmake CMakeLists.txt
    make -j tiramisu

You need to add the Halide library path to your system library path (DYLD_LIBRARY_PATH on Mac OS).

    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<TIRAMISU_ROOT_DIRECTORY>/Halide/lib/


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

Tutorials
-----------

- [tutorial_01](tutorials/tutorial_01.cpp): A simple example of how to use Tiramisu (a simple assignment).
- [tutorial 02](tutorials/tutorial_02.cpp): blurxy.
- [tutorial 03](tutorials/tutorial_03.cpp): matrix multiplication.
- [tutorial 05](tutorials/tutorial_05.cpp): simple sequence of computations.
- [tutorial 06](tutorials/tutorial_06.cpp): reduction example.
- [tutorial 08](tutorials/tutorial_08.cpp): update example.
- [tutorial 09](tutorials/tutorial_09.cpp): complicate reduction/update example.

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.
