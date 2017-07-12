Tiramisu Optimization Framework
----------------------------------
Tiramisu is a library that is designed to simplify code optimization and code generation.  The user can express his code in the Tiramisu intermediate representation (Tiramisu IR), he can use the Tiramisu API to perform different optimizations and finaly he can generate an LLVM code from the optimized Tiramisu IR.

Current optimizations include:
- Affine loop nest transformations (non-parametric tiling, loop fusion/distribution, spliting, interchange, shifting, ...),
- For shared memory systems:
  - Loop parallelization, and
  - Loop vectorization.

Current code generators:
- LLVM IR for shared memory systems.


Compiling Tiramisu
----------------------
#### Prerequisites

- autoconf and libtool. On Ubuntu you can simply install them using

        sudo apt-get install autoconf
        sudo apt-get install libtool

- LLVM-3.7 or greater (required by the [Halide] (https://github.com/halide/Halide) framework,
  check the section "Acquiring LLVM" in the Halide [README] (https://github.com/halide/Halide/blob/master/README.md) for details on how to get LLVM and install it).

#### Short Version
        git clone https://github.com/rbaghdadi/tiramisu.git
        cd tiramisu
        ./get_and_install_tiramisu_dependencies.sh
        make -j

#### Long Version

##### Compiling Tiramisu
Install the [ISL] (http://repo.or.cz/w/isl.git) Library.  Check the ISL [README] (http://repo.or.cz/isl.git/blob/HEAD:/README) for details. Be sure to have installed autoconf and libtool installed on your machine before building ISL by running the following commands:

        sudo apt-get install autoconf
        sudo apt-get install libtool


You need to specify the following paths in the Makefile

    ISL_INCLUDE_DIRECTORY: path to the ISL include directory
    ISL_LIB_DIRECTORY: path to the ISL library (lib/)

To get the Halide submodule and compile it run the following commands (in the Tiramisu root directory)

    git submodule update --init --remote
    cd Halide
    git checkout tiramisu
    make

Otherwise, you can use the script retrieve_and_compile_halide.sh to retrieve
and compile Halide. Note that you need to set the path to an installed LLVM
first. To do so, set the variable LLVM_PREFIX in the script.

You may get an access rights error from git when running trying to retrieve Halide. To solve this error, be sure to have your machine's ssh key added to your github account, the steps to do so could be found [HERE](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).

To build Tiramisu

    make

You need to add the Halide library path to your system library path (DYLD_LIBRARY_PATH on Mac OS).

    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<TIRAMISU_ROOT_DIRECTORY>/Halide/lib/

To build documentation (doxygen required)

    make doc

#### Run tutorials

    make tutorials

#### Run tests

    make tests


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

A full example of how Tiramisu should be used is provided in [tutorial_01](tutorials/tutorial_01.cpp).
