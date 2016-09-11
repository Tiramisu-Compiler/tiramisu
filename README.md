Computation Optimization Library (COLi)
----------------------------------
COLi is a library that is designed to simplify code optimization and code generation.  The user can express his code in the COLi intermediate representation (COLi IR), he can use the COLi API to perform different optimizations and finaly he can generate an LLVM code from the optimized COLi IR.

Current optimizations include:
- Affine loop nest transformations (non-parametric tiling, loop fusion/distribution, spliting, interchange, shifting, ...),
- For shared memory systems:
  - Loop parallelization, and
  - Loop vectorization.

Current code generators:
- LLVM IR for shared memory systems.


Compiling the COLi Library
----------------------------
#### Prerequisites

- [ISL] (http://repo.or.cz/w/isl.git) Library.
  Check the ISL [README] (http://repo.or.cz/isl.git/blob/HEAD:/README) for details.
- [Halide] (https://github.com/halide/Halide) compiler.
  Halide itself requires llvm-3.7 or greater. Check the Halide [README] (https://github.com/halide/Halide/blob/master/README.md) for details.

#### Compiling COLi
You need to specify the following paths

    ISL_INCLUDE_DIRECTORY: path to the ISL include directory
    ISL_LIB_DIRECTORY: path to the ISL library (lib/)
    HALIDE_SOURCE_DIRECTORY: path to the Halide source code directory
    HALIDE_LIB_DIRECTORY: path to the Halide library (bin/)

You need also to add the Halide library path to your system library path (DYLD_LIBRARY_PATH on Mac OS).

    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<HALIDE_LIB_DIRECTORY>

To build COLi

    make

To build documentation (doxygen required)

    make doc

#### Run Hello World Example

    make tutorial

How To Use COLi
-----------------
COLi provides few classes to enable users to represent their program:
- The `coli::computation` class: a computation is composed of an expression and an iteration space but is not associated with any memory location.
- The `coli::function` class: a function is composed of multiple computations and a vector of arguments (functions arguments).
- The `coli::buffer`: a class to represent memory buffers.

In general, in order to use COLi to optimize, all what a user needs to do is the following:
- Represent the program that needs to be optimized
    - Instantiate a `cori::function`,
    - Instantiate a set of `cori::computation` objects for each function,
- Provide the list of optimizations (memory mapping and schedule)
    - Provide the mapping of each `cori::computation` to memory (i.e. where each computation should be stored in memory),
    - Provide the schedule of each `cori::computation` (a list of loop nest transformations and optimizations such as tiling, parallelization, vectorization, fusion, ...),
- Generate code
    - Generate an AST (Abstract Syntax Tree),
    - Generate target code (an object file),

A full example of how COLi should be used is provided in [tutorial_01](tutorials/tutorial_01.cpp).
