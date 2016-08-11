Code Optimization Library (COLi)
----------------------------------
COLi is a library that is designed to simplify code optimization and code generation.  The user can express his code in the COLi intermediate representation (COLi IR), he can use the COLi API to perform different optimizations and finaly he can generate an LLVM code from the optimized COLi IR.

Current optimizations include:
- Affine loop nest transformations (non-parametric tiling, loop fusion/distribution, spliting, interchange, shifting, ...),
- For shared memory systems:
  - Loop parallelization, and
  - Loop vectorization.

Current code generators:
- LLVM IR for shared memory systems.

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

To compile
    make

#### Run Tests
    make test
