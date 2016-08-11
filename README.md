Code Optimization Library (COLi)
----------------------------------
COLi is a library that is designed to simplify code optimization
and code generation.  The user can express his code in the COLi
intermediate representation and can use the COLi API to specify
the code optimizations that he wants to apply on the code.
Finaly the user can generate an optimized code that can be
executed.

Current optimizations include:
- All the affine loop nest transformations (non-parametric
  tiling, loop fusion/distribution, spliting,
  interchange, shifting, ...),
- Shared memory systems:
  - Loop parallelization and vectorization,
  - Loop vectorization,

Current code generators:
- LLVM IR for shared memory systems.

Prerequisites
--------------
- [ISL] (http://repo.or.cz/w/isl.git) Library.
  Check the ISL [README] (http://repo.or.cz/isl.git/blob/HEAD:/README) for details.
- [Halide] (https://github.com/halide/Halide) compiler.
  Halide itself requires llvm-3.7 or greater. Check the Halide [README] (https://github.com/halide/Halide/blob/master/README.md) for details.
