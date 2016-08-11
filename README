Code Optimization Library (COLi)
----------------------------------
COLi is a library that is designed to simplify code optimization
and code generation.  The user can express his code in the COLi
intermediate representation and can use the COLi API to specify
the code optimizations that he wants to apply on the code.
Finaly the user can generate an optimized code that can be
executed.

The optimization that can be done currently are:
- All the affine loop nest transformations (non-parametric
  tiling, loop fusion/distribution, loop spliting,
  loop interchange, loop shifting, ...),
- Loop parallelization (for shared memory systems),
- Loop vectorization,

Current code generators:
- LLVM IR for shared memory systems (with loop parallelization
  and vectorization),


¡----------------------¡   ¡-----------------------------------¡
¡ Set of Optimizations ¡ + ¡ COLi Intermediate Representation  ¡
¡----------------------¡   ¡-----------------------------------¡
                      \     /
                       \   /
                        \ /
       ¡-------------------------------------¡
       ¡ COLi Library: applies the requested ¡
       ¡ optimizations on the COLi IR and    ¡
       ¡ generates optimized code.           ¡
       ¡-------------------------------------¡
                      \     /
                       \   /
                        \ /
       ¡-------------------------------------¡
       ¡ Optimized code generated. Many      ¡
       ¡ backends exist: LLVM IR, C code ... ¡
       ¡-------------------------------------¡

Prerequisites
--------------
- ISL Library (http://repo.or.cz/w/isl.git).
  Check http://repo.or.cz/isl.git/blob/HEAD:/README for information
  about how to compile and install ISL.
- Halide compiler (https://github.com/halide/Halide).
  Check https://github.com/halide/Halide/blob/master/README.md for
  information about how to compile and install Halide.  Halide itself
  requires LLVM.
