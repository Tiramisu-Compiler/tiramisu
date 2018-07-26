/** \file
 * This file only exists to contain the front-page of the documentation
 */

/** \mainpage Tiramisu Optimization Framework 
 * Tiramisu is a library that is designed to simplify code optimization and code generation.  The user can express his code in the Tiramisu intermediate representation (Tiramisu IR), he can use the Tiramisu API to perform different optimizations and finaly he can generate an LLVM code from the optimized Tiramisu IR.
 *
 * Tiramisu provides few classes to enable users to represent their program:
 * - The \ref tiramisu::function class: a function is composed of multiple computations and a vector of arguments (functions arguments).
 * - The \ref tiramisu::computation class: a computation is composed of an expression and an iteration space but is not associated with any memory location.
 * - The \ref tiramisu::buffer class: a class to represent memory buffers.
 *
 * \example tutorials/tutorial_01.cpp 
 * \example tutorials/tutorial_02.cpp 
 * \example tutorials/tutorial_03.cpp 
 * \example tutorials/tutorial_04.cpp 
 * \example tutorials/tutorial_05.cpp 
 * \example tutorials/tutorial_06.cpp 
 */
