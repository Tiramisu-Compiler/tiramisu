/** \file
 * This file only exists to contain the front-page of the documentation
 */

/** \mainpage Documentation of the API if the Tiramisu Compiler
 *
 * Tiramisu provides few classes to enable users to represent their program:
 * - The \ref tiramisu::function class: a function in Tiramisu is equivalent to a function in C. It is composed of multiple computations. Each computation is the equivalent of a statement in C.
 * - The \ref tiramisu::input class: an input is used to represent inputs passed to Tiramisu.  An input can represent a buffer or a scalar.
 * - The \ref tiramisu::constant class: a constant is designed to represent constants that are supposed to be declared at the beginning of a Tiramisu function.
 * - The \ref tiramisu::var class: used to represent loop iterators. Usually we declare a var (a loop iterator) and then use it for the declaration of computations. The range of that variable defines the loop range. When use witha buffer it defines the buffer size and when used with an input it defines the input size.
 * - The \ref tiramisu::computation class: a computation in Tiramisu is the equivalent of a statement in C. It is composed of an expression and an iteration domain.
 * - The \ref tiramisu::buffer class: a class to represent memory buffers.
 */
