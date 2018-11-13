/** \file
 * This file only exists to contain the front-page of the documentation
 */

/** \mainpage Documentation of the Tiramisu Compiler API
 *
 * Tiramisu provides few classes to enable users to represent their program:
 * - The \ref tiramisu::function class: used to declare Tiramisu functions.  A function in Tiramisu is equivalent to a function in C.  It is composed of multiple computations where each computation is the equivalent of a statement in C.
 * - The \ref tiramisu::input class: used to represent inputs passed to Tiramisu.  An input can represent a buffer or a scalar.
 * - The \ref tiramisu::constant class: a constant is designed to represent constants that are supposed to be declared at the beginning of a Tiramisu function. This can be used only to declare constant scalars.
 * - The \ref tiramisu::var class: used to represent loop iterators. Usually we declare a var (a loop iterator) and then use it for the declaration of computations. The range of that variable defines the range of the loop around the computation (its iteration domain).  When used to declare a buffer it defines the buffer size and when used with an input it defines the input size.
 * - The \ref tiramisu::computation class: used to declare a computation which is the equivalent of a statement in C.  A computation has an expression (tiramisu::expr) and iteration domain defined using an iterator variable.
 * - The \ref tiramisu::buffer class: a class to represent memory buffers.
 * - The \ref tiramisu::expr class: used to declare Tiramisu expressions (e.g., 4, 4 + 4, 4 * i, A(i, j), ...).
 * - The \ref tiramisu::view class: used to declare a view on a buffer
 */
