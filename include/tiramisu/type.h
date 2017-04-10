#ifndef _H_TIRAMISU_TYPE_
#define _H_TIRAMISU_TYPE_

#include <string.h>
#include <stdint.h>

namespace tiramisu {

    /**
     * The possible types of an expression.
     * "e_" stands for expression.
     */
    enum expr_t {
        e_val,          // literal value, like 1, 2.4, 10, ...
        e_var,          // a variable of a primitive type (i.e., an identifier holding one value),
        e_op,           // an operation: add, mul, div, ...
        e_none          // undefined expression. The existence of an expression of e_none type means an error.
    };

    /**
      * tiramisu data types.
      * "p_" stands for primitive.
      */
    enum primitive_t {
        p_uint8,
        p_uint16,
        p_uint32,
        p_uint64,
        p_int8,
        p_int16,
        p_int32,
        p_int64,
        p_float32,
        p_float64,
        p_boolean,
        p_none
    };

    /**
      * Types of tiramisu operators.
      * If the expression is of type e_op then it should
      * have an operator type. This is the operator type.
      *
      * "o_" stands for operator.
      */
    enum op_t {
        o_minus,
        o_add,
        o_sub,
        o_mul,
        o_div,
        o_mod,

        o_logical_and,
        o_logical_or,
        o_logical_not,
        o_select,
        o_cond,
        o_eq,
        o_ne,
        o_le,
        o_lt,
        o_ge,
        o_gt,

        o_max,
        o_min,
        o_right_shift,
        o_left_shift,
        o_floor,
        o_cast,

        o_sin,
        o_cos,
        o_tan,
        o_asin,
        o_acos,
        o_atan,
        o_abs,
        o_sqrt,
        o_expo, // exponential
        o_log,
        o_ceil,
        o_round,
        o_trunc,

        o_call,
        o_access,

        o_none
    };

    /**
     * Types of function arguments.
     * "a_" stands for argument.
     */
    enum argument_t {
        a_input,
        a_output,
        a_temporary
    };

    /**
     * Convert a Tiramisu type into the equivalent Halide type (if it exists),
     * otherwise show an error message (no automatic type conversion is performed).
     */
    Halide::Type halide_type_from_tiramisu_type(tiramisu::primitive_t type);

    /**
     * Convert a Halide type into the equivalent Tiramisu type (if it exists),
     * otherwise show an error message (no automatic type conversion is performed).
     */
    tiramisu::primitive_t halide_type_to_tiramisu_type(Halide::Type type);
}

#endif
