#ifndef _H_TIRAMISU_TYPE_
#define _H_TIRAMISU_TYPE_

#include <string.h>
#include <stdint.h>

namespace tiramisu
{

/**
  * The possible types of an expression.
  * "e_" stands for expression.
  */
enum expr_t
{
    e_val,          // literal value, like 1, 2.4, 10, ...
    e_var,          // a variable of a primitive type (i.e., an identifier holding one value),
    e_sync,         // syncs parallel computations. Currently used in the context of GPUs.
    e_op,           // an operation: add, mul, div, ...
    e_none          // undefined expression. The existence of an expression of e_none type means an error.
};

/**
  * tiramisu data types.
  * "p_" stands for primitive.
  */
enum primitive_t
{
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
    p_async,
    p_wait_ptr,
    p_none
};


/**
  * Types of tiramisu operators.
  * If the expression is of type e_op then it should
  * have an operator type. This is the operator type.
  *
  * "o_" stands for operator.
  */
enum op_t
{
    // Unary operators
    // The argument of the following operators is a tiramisu::expr.
    o_minus,
    o_floor,
    o_sin,
    o_cos,
    o_tan,
    o_asin,
    o_acos,
    o_atan,
    o_sinh,
    o_cosh,
    o_tanh,
    o_asinh,
    o_acosh,
    o_atanh,
    o_abs,
    o_sqrt,
    o_expo, // exponential
    o_log,
    o_ceil,
    o_round,
    o_trunc,
    // The argument of the following operators is a string representing
    // the name of the buffer to allocate.
    o_allocate,
    o_free,
    // Other arguments
    o_cast, // The argument is an expression and a type.
    o_address, // The argument is a tiramisu::var() that represents a buffer.


    // Binary operator
    // The arguments are tiramisu::expr.
    o_add,
    o_sub,
    o_mul,
    o_div,
    o_mod,
    o_logical_and,
    o_logical_or,
    o_logical_not,
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
    o_memcpy,


    // Ternary operators
    // The arguments are tiramisu::expr.
    o_select,
    o_cond,
    o_lerp,

    // Operators taking a name and a vector of expressions.
    o_call,
    o_access,
    o_address_of,
    o_lin_index,
    o_type,
    o_dummy,
    // just pass in the buffer
    o_buffer,

    o_none,
};

/**
  * Types of function arguments.
  * "a_" stands for argument.
  */
enum argument_t
{
    a_input,
    a_output,
    a_temporary
};

/**
  * Types of ranks in a distributed communication
  * "r_" stands for rank.
  */
enum class rank_t
{
    r_sender,
    r_receiver
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
   /**
     * return true if the type is an integer
     */
static bool is_int(tiramisu::primitive_t type) 
    {
        bool result = false;

        if (type == tiramisu::p_uint8)
        {
            result = true;
        }
        else if (type == tiramisu::p_int8)
        {
            result = true;
        }
        else if (type == tiramisu::p_uint16)
        {
            result = true;
        }
        else if (type == tiramisu::p_int16)
        {
            result = true;
        }
        else if (type == tiramisu::p_uint32)
        {
            result = true;
        }
        else if (type == tiramisu::p_int32)
        {
            result = true;
        }
        else if (type == tiramisu::p_uint64)
        {
            result = true;
        }
        else if (type == tiramisu::p_int64)
        {
            result = true;
        }
        return result;
    }
     /**
     * return true if the type is a float
     */

static bool is_float(tiramisu::primitive_t type) 
    {
        bool result = false;

        if (type == tiramisu::p_float32)
        {
            result = true;
        }
        else if (type == tiramisu::p_float64)
        {
            result = true;
        }
        return result;
    }
/**
  * Structures for the computations features. 
  * 
  */
struct iterator_features{ // loop iterator features
   std::string it_name; 
   int it_level;
   int lower_bound; 
   int upper_bound; 
   bool parallelized; // true if the loop level is parallelized
   int vectorization_factor; // the vectorization factor if this optimization is applied,  0 else
   std::vector<int> dependencies_lower; // list of iterators levels that this level (lewer bound) depends on
   std::vector<int> dependencies_upper; // list of iterators levels that this level (upper bound) depends on
};
struct operation_features{  
       int op_loop_level; // the loop level where this operation is declared 
       int op_rank;   // the rank of this operation
       int nb_var; // Number of variables
       int nb_constant; // Number of constantes in the 
       int data_type; // 0 for int type and 1 for float type
       int nb_operands; // Number of operands of the operatiion ( including direct values)
       std::vector<int>  histograme_loads ; // Number of load ops. i.e. acces to inputs per type
       std::vector<int>  histograme_stores;  // Number of store ops. i.e. write in inputs per type 
       int nb_library_call; // Number of the computation library_calls   
       std::vector<int>  hitograme_int_ops; // number of arithmetic (+  -  *  /   o_min  o_max )operations for int values 
       std::vector<int>  hitograme_double_ops; // number of arithmetic operations (+  -  *  /   o_min  o_max ) for double values  
        // TODO: other op types (o_sin;....) 
    }; 
struct local_schedule_features{ 
     std::vector<int> factors;
     std::vector<int> levels;      
};
struct global_schedule_features{ 
    std::vector<int> factors; // list of optimizations factors
    std::vector<int> levels;  // list of loop levels on which the optimization is applied
    std::vector<int> related_computation;   // list of related computations
};
        
struct computation_features_struct{
    std::string coputation_name; 
    int loop_levels;  // Number of nest levels
    std::vector<iterator_features> iterators; // list des iterators 
    std::vector<operation_features> operations_features; // list of opererations features in the computation,  the index represent the loop level asigned to the operation
    int is_pedicate;  // set to 1 if a predicate statement is assigned to This computation (executed  only if this predicate is true) 
    int nb_dependencies_intern; // Number of dependencies between loops levels in the computation  
    std::map<std::string,std::vector<local_schedule_features>> local_schedule;// features describing the optimizations applied on the computation vectorize(1); split(3); tile(4) ; skew(5); ---> local optimizations
    //TODO:  after  before  compute_at  store_at fuse  ---> global computations 
    std::map<std::string,std::vector<global_schedule_features>> global_schedule; 
    
}; 

#endif
