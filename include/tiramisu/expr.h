#ifndef _H_TIRAMISU_EXPR_
#define _H_TIRAMISU_EXPR_

#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>

#include <map>
#include <vector>
#include <string.h>
#include <stdint.h>

#include <Halide.h>
#include <tiramisu/debug.h>
#include <tiramisu/type.h>


namespace tiramisu
{

class computation;

std::string str_from_tiramisu_type_expr(tiramisu::expr_t type);
std::string str_tiramisu_type_op(tiramisu::op_t type);
std::string str_from_tiramisu_type_primitive(tiramisu::primitive_t type);

class buffer;
class var;
class global;

/**
  * A class that holds all the global variables necessary for Tiramisu.
  * It also holds Tiramisu options.
  */
class global
{
private:
    /**
      * Perform automatic data mapping ?
      */
    static bool auto_data_mapping;

public:
    /**
      * If this option is set to true, Tiramisu automatically
      * modifies the computation data mapping whenever a new
      * schedule is applied to a computation.
      * If it is set to false, it is up to the user to set
      * the right data mapping before code generation.
      */
    static void set_auto_data_mapping(bool v)
    {
        global::auto_data_mapping = v;
    }

    /**
      * Return whether auto data mapping is set.
      * If auto data mapping is set, Tiramisu automatically
      * modifies the computation data mapping whenever a new
      * schedule is applied to a computation.
      * If it is set to false, it is up to the user to set
      * the right data mapping before code generation.
      */
    static bool is_auto_data_mapping_set()
    {
        return global::auto_data_mapping;
    }

    static void set_default_tiramisu_options()
    {
        set_auto_data_mapping(true);
    }

    static primitive_t get_loop_iterator_default_data_type()
    {
        return tiramisu::p_int32;
    }

    global()
    {
        set_default_tiramisu_options();
    }
};


/**
  * A class to represent tiramisu expressions.
  */
class expr
{
    /**
      * The type of the operator.
      */
    tiramisu::op_t _operator;

    /**
      * The value of the 1st, 2nd and 3rd operands of the expression.
      * op[0] is the 1st operand, op[1] is the 2nd, ...
      */
    std::vector<tiramisu::expr> op;

    /**
      * The value of the expression.
      */
    union
    {
        uint8_t     uint8_value;
        int8_t      int8_value;
        uint16_t    uint16_value;
        int16_t     int16_value;
        uint32_t    uint32_value;
        int32_t     int32_value;
        uint64_t    uint64_value;
        int64_t     int64_value;
        float       float32_value;
        double      float64_value;
    };

    /**
      * A vector of expressions representing buffer accesses,
      * or computation accesses.
      * For example for the computation C0(i,j), the access is
      * the vector {i, j}.
      */
    std::vector<tiramisu::expr> access_vector;

    /**
      * A vector of expressions representing arguments of an
      * external function.
      * For example, to call the function foo() with the following
      * three arguments as input
      *     the integer 1, the result of the computation C1(0,0), and
      *     the computation C0 (i.e., its buffer).
      * \p vector should be {tiramisu::expr(1), C1(0,0), tiramisu::expr(o_address, tiramisu::var("C0"))}.
      */
    std::vector<tiramisu::expr> argument_vector;

    /**
      * Is this expression defined?
      */
    bool defined;

protected:
    /**
      * Identifier name.
      */
    std::string name;

    /**
      * Data type.
      */
    tiramisu::primitive_t dtype;

    /**
      * The type of the expression.
      */
    tiramisu::expr_t etype;

public:

    /**
      * Create an undefined expression.
      */
    expr()
    {
        this->defined = false;

        this->_operator = tiramisu::o_none;
        this->etype = tiramisu::e_none;
        this->dtype = tiramisu::p_none;
    }

    /**
      * Create a cast expression to type \p t (a unary operator).
      */
    expr(tiramisu::op_t o, tiramisu::primitive_t dtype, tiramisu::expr expr0)
    {
        assert((o == tiramisu::o_cast) && "Only support cast operator.");

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = dtype;
        this->defined = true;

        this->op.push_back(expr0);
    }

    /**
      * Create a expression for a unary operator.
      */
    expr(tiramisu::op_t o, tiramisu::expr expr0)
    {
        if (o == tiramisu::o_floor)
        {
            assert(((expr0.get_data_type() == tiramisu::p_float32) ||
                    (expr0.get_data_type() == tiramisu::p_float64)) &&
                   "Can only do floor on float32 or float64.");
        }

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = expr0.get_data_type();
        this->defined = true;

        this->op.push_back(expr0);
    }

    expr(tiramisu::op_t o, tiramisu::expr expr0, tiramisu::expr expr1)
    {
        DEBUG(10, tiramisu::str_dump("Expr0:"); expr0.dump(false));
        DEBUG(10, tiramisu::str_dump("Expr1:"); expr1.dump(false));

        assert(expr0.get_data_type() == expr1.get_data_type()
               && "expr0 and expr1 should be of the same type. Dumping expr0 and expr1.");

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = expr0.get_data_type();
        this->defined = true;

        this->op.push_back(expr0);
        this->op.push_back(expr1);
    }

    expr(tiramisu::op_t o, tiramisu::expr expr0, tiramisu::expr expr1, tiramisu::expr expr2)
    {
        assert(expr1.get_data_type() == expr2.get_data_type() &&
               "expr1 and expr2 should be of the same type.");

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = expr1.get_data_type();
        this->defined = true;

        this->op.push_back(expr0);
        this->op.push_back(expr1);
        this->op.push_back(expr2);
    }

    expr(tiramisu::op_t o, std::string name,
         std::vector<tiramisu::expr> vec,
         tiramisu::primitive_t type)
    {
        assert(((o == tiramisu::o_access) || (o == tiramisu::o_call)) &&
               "The operator is not an access operator.");
        assert(vec.size() > 0);
        assert(name.size() > 0);

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = type;
        this->defined = true;

        if (o == tiramisu::o_access)
        {
            this->set_access(vec);
        }
        else if (o == tiramisu::o_call)
        {
            this->set_arguments(vec);
        }
        else
        {
            tiramisu::error("Type of operator is not o_access or o_call.", true);
        }

        this->name = name;
    }

    /**
      * Construct an unsigned 8-bit integer expression.
      */
    expr(uint8_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_uint8;
        this->uint8_value = val;
    }

    /**
      * Construct a signed 8-bit integer expression.
      */
    expr(int8_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_int8;
        this->int8_value = val;
    }

    /**
      * Construct an unsigned 16-bit integer expression.
      */
    expr(uint16_t val)
    {
        this->defined = true;
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;

        this->dtype = tiramisu::p_uint16;
        this->uint16_value = val;
    }

    /**
      * Construct a signed 16-bit integer expression.
      */
    expr(int16_t val)
    {
        this->defined = true;
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;

        this->dtype = tiramisu::p_int16;
        this->int16_value = val;
    }

    /**
      * Construct an unsigned 32-bit integer expression.
      */
    expr(uint32_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_uint32;
        this->uint32_value = val;
    }

    /**
      * Construct a signed 32-bit integer expression.
      */
    expr(int32_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_int32;
        this->int32_value = val;
    }

    /**
      * Construct an unsigned 64-bit integer expression.
      */
    expr(uint64_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_uint64;
        this->uint64_value = val;
    }

    /**
      * Construct a signed 64-bit integer expression.
      */
    expr(int64_t val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_int64;
        this->int64_value = val;
    }

    /**
      * Construct a 32-bit float expression.
      */
    expr(float val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_float32;
        this->float32_value = val;
    }

    /**
      * Construct a 64-bit float expression.
      */
    expr(double val)
    {
        this->etype = tiramisu::e_val;
        this->_operator = tiramisu::o_none;
        this->defined = true;

        this->dtype = tiramisu::p_float64;
        this->float64_value = val;
    }

    /**
      * Return true if the expression is defined.
      */
    bool is_defined() const
    {
        return defined;
    }

    /**
      * Return the actual value of the expression.
      */
    // @{
    uint8_t get_uint8_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_uint8);

        return uint8_value;
    }

    int8_t get_int8_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_int8);

        return int8_value;
    }

    uint16_t get_uint16_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_uint16);

        return uint16_value;
    }

    int16_t get_int16_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_int16);

        return int16_value;
    }

    uint32_t get_uint32_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_uint32);

        return uint32_value;
    }

    int32_t get_int32_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_int32);

        return int32_value;
    }

    uint64_t get_uint64_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_uint64);

        return uint64_value;
    }

    int64_t get_int64_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_int64);

        return int64_value;
    }

    float get_float32_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_float32);

        return float32_value;
    }

    double get_float64_value() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);
        assert(this->get_data_type() == tiramisu::p_float64);

        return float64_value;
    }
    // @}

    int64_t get_int_val() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);

        int64_t result = 0;

        if (this->get_data_type() == tiramisu::p_uint8)
        {
            result = this->get_uint8_value();
        }
        else if (this->get_data_type() == tiramisu::p_int8)
        {
            result = this->get_int8_value();
        }
        else if (this->get_data_type() == tiramisu::p_uint16)
        {
            result = this->get_uint16_value();
        }
        else if (this->get_data_type() == tiramisu::p_int16)
        {
            result = this->get_int16_value();
        }
        else if (this->get_data_type() == tiramisu::p_uint32)
        {
            result = this->get_uint32_value();
        }
        else if (this->get_data_type() == tiramisu::p_int32)
        {
            result = this->get_int32_value();
        }
        else if (this->get_data_type() == tiramisu::p_uint64)
        {
            result = this->get_uint64_value();
        }
        else if (this->get_data_type() == tiramisu::p_int64)
        {
            result = this->get_int64_value();
        }
        else if (this->get_data_type() == tiramisu::p_float32)
        {
            result = this->get_float32_value();
        }
        else if (this->get_data_type() == tiramisu::p_float64)
        {
            result = this->get_float64_value();
        }
        else
        {
            tiramisu::error("Calling get_int_val() on a non integer expression.", true);
        }

        return result;
    }

    double get_double_val() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);

        int64_t result = 0;

        if (this->get_data_type() == tiramisu::p_float32)
        {
            result = this->get_float32_value();
        }
        else if (this->get_data_type() == tiramisu::p_float64)
        {
            result = this->get_float64_value();
        }
        else
        {
            tiramisu::error("Calling get_double_val() on a non double expression.", true);
        }

        return result;
    }

    /**
      * Return the value of the \p i 'th operand of the expression.
      * \p i can be 0, 1 or 2.
      */
    const tiramisu::expr &get_operand(int i) const
    {
        assert(this->get_expr_type() == tiramisu::e_op);
        assert((i < (int)this->op.size()) && "Operand index is out of bounds.");

        return this->op[i];
    }

    /**
      * Return the number of arguments of the operator.
      */
    int get_n_arg() const
    {
        assert(this->get_expr_type() == tiramisu::e_op);

        return this->op.size();
    }

    /**
      * Return the type of the expression (tiramisu::expr_type).
      */
    tiramisu::expr_t get_expr_type() const
    {
        return etype;
    }

    /**
      * Get the data type of the expression.
      */
    tiramisu::primitive_t get_data_type() const
    {
        return dtype;
    }

    /**
      * Get the name of the ID or the variable represented by this expressions.
      */
    const std::string &get_name() const
    {
        assert((this->get_expr_type() == tiramisu::e_var) ||
               (this->get_op_type() == tiramisu::o_access) ||
               (this->get_op_type() == tiramisu::o_call));

        return name;
    }

    /**
      * Get the type of the operator (tiramisu::op_t).
      */
    tiramisu::op_t get_op_type() const
    {
        return _operator;
    }

    /**
      * Return a vector of the access of the computation
      * or array.
      * For example, for the computation C0(i,j), this
      * function will return the vector {i, j} where i and j
      * are both tiramisu expressions.
      * For a buffer access A[i+1,j], it will return also {i+1, j}.
      */
    const std::vector<tiramisu::expr> &get_access() const
    {
        assert(this->get_expr_type() == tiramisu::e_op);
        assert(this->get_op_type() == tiramisu::o_access);

        return access_vector;
    }

    /**
      * Return the arguments of an external function call.
      */
    const std::vector<tiramisu::expr> &get_arguments() const
    {
        assert(this->get_expr_type() == tiramisu::e_op);
        assert(this->get_op_type() == tiramisu::o_call);

        return argument_vector;
    }

    /**
      * Get the number of dimensions in the access vector.
      */
    int get_n_dim_access() const
    {
        assert(this->get_expr_type() == tiramisu::e_op);
        assert(this->get_op_type() == tiramisu::o_access);

        return access_vector.size();
    }

    /**
      * Addition.
      */
    template<typename T> friend tiramisu::expr operator+(tiramisu::expr e1, T val)
    {
        if ((std::is_same<T, uint8_t>::value) ||
                (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, uint16_t>::value) ||
                (std::is_same<T, int16_t>::value) ||
                (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_add, e1, tiramisu::expr((T) val));
        }
        else
        {
            return tiramisu::expr(tiramisu::o_add, e1, val);
        }
    }

    /**
      * Subtraction.
      */
    template<typename T> friend tiramisu::expr operator-(tiramisu::expr e1, T val)
    {
        if ((std::is_same<T, uint8_t>::value) ||
                (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, uint16_t>::value) ||
                (std::is_same<T, int16_t>::value) ||
                (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_sub, e1, tiramisu::expr((T) val));
        }
        else
        {
            return tiramisu::expr(tiramisu::o_sub, e1, val);
        }
    }

    /**
      * Division.
      */
    template<typename T> friend tiramisu::expr operator/(tiramisu::expr e1, T val)
    {
        if ((std::is_same<T, uint8_t>::value) ||
                (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, uint16_t>::value) ||
                (std::is_same<T, int16_t>::value) ||
                (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_div, e1, tiramisu::expr((T) val));
        }
        else
        {
            return tiramisu::expr(tiramisu::o_div, e1, val);
        }
    }

    /**
      * Multiplication.
      */
    template<typename T> friend tiramisu::expr operator*(tiramisu::expr e1, T val)
    {
        if ((std::is_same<T, uint8_t>::value) ||
                (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, uint16_t>::value) ||
                (std::is_same<T, int16_t>::value) ||
                (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_mul, e1, tiramisu::expr((T) val));
        }
        else
        {
            return tiramisu::expr(tiramisu::o_mul, e1, val);
        }
    }

    /**
      * Modulo.
      */
    template<typename T> friend tiramisu::expr operator%(tiramisu::expr e1, T val)
    {
        if ((std::is_same<T, uint8_t>::value) ||
                (std::is_same<T, int8_t>::value) ||
                (std::is_same<T, uint16_t>::value) ||
                (std::is_same<T, int16_t>::value) ||
                (std::is_same<T, int32_t>::value) ||
                (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_mod, e1, tiramisu::expr((T) val));
        }
        else
        {
            return tiramisu::expr(tiramisu::o_mod, e1, val);
        }
    }

    /**
      * Right shift operator.
      */
    template<typename T> tiramisu::expr operator>>(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_right_shift, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_right_shift, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Right shift of a tiramisu expression by a non supported type.\n", true);
        }
    }

    /**
      * Left shift operator.
      */
    template<typename T> tiramisu::expr operator<<(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_left_shift, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_left_shift, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Left shift of a tiramisu expression by a non supported type.\n", true);
        }
    }

    /**
      * Logical and of two expressions.
      */
    tiramisu::expr operator&&(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_logical_and, *this, e1);
    }

    /**
      * Logical and of two expressions.
      */
    tiramisu::expr operator||(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_logical_or, *this, e1);
    }

    /**
      * Expression multiplied by (-1).
      */
    tiramisu::expr operator-() const
    {
        return tiramisu::expr(tiramisu::o_minus, *this);
    }

    /**
      * Logical NOT of an expression.
      */
    tiramisu::expr operator!() const
    {
        return tiramisu::expr(tiramisu::o_logical_not, *this);
    }

    /**
      * Comparison operator.
      */
    // @{
    tiramisu::expr operator==(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_eq, *this, e1);
    }
    tiramisu::expr operator!=(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_ne, *this, e1);
    }
    // @}

    /**
      * Less than operator.
      */
    tiramisu::expr operator<(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_lt, *this, e1);
    }

    /**
      * Less than or equal operator.
      */
    tiramisu::expr operator<=(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_le, *this, e1);
    }

    /**
      * Greater than operator.
      */
    tiramisu::expr operator>(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_gt, *this, e1);
    }

    /**
      * Greater than or equal operator.
      */
    tiramisu::expr operator>=(tiramisu::expr e1) const
    {
        return tiramisu::expr(tiramisu::o_ge, *this, e1);
    }

    /**
      * Set the access of a computation or an array.
      * For example, for the computation C0, this
      * function can set the vector {i, j} as an access vector.
      * The result is that the computation C0 is accessed
      * with C0(i,j).
      */
    void set_access(std::vector<tiramisu::expr> vector)
    {
        access_vector = vector;
    }

    /**
      * Set an element of the vector of accesses of a computation.
      * This changes only one dimension of the access vector.
      */
    void set_access_dimension(int i, tiramisu::expr acc)
    {
        assert((i < (int)this->access_vector.size()) && "index is out of bounds.");
        access_vector[i] = acc;
    }

    /**
      * Set the arguments of an external function call.
      * For example, for the call my_external(C0, 1, C1(i,j)),
      * \p vector should be {C0, 1, C1(i,j)}.
      */
    void set_arguments(std::vector<tiramisu::expr> vector)
    {
        argument_vector = vector;
    }

    /**
      * Dump the object on standard output (dump most of the fields of
      * the expression class). This is mainly useful for debugging.
      * If \p exhaustive is set to true, all the fields of the class are
      * printed. This is useful to find potential initialization problems.
      */
    void dump(bool exhaustive) const
    {
        if (this->get_expr_type() != e_none)
        {
            if (exhaustive == true)
            {
                if (ENABLE_DEBUG && (this->is_defined()))
                {
                    std::cout << "Expression:" << std::endl;
                    std::cout << "Expression type:" << str_from_tiramisu_type_expr(this->etype) << std::endl;
                    switch (this->etype)
                    {
                    case tiramisu::e_op:
                    {
                        std::cout << "Expression operator type:" << str_tiramisu_type_op(this->_operator) << std::endl;
                        if (this->get_n_arg() > 0)
                        {
                            std::cout << "Number of operands:" << this->get_n_arg() << std::endl;
                            std::cout << "Dumping the operands:" << std::endl;
                            for (int i = 0; i < this->get_n_arg(); i++)
                            {
                                std::cout << "Operand " << std::to_string(i) << "." << std::endl;
                                this->op[i].dump(exhaustive);
                            }
                        }
                        if ((this->get_op_type() == tiramisu::o_access))
                        {
                            std::cout << "Access to " +  this->get_name() + ". Access expressions:" << std::endl;
                            for (const auto &e : this->get_access())
                            {
                                e.dump(exhaustive);
                            }
                        }
                        if ((this->get_op_type() == tiramisu::o_call))
                        {
                            std::cout << "call to " +  this->get_name() + ". Argument expressions:" << std::endl;
                            for (const auto &e : this->get_arguments())
                            {
                                e.dump(exhaustive);
                            }
                        }
                        if ((this->get_op_type() == tiramisu::o_address))
                        {
                            std::cout << "Address of the following access : " << std::endl;
                            this->get_operand(0).dump(true);
                        }
                        break;
                    }
                    case (tiramisu::e_val):
                    {
                        std::cout << "Expression value type:" << str_from_tiramisu_type_primitive(this->dtype) << std::endl;

                        if (this->get_data_type() == tiramisu::p_uint8)
                        {
                            std::cout << "Value:" << this->get_uint8_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_int8)
                        {
                            std::cout << "Value:" << this->get_int8_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_uint16)
                        {
                            std::cout << "Value:" << this->get_uint16_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_int16)
                        {
                            std::cout << "Value:" << this->get_int16_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_uint32)
                        {
                            std::cout << "Value:" << this->get_uint32_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_int32)
                        {
                            std::cout << "Value:" << this->get_int32_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_uint64)
                        {
                            std::cout << "Value:" << this->get_uint64_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_int64)
                        {
                            std::cout << "Value:" << this->get_int64_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_float32)
                        {
                            std::cout << "Value:" << this->get_float32_value() << std::endl;
                        }
                        else if (this->get_data_type() == tiramisu::p_float64)
                        {
                            std::cout << "Value:" << this->get_float64_value() << std::endl;
                        }
                        break;
                    }
                    case (tiramisu::e_var):
                    {
                        std::cout << "Var name:" << this->get_name() << std::endl;
                        std::cout << "Expression value type:" << str_from_tiramisu_type_primitive(this->dtype) << std::endl;
                        break;
                    }
                    default:
                        tiramisu::error("Expression type not supported.", true);

                    }
                }
            }
            else
            {
                switch (this->etype)
                {
                case tiramisu::e_op:
                {
                    switch (this->get_op_type())
                    {
                    case tiramisu::o_logical_and:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") &&" << "(";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_logical_or:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ||" << "(";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_max:
                        std::cout << "max(";
                        this->get_operand(0).dump(false);
                        std::cout << ", ";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_min:
                        std::cout << "min(";
                        this->get_operand(0).dump(false);
                        std::cout << ", ";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_minus:
                        std::cout << "-(";
                        this->get_operand(0).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_add:
                        std::cout << "";
                        this->get_operand(0).dump(false);
                        std::cout << " + ";
                        this->get_operand(1).dump(false);
                        std::cout << "";
                        break;
                    case tiramisu::o_sub:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") - (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_mul:
                        std::cout << "";
                        this->get_operand(0).dump(false);
                        std::cout << "*";
                        this->get_operand(1).dump(false);
                        std::cout << "";
                        break;
                    case tiramisu::o_div:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ")/(";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_mod:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ")%(";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_select:
                        std::cout << "select(";
                        this->get_operand(0).dump(false);
                        std::cout << ", ";
                        this->get_operand(1).dump(false);
                        std::cout << ", ";
                        this->get_operand(2).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_le:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") <= (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_lt:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") < (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_ge:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") >= (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_gt:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") > (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_logical_not:
                        std::cout << "!(";
                        this->get_operand(0).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_eq:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") == (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_ne:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") != (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_right_shift:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") >> (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_left_shift:
                        std::cout << "(";
                        this->get_operand(0).dump(false);
                        std::cout << ") << (";
                        this->get_operand(1).dump(false);
                        std::cout << ")";
                        break;
                    case tiramisu::o_floor:
                        std::cout << "floor(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_sin:
                        std::cout << "sin(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_cos:
                        std::cout << "cos(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_tan:
                        std::cout << "tan(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_atan:
                        std::cout << "atan(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_acos:
                        std::cout << "acos(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_asin:
                        std::cout << "asin(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_abs:
                        std::cout << "abs(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_sqrt:
                        std::cout << "sqrt(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_expo:
                        std::cout << "exp(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_log:
                        std::cout << "log(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_ceil:
                        std::cout << "ceil(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_round:
                        std::cout << "round(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_trunc:
                        std::cout << "trunc(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_cast:
                        std::cout << "cast(";
                        this->get_operand(0).dump(false);
                        std::cout << ") ";
                        break;
                    case tiramisu::o_access:
                        std::cout << this->get_name() << "(";
                        for (int k = 0; k < this->get_access().size(); k++)
                        {
                            if (k != 0)
                            {
                                std::cout << ", ";
                            }
                            this->get_access()[k].dump(false);
                        }
                        std::cout << ")";
                        break;
                    case tiramisu::o_call:
                        std::cout << this->get_name() << "(";
                        for (int k = 0; k < this->get_arguments().size(); k++)
                        {
                            if (k != 0)
                            {
                                std::cout << ", ";
                            }
                            this->get_arguments()[k].dump(false);
                        }
                        std::cout << ")";
                        break;
                    case tiramisu::o_address:
                        std::cout << "&" << this->get_operand(0).get_name();
                        break;
                    default:
                        tiramisu::error("Dumping an unsupported tiramisu expression.", 1);
                    }
                    break;
                }
                case (tiramisu::e_val):
                {
                    if (this->get_data_type() == tiramisu::p_uint8)
                    {
                        std::cout << (int)this->get_uint8_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_int8)
                    {
                        std::cout << (int)this->get_int8_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_uint16)
                    {
                        std::cout << this->get_uint16_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_int16)
                    {
                        std::cout << this->get_int16_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_uint32)
                    {
                        std::cout << this->get_uint32_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_int32)
                    {
                        std::cout << this->get_int32_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_uint64)
                    {
                        std::cout << this->get_uint64_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_int64)
                    {
                        std::cout << this->get_int64_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_float32)
                    {
                        std::cout << this->get_float32_value();
                    }
                    else if (this->get_data_type() == tiramisu::p_float64)
                    {
                        std::cout << this->get_float64_value();
                    }
                    break;
                }
                case (tiramisu::e_var):
                {
                    std::cout << this->get_name();
                    break;
                }
                default:
                    tiramisu::error("Expression type not supported.", true);

                }
            }
        }
    }
};

/**
  * A class that represents index expressions
  */
class idx: public tiramisu::expr
{
public:
    /**
      * Construct an expression that represents an id.
      */
};


/**
  * A class that represents constant variable references
  */
class var: public tiramisu::expr
{
public:
    /**
      * Construct an expression that represents a variable.
      * \p type is the type of the variable and \p name is its name.
      */
    var(tiramisu::primitive_t type, std::string name)
    {
        assert(!name.empty());

        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = type;
    }

    /**
     * Construct an expression that represents an untyped variable.
     * For example to declare the variable "t", use
     * tiramisu::var("t");
     *
     * To declare an expression that represents the reference of a buffer
     * one can declare a variable as follows
     *
     * tiramisu::expr(o_address, "buf0");
     *
     */
    var(std::string name)
    {
        assert(!name.empty());

        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = global::get_loop_iterator_default_data_type();
    }
};

/**
  * Convert a Tiramisu expression into a Halide expression.
  */
Halide::Expr halide_expr_from_tiramisu_expr(
    const tiramisu::computation *comp,
    std::vector<isl_ast_expr *> &index_expr,
    const tiramisu::expr &tiramisu_expr);

}

#endif
