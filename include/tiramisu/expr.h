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
      * The value of the 1st, 2nd and 3rd operators of the expression.
      * op[0] is the 1st operator, op[1] is the 2nd, ...
      */
    std::vector<tiramisu::expr> op;

    /**
      * The value of the expression.
      */
    union {
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
      * Identifier name.
      */
    std::string name;

    /**
     * Is this expression defined ?
     */
    bool defined;

protected:
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
      * Create a expression of type \p t (a unary operator).
      */
    expr(tiramisu::op_t o, tiramisu::expr expr0)
    {
        assert(((o == tiramisu::o_minus) || (o == tiramisu::o_floor)) &&
               "The only unary operators are the minus and floor operator.");
        if (o == tiramisu::o_floor) {
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
        assert((o != tiramisu::o_minus) &&
               (o != tiramisu::o_call) &&
               (o != tiramisu::o_access) &&
               (o != tiramisu::o_cond) &&
               "The operator is not an binary operator.");

        assert(expr0.get_data_type() == expr1.get_data_type()
               && "expr0 and expr1 should be of the same type.");

        this->_operator = o;
        this->etype = tiramisu::e_op;
        this->dtype = expr0.get_data_type();
        this->defined = true;

        this->op.push_back(expr0);
        this->op.push_back(expr1);
    }

    expr(tiramisu::op_t o, tiramisu::expr expr0, tiramisu::expr expr1, tiramisu::expr expr2)
    {
        assert((o == tiramisu::o_cond) && "The operator is not a ternary operator.");
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

    expr(tiramisu::op_t o, tiramisu::expr id_expr,
         std::vector<tiramisu::expr> access_expressions,
         tiramisu::primitive_t type)
    {
        assert((o == tiramisu::o_access) && "The operator is not an access operator.");
        assert(access_expressions.size() > 0);
        assert(id_expr.get_expr_type() == tiramisu::e_id);

        this->_operator = tiramisu::o_access;
        this->etype = tiramisu::e_op;
        this->dtype = type;
        this->defined = true;

        this->set_access(access_expressions);
        this->op.push_back(id_expr);
    }

    /**
     * Construct an expression that represents an id.
     */
    expr(std::string name)
    {
        assert(name.length() > 0);

        this->etype = tiramisu::e_id;
        this->name = name;
        this->defined = true;

        this->_operator = tiramisu::o_none;
        this->dtype = tiramisu::p_none;
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
            result = this->get_uint8_value();
        else if (this->get_data_type() == tiramisu::p_int8)
            result = this->get_int8_value();
        else if (this->get_data_type() == tiramisu::p_uint16)
            result = this->get_uint16_value();
        else if (this->get_data_type() == tiramisu::p_int16)
            result = this->get_int16_value();
        else if (this->get_data_type() == tiramisu::p_uint32)
            result = this->get_uint32_value();
        else if (this->get_data_type() == tiramisu::p_int32)
            result = this->get_int32_value();
        else if (this->get_data_type() == tiramisu::p_uint64)
            result = this->get_uint64_value();
        else if (this->get_data_type() == tiramisu::p_int64)
            result = this->get_int64_value();
        else if (this->get_data_type() == tiramisu::p_float32)
            result = this->get_float32_value();
        else if (this->get_data_type() == tiramisu::p_float64)
            result = this->get_float64_value();
        else
            tiramisu::error("Calling get_int_val() on a non integer expression.", true);

        return result;
    }

    double get_double_val() const
    {
        assert(this->get_expr_type() == tiramisu::e_val);

        int64_t result = 0;

        if (this->get_data_type() == tiramisu::p_float32)
            result = this->get_float32_value();
        else if (this->get_data_type() == tiramisu::p_float64)
            result = this->get_float64_value();
        else
            tiramisu::error("Calling get_double_val() on a non double expression.", true);

        return result;
    }

    /**
      * Return the value of the \p i 'th operand of the expression.
      * \p i can be 0, 1 or 2.
      */
    tiramisu::expr get_operand(int i) const
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
    std::string get_name() const
    {
        assert((this->get_expr_type() == tiramisu::e_id) ||
               (this->get_expr_type() == tiramisu::e_var));

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
    std::vector<tiramisu::expr> get_access() const
    {
        assert(this->get_expr_type() == tiramisu::e_op);
        assert(this->get_op_type() == tiramisu::o_access);

        return access_vector;
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
    template<typename T> tiramisu::expr operator+(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value) ||
            (std::is_same<T, tiramisu::var>::value))
        {
            return tiramisu::expr(tiramisu::o_add, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_add, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Adding a tiramisu expression to a non supported type.\n",
                        true);
        }
    }

    /**
     * Substruction.
     */
    template<typename T> tiramisu::expr operator-(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_sub, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_sub, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Substructing a tiramisu expression from a non supported type.\n",
                        true);
        }
    }

    /**
     * Division.
     */
    template<typename T> tiramisu::expr operator/(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_div, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_div, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Dividing a tiramisu expression by a non supported type.\n",
                        true);
        }
    }

    /**
     * Multiplication.
     */
    template<typename T> tiramisu::expr operator*(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_mul, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_mul, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Multiplying a tiramisu expression by a non supported type.\n",
                        true);
        }
    }

    /**
     * Modulo.
     */
    template<typename T> tiramisu::expr operator%(T val) const
    {
        if ((std::is_same<T, tiramisu::expr>::value))
        {
            return tiramisu::expr(tiramisu::o_mod, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return tiramisu::expr(tiramisu::o_mod, *this, tiramisu::expr((T) val));
        }
        else
        {
            tiramisu::error("Modulo of a tiramisu expression by a non supported type.\n",
                        true);
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
            tiramisu::error("Right shift of a tiramisu expression by a non supported type.\n",
                        true);
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
            tiramisu::error("Left shift of a tiramisu expression by a non supported type.\n",
                        true);
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
        return tiramisu::expr(tiramisu::o_not, *this);
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
      * For example, for the computation C0(i,j), this
      * function will return the vector {i, j} where i and j
      * are both tiramisu expressions.
      * For a buffer access A[i+1,j], it will return also {i+1, j}.
      */
    void set_access(std::vector<tiramisu::expr> vector)
    {
        access_vector = vector;
    }

    /**
      * Dump the object on standard output (dump most of the fields of
      * the expression class).  This is mainly useful for debugging.
      * If \p exhaustive is set to true, all the fields of the class are
      * printed.  This is useful to find potential initialization problems.
      */
    void dump(bool exhaustive) const
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
                    std::cout << "Number of operands:" << this->get_n_arg() << std::endl;
                    std::cout << "Dumping the operands:" << std::endl;
                    for (int i = 0; i < this->get_n_arg(); i++)
                    {
                        std::cout << "Operand " << std::to_string(i) << "." << std::endl;
                        this->op[i].dump(exhaustive);
                    }
                    if ((this->get_op_type() == tiramisu::o_access) || (this->get_op_type() == tiramisu::o_call))
                    {
                        std::cout << "Access expressions:" << std::endl;
                        for (const auto &e: this->get_access())
                        {
                            e.dump(exhaustive);
                        }
                    }
                    break;
                }
                case (tiramisu::e_val):
                {
                    std::cout << "Expression value type:" << str_from_tiramisu_type_primitive(this->dtype) << std::endl;

                    if (this->get_data_type() == tiramisu::p_uint8)
                        std::cout << "Value:" << this->get_uint8_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_int8)
                        std::cout << "Value:" << this->get_int8_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_uint16)
                        std::cout << "Value:" << this->get_uint16_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_int16)
                        std::cout << "Value:" << this->get_int16_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_uint32)
                        std::cout << "Value:" << this->get_uint32_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_int32)
                        std::cout << "Value:" << this->get_int32_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_uint64)
                        std::cout << "Value:" << this->get_uint64_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_int64)
                        std::cout << "Value:" << this->get_int64_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_float32)
                        std::cout << "Value:" << this->get_float32_value() << std::endl;
                    else if (this->get_data_type() == tiramisu::p_float64)
                        std::cout << "Value:" << this->get_float64_value() << std::endl;
                    break;
                }
                case (tiramisu::e_id):
                {
                    std::cout << "Id name:" << this->get_name() << std::endl;
                    break;
                }
                default:
                    tiramisu::error("Expression type not supported.", true);

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
    idx(std::string name): expr(name)
    {
        assert(name.length() > 0);

        //this->etype = tiramisu::e_var;
        this->dtype = global::get_loop_iterator_default_data_type();
    }
};


/**
 * A class that represents constant variable references
 */
class var: public tiramisu::expr
{
public:
    /**
     * Construct an expression that represents an id.
     */
    var(tiramisu::primitive_t type,
        std::string name): expr(name)
    {
        assert(name.length() > 0);

        this->etype = tiramisu::e_var;
        this->dtype = type;
    }
};

}

#endif
