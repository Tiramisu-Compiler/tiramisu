#ifndef _H_COLI_EXPR_
#define _H_COLI_EXPR_

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
#include <coli/debug.h>
#include <coli/type.h>

namespace coli
{

std::string coli_type_expr_to_str(coli::expr_t type);
std::string coli_type_op_to_str(coli::op_t type);
std::string coli_type_primitive_to_str(coli::primitive_t type);

/**
  * A class to represent coli expressions.
  */
class expr
{
    /**
      * The type of the expression.
      */
    coli::expr_t etype;

    /**
      * The type of the operator.
      */
    coli::op_t _operator;

    /**
      * The value of the 1st, 2nd and 3rd operators of the expression.
      * op[0] is the 1st operator, op[1] is the 2nd, ...
      */
    std::vector<coli::expr> op;

    /**
      * The value of the expression.
      */
    union {
        uint8_t     uint8_value;
        int8_t      int8_value;
        uint32_t    uint32_value;
        int32_t     int32_value;
        uint64_t    uint64_value;
        int64_t     int64_value;
    };

    /**
      * A vector of expressions representing buffer accesses,
      * or computation accesses.
      * For example for the computation C0(i,j), the access is
      * the vector {i, j}.
      */
    std::vector<coli::expr> access_vector;

    /**
      * Identifier name.
      */
    std::string id_name;

protected:
    /**
      * Data type.
      */
    coli::primitive_t dtype;

public:

    /**
      * Create a expression of type \p t (a unary operator).
      */
    expr(coli::op_t o, coli::expr expr0)
    {
        assert((o == coli::o_minus) && "The only unary operator is the minus operator.");

        this->_operator = o;
        this->etype = coli::e_op;
        this->dtype = expr0.get_data_type();

        this->op.push_back(expr0);
    }

    expr(coli::op_t o, coli::expr expr0, coli::expr expr1)
    {
        assert((o != coli::o_minus) &&
               (o != coli::o_call) &&
               (o != coli::o_access) &&
               (o != coli::o_cond) &&
               "The operator is not an binary operator.");
        assert(expr0.get_data_type() == expr1.get_data_type() &&
               "expr0 and expr1 should be of the same type.");

        this->_operator = o;
        this->etype = coli::e_op;
        this->dtype = expr0.get_data_type();

        this->op.push_back(expr0);
        this->op.push_back(expr1);
    }

    expr(coli::op_t o, coli::expr expr0, coli::expr expr1, coli::expr expr2)
    {
        assert((o == coli::o_cond) && "The operator is not a ternary operator.");
        assert(expr1.get_data_type() == expr2.get_data_type() &&
               "expr1 and expr2 should be of the same type.");

        this->_operator = o;
        this->etype = coli::e_op;
        this->dtype = expr1.get_data_type();

        this->op.push_back(expr0);
        this->op.push_back(expr1);
        this->op.push_back(expr2);
    }

    expr(coli::primitive_t type, coli::op_t o, coli::expr id_expr,
                            std::vector<coli::expr> access_expressions)
    {
        assert((o == coli::o_access) && "The operator is not an access operator.");
        assert(access_expressions.size() > 0);
        assert(id_expr.get_expr_type() == coli::e_id);

        this->_operator = coli::o_access;
        this->etype = coli::e_op;
        this->dtype = type;

        this->set_access(access_expressions);
        this->op.push_back(id_expr);
    }

    /**
    * Construct an expression that represents an id.
    */
    expr(std::string name)
    {
        assert(name.length() > 0);

        this->etype = coli::e_id;
        this->id_name = name;

        this->_operator = coli::o_none;
        this->dtype = coli::p_none;
    }

    /**
      * Construct an unsigned 8-bit integer expression.
      */
    expr(uint8_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_uint8;
        this->uint8_value = val;
    }

    /**
      * Construct a signed 8-bit integer expression.
      */
    expr(int8_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_int8;
        this->int8_value = val;
    }

    /**
      * Construct an unsigned 16-bit integer expression.
      */
    expr(uint16_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_uint16;
        this->uint8_value = val;
    }

    /**
      * Construct a signed 16-bit integer expression.
      */
    expr(int16_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_int16;
        this->int8_value = val;
    }

    /**
      * Construct an unsigned 32-bit integer expression.
      */
    expr(uint32_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_uint32;
        this->uint32_value = val;
    }

    /**
      * Construct a signed 32-bit integer expression.
      */
    expr(int32_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_int32;
        this->int32_value = val;
    }

    /**
      * Construct an unsigned 64-bit integer expression.
      */
    expr(uint64_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_uint64;
        this->uint64_value = val;
    }

    /**
      * Construct a signed 64-bit integer expression.
      */
    expr(int64_t val)
    {
        this->etype = coli::e_val;
        this->_operator = coli::o_none;

        this->dtype = coli::p_int64;
        this->int64_value = val;
    }

    /**
      * Return the actual value of the expression.
      */
    //@
    uint8_t get_uint8_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_uint8);

        return uint8_value;
    }

    int8_t get_int8_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_int8);

        return int8_value;
    }

    uint16_t get_uint16_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_uint16);

        return uint8_value;
    }

    int8_t get_int16_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_int16);

        return int8_value;
    }

    uint32_t get_uint32_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_uint32);

        return uint32_value;
    }

    int32_t get_int32_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_int32);

        return int32_value;
    }

    uint64_t get_uint64_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_uint64);

        return uint64_value;
    }

    int64_t get_int64_value() const
    {
        assert(this->get_expr_type() == coli::e_val);
        assert(this->get_data_type() == coli::p_int64);

        return int64_value;
    }
    //@

    /**
      * Return the value of the \p i 'th operand of the expression.
      * \p i can be 0, 1 or 2.
      */
    coli::expr get_operand(int i) const
    {
        assert(this->get_expr_type() == coli::e_op);
        assert((i < (int)this->op.size()) && "Operand index is out of bounds.");

        return this->op[i];
    }

    /**
      * Return the number of arguments of the operator.
      */
    int get_n_arg() const
    {
        assert(this->get_expr_type() == coli::e_op);

        return this->op.size();
    }

    /**
      * Return the type of the expression (coli::expr_type).
      */
    coli::expr_t get_expr_type() const
    {
        return etype;
    }

    /**
      * Get the data type of the expression.
      */
    coli::primitive_t get_data_type() const
    {
        return dtype;
    }

    /**
      * Get the name of the ID.
      */
    std::string get_id_name() const
    {
        assert(this->get_expr_type() == coli::e_id);

        return id_name;
    }

    /**
      * Get the type of the operator (coli::op_t).
      */
    coli::op_t get_op_type() const
    {
        return _operator;
    }

    /**
      * Return a vector of the access of the computation
      * or array.
      * For example, for the computation C0(i,j), this
      * function will return the vector {i, j} where i and j
      * are both coli expressions.
      * For a buffer access A[i+1,j], it will return also {i+1, j}.
      */
    std::vector<coli::expr> get_access() const
    {
        assert(this->get_expr_type() == coli::e_op);
        assert(this->get_op_type() == coli::o_access);

        return access_vector;
    }

    /**
      * Get the number of dimensions in the access vector.
      */
    int get_n_dim_access() const
    {
        assert(this->get_expr_type() == coli::e_op);
        assert(this->get_op_type() == coli::o_access);

        return access_vector.size();
    }

    /**
     * Addition.
     */
    template<typename T> coli::expr operator+(T val) const
    {
        if ((std::is_same<T, coli::expr>::value))
        {
            return coli::expr(coli::o_add, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return coli::expr(coli::o_add, *this, coli::expr((T) val));
        }
        else
        {
            coli::error("Adding a coli expression to a non supported type.\n",
                        true);
        }
    }

    /**
     * Substruction.
     */
    template<typename T> coli::expr operator-(T val) const
    {
        if ((std::is_same<T, coli::expr>::value))
        {
            return coli::expr(coli::o_sub, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return coli::expr(coli::o_sub, *this, coli::expr((T) val));
        }
        else
        {
            coli::error("Substructing a coli expression from a non supported type.\n",
                        true);
        }
    }

    /**
     * Division.
     */
    template<typename T> coli::expr operator/(T val) const
    {
        if ((std::is_same<T, coli::expr>::value))
        {
            return coli::expr(coli::o_div, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return coli::expr(coli::o_div, *this, coli::expr((T) val));
        }
        else
        {
            coli::error("Dividing a coli expression by a non supported type.\n",
                        true);
        }
    }

    /**
     * Multiplication.
     */
    template<typename T> coli::expr operator*(T val) const
    {
        if ((std::is_same<T, coli::expr>::value))
        {
            return coli::expr(coli::o_mul, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return coli::expr(coli::o_mul, *this, coli::expr((T) val));
        }
        else
        {
            coli::error("Multiplying a coli expression by a non supported type.\n",
                        true);
        }
    }

    /**
     * Modulo.
     */
    template<typename T> coli::expr operator%(T val) const
    {
        if ((std::is_same<T, coli::expr>::value))
        {
            return coli::expr(coli::o_mod, *this, val);
        }
        else if ((std::is_same<T, uint8_t>::value) ||
                 (std::is_same<T, int8_t>::value) ||
                 (std::is_same<T, uint16_t>::value) ||
                 (std::is_same<T, int16_t>::value) ||
                 (std::is_same<T, int32_t>::value) ||
                 (std::is_same<T, uint32_t>::value))
        {
            return coli::expr(coli::o_mod, *this, coli::expr((T) val));
        }
        else
        {
            coli::error("Modulo of a coli expression by a non supported type.\n",
                        true);
        }
    }

    /**
     * Logical and of two expressions.
     */
    coli::expr operator&&(coli::expr e1) const
    {
      return coli::expr(coli::o_logical_and, *this, e1);
    }

    /**
     * Logical and of two expressions.
     */
    coli::expr operator||(coli::expr e1) const
    {
        return coli::expr(coli::o_logical_or, *this, e1);
    }

    /**
     * Expression multiplied by (-1).
     */
    coli::expr operator-() const
    {
        return coli::expr(coli::o_minus, *this);
    }

    /**
     * Comparison operator.
     */
    coli::expr operator==(coli::expr e1) const
    {
        return coli::expr(coli::o_eq, *this, e1);
    }

    /**
     * Less than operator.
     */
    coli::expr operator<(coli::expr e1) const
    {
        return coli::expr(coli::o_lt, *this, e1);
    }

    /**
     * Less than or equal operator.
     */
    coli::expr operator<=(coli::expr e1) const
    {
        return coli::expr(coli::o_le, *this, e1);
    }

    /**
     * Greater than operator.
     */
    coli::expr operator>(coli::expr e1) const
    {
        return coli::expr(coli::o_gt, *this, e1);
    }

    /**
     * Greater than or equal operator.
     */
    coli::expr operator>=(coli::expr e1) const
    {
        return coli::expr(coli::o_ge, *this, e1);
    }

    /**
      * Set the access of a computation or an array.
      * For example, for the computation C0(i,j), this
      * function will return the vector {i, j} where i and j
      * are both coli expressions.
      * For a buffer access A[i+1,j], it will return also {i+1, j}.
      */
    void set_access(std::vector<coli::expr> vector)
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
        if (ENABLE_DEBUG)
        {
            std::cout << "Expression:" << std::endl;
            std::cout << "Expression type:" << coli_type_expr_to_str(this->etype) << std::endl;
            switch (this->etype)
            {
                case coli::e_op:
                {
                    std::cout << "Expression operator type:" << coli_type_op_to_str(this->_operator) << std::endl;
                    std::cout << "Number of operands:" << this->get_n_arg() << std::endl;
                    for (int i = 0; i < this->get_n_arg(); i++)
                    {
                        this->op[i].dump(exhaustive);
                    }
                    if ((this->get_op_type() == coli::o_access) || (this->get_op_type() == coli::o_call))
                    {
                        std::cout << "Access expressions:" << std::endl;
                        for (const auto &e: this->get_access())
                        {
                            e.dump(exhaustive);
                        }
                    }
                    break;
                }
                case (coli::e_val):
                {
                    std::cout << "Expression value type:" << coli_type_primitive_to_str(this->dtype) << std::endl;

                    if (this->get_data_type() == coli::p_uint8)
                        std::cout << "Value:" << this->get_uint8_value() << std::endl;
                    else if (this->get_data_type() == coli::p_int8)
                        std::cout << "Value:" << this->get_int8_value() << std::endl;
                    else if (this->get_data_type() == coli::p_uint16)
                        std::cout << "Value:" << this->get_uint16_value() << std::endl;
                    else if (this->get_data_type() == coli::p_int16)
                        std::cout << "Value:" << this->get_int16_value() << std::endl;
                    else if (this->get_data_type() == coli::p_uint32)
                        std::cout << "Value:" << this->get_uint32_value() << std::endl;
                    else if (this->get_data_type() == coli::p_int32)
                        std::cout << "Value:" << this->get_int32_value() << std::endl;
                    else if (this->get_data_type() == coli::p_uint64)
                        std::cout << "Value:" << this->get_uint64_value() << std::endl;
                    else if (this->get_data_type() == coli::p_int64)
                        std::cout << "Value:" << this->get_int64_value() << std::endl;

                    break;
                }
                case (coli::e_id):
                {
                    std::cout << "Id name:" << this->get_id_name() << std::endl;
                    break;
                }
                default:
                    coli::error("Expression type not supported.", true);

            }

            std::cout << std::endl;
        }
    }
};

/**
 * A class that represents index expressions
 */
class idx: public coli::expr
{
public:
    /**
     * Construct an expression that represents an id.
     */
    idx(std::string name): expr(name)
    {
        assert(name.length() > 0);

        this->dtype = coli::p_int32;
    }
};

}

#endif
