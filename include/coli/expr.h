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
#include <string.h>
#include <stdint.h>

#include <Halide.h>
#include <coli/debug.h>
#include <coli/type.h>

namespace coli
{

/**
  * A class to represent coli expressions.
  */
class expr
{
	/**
	  * The value of the 1st, 2nd and 3rd operators of the expression.
	  * op[0] is the 1st operator, op[1] is the 2nd, ...
	  */
	coli::expr *op;

	/**
	  * The type of the expression.
	  */
	coli::type::expr etype;

	/**
	  * The type of the operator.
	  */
	coli::type::op _operator;

	/**
	  * The value of the expression. 
	  */
	union {
		uint8_t		uint8_value;
		int8_t		int8_value;
		uint32_t	uint32_value;
		int32_t		int32_value;
		uint64_t	uint64_value;
		int64_t		int64_value;
	};

	/**
	  * Data type.
	  */
	coli::type::primitive dtype;

	/**
	  * Number of operator arguments.
	  */
	int n_arg;

	/**
	  * Identifier expression.
	  * This is the identifier of the buffer, function, or computation
	  * if the operator is a buffer access, a function call or a
	  * computation access.
	  */
	coli::expr *id_expr;

	/**
	  * A vector of expressions representing buffer accesses,
	  * or computation accesses.
	  * For example for the computation C0(i,j), the access is
	  * the vector {i, j}.
	  */
	std::vector<coli::expr*> access_vector;

public:

	/**
	  * Create a expression of type \p t (a uniary operator).
	  */
	expr(coli::type::op o, coli::expr expr0)
	{
		assert((o == coli::type::op::minus) && "The only unary operator is the minus operator.");

		this->set_op_type(o);
		this->set_expr_type(coli::type::expr::op);

		this->op = (coli::expr *) malloc(sizeof(coli::expr));
		this->op[0] = expr0;
		this->n_arg = 1;
	}

	expr(coli::type::op o, coli::expr expr0, coli::expr expr1)
	{
		assert((o != coli::type::op::minus) &&
			(o != coli::type::op::call) &&
			(o != coli::type::op::access) &&
			(o != coli::type::op::cond) &&
			"The operator is not an binay operator.");

		this->set_op_type(o);
		this->set_expr_type(coli::type::expr::op);

		this->op = (coli::expr *) malloc(2*sizeof(coli::expr));
		this->op[0] = expr0;
		this->op[1] = expr1;
		this->n_arg = 2;
	}

	expr(coli::type::op o, coli::expr expr0, coli::expr expr1, coli::expr expr2)
	{
		assert((o == coli::type::op::cond) && "The operator is not a ternary operator.");

		this->set_op_type(o);
		this->set_expr_type(coli::type::expr::op);

		this->op = (coli::expr *) malloc(2*sizeof(coli::expr));
		this->op[0] = expr0;
		this->op[1] = expr1;
		this->op[2] = expr2;
		this->n_arg = 3;
	}

	expr(coli::type::op o, coli::expr id_expr, std::vector<coli::expr*> access_expressions)
	{
		assert((o == coli::type::op::access) && "The operator is not an access operator.");
		assert(access_expressions.size() > 0);
		assert(id_expr.get_expr_type() ==
				coli::type::expr::id);

		this->set_op_type(coli::type::op::access);
		this->set_expr_type(coli::type::expr::op);

		this->set_identifier(id_expr);
		this->set_access(access_expressions);
	}

	/**
	  * Construct an unsigned 8bit integer expression.
	  */
	expr(uint8_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::uint8);
		uint8_value = val;
	}

	/**
	  * Construct a signed 8bit integer expression.
	  */
	expr(int8_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::int8);
		int8_value = val;
	}

	/**
	  * Construct an unsigned 32bit integer expression.
	  */
	expr(uint32_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::uint32);
		uint32_value = val;
	}

	/**
	  * Construct a signed 32bit integer expression.
	  */
	expr(int32_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::int32);
		int32_value = val;
	}

	/**
	  * Construct an unsigned 64bit integer expression.
	  */
	expr(uint64_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::uint64);
		uint64_value = val;
	}

	/**
	  * Construct a signed 64bit integer expression.
	  */
	expr(int64_t val)
	{
		this->set_expr_type(coli::type::expr::val);
		this->set_data_type(coli::type::primitive::int64);
		int64_value = val;
	}

	/**
	  * Return the actual value of the expression.
	  */
	//@
	uint8_t get_uint8_value()
	{
		return uint8_value;
	}

	int8_t get_int8_value()
	{
		return int8_value;
	}

	uint32_t get_uint32_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return uint32_value;
	}

	int32_t get_int32_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return int32_value;
	}

	uint64_t get_uint64_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return uint64_value;
	}

	int64_t get_int64_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return int64_value;
	}	
	//@

	/**
	  * Return the value of the \p i 'th operator of the expression.
	  * \p i can be 0, 1 or 2.
	  */
	coli::expr get_operator(int i)
	{
		assert(this->get_expr_type() == coli::type::expr::op);

		assert((i<3) && "The expression has only 3 operators.");
		return op[i];
	}

	/**
	  * Return the number of arguments of the operator.
	  */
	int get_n_arg()
	{
		assert(this->get_expr_type() == coli::type::expr::op);

		return n_arg;
	}

	/**
	  * Return the type of the expression (coli::expr_type).
	  */
	coli::type::expr get_expr_type()
	{
		return etype;
	}

	/**
	  * Get the data type of the expression.
	  */
	coli::type::primitive get_data_type()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return dtype;
	}

	/**
	  * Get the type of the operator (coli::type::op).
	  */
	coli::type::op get_op_type()
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
	std::vector<coli::expr*> get_access()
	{
		assert(this->get_expr_type() == coli::type::expr::op);
		assert(this->get_op_type() == coli::type::op::access);

		return access_vector;
	}

	/**
	  * Get the number of dimensions in the access vector.
	  */
	int get_n_dim_access()
	{
		assert(this->get_expr_type() == coli::type::expr::op);
		assert(this->get_op_type() == coli::type::op::access);

		return access_vector.size();
	}

	/**
	  * Get the identifier of the access operator or the
	  * call operator.
	  */
	coli::expr get_identifier()
	{
		assert(this->get_expr_type() == coli::type::expr::op);
		assert(this->get_op_type() == coli::type::op::access ||
			this->get_op_type() == coli::type::op::call);

		return *(this->id_expr);
	}

	// TODO: Tes this function
	/**
	  * Set the identifier of the access operator or the
	  * call operator.
	  */
	void set_identifier(coli::expr identifier)
	{
		assert(this->get_expr_type() == coli::type::expr::op);
		assert(this->get_op_type() == coli::type::op::access ||
			this->get_op_type() == coli::type::op::call);

		this->id_expr = (coli::expr *) malloc(sizeof(coli::expr));
		*this->id_expr = identifier;
	}

	/**
	  * Set the access of a computation or an array.
	  * For example, for the computation C0(i,j), this
	  * function will return the vector {i, j} where i and j
	  * are both coli expressions.
	  * For a buffer access A[i+1,j], it will return also {i+1, j}.
	  */
	void set_access(std::vector<coli::expr*> vector)
	{
		access_vector = vector;
	}

	/**
	  * Set the type of the expression (coli::expr_type).
	  */
	void set_expr_type(coli::type::expr t)
	{
		etype = t;
	}

	/**
	  * Set the type of the operator (coli::type::op).
	  */
	void set_op_type(coli::type::op op)
	{
		_operator = op;
	}

	/**
	  * Set the data type of the expression.
	  */
	void set_data_type(coli::type::primitive t)
	{
		dtype = t;
	}
};

}

#endif
