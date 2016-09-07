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

public:

	/**
	  * Create a expression of type \p t (a uniary operator).
	  */
	expr(coli::type::expr t, coli::expr expr0)
	{
		assert((t == coli::type::expr::minus) && "The only unary operator is the minus operator.");

		this->set_expr_type(t);
		op = (coli::expr *) malloc(sizeof(coli::expr));
		op[0] = expr0;
	}

	expr(coli::type::expr t, coli::expr expr0, coli::expr expr1)
	{
		assert((t != coli::type::expr::minus) &&
			(t != coli::type::expr::call) &&
			(t != coli::type::expr::computation) &&
			(t != coli::type::expr::cond) &&
			"The operator is not an binay operator.");

		this->set_expr_type(t);
		op = (coli::expr *) malloc(2*sizeof(coli::expr));
		op[0] = expr0;
		op[1] = expr1;
	}

	expr(coli::type::expr t, coli::expr expr0, coli::expr expr1, coli::expr expr2)
	{
		assert((t == coli::type::expr::cond) && "The operator is not a ternary operator.");

		this->set_expr_type(t);
		op = (coli::expr *) malloc(2*sizeof(coli::expr));
		op[0] = expr0;
		op[1] = expr1;
		op[2] = expr2;
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
		return uint32_value;
	}

	int32_t get_int32_value()
	{
		return int32_value;
	}

	uint64_t get_uint64_value()
	{
		return uint64_value;
	}

	int64_t get_int64_value()
	{
		return int64_value;
	}	
	//@

	/**
	  * Return the value of the \p i 'th operator of the expression.
	  * \p i can be 0, 1 or 2.
	  */
	coli::expr get_op(int i)
	{
		assert((i<3) && "The expression has only 3 operators.");
		return op[i];
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
		return dtype;
	}
	/**
	  * Set the type of the expression (coli::expr_type).
	  */
	void set_expr_type(coli::type::expr t)
	{
		etype = t;
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
