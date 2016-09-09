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

std::string coli_type_expr_to_str(coli::type::expr type);
std::string coli_type_op_to_str(coli::type::op type);
std::string coli_type_primitive_to_str(coli::type::primitive type);

/**
  * A class to represent coli expressions.
  */
class expr
{
	/**
	  * The type of the expression.
	  */
	coli::type::expr etype;

	/**
	  * The type of the operator.
	  */
	coli::type::op _operator;

	/**
	  * The value of the 1st, 2nd and 3rd operators of the expression.
	  * op[0] is the 1st operator, op[1] is the 2nd, ...
	  */
	std::vector<coli::expr*> op;

	/**
	  * Data type.
	  */
	coli::type::primitive dtype;

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
	  * A vector of expressions representing buffer accesses,
	  * or computation accesses.
	  * For example for the computation C0(i,j), the access is
	  * the vector {i, j}.
	  */
	std::vector<coli::expr*> access_vector;

	/**
	  * Identifier name.
	  */
	std::string id_name;

public:

	/**
	  * Create a expression of type \p t (a unary operator).
	  */
	static coli::expr *make(coli::type::op o, coli::expr *expr0)
	{
		assert((o == coli::type::op::minus) && "The only unary operator is the minus operator.");

		coli::expr *new_expression = new coli::expr();

		new_expression->_operator = o;
		new_expression->etype = coli::type::expr::op;
		new_expression->dtype = coli::type::primitive::none;

		new_expression->op.push_back(expr0);

		return new_expression;
	}

	static coli::expr *make(coli::type::op o, coli::expr *expr0, coli::expr *expr1)
	{
		assert((o != coli::type::op::minus) &&
			(o != coli::type::op::call) &&
			(o != coli::type::op::access) &&
			(o != coli::type::op::cond) &&
			"The operator is not an binay operator.");

		coli::expr *new_expression = new coli::expr();

		new_expression->_operator = o;
		new_expression->etype = coli::type::expr::op;
		new_expression->dtype = coli::type::primitive::none;

		new_expression->op.push_back(expr0);
		new_expression->op.push_back(expr1);

		return new_expression;
	}

	static coli::expr *make(coli::type::op o, coli::expr *expr0, coli::expr *expr1, coli::expr *expr2)
	{
		assert((o == coli::type::op::cond) && "The operator is not a ternary operator.");

		coli::expr *new_expression = new coli::expr();

		new_expression->_operator = o;
		new_expression->etype = coli::type::expr::op;
		new_expression->dtype = coli::type::primitive::none;

		new_expression->op.push_back(expr0);
		new_expression->op.push_back(expr1);
		new_expression->op.push_back(expr2);

		return new_expression;
	}

	static coli::expr *make(coli::type::op o, coli::expr *id_expr, std::vector<coli::expr*> access_expressions)
	{
		assert((o == coli::type::op::access) && "The operator is not an access operator.");
		assert(access_expressions.size() > 0);
		assert(id_expr->get_expr_type() ==
				coli::type::expr::id);

		coli::expr *new_expression = new coli::expr();

		new_expression->_operator = coli::type::op::access;
		new_expression->etype = coli::type::expr::op;
		new_expression->dtype = coli::type::primitive::none;

		new_expression->set_access(access_expressions);
		new_expression->op.push_back(id_expr);

		return new_expression;
	}

	/**
	* Construct an expression that represents an id.
	*/
	static coli::expr *make(std::string name)
	{
		assert(name.length() > 0);

		coli::expr *new_expression = new coli::expr();
		new_expression->etype = coli::type::expr::id;
		new_expression->id_name = name;

		new_expression->_operator = coli::type::op::none;
		new_expression->dtype = coli::type::primitive::none;

		return new_expression;
	}

	/**
	  * Construct an unsigned 8bit integer expression.
	  */
	static coli::expr *make(uint8_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::uint8;
		new_expression->uint8_value = val;

		return new_expression;
	}

	/**
	  * Construct a signed 8bit integer expression.
	  */
	static coli::expr *make(int8_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::int8;
		new_expression->int8_value = val;

		return new_expression;
	}

	/**
	  * Construct an unsigned 32bit integer expression.
	  */
	static coli::expr *make(uint32_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::uint32;
		new_expression->uint32_value = val;

		return new_expression;
	}

	/**
	  * Construct a signed 32bit integer expression.
	  */
	static coli::expr *make(int32_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::int32;
		new_expression->int32_value = val;

		return new_expression;
	}

	/**
	  * Construct an unsigned 64bit integer expression.
	  */
	static coli::expr *make(uint64_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::uint64;
		new_expression->uint64_value = val;

		return new_expression;
	}

	/**
	  * Construct a signed 64bit integer expression.
	  */
	static coli::expr *make(int64_t val)
	{
		coli::expr *new_expression = new coli::expr();

		new_expression->etype = coli::type::expr::val;
		new_expression->_operator = coli::type::op::none;

		new_expression->dtype = coli::type::primitive::int64;
		new_expression->int64_value = val;

		return new_expression;
	}

	/**
	  * Return the actual value of the expression.
	  */
	//@
	uint8_t get_uint8_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

		return uint8_value;
	}

	int8_t get_int8_value()
	{
		assert(this->get_expr_type() == coli::type::expr::val);

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
	coli::expr *get_operator(int i)
	{
		assert(this->get_expr_type() == coli::type::expr::op);

		assert((i<3) && "The expression has only 3 operators.");
		return this->op[i];
	}

	/**
	  * Return the number of arguments of the operator.
	  */
	int get_n_arg()
	{
		assert(this->get_expr_type() == coli::type::expr::op);

		return this->op.size();
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
	  * Get the name of the ID.
	  */
	std::string get_id_name()
	{
		assert(this->get_expr_type() == coli::type::expr::id);

		return id_name;
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

	/**
	  * Dump the object on standard output (dump most of the fields of
	  * the expression class).  This is mainly useful for debugging.
	  * If \p exhaustive is set to true, all the fields of the class are
	  * printed.  This is useful to find potential initialization problems.
	  */
	void dump(bool exhaustive)
	{
		if (DEBUG)
		{
			std::cout << "Expression:" << std::endl;
			std::cout << "Expression type:" << coli_type_expr_to_str(this->etype) << std::endl;
			switch (this->etype)
			{
				case coli::type::expr::op:
				{
					std::cout << "Expression operator type:" << coli_type_op_to_str(this->_operator) << std::endl;
					std::cout << "Number of operands:" << this->get_n_arg() << std::endl;
					for (int i = 0; i < this->get_n_arg(); i++)
						this->op[i]->dump(exhaustive);
					if ((this->get_op_type() == coli::type::op::access) || (this->get_op_type() == coli::type::op::call))
					{
						std::cout << "Access expressions:" << std::endl;
						for (auto e: this->get_access())
							e->dump(exhaustive);
					}
					break;
				}
				case (coli::type::expr::val):
				{
					std::cout << "Expression value type:" << coli_type_primitive_to_str(this->dtype) << std::endl;

					if (this->get_data_type() == coli::type::primitive::uint8)
						std::cout << "Value:" << this->get_uint8_value() << std::endl;
					else if (this->get_data_type() == coli::type::primitive::int8)
						std::cout << "Value:" << this->get_int8_value() << std::endl;
					else if (this->get_data_type() == coli::type::primitive::uint32)
						std::cout << "Value:" << this->get_uint32_value() << std::endl;
					else if (this->get_data_type() == coli::type::primitive::int32)
						std::cout << "Value:" << this->get_int32_value() << std::endl;
					else if (this->get_data_type() == coli::type::primitive::uint64)
						std::cout << "Value:" << this->get_uint64_value() << std::endl;
					else if (this->get_data_type() == coli::type::primitive::int64)
						std::cout << "Value:" << this->get_int64_value() << std::endl;

					break;
				}
				case (coli::type::expr::id):
				{
					//std::cout << "Id name:" << this->id_name << std::endl;
					coli::error("ID expressions still not supported.", true);
					break;
				}
				default:
					coli::error("Expression type not supported.", true);

			}

			std::cout << std::endl;
		}
	}
};

}

#endif
