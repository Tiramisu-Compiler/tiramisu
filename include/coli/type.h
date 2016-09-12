#ifndef _H_COLI_TYPE_
#define _H_COLI_TYPE_

#include <string.h>
#include <stdint.h>

namespace coli {
namespace type {

/**
* The possible types of an expression.
*/
	enum class expr {
		val,
		id,
		op,
		none
	};

	/**
	  * coli data types.
	  */
	enum class primitive {
		uint8,
		int8,
		uint32,
		int32,
		uint64,
		int64,
		none
	};

	/**
	  * Types of coli operators.
	  */
	enum class op {
		logical_and,
		logical_or,
		max,
		min,
		minus,
		add,
		sub,
		mul,
		div,
		mod,
		cond,
		eq,
		le,
		lt,
		ge,
		gt,
		call,
		access,
		none
	};

	/**
	* Types of function arguments.
	*/
	enum class argument {
		input,
		output,
		none
	};
} }

#endif
