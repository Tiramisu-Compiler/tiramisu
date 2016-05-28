#ifndef _H_IR_
#define _H_IR_

#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>

#include <map>
#include <String.h>

#include <Halide.h>
#include <DebugIR.h>
#include <IR.h>

isl_union_set *create_time_space(
		__isl_take isl_union_set *set,
		__isl_take isl_union_map *umap);

isl_schedule *create_schedule_tree(isl_ctx *ctx,
		   isl_union_set *udom,
		   isl_union_map *sched_map);

isl_ast_node *generate_code(isl_ctx *ctx,
		   isl_schedule *sched_tree);

isl_union_map *create_schedule_map(isl_ctx *ctx,
		   std::string map);

isl_ast_node *stmt_halide_code_generator(isl_ast_node *node,
		isl_ast_build *build, void *user);

isl_ast_node *for_halide_code_generator_after_for(isl_ast_node *node,
		isl_ast_build *build, void *user);

class Computation {
public:
	/**
	  * Iteration space of the computation.
	 */
	isl_union_set *iter_space;

	/**
	  * Halide expression that represents the computation.
	  */
	Halide::Expr expression;
	Halide::Internal::Stmt *s;

	Computation(Halide::Expr expression, isl_union_set *iter_space) : iter_space(iter_space), expression(expression) { };

	Computation(isl_ctx *ctx, std::string iteration_space_str) {
		iter_space = isl_union_set_read_from_str(ctx, iteration_space_str.c_str());
		this->expression = expression;
		const char *tuple_name = isl_space_get_tuple_name(isl_union_set_get_space(iter_space), isl_dim_type::isl_dim_set);
		unsigned char stmt_name_length = strlen(tuple_name);
		char *stmt_name = (char *) malloc(stmt_name_length*sizeof(char));
		strcpy(stmt_name, tuple_name);
	}

	void dump();
};

void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node);
void halide_IR_dump(Halide::Internal::Stmt s);

#endif
