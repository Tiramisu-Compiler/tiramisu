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

extern std::map<std::string, Halide::Internal::Stmt> stmts_list;
class IRProgram;
class IRFunction;
class Computation;

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


class IRProgram
{
private:
	std::string name;
	std::vector<IRFunction *> functions;
public:
	IRProgram(std::string name): name(name) { };
	void add_function(IRFunction *fct);
};


class IRFunction
{
private:
	std::string name;

	/**
	  * Function signature (a vector of computations).
	  */
	std::vector<Computation *> signature;

	/**
	  * Body of the function (a vector of functions).
	  * The order of the computations in the vector do not have any
	  * effect on the actual order of execution of the computations.
	  * The order of execution of computations is specified through the
	  * schedule.
	  */
	std::vector<Computation *> body;

public:
	void add_computation_to_body(Computation *cpt);
	void add_computation_to_signature(Computation *cpt);

	IRFunction(std::string name): name(name) { };
};


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
	Halide::Internal::Stmt stmt;

	Computation(Halide::Expr expression, isl_union_set *iter_space) : iter_space(iter_space), expression(expression) { };

	Computation(isl_ctx *ctx,
		    Halide::Internal::Stmt given_stmt,
		    std::string iteration_space_str, IRFunction fct) {
		iter_space = isl_union_set_read_from_str(ctx, iteration_space_str.c_str());
		const char *tuple_name = isl_space_get_tuple_name(isl_union_set_get_space(iter_space), isl_dim_type::isl_dim_set);
		this->stmt = given_stmt;
		stmts_list.insert(std::pair<std::string, Halide::Internal::Stmt>(tuple_name, this->stmt));
		fct.add_computation_to_body(this);
	}

	void dump();
};

void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node);
void halide_IR_dump(Halide::Internal::Stmt s);

#endif
