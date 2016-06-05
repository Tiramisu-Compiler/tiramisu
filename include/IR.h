#ifndef _H_IR_
#define _H_IR_

#include <isl/set.h>
#include <isl/map.h>
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

void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node);


//IRProgram

class IRProgram
{
private:
	std::string name;
	std::vector<IRFunction *> functions;
public:
	std::map<std::string, int> parallel_dimensions;
	std::map<std::string, int> vector_dimensions;

	IRProgram(std::string name): name(name) { };
	void add_function(IRFunction *fct);

	void tag_parallel_dimension(std::string stmt_name, int dim);
	void tag_vector_dimension(std::string stmt_name, int dim);

	isl_union_set *get_iteration_spaces();
	isl_union_map *get_schedule_map();

	void dump_ISIR();
	void dump_schedule();
	void dump();
};


// IRFunction

class IRFunction
{
public:
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

	IRFunction(std::string name, IRProgram *pgm): name(name) {
		pgm->add_function(this);
	};
	void dump_ISIR();
	void dump_schedule();
	void dump();
};

// Computation

class Computation {
public:
	isl_ctx *ctx;

	/**
	  * Iteration space of the computation.
	 */
	isl_set *iter_space;

	/**
	  *
	  */
	isl_map *schedule;

	/**
	  * The name of this computation.
	  */
	std::string name;

	/**
	  * Halide expression that represents the computation.
	  */
	Halide::Expr expression;
	Halide::Internal::Stmt stmt;

	Computation(Halide::Expr expression, isl_set *iter_space) : iter_space(iter_space), expression(expression) { };

	Computation(isl_ctx *ctx,
		    Halide::Internal::Stmt given_stmt,
		    std::string iteration_space_str, IRFunction *fct) {
		this->ctx = ctx;
		iter_space = isl_set_read_from_str(ctx, iteration_space_str.c_str());
		name = std::string(isl_space_get_tuple_name(isl_set_get_space(iter_space), isl_dim_type::isl_dim_set));
		this->stmt = given_stmt;
		stmts_list.insert(std::pair<std::string, Halide::Internal::Stmt>(name, this->stmt));
		fct->add_computation_to_body(this);

		std::string domain = isl_space_to_str(isl_set_get_space(iter_space));
		std::string schedule_map_str = isl_set_to_str(iter_space);
		domain = domain.erase(domain.find("{"), 1);
		domain = domain.erase(domain.find("}"), 1);
		std::string domain_without_name = domain;
		domain_without_name.erase(domain.find(name), name.length()); 
		schedule_map_str.insert(schedule_map_str.find(":"), " -> " + domain_without_name);
		this->schedule = isl_map_read_from_str(ctx, schedule_map_str.c_str());
	}

	void Tile(std::string inDim0, std::string inDim1, std::string outDim0,
			std::string outDim1, std::string outDim2,
			std::string outDime3, int sizeX, int sizeY);

	void Split(int inDim0, int sizeX);
	void Interchange(int inDim0, int inDim1);

	void Schedule(std::string umap_str);

	void dump_ISIR();
	void dump_schedule();
	void dump();
};


// Halide IR specific functions

void halide_IR_dump(Halide::Internal::Stmt s);
Halide::Internal::Stmt generate_Halide_stmt_from_isl_node(IRProgram pgm, isl_ast_node *node,
		int level, std::vector<std::string> &generated_stmts);

#endif
