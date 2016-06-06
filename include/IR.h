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

void split_string(std::string str, std::string delimiter,
		  std::vector<std::string> &vector);

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


// A class to hold parsed tokens of isl_space

class isl_space_tokens
{
public:
	std::vector<std::string> dimensions;

	isl_space_tokens(std::string isl_space_str)
	{
		assert(isl_space_str.empty() == false);
		this->Parse(isl_space_str);
	};

	isl_space_tokens() {};

	std::string get_str()
	{
		std::string result;

		for (int i=0; i<dimensions.size(); i++)
		{
			if (i != 0)
				result = result + ",";
			result = result + dimensions.at(i);
		}

		return result;
	};

	void replace(std::string in, std::string out1, std::string out2)
	{
		std::vector<std::string> new_dimensions;

		for (auto dim:dimensions)
		{
			if (dim.compare(in) == 0)
			{
				new_dimensions.push_back(out1);
				new_dimensions.push_back(out2);
			}
			else
				new_dimensions.push_back(dim);
		}

		dimensions = new_dimensions;
	}

	void Parse(std::string space);
	bool empty() {return dimensions.empty();};
};


// A class to hold parsed tokens of isl_constraints

class constraint_tokens
{
public:
	std::vector<std::string> constraints;
	constraint_tokens() { };

	void Parse(std::string str)
	{
		assert(str.empty() == false);

		split_string(str, "and", this->constraints);
	};

	void add(std::string str)
	{
		assert(str.empty() == false);
		constraints.push_back(str);
	}

	std::string get_str()
	{
		std::string result;

		for (int i=0; i<constraints.size(); i++)
		{
			if (i != 0)
				result = result + " and ";
			result = result + constraints.at(i);
		}

		return result;
	};

	bool empty() {return constraints.empty();};
};


// A class to hold parsed tokens of isl_maps

class isl_map_tokens
{
public:
	isl_space_tokens parameters;
	std::string domain_name;
	isl_space_tokens domain;
	isl_space_tokens range;
	constraint_tokens constraints;

	isl_map_tokens(std::string map_str)
	{
		int map_begin =  map_str.find("{")+1;
		int map_end   =  map_str.find("}")-1;

		assert(map_begin != std::string::npos);
		assert(map_end != std::string::npos);

		int domain_space_begin = map_str.find("[", map_begin)+1;
		int domain_space_begin_pre_bracket = map_str.find("[", map_begin)-1;
		int domain_space_end   = map_str.find("]", map_begin)-1;

		assert(domain_space_begin != std::string::npos);
		assert(domain_space_end != std::string::npos);

		domain_name = map_str.substr(map_begin,
		 		             domain_space_begin_pre_bracket-map_begin+1);

		std::string domain_space_str =
			map_str.substr(domain_space_begin,
		 		       domain_space_end-domain_space_begin+1);

		domain.Parse(domain_space_str);

		int pos_arrow = map_str.find("->", domain_space_end);

		assert(pos_arrow != std::string::npos);

		int range_space_begin = map_str.find("[", pos_arrow)+1;
		int range_space_end = map_str.find("]",pos_arrow)-1;

		assert(range_space_begin != std::string::npos);
		assert(range_space_end != std::string::npos);

		std::string range_space_str = map_str.substr(range_space_begin,
							 range_space_end-range_space_begin+1);
		range.Parse(range_space_str);
		int column_pos = map_str.find(":")+1;

		if (column_pos != std::string::npos)
		{
			std::string constraints_str = map_str.substr(column_pos,
								     map_end-column_pos+1);
			constraints.Parse(constraints_str);
		}
	};

	std::string get_str()
	{
		std::string result;

		result = "{" + domain_name + "[" + domain.get_str() + "] -> [" +
			  range.get_str() + "]";

		if (constraints.empty() == false)
			result = result + " : " + constraints.get_str();

		result = result + " }";

		return result;
	};

	isl_map *get_isl_map(isl_ctx *ctx)
	{
		return isl_map_read_from_str(ctx, this->get_str().c_str());
	};
};


// Halide IR specific functions

void halide_IR_dump(Halide::Internal::Stmt s);
Halide::Internal::Stmt generate_Halide_stmt_from_isl_node(IRProgram pgm, isl_ast_node *node,
		int level, std::vector<std::string> &generated_stmts);

#endif
