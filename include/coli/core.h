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
#include <coli/debug.h>

namespace coli
{

class program;
class function;
class computation;
class buffer;

extern std::map<std::string, computation *> computations_list;

void split_string(std::string str, std::string delimiter,
		  std::vector<std::string> &vector);

/* Schedule the iteration space.  */
isl_union_set *create_time_space_representation(
		__isl_take isl_union_set *set,
		__isl_take isl_union_map *umap);

isl_schedule *create_schedule_tree(isl_ctx *ctx,
		   isl_union_set *udom,
		   isl_union_map *sched_map);

isl_ast_node *generate_isl_ast_node(isl_ctx *ctx,
		   isl_schedule *sched_tree);

isl_union_map *create_schedule_map(isl_ctx *ctx,
		   std::string map);

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation).  Store the access in computation->access.
 */
isl_ast_node *stmt_halide_code_generator(isl_ast_node *node,
		isl_ast_build *build, void *user);

isl_ast_node *for_halide_code_generator_after_for(isl_ast_node *node,
		isl_ast_build *build, void *user);

void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node);


/**
  * A class to represent a full program.  A program is composed of
  * functions (of type coli::function).
  */
class program
{
private:
	std::string name;
	std::vector<coli::function *> functions;
public:
	std::map<std::string, int> parallel_dimensions;
	std::map<std::string, int> vector_dimensions;

	/**
	 * Program name.
	 */
	program(std::string name): name(name) { };

	/**
	  * Add a function to the program.
	  */
	void add_function(coli::function *fct);

	void tag_parallel_dimension(std::string stmt_name, int dim);
	void tag_vector_dimension(std::string stmt_name, int dim);

	isl_union_set *get_iteration_spaces();
	isl_union_map *get_schedule_map();

	void dump_ISIR();
	void dump_schedule();
	void dump();
};


/**
  * A class to represent functions.  A function is composed of
  * computations (of type coli::computation).
  */
class function
{
public:
	/**
	  * The name of the function.
	  */
	std::string name;

	/**
	  * List of buffers of the function.  Some of these buffers are passed
	  * to the function as arguments and some are declared and allocated
	  * within the function itself.
	  */
	std::map<std::string, Halide::Buffer *> buffers_list;

	/**
	  * Body of the function (a vector of computations).
	  * The order of the computations in the vector do not have any
	  * effect on the actual order of execution of the computations.
	  * The order of execution of computations is specified through the
	  * schedule.
	  */
	std::vector<computation *> body;

	/**
	  * Function arguments.  These are the buffers or scalars that are
	  * passed to the function.
	  */
	std::vector<Halide::Argument> arguments;

public:
	void add_computation_to_body(computation *cpt);

	/**
	  * Add an argument to the list of arguments of the function.
	  * The order in which the arguments are added is important.
	  * (the first added argument is the first function argument, ...).
	  */
	void add_argument(coli::buffer buf);

	function(std::string name, coli::program *pgm): name(name) {
		pgm->add_function(this);
	};

	std::vector<Halide::Argument> get_args()
	{
		return arguments;
	}

	void dump_ISIR();
	void dump_schedule();
	void dump();
};

/**
  * A class that represents computations.
  */
class computation {
public:
	isl_ctx *ctx;

	/**
	  * Iteration space of the computation.
	 */
	isl_set *iter_space;

	/**
	  * Schedule of the computation.
	  */
	isl_map *schedule;

	/**
	  * The function where this computation is declared.
	  */
	coli::function *function;

	/**
	  * The name of this computation.
	  */
	std::string name;

	/**
	  * Halide expression that represents the computation.
	  */
	Halide::Expr expression;

	/**
	  * Halide statement that assigns the computation to a buffer location.
	  */
	Halide::Internal::Stmt stmt;

	/**
	  * Access function.  A map indicating how each computation should be stored
	  * in memory.
	  */
	isl_map *access;

	/**
	  * An isl_ast_expr representing the index of the array where
	  * the computation will be stored.  This index is computed after the scheduling is done.
	  */
	isl_ast_expr *index_expr;

	computation(Halide::Expr expression, isl_set *iter_space) : iter_space(iter_space), expression(expression) { };

	computation(isl_ctx *ctx,
		    Halide::Expr expr,
		    std::string iteration_space_str, coli::function *fct) {
		// Initialize all the fields to NULL (useful for later asserts)
		index_expr = NULL;
		access = NULL;
		schedule = NULL;
		stmt = Halide::Internal::Stmt();

		this->ctx = ctx;
		iter_space = isl_set_read_from_str(ctx, iteration_space_str.c_str());
		name = std::string(isl_space_get_tuple_name(isl_set_get_space(iter_space), isl_dim_type::isl_dim_set));
		this->expression = expr;
		computations_list.insert(std::pair<std::string, computation *>(name, this));
		function = fct;
		function->add_computation_to_body(this);

		std::string domain = isl_set_to_str(iter_space);
		std::string schedule_map_str = isl_set_to_str(iter_space);
		domain = domain.erase(domain.find("{"), 1);
		domain = domain.erase(domain.find("}"), 1);
		if (schedule_map_str.find(":") != std::string::npos)
			domain = domain.erase(domain.find(":"), domain.length() - domain.find(":"));
		std::string domain_without_name = domain;
		domain_without_name.erase(domain.find(name), name.length());

		if (schedule_map_str.find(":") != std::string::npos)
			schedule_map_str.insert(schedule_map_str.find(":"), " -> " + domain_without_name);
		else
			schedule_map_str.insert(schedule_map_str.find("]")+1, " -> " + domain_without_name);

		this->schedule = isl_map_read_from_str(ctx, schedule_map_str.c_str());
	}

	void SetAccess(std::string access_str)
	{
		this->access = isl_map_read_from_str(this->ctx, access_str.c_str());
	}

	void create_halide_assignement(std::vector<std::string> &iterators);

	void Tile(int inDim0, int inDim1, int sizeX, int sizeY);

	/**
	 * Modify the schedule of this computation so that it splits the
	 * dimension inDim0 of the iteration space into two new dimensions.
	 * The size of the inner dimension created is sizeX.
	 */
	void Split(int inDim0, int sizeX);

	/**
	 * Modify the schedule of this computation so that the two dimensions
	 * inDim0 and inDime1 are interchanged (swaped).
	 */
	void Interchange(int inDim0, int inDim1);

	void Schedule(std::string umap_str);

	void dump_ISIR();
	void dump_schedule();
	void dump();
};

/**
  * A class that represents a buffer.  The result of a computation
  * can be stored in a buffer.  A computation can also be a binding
  * to a buffer (i.e. a buffer element is represented as a computation).
  */
class buffer
{
	/**
	  * The name of the buffer.
	  */
	std::string name;

	/**
	  * The number of dimensions of the buffer.
	  */
	int nb_dims;

	/**
	  * The size of buffer dimensions.  Assuming the following
	  * buffer: buf[N0][N1][N2].  The first vector element represents the
	  * leftmost dimension of the buffer (N0), the second vector element
	  * represents N1, ...
	  */
	std::vector<int> dim_sizes;

	/**
	  * The type of the elements of the buffer.
	  */
	Halide::Type type;

	/**
	  * Buffer data.
	  */
	uint8_t *data;

	/**
	  * The coli function where this buffer is declared or where the
	  * buffer is an argument.
	  */
	coli::function *fct;

public:
	buffer(std::string name, int nb_dims, std::vector<int> dim_sizes,
			Halide::Type type, uint8_t *data, coli::function *fct):
		name(name), nb_dims(nb_dims), dim_sizes(dim_sizes), type(type),
		data(data), fct(fct)
		{
			Halide::Buffer *buf = new Halide::Buffer(type, dim_sizes, data, name);
			fct->buffers_list.insert(std::pair<std::string, Halide::Buffer *>(buf->name(), buf));
		};

	std::string get_name()
	{
		return name;
	}

	Halide::Type get_type()
	{
		return type;
	}

	int get_n_dims()
	{
		return nb_dims;
	}
};

/**
  * A class to hold parsed tokens of isl_space.
  */
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


/**
 * A class to hold parsed tokens of isl_constraints.
 */
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


/**
  * A class to hold parsed tokens of isl_maps.
  */
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

/**
  * Generate a Halide statement from an ISL ast node object in the ISL ast
  * tree.
  * Level represents the level of the node in the schedule.  0 means root.
  */
Halide::Internal::Stmt generate_Halide_stmt_from_isl_node(coli::program pgm, isl_ast_node *node,
		int level, std::vector<std::string> &generated_stmts,
		std::vector<std::string> &iterators);

}
#endif
