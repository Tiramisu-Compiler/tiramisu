#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string>

namespace coli
{

std::map<std::string, computation *> computations_list;
bool global::auto_data_mapping;

// Used for the generation of new variable names.
int id_counter = 0;

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation).  Store the access in computation->access.
 */
isl_ast_node *stmt_code_generator(isl_ast_node *node,
		isl_ast_build *build, void *user);

isl_ast_node *for_code_generator_after_for(isl_ast_node *node,
		isl_ast_build *build, void *user);


/**
  * Generate an isl AST for the library.
  */
void library::gen_isl_ast()
{
	// Check that time_processor representation has already been computed,
	// that the time_processor identity relation can be computed without any
	// issue and check that the access was provided.
	assert(this->get_schedule() != NULL);

	isl_ctx *ctx = this->get_ctx();
	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &coli::for_code_generator_after_for, NULL);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &coli::stmt_code_generator, this);

	this->align_schedules();

	// Intersect the iteration domain with the domain of the schedule.
	isl_union_map *umap =
		isl_union_map_intersect_domain(
			isl_union_map_copy(this->get_schedule()),
			isl_union_set_copy(this->get_iteration_domain()));

	this->ast = isl_ast_build_node_from_schedule_map(ast_build, umap);

	isl_ast_build_free(ast_build);
}


/**
  * A helper function to split a string.
  */
// TODO: Test this function
void split_string(std::string str, std::string delimiter,
		  std::vector<std::string> &vector)
{
	size_t pos = 0;
	std::string token;
	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		vector.push_back(token);
		str.erase(0, pos + delimiter.length());
	}
	token = str.substr(0, pos);
	vector.push_back(token);
}

void coli::parser::constraint::parse(std::string str)
{
	assert(str.empty() == false);

	split_string(str, "and", this->constraints);
};

void coli::parser::space::parse(std::string space)
{
	std::vector<std::string> vector;
	split_string(space, ",", vector);

	// Check if the vector has constraints
	for (int i=0; i<vector.size(); i++)
		if (vector[i].find("=") != std::string::npos)
		{
			vector[i] = vector[i].erase(0, vector[i].find("=")+1);
		}

	this->dimensions = vector;
}

std::string generate_new_variable_name()
{
	return "c" + std::to_string(id_counter++);
}

/**
  * Methods for the computation class.
  */

void coli::computation::tag_parallel_dimension(int par_dim)
{
	assert(par_dim >= 0);
	assert(this->get_name().length() > 0);
	assert(this->get_function() != NULL);
	assert(this->get_function()->get_library() != NULL);

	this->get_function()->get_library()->add_parallel_dimension(this->get_name(), par_dim);
}

void coli::computation::tag_vector_dimension(int par_dim)
{
	assert(par_dim >= 0);
	assert(this->get_name().length() > 0);
	assert(this->get_function() != NULL);
	assert(this->get_function()->get_library() != NULL);

	this->get_function()->get_library()->add_vector_dimension(this->get_name(), par_dim);
}

void computation::dump_iteration_domain()
{
	if (DEBUG)
	{
		isl_set_dump(this->get_iteration_domain());
	}
}

void library::dump_halide_stmt()
{
	for (auto func: this->get_functions())
		if (DEBUG)
		{
			coli::str_dump("\n\n");
			coli::str_dump("\nGenerated Halide Low Level IR:\n");
			std::cout << func->get_halide_stmt();
			coli::str_dump("\n\n\n\n");
		}
}


void library::dump_time_processor_domain()
{
	// Create time space domain

	if (DEBUG)
	{
		coli::str_dump("\n\nTime-processor domain:\n");

		for (auto func: this->get_functions())
		{
			coli::str_dump("Function " + func->get_name() + ":\n");
			for (auto comp: func->get_computations())
				isl_set_dump(
					comp->get_time_processor_domain());
		}

		coli::str_dump("\n\n");
	}
}

void library::gen_time_processor_domain()
{
	for (auto func: this->get_functions())
		for (auto comp: func->get_computations())
			comp->gen_time_processor_domain();
}

void computation::dump_schedule()
{
	if (DEBUG)
	{
		isl_map_dump(this->schedule);
	}
}

void computation::dump()
{
	if (DEBUG)
	{
		std::cout << "computation \"" << this->name << "\"" << std::endl;
		isl_set_dump(this->get_iteration_domain());
		std::cout << "Schedule " << std::endl;
		isl_map_dump(this->schedule);
		coli::str_dump("Halide statement:\n");
		if (this->stmt.defined())
		{
			std::cout << this->stmt;
		}
		else
		{
			coli::str_dump("NULL");
		}
		coli::str_dump("\n");
	}
}

void computation::set_schedule(std::string map_str)
{
	assert(map_str.length() > 0);
	assert(this->ctx != NULL);

	isl_map *map = isl_map_read_from_str(this->ctx,
			map_str.c_str());

	assert(map != NULL);

	this->set_schedule(map);
}

/**
  * Add a dimension to the map in the specified position.
  * A constraint that indicates that the dim is equal to a constant
  * is added.
  */
isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map,
		int dim_pos, int constant)
{
	assert(map != NULL);
	assert(dim_pos+1 >= 0);
	assert(dim_pos < (signed int) isl_map_dim(map, isl_dim_out));

	map = isl_map_insert_dims(map, isl_dim_out, dim_pos+1, 1);

	isl_space *sp = isl_map_get_space(map);
	isl_local_space *lsp =
		isl_local_space_from_space(isl_space_copy(sp));
	isl_constraint *cst = isl_constraint_alloc_equality(lsp);
	cst = isl_constraint_set_coefficient_si(cst,
			isl_dim_out, dim_pos+1, 1);
	cst = isl_constraint_set_constant_si(cst, (-1)*constant);
	map = isl_map_add_constraint(map, cst);

	return map;
}

void computation::after(computation &comp, int dim)
{
	isl_map *sched1 = comp.get_schedule();
	isl_map *sched2 = this->get_schedule();

	assert(sched1 != NULL);
	assert(sched2 != NULL);
	std::cout << "dim = " << dim << ",  isl_map_dim(sched1, isl_dim_out) = " << isl_map_dim(sched1, isl_dim_out) << std::endl;
	assert(dim < (signed int) isl_map_dim(sched1, isl_dim_out));
	assert(dim >= computation::root_dimension);
	assert(dim < (signed int) isl_map_dim(sched2, isl_dim_out));

	sched1 = isl_map_add_dim_and_eq_constraint(sched1, dim, 0);
	sched2 = isl_map_add_dim_and_eq_constraint(sched2, dim, 1);

	comp.set_schedule(sched1);
	this->set_schedule(sched2);
}

void computation::tile(int inDim0, int inDim1,
			int sizeX, int sizeY)
{
	// Check that the two dimensions are consecutive.
	// Tiling only applies on a consecutive band of loop dimensions.
	assert((inDim0 == inDim1+1) || (inDim1 == inDim0+1));
	assert(sizeX > 0);
	assert(sizeY > 0);
	assert(inDim0 >= 0);
	assert(inDim1 >= 0);
	assert(this->get_iteration_domain() != NULL);
	assert(inDim1 < isl_space_dim(isl_map_get_space(this->schedule),
							isl_dim_out));

	this->split(inDim0, sizeX);
	this->split(inDim1+1, sizeY);
	this->interchange(inDim0+1, inDim1+1);
}

/**
 * Modify the schedule of this computation so that the two dimensions
 * inDim0 and inDime1 are interchanged (swaped).
 */
void computation::interchange(int inDim0, int inDim1)
{
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->schedule),
							isl_dim_out));
	assert(inDim1 >= 0);
	assert(inDim1 < isl_space_dim(isl_map_get_space(this->schedule),
				          		isl_dim_out));

	IF_DEBUG2(str_dump("\nDebugging interchange()"));

	isl_map *schedule = this->get_schedule();

	IF_DEBUG2(coli::str_dump("\nOriginal schedule: ", isl_map_to_str(schedule)));

	int n_dims = isl_map_dim(schedule, isl_dim_out);
	std::string inDim0_str = isl_map_get_dim_name(schedule, isl_dim_out,
			inDim0);
	std::string inDim1_str = isl_map_get_dim_name(schedule, isl_dim_out,
			inDim1);

	std::vector<isl_id *> dimensions;

	std::string map = "{[";

	for (int i=0; i<n_dims; i++)
	{
		map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "] -> [";

	for (int i=0; i<n_dims; i++)
	{
		if ((i != inDim0) && (i != inDim1))
		{
			map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
			dimensions.push_back(isl_map_get_dim_id(schedule,
						isl_dim_out, i));
		}
		else if (i == inDim0)
		{
			map = map + inDim1_str;
			isl_id *id1 = isl_id_alloc(this->get_ctx(),
					inDim1_str.c_str(), NULL);
			dimensions.push_back(id1);
		}
		else if (i == inDim1)
		{
			map = map + inDim0_str;
			isl_id *id1 = isl_id_alloc(this->get_ctx(),
					inDim0_str.c_str(), NULL);
			dimensions.push_back(id1);
		}

		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "]}";

	IF_DEBUG2(coli::str_dump("\nTransformation map = ", map.c_str()));

	isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());
	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
	isl_id *id_range = isl_id_alloc(this->get_ctx(), "", NULL);
	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_out, id_range);
	schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

	IF_DEBUG2(coli::str_dump("\nSchedule after interchange: ", isl_map_to_str(schedule)));

	this->set_schedule(schedule);
}

/**
 * Modify the schedule of this computation so that it splits the
 * dimension inDim0 of the iteration space into two new dimensions.
 * The size of the inner dimension created is sizeX.
 */
void computation::split(int inDim0, int sizeX)
{
	assert(this->get_schedule() != NULL);
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()),
					isl_dim_out));
	assert(sizeX >= 1);

	IF_DEBUG2(str_dump("\nDebugging split()"));

	isl_map *schedule = this->get_schedule();

	std::string inDim0_str(isl_map_get_dim_name(schedule,
				isl_dim_out, inDim0)); 
	std::string outDim0_str = generate_new_variable_name();
	std::string outDim1_str = generate_new_variable_name();

	IF_DEBUG2(coli::str_dump("\nOriginal schedule: ", isl_map_to_str(schedule)));

	int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
	std::string map = "{[";

	std::vector<isl_id *> dimensions;

	for (int i=0; i<n_dims; i++)
	{
		map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "] -> [";

	for (int i=0; i<n_dims; i++)
	{
		if (i != inDim0)
		{
			map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
			dimensions.push_back(isl_map_get_dim_id(schedule,
						isl_dim_out, i));
		}
		else
		{
			map = map + outDim0_str + "," + outDim1_str;
			isl_id *id0 = isl_id_alloc(this->get_ctx(),
					outDim0_str.c_str(), NULL);
			isl_id *id1 = isl_id_alloc(this->get_ctx(),
					outDim1_str.c_str(), NULL);
			dimensions.push_back(id0);
			dimensions.push_back(id1);
		}

		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "] : " + outDim0_str + " = floor(" + inDim0_str + "/" +
		std::to_string(sizeX) + ") and " + outDim1_str + " = (" +
		inDim0_str + "%" + std::to_string(sizeX) + ")}";

	IF_DEBUG2(coli::str_dump("\nTransformation map = ", map.c_str()));

	isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

	for (int i=0; i< dimensions.size(); i++)
		transformation_map = isl_map_set_dim_id(transformation_map,
				isl_dim_out, i, isl_id_copy(dimensions[i]));

	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
	isl_id *id_range = isl_id_alloc(this->get_ctx(), " ", NULL);
	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_out, id_range);
	schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

	IF_DEBUG2(coli::str_dump("\nSchedule after splitting: ", isl_map_to_str(schedule)));

	this->set_schedule(schedule);
}

// Methods related to the coli::function class.

int coli::function::get_max_schedules_range_dim()
{
	int max_dim = 0;

	for (auto comp: this->get_computations())
	{
		int m = isl_map_dim(comp->get_schedule(), isl_dim_out);
		max_dim = std::max(max_dim, m);
	}

	return max_dim;
}

isl_map *isl_map_align_range_dims(isl_map *map, int max_dim)
{
	assert(map != NULL);
	int mdim = isl_map_dim(map, isl_dim_out);
	assert(max_dim >= mdim);

	IF_DEBUG2(coli::str_dump("\n\tDebugging isl_map_align_range_dims()."));
	IF_DEBUG2(coli::str_dump("\n\tInput map:", isl_map_to_str(map)));

	map = isl_map_add_dims(map, isl_dim_out, max_dim - mdim);

	for (int i=mdim; i<max_dim; i++)
	{
		isl_space *sp = isl_map_get_space(map);
		isl_local_space *lsp =
			isl_local_space_from_space(isl_space_copy(sp));
		isl_constraint *cst = isl_constraint_alloc_equality(lsp);
		cst = isl_constraint_set_coefficient_si(cst,
				isl_dim_out, i, 1);
		map = isl_map_add_constraint(map, cst);
	}

	IF_DEBUG2(coli::str_dump("\n\tAfter alignement, map:",
				isl_map_to_str(map)));

	return map;
}

void coli::function::align_schedules()
{
	IF_DEBUG2(coli::str_dump("\nDebugging align_schedules()."));

	int max_dim = this->get_max_schedules_range_dim();

	for (auto &comp: this->get_computations())
	{
		isl_map *sched = comp->get_schedule();
		assert((sched != NULL) && "Schedules should be set before calling align_schedules");
		sched = isl_map_align_range_dims(sched, max_dim);
		comp->set_schedule(sched);
	}
}

void coli::function::add_invariant(coli::invariant invar)
{
	invariants.push_back(invar);
}

void coli::function::add_computation(computation *cpt)
{
	assert(cpt != NULL);

	this->body.push_back(cpt);
}

void coli::function::dump()
{
	if (DEBUG)
	{
		std::cout << "Function \"" << this->name << "\"" << std::endl;
		std::cout << "Body " << std::endl;

		for (auto cpt : this->body)
		       cpt->dump();

		std::cout << "Buffers" << std::endl;

		for (auto buf : this->buffers_list)
		       std::cout << "Buffer name: " << buf.second->name()
				<< std::endl;

		std::cout << "Arguments" << std::endl;

		for (auto arg : this->get_arguments())
		       std::cout << "Argument name: " << arg.name
				<< std::endl;

		std::cout << std::endl;
	}
}

void coli::function::dump_iteration_domain()
{
	if (DEBUG)
	{
		for (auto cpt : this->body)
		       cpt->dump_iteration_domain();
	}
}

void coli::function::dump_schedule()
{
	if (DEBUG)
	{
		for (auto cpt : this->body)
		       cpt->dump_schedule();
	}
}

void coli::function::add_argument(coli::buffer buf)
{
	Halide::Argument buffer_arg(buf.get_name(), Halide::Argument::OutputBuffer,
			buf.get_type(), buf.get_n_dims());
	arguments.push_back(buffer_arg);
}


// Library related methods

void coli::library::align_schedules()
{
	for (auto fct: this->get_functions())
		fct->align_schedules();
}

void coli::library::add_vector_dimension(std::string stmt_name,
		int vec_dim)
{
	assert(vec_dim >= 0);
	assert(stmt_name.length() > 0);

	this->vector_dimensions.insert(
		std::pair<std::string,int>(stmt_name, vec_dim));
}

void coli::library::add_parallel_dimension(std::string stmt_name,
		int vec_dim)
{
	assert(vec_dim >= 0);
	assert(stmt_name.length() > 0);

	this->parallel_dimensions.insert(
		std::pair<std::string,int>(stmt_name, vec_dim));
}

void coli::library::dump_iteration_domain()
{
	if (DEBUG)
	{
		coli::str_dump("\nIteration domain:\n");
		for (const auto &fct : this->functions)
		{
			coli::str_dump("Function " + fct->get_name() + ":\n");
			fct->dump_iteration_domain();
		}
		coli::str_dump("\n");
	}
}

void coli::library::dump_schedule()
{
	if (DEBUG)
	{
		coli::str_dump("\nSchedule:\n");
		for (const auto &fct : this->functions)
		       fct->dump_schedule();

		std::cout << "Parallel dimensions: ";
		for (auto par_dim: parallel_dimensions)
			std::cout << par_dim.first << "(" << par_dim.second << ") ";

		std::cout << std::endl;

		std::cout << "Vector dimensions: ";
		for (auto vec_dim: vector_dimensions)
			std::cout << vec_dim.first << "(" << vec_dim.second << ") ";

		std::cout<< std::endl << std::endl << std::endl;
	}
}

void coli::library::dump()
{
	if (DEBUG)
	{
		std::cout << "Program \"" << this->name << "\"" << std::endl
			  <<
			std::endl;

		for (const auto &fct : this->functions)
		       fct->dump();

		std::cout << "Parallel dimensions: ";
		for (auto par_dim: parallel_dimensions)
			std::cout << par_dim.first << "(" << par_dim.second << ") ";

		std::cout << std::endl;

		std::cout << "Vector dimensions: ";
		for (auto vec_dim: vector_dimensions)
			std::cout << vec_dim.first << "(" << vec_dim.second << ") ";

		std::cout<< std::endl << std::endl;
	}
}

void coli::library::add_function(coli::function *fct)
{
	assert(fct != NULL);

	this->functions.push_back(fct);
}

isl_union_set * coli::library::get_time_processor_domain()
{
	isl_union_set *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
			&& (this->functions[0]->body.empty() == false))
	{
		space = isl_set_get_space(this->functions[0]->body[0]->get_iteration_domain());
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_set_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_set *cpt_iter_space = isl_set_copy(cpt->get_time_processor_domain());
			result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
		}

	return result;
}


isl_union_set * coli::library::get_iteration_domain()
{
	isl_union_set *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
			&& (this->functions[0]->body.empty() == false))
	{
		space = isl_set_get_space(this->functions[0]->body[0]->get_iteration_domain());
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_set_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_set *cpt_iter_space = isl_set_copy(cpt->get_iteration_domain());
			result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
		}

	return result;
}

std::vector<Halide::Internal::Stmt> library::get_halide_stmts()
{
	std::vector<Halide::Internal::Stmt> stmts;

	for (auto func: this->get_functions())
		stmts.push_back(func->get_halide_stmt());

	return stmts;
}

isl_union_map * coli::library::get_schedule()
{
	isl_union_map *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
		&& (this->functions[0]->body.empty() == false))
	{
		space = isl_map_get_space(this->functions[0]->body[0]->schedule);
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_map_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_map *m = isl_map_copy(cpt->schedule);
			result = isl_union_map_union(isl_union_map_from_map(m), result);
		}

	return result;
}

}
