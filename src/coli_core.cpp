#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
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
bool context::auto_data_mapping;

// Used for the generation of new variable names.
int id_counter = 0;


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

void computation::dump_iteration_space_IR()
{
	if (DEBUG)
	{
		isl_set_dump(this->iter_space);
	}
}

void library::dump_halide_IR()
{
	for (auto func: this->get_functions())
		coli::halide_IR_dump(func->get_halide_stmt());
}


void library::dump_time_processor_IR()
{
	// Create time space IR

	if (DEBUG)
	{
		coli::str_dump("\n\nTime Space IR:\n");

		for (auto func: this->get_functions())
		{
			coli::str_dump("Function " + func->get_name() + ":\n");
			for (auto comp: func->get_computations())
				isl_set_dump(
					comp->get_time_processor_representation());
		}

		coli::str_dump("\n\n");
	}
}


isl_union_map *library::get_time_processor_identity_relation()
{
	isl_union_map *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
		&& (this->functions[0]->body.empty() == false))
	{
		space = isl_map_get_space(this->functions[0]->body[0]->get_time_processor_identity_relation());
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_map_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_map *m = isl_map_copy(cpt->get_time_processor_identity_relation());
			result = isl_union_map_union(isl_union_map_from_map(m), result);
		}

	return result;
}

void library::gen_time_processor_IR()
{
	for (auto func: this->get_functions())
	{
		for (auto comp: func->get_computations())
			comp->gen_time_processor_IR();
	}
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
		isl_set_dump(this->iter_space);
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
	assert(this->iter_space != NULL);
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

	coli::parser::map map(isl_map_to_str(this->schedule));

	std::iter_swap(map.range.dimensions.begin()+inDim0,
			map.range.dimensions.begin()+inDim1);

	this->schedule = isl_map_read_from_str(this->ctx, map.get_str().c_str());
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

	IF_DEBUG2(coli::str_dump("\nOriginal schedule: ", isl_map_to_str(schedule)));

	int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
	std::string map = "{[";

	for (int i=0; i<n_dims; i++)
	{
		map = map + "i" + std::to_string(i);
		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "] -> [";

	for (int i=0; i<n_dims; i++)
	{
		if (i != inDim0)
			map = map + "i" + std::to_string(i);
		else
			map = map + "c0,c1";

		if (i != n_dims-1)
			map = map + ",";
	}

	map = map + "] : c0 = floor(i" + std::to_string(inDim0) + "/" +
		std::to_string(sizeX) + ") and c1 = (i" +
		std::to_string(inDim0) + "%" + std::to_string(sizeX) +
		")}";

	std::cout << "\nmap = " << map << std::endl;
	isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());
	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
	isl_id *id_range = isl_id_alloc(this->get_ctx(), "", NULL);
	transformation_map = isl_map_set_tuple_id(transformation_map,
			isl_dim_out, id_range);
	schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

	IF_DEBUG2(coli::str_dump("\nSchedule after splitting: ", isl_map_to_str(schedule)));

	this->set_schedule(schedule);
}

// Methods related to the coli::function class.

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

void coli::function::dump_iteration_space_IR()
{
	if (DEBUG)
	{
		for (auto cpt : this->body)
		       cpt->dump_iteration_space_IR();
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

void coli::library::dump_iteration_space_IR()
{
	if (DEBUG)
	{
		coli::str_dump("\nIteration Space IR:\n");
		for (const auto &fct : this->functions)
		{
			coli::str_dump("Function " + fct->get_name() + ":\n");
			fct->dump_iteration_space_IR();
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

isl_union_set * coli::library::get_time_processor_representation()
{
	isl_union_set *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
			&& (this->functions[0]->body.empty() == false))
	{
		space = isl_set_get_space(this->functions[0]->body[0]->iter_space);
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_set_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_set *cpt_iter_space = isl_set_copy(cpt->get_time_processor_representation());
			result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
		}

	return result;
}


isl_union_set * coli::library::get_iteration_spaces()
{
	isl_union_set *result = NULL;
	isl_space *space = NULL;

	if ((this->functions.empty() == false)
			&& (this->functions[0]->body.empty() == false))
	{
		space = isl_set_get_space(this->functions[0]->body[0]->iter_space);
	}
	else
		return NULL;

	assert(space != NULL);
	result = isl_union_set_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_set *cpt_iter_space = isl_set_copy(cpt->iter_space);
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

isl_union_map * coli::library::get_schedule_map()
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


// Halide IR related methods

void halide_IR_dump(Halide::Internal::Stmt s)
{
	if (DEBUG)
	{
		coli::str_dump("\n\n");
		coli::str_dump("\nGenerated Halide Low Level IR:\n");
		std::cout << s;
		coli::str_dump("\n\n\n\n");
	}
}

}
