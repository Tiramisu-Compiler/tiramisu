#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/ir.h>

#include <string>


std::string generate_new_variable_name();

/**
  * Methods for the computation class.
  */

void Computation::dump_ISIR()
{
	if (DEBUG)
	{
		isl_set_dump(this->iter_space);
	}
}

void Computation::dump_schedule()
{
	if (DEBUG)
	{
		isl_map_dump(this->schedule);
	}
}

void Computation::dump()
{
	if (DEBUG)
	{
		std::cout << "Computation \"" << this->name << "\"" << std::endl;
		isl_set_dump(this->iter_space);
		std::cout << "Schedule " << std::endl;
		isl_map_dump(this->schedule);
		coli_str_dump("Halide statement:\n");
		Halide::Internal::IRPrinter pr(std::cout);
	    	pr.print(this->stmt);
		coli_str_dump("\n");

	}
}

void Computation::Schedule(std::string map_str)
{
	isl_map *map = isl_map_read_from_str(this->ctx,
			map_str.c_str());

	this->schedule = map;
}

void Computation::Tile(int inDim0, int inDim1,
			int sizeX, int sizeY)
{
	assert((inDim0 == inDim1+1) || (inDim1 == inDim0+1));

	this->Split(inDim0, sizeX);
	this->Split(inDim1+1, sizeY);
	this->Interchange(inDim0+1, inDim1+1);
}

/**
 * Modify the schedule of this computation so that the two dimensions
 * inDim0 and inDime1 are interchanged (swaped).
 */
void Computation::Interchange(int inDim0, int inDim1)
{
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->schedule),
							isl_dim_out));
	assert(inDim1 >= 0);
	assert(inDim1 < isl_space_dim(isl_map_get_space(this->schedule),
				          		isl_dim_out));

	isl_map_tokens map(isl_map_to_str(this->schedule));

	std::iter_swap(map.range.dimensions.begin()+inDim0,
			map.range.dimensions.begin()+inDim1);

	this->schedule = isl_map_read_from_str(this->ctx, map.get_str().c_str());
}

/**
 * Modify the schedule of this computation so that it splits the
 * dimension inDim0 of the iteration space into two new dimensions.
 * The size of the inner dimension created is sizeX.
 */
void Computation::Split(int inDim0, int sizeX)
{
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->schedule),
				          isl_dim_out));
	assert(sizeX >= 1);


	isl_map_tokens map(isl_map_to_str(this->schedule));

	std::string inDim0_str = map.range.dimensions.at(inDim0);
	std::string outDim0 = generate_new_variable_name(); 
	std::string outDim1 = generate_new_variable_name();
	std::string outDimensions = outDim0 + "," + outDim1;

	map.range.replace(inDim0_str, outDim0, outDim1);

	// Add the relations
	std::string relation1 = outDim0 + "=floor(" + inDim0_str + "/" +
		std::to_string(sizeX) + ") ";
	std::string relation2 = outDim1 + "=" + inDim0_str + "%" +
	 	std::to_string(sizeX);

	map.constraints.add(relation1);
	map.constraints.add(relation2);

	this->schedule = isl_map_read_from_str(this->ctx, map.get_str().c_str());
}
