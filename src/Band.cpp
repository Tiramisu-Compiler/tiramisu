#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>

#include <Band>

void Band::set_n_dims(int value)
{
	n_dims = value;
}

int Band::get_n_dims()
{
	return n_dims;
}

void Band::add_dim(std::string name)
{
	dim_names.push_back(name);
}

std::string Band::get_dim(int pos)
{
	return dim_names[pos];
}

void set_tile_size(std::string dim, int size)
{
	tile_sizes[dim] = size;
}

int get_tile_size(std::string dim)
{
	return tile_sizes[dim];
}
