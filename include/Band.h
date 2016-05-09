#ifndef _H_BAND_
#define _H_BAND_

#include <iostream>
#include <vector>

#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>

/*
   A class that represents loop bands (a band is a set of consecutive
   loop dimensions).
   If a band is permutable then that band is tilable it.
   */
class Band
{
public:	
	void set_n_dims(int value);
	int get_n_dims();
	void add_dim(std::string name);
	std::string get_dim(int pos);
	void set_tile_size(std::string dim, int size);
	int get_tile_size(std::string dim);
	bool get_tile_band();
	void set_tile_band();

private:
	std::vector<std::string> dim_names;
	int n_dims;
	
	/* if set to true, the band is tiled.
	   tile_sizes should be set in this case.
	 */
	bool tile_band;
	
	std::map<std::string, int> tile_sizes;
};

#endif
