#include "Halide.h"
#include "wrapper_test_02.h"

#include <coli/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 100
#define MM 100

int main(int, char**)
{
	buffer_t reference_buf = allocate_2D_buffer(NN, MM);
	init_2D_buffer_val(&reference_buf, NN, MM, 7);

	buffer_t output_buf = allocate_2D_buffer(NN, MM);

	init_2D_buffer_val(&output_buf, NN, MM, 13);
	assign_7_to_100x100_2D_array_with_tiling_parallelism(&output_buf);
	compare_2_2D_arrays("assign_7_to_100x100_2D_array_with_tiling_parallelism",
			   	   	   	output_buf.host, reference_buf.host, NN, MM);

	return 0;
}
