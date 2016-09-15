#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>

void generate_function_1(std::string name, int size, int val0, int val1)
{
	coli::global::set_default_coli_options();

	coli::function function0(name);
	coli::invariant N("N", coli::expr::make((int32_t) size), &function0);
	coli::expr *e1 = coli::expr::make(coli::type::op::add,
						coli::expr::make((uint8_t) val0),
						coli::expr::make((uint8_t) val1));
	coli::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", e1, true, &function0);

	coli::buffer buf0("buf0", 2, {size,size}, coli::type::primitive::uint8, NULL,
						coli::type::argument::output, &function0);
	S0.set_access("{S0[i,j]->buf0[i,j]}");
	S0.tile(0,1,2,2);
	S0.tag_parallel_dimension(0);

	function0.set_arguments({&buf0});
	function0.gen_isl_ast();
	function0.gen_halide_stmt();
	function0.gen_halide_obj("build/generated_fct_test_03.o");
}


int main(int argc, char **argv)
{
	generate_function_1("assign_7_to_1000x1000_2D_array_with_tiling_parallelism", 1000, 3, 4);

	return 0;
}
