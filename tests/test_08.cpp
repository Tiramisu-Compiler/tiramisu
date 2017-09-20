#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

using namespace tiramisu;

void generate_function_1(std::string name, int size, int val0, int val1)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::expr e_N = tiramisu::expr((int32_t) size);
    tiramisu::constant N("N", e_N, p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");
    tiramisu::expr e1 = tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8,
                                       tiramisu::expr(tiramisu::o_floor, tiramisu::expr((float) val0) / tiramisu::expr((float) val1)));
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", e1, true, p_uint8, &function0);

    tiramisu::buffer buf0("buf0", 2, {size, size}, tiramisu::p_uint8, NULL,
                          a_output, &function0);
    S0.set_access("{S0[i,j]->buf0[i,j]}");
    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.tag_parallel_level(i0);

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.dump_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_08.o");
}


int main(int argc, char **argv)
{
    generate_function_1("test_floor_operator", 10, 37, 3);

    return 0;
}
