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

void generate_function_1(std::string name, int size)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");
    tiramisu::computation Si("{Si[i,j]}", i + j / expr(2), false, p_int32, &function0);
    assert(!Si.is_inline_computation());
    Si.set_inline();
    assert(Si.is_inline_computation());
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", Si(j, i), true, p_int32,
                             &function0);
    assert(!S0.is_inline_computation());

    tiramisu::buffer buf0("buf0", {size, size}, tiramisu::p_int32, a_output, &function0);

    S0.set_access("{S0[i,j]->buf0[i,j]}");
    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.tag_parallel_level(i0);

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_78.o");
}


int main(int argc, char **argv)
{
    generate_function_1("test_inlining_2", 10);

    return 0;
}
