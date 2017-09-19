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

    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr(), false, p_uint8,
            &function0);

    tiramisu::var i("i"), j("j");
    tiramisu::expr e_blur = (S0(i - 1, j - 1) + S0(i, j - 1) + S0(i + 1, j - 1) +
                             S0(i - 1, j    ) + S0(i, j    ) + S0(i + 1, j    ) +
                             S0(i - 1, j + 1) + S0(i, j + 1) + S0(i + 1, j + 1)) / ((uint8_t) 9);


    tiramisu::computation S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", e_blur, true, p_uint8,
            &function0);

    tiramisu::buffer buf0("buf0", 2, {size, size}, tiramisu::p_uint8, NULL, a_input, &function0);
    tiramisu::buffer buf1("buf1", 2, {size, size}, tiramisu::p_uint8, NULL, a_output, &function0);

    S0.set_access("[N]->{S0[i,j]->buf0[min(max(i, 0), N - 1), min(max(j, 0), N - 1)]}");
    S1.set_access("{S1[i,j]->buf1[i,j]}");

    S1.tile(i, j, 2, 2);
    S1.tag_parallel_level(0);

    function0.set_arguments({&buf0, &buf1});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_56.o");
}

int main(int argc, char **argv)
{
    generate_function_1("blur_100x100_2D_array_with_tiling_parallelism", 100);

    return 0;
}
