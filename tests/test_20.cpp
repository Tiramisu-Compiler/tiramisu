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

#include "wrapper_test_20.h"

using namespace tiramisu;

/**
 * Test non-affine accesses.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M("M", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", {size, size}, tiramisu::p_uint8, a_output, &function0);

    tiramisu::expr e0 = tiramisu::expr((uint8_t) val0);
    tiramisu::computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<N}", e0, true, p_uint8, &function0);

    tiramisu::var i = tiramisu::var("i");
    tiramisu::var j = tiramisu::var("j");
    tiramisu::expr e1 = tiramisu::expr(S0(i, j)) +
                        tiramisu::expr(S0(tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32,
                                          tiramisu::expr(tiramisu::o_abs, tiramisu::expr(9))), 9));
    tiramisu::computation S1("[N,M]->{S1[i,j]: 0<=i<M and 0<=j<M}", e1, true, p_uint8, &function0);

    tiramisu::expr e2 = tiramisu::expr(S1(i, j)) +
                        tiramisu::expr(S0(tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32,
                                          tiramisu::expr(tiramisu::o_abs, tiramisu::expr(2)) +
                                          tiramisu::expr(tiramisu::o_abs, tiramisu::expr(7))),
                                          9));
    tiramisu::computation S2("[N,M]->{S2[i,j]: 0<=i<M and 0<=j<M}", e2, true, p_uint8, &function0);


    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");
    S1.set_access("[N,M]->{S1[i,j]->buf0[i,j]: 0<=i<M and 0<=j<M}");
    S2.set_access("[N,M]->{S2[i,j]->buf0[i,j]: 0<=i<M and 0<=j<M}");

    S1.after(S0, computation::root);
    S2.after(S1, computation::root);

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("test_" + std::string(TEST_NAME_STR), SIZE0, 1);

    return 0;
}
