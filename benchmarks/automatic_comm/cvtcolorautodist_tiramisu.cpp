#include <isl/set.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/utils.h>
#include "halide_image_io.h"
#include "wrapper_cvtcolorautodist.h"

using namespace tiramisu;
int main() {

    global::set_default_tiramisu_options();

    function cvtcolor_dist("cvtcolorautodist_tiramisu");

    cvtcolor_dist.add_context_constraints("[ROWS]->{:ROWS = "+std::to_string(_ROWS)+"}");

    constant CHANNELS("CHANNELS", expr(3), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant ROWS("ROWS", expr(_ROWS), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant COLS("COLS", expr(_COLS), p_int32, true, NULL, 0, &cvtcolor_dist);

    var i("i"), j("j"), c("c");
    var i1("i1"), i2("i2");

    computation input("[CHANNELS, ROWS, COLS]->{input[i, j, c]: 0<=c<CHANNELS and 0<=i<ROWS and 0<=j<COLS}", expr(), false, p_uint8, &cvtcolor_dist);
    expr rgb_expr(tiramisu::o_cast,
    tiramisu::p_uint8, (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                    input(i, j, 2)) *
                                                             tiramisu::expr((uint32_t)1868)) +
                                                            (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                                                                            input(i, j, 1)) *
                                                             tiramisu::expr((uint32_t)9617))) +
                                                           (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                                                                           input(i, j, 0)) *
                                                            tiramisu::expr((uint32_t)4899))) +
                                                          tiramisu::expr((uint32_t)8192)) /
                                                         tiramisu::expr((uint32_t)16384)));


    computation RGB2Gray_s0("[ROWS, COLS]->{RGB2Gray_s0[i, j]: 0<=i<ROWS and 0<=j<COLS}",
                            expr(o_cast, p_uint8, rgb_expr), true, p_uint8,
                            &cvtcolor_dist);

    input.split(i, _ROWS / NODES, i1, i2);
    input.tag_distribute_level(i1);
    input.drop_rank_iter(i1);

    RGB2Gray_s0.split(i, _ROWS / NODES, i1, i2);
    RGB2Gray_s0.tag_distribute_level(i1);
    RGB2Gray_s0.drop_rank_iter(i1);
    RGB2Gray_s0.tag_parallel_level(i2);

    buffer buff_input("buff_input", {_ROWS / NODES, COLS, 3}, p_uint8, a_input,
                      &cvtcolor_dist);
    buffer buff_RGB2Gray("buff_RGB2Gray", {_ROWS / NODES, COLS}, p_uint8, a_output,
                         &cvtcolor_dist);

    input.set_access("{input[i, j, c]->buff_input[i, j, c]}");
    RGB2Gray_s0.set_access("{RGB2Gray_s0[i, j]->buff_RGB2Gray[i, j]}");

    RGB2Gray_s0.gen_communication();

    cvtcolor_dist.codegen({&buff_input, &buff_RGB2Gray}, "build/generated_fct_cvtcolorautodist.o");

    return 0;
}
