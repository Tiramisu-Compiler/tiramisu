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

#include "wrapper_test_54.h"

using namespace tiramisu;

/**
 * Test predicates with reduction.  We want to implement the following example
 * in Tiramisu
 *
 * // Initialization
 * for (ry=0; ry<7; ry++)
 *      for (rx=0; rx<7; rx++)
 *              S0[ry,rx] = 5;
 *
 * // Reduction with predicate
 * for (ry=0; ry<7; ry++)
 *      for (rx=0; rx<7; rx++)
 *              if ((r.x - 3)*(r.x - 3) + (r.y - 3)*(r.y - 3) <= 10)
 *                  S0[ry,rx] = S0[ry,rx] * 2;
 *
 * According to the rules stated in "how_to_translate_halide_commands_tiramisu.txt", the above
 * example is an update/reduction. The same rules apply for both (updates and reductions).
 * S0[ry,rx] should be expanded by one dimension.
 *
 * The Tiramisu code that we should write would be as follows
 * First, since the predicate is not affine we define a constant t that holds
 * the value of the predicate and use that constant as a parameter to the computation
 * S0[ry][rx][1]. We also add (t=1) to the iteration domain of S0[ry][rx][1]. This would
 * mean that S0[ry][rx][1] is defined only when t==1 i.e., when
 * ((r.x - 3)*(r.x - 3) + (r.y - 3)*(r.y - 3) <= 10) is true.
 * If the predicate was affine we could just add it to the iteration
 * domain of the computation directly without wrapping in t. We are wrapping it
 * in t because it is not affine.
 *
 * {S0[ry][rx][0]: 0<=rx<7 and 0<=ry<7}: 0
 * t = ((r.x - 3)*(r.x - 3) + (r.y - 3)*(r.y - 3) <= 10)
 * [t]->{S0[ry][rx][1]: 0<=rx<7 and 0<=ry<7 and (t=1)}: S0[ry][rx][0] * 2
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();
    

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);

    tiramisu::var rx("rx");
    tiramisu::var ry("ry");

    // Computation to initialize S0
    tiramisu::computation S0("{S0[ry,rx,0]: 0<=ry<7 and 0<=rx<7}", tiramisu::expr((uint8_t) 5), true, p_uint8, &function0);

    tiramisu::expr predicate_0 = (((rx - tiramisu::expr(3)) * (rx - tiramisu::expr(3)) +
                                   (ry - tiramisu::expr(3)) * (ry - tiramisu::expr(3))) <= tiramisu::expr(10));
    S0.add_definitions("[t]->{S0[ry,rx,1]: 0<=ry<7 and 0<=rx<7}", S0(ry,rx,0) * tiramisu::expr((uint8_t) 2), true, p_uint8, &function0);
    S0.get_update(1).add_predicate(predicate_0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S0.get_update(1).after(S0,computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf0("buf0", {7,7}, tiramisu::p_uint8, a_output, &function0);
    S0.set_access("{S0[ry,rx,k]->buf0[ry,rx]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 7);

    return 0;
}
