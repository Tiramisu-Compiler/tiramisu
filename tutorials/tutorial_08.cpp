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

#include "wrapper_tutorial_08.h"

using namespace tiramisu;

/**
 * Update
 *
 * We want to represent the following program
 *
 * for i = 1 to N
 *      C[i] = 10
 *
 * for i = 1 to N
 *      C[i] += 10
 *
 * for i to N
 *      out[i] = C[i] + 1
 *
 * In order to implement this program, we create the following
 * computations
 *
 * {C[i]: 0<=i<N}:                   10
 * {C[i]: 0<=i<N and 0<=j<N}:        C[i] + 10
 * {out[i]: 0<=i<N}:                 C[i] + 1
 *
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------


    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::var i = tiramisu::var("i");
    tiramisu::var j = tiramisu::var("j");

    tiramisu::computation C("[N]->{C[i]: 0<=i<N}", tiramisu::expr((uint8_t) 10), true, p_uint8, &function0);
    tiramisu::computation *C2 = C.add_computations("[N]->{C[i]: 0<=i<N}", C(i) + tiramisu::expr((uint8_t) 10), true, p_uint8, &function0);
    tiramisu::computation out("[N]->{out[i]: 0<=i<N}", C(i) + tiramisu::expr((uint8_t) 1), true, p_uint8, &function0);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    C2->after(C, computation::root_dimension);
    out.after((*C2), computation::root_dimension);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------


    tiramisu::buffer C_buff("C_buff", 1, {size}, tiramisu::p_uint8, NULL, a_temporary, &function0);
    tiramisu::buffer out_buff("out_buff", 1, {size}, tiramisu::p_uint8, NULL, a_output, &function0);
    C.set_access("[N]->{C[i]->C_buff[i]}");
    C2->set_access("[N]->{C[i]->C_buff[i]}");
    out.set_access("[N]->{out[i]->out_buff[i]}");


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------


    function0.set_arguments({&out_buff});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_tutorial_08.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
