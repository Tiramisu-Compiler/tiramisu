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
 *      C[i] = C[i] + 10
 *
 * for i to N
 *      out[i] = C[i] + 1
 *
 * In order to implement this program, we create the following
 * computations
 *
 * {C[0,i]: 0<=i<N}:                 10
 * {C[1,i]: 0<=i<N}:                 C[0,i] + 10
 * {out[i]: 0<=i<N}:                 C[1,i] + 1
 *
 */

void generate_function(std::string name, int size, int val0)
{
    global::set_default_tiramisu_options();


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------


    function function0(name);
    constant N("N", expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    var i = var("i");
    var j = var("j");

    computation C("[N]->{C[0,i]: 0<=i<N}", expr((uint8_t) 10), true, p_uint8, &function0);
    C.add_definitions("[N]->{C[1,i]: 0<=i<N}", C(0, i) + expr((uint8_t) 10), true, p_uint8, &function0);
    computation out("[N]->{out[i]: 0<=i<N}", C(1, i) + expr((uint8_t) 1), true, p_uint8, &function0);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    C.get_update(1).after(C, computation::root);
    out.after(C.get_update(1), computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------


    buffer C_buff("C_buff", {size}, p_uint8, a_temporary, &function0);
    buffer out_buff("out_buff", {size}, p_uint8, a_output, &function0);
    // Important: note that the access relations of the two computation C and C2 are identical.
    // The Tiramisu code generator assumes that the access relations of computations that have the same
    // name are identical.  In this case, the two relations are equal to "{C[j,i]->C_buff[i]}".
    C.set_access("{C[j,i]->C_buff[i]}");
    C.get_update(1).set_access("{C[j,i]->C_buff[i]}");
    out.set_access("{out[i]->out_buff[i]}");


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------


    function0.set_arguments({&out_buff});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_developers_tutorial_08.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
