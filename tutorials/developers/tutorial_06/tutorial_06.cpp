/**
 * Update
 *
 * We want to represent the following program
 *
 * for i = 0 to N
 *      C[i] = 10
 *
 * for i = 0 to N
 *      C[i] = C[i] + 10
 *
 * for i = 0 to N
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

#include <tiramisu/tiramisu.h>
#include "wrapper_tutorial_06.h"

using namespace tiramisu;

void generate_function(std::string name, int size, int val0)
{
    // Set default tiramisu options.
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function function0(name);
    constant N("N", expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    var i = var("i");
    var j = var("j");

    computation C_0("[N]->{C_0[0,i]: 0<=i<N}", expr((uint8_t) 10), true, p_uint8, &function0);
    computation C_1("[N]->{C_1[1,i]: 0<=i<N}", C_0(0, i) + expr((uint8_t) 10), true, p_uint8, &function0);
    computation out("[N]->{out[i]: 0<=i<N}", C_1(1, i) + expr((uint8_t) 1), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    C_1.after(C_0, computation::root);
    out.after(C_1, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer C_buff("C_buff", {size}, p_uint8, a_temporary, &function0);
    buffer out_buff("out_buff", {size}, p_uint8, a_output, &function0);
    C_0.set_access("{C_0[j,i]->C_buff[i]}");
    C_1.set_access("{C_1[j,i]->C_buff[i]}");
    out.set_access("{out[i]->out_buff[i]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.codegen({&out_buff}, "build/generated_fct_developers_tutorial_06.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
