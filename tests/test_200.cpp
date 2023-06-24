#include <tiramisu/tiramisu.h>

#include "wrapper_test_200.h"

using namespace tiramisu;

/**
 * Test auto skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-1), j("j", 1, N-1);


    tiramisu::var i1("i1"), j1("j1");
    tiramisu::var i2("i2"), j2("j2");

    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i-1, j)+ A(i-1, j));

    tiramisu::computation addition({i,j}, A(i, j) + 1);

    tiramisu::computation addition2({i,j}, A(i, j) + 2);


    result.tile(i, j, 4, 4, i1, i2, j1, j2);
    addition.tile(i, j, 4, 4, i1, i2, j1, j2);
    addition2.tile(i, j, 4, 4, i1, i2, j1, j2);

    // the order must be defined
    // result & addition must be fused in the innermost loop level
    result.then(addition, j2).then(addition2, j2);

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);
    addition.store_in(&buff_A);
    addition2.store_in(&buff_A);

    // required method before clustering
    prepare_schedules_for_legality_checks();
    cluster_statements_automatically();

    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);
    return 0;
}
