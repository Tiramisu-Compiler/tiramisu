#include <tiramisu/tiramisu.h>

#include "wrapper_test_183.h"

using namespace tiramisu;



void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-1), j("j", 1, N-2);
    tiramisu::var i1("i1"), j1("j1");
    tiramisu::var i2("i2"), j2("j2");
    tiramisu::input A("A", {i, j}, p_uint8);

    // Contains deps that disable tiling
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i, j)+ A(i-1, j+1));


    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    // Analysis
    perform_full_dependency_analysis() ;

    // start adding optimizations
    result.tile(i,j,4,4,i1,j1,i2,j2);

    // legality check of function : tiling should be illegal here
    prepare_schedules_for_legality_checks() ;

    // The check must fail here
    assert(check_legality_of_function() == false) ;

    // Code generation
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
