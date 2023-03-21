#include <tiramisu/tiramisu.h>

#include "wrapper_test_191.h"

using namespace tiramisu;

/**
 * Test shifting, the case of impossible fusion.
 * The buffers values doesn't matter,
 * We essentially check validity of legality assertions.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);
    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-2), j("j", 1, N-2);
    tiramisu::var i1("i1"), j1("j1");
    tiramisu::var i2("i2"), j2("j2");

    tiramisu::input A("A", {i, j}, tiramisu::p_uint8);


    //Computations
    tiramisu::computation B_out("B_out", {i,j}, A(i-1,j) + A(i+1,j) + A(i,j-1) +A(i,j+1)  +  A(i,3)  );
    
    /* A(i,0) would not be within the execution range of B_out, so the fusion with shifting would be possible
        not the case of A(i,3)
    */

    tiramisu::computation A_out("A_out", {i,j}, B_out(i,j) + 5);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    B_out.then(A_out,tiramisu::computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    tiramisu::buffer b_A("b_A", {N,N}, tiramisu::p_uint8, tiramisu::a_input);    

    //Store inputs
    A.store_in(&b_A);

    //Store computations
    A_out.store_in(&b_A, {i,j});
    B_out.store_in(&b_A, {i,j});



    tiramisu::perform_full_dependency_analysis();

    //tests

    tiramisu::function * fct = tiramisu::global::get_implicit_function();

    assert(tiramisu::check_legality_of_function() == true);
    assert(fct->check_partial_legality_in_function({&A_out,&B_out}) == true);

    

    B_out.then(A_out,j);
    //fuse every thing and solve


    tiramisu::prepare_schedules_for_legality_checks(true);

    assert(fct->check_partial_legality_in_function({&A_out,&B_out}) == false);
   

    //shift A_out, impossible A(i,0) 
    auto shiftings = fct->correcting_loop_fusion_with_shifting({&B_out},A_out,{i,j});
    assert(shiftings.size() == 0);

    //2nd try remove j
    
    auto shiftings2 = fct->correcting_loop_fusion_with_shifting({&B_out},A_out,{i});
    assert(shiftings2.size() > 0);

    for(auto const& tup:shiftings2)
    {
        A_out.shift(
            std::get<0>(tup),
            std::get<1>(tup)
            );
    }

    assert(tiramisu::check_legality_of_function() == true);
    assert(fct->check_partial_legality_in_function({&A_out,&B_out}) == true);
    

    // Code generation
    tiramisu::codegen({&b_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
