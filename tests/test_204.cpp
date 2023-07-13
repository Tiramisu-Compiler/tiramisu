#include <tiramisu/tiramisu.h>

#include "wrapper_test_204.h"

using namespace tiramisu;

/**
 * Test single loop tiling aka loop splitting
 */

void generate_function(std::string name, int size0, int size1, int val0)
{
    tiramisu::init(name);

    // Algorithm
    var i("i", 0, size0), i1("i1", 0, size1), j("j", 0, size1);
    
    tiramisu::input A("A", {i1, j}, p_uint8);
    computation comp00({i, i1}, A(0, i1));
    computation comp01({i, i1, j}, A(j, i1));
	// computation comp00("{comp00[i, i1]: 0<=i<3 and 0<=i1<32}",  A(0, i1), true, p_uint8, global::get_implicit_function());
	// computation comp01("{comp01[i, i1, j]:  0<=i<3 and 0<=i1<32 and 0<=j<32}", A(j, i1), true, p_uint8, global::get_implicit_function());
	comp00.then(comp01,1);
	buffer buf00("buf00", {size1}, p_uint8, a_output);
	buffer buf01("buf01", {size1, size1}, p_uint8, a_output);
    buffer bufi("bufi", {size1, size1}, p_uint8, a_input);
    A.store_in(&bufi);
	comp00.store_in(&buf00, {i1});
	comp01.store_in(&buf01, {i1, j});

    // legality check of function
    prepare_schedules_for_legality_checks();
    // analysis
    performe_full_dependency_analysis();     

    // Tile both computations at level 1
    comp00.tile(1, 4);
    comp01.tile(1, 4);

    // Tile computations 1 at level 2 (now 3rd iterator since tiling adds a new loop)
    comp01.tile(3, 4);

    assert(check_legality_of_function() == true);

    // Code generation
    tiramisu::codegen({&bufi, &buf00, &buf01}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE0, SIZE1, 0);

    return 0;
}
