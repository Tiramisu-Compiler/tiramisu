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

#include "wrapper_test_84.h"

using namespace tiramisu;

/**
 * Test that predecessor works.
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);

    var i("i");

    constant N("N", expr((int32_t) SIZE), p_int32, true, nullptr, computation::root_dimension, &function0);
    computation S0("[N] -> {S0[i] : 0 <= i < N}", expr((int32_t) 5), true, p_int32, &function0);
    computation S1("[N] -> {S1[i] : 0 <= i < N}", S0(i) + expr((int32_t) 1), true, p_int32, &function0);
    computation S2("[N] -> {S2[i] : 0 <= i < N}", S0(i) + S1(i), true, p_int32, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    assert(!S2.get_predecessor() && "S2 should not have a predecessor yet.");
    S2.after(S0, i);
    assert(S2.get_predecessor() == &S0 && "S2's predecessor should be S0.");
    S1.between(*(S2.get_predecessor()), i, S2, i);
    assert(S2.get_predecessor() == &S1 && "S2's predecessor should be S1.");
    assert(S1.get_predecessor() == &S0 && "S1's predecessor should be S0.");
    assert(!S0.get_predecessor() && "S0 should not have a predecessor.");

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    std::vector<expr> size = {expr((int32_t) SIZE)};
    buffer B0("B0", size, p_int32, a_input, &function0);
    buffer B1("B1", size, p_int32, a_temporary, &function0);
    buffer B2("B2", size, p_int32, a_output, &function0);

    S0.set_access("{S0[i] -> B0[i]}");
    S1.set_access("{S1[i] -> B1[i]}");
    S2.set_access("{S2[i] -> B2[i]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&B0, &B2});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
