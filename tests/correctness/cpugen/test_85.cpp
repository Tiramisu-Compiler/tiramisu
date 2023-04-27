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

#include "wrapper_test_85.h"

using namespace tiramisu;

/**
 * Test variable size input
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    computation Sdims("{Sdims[i] : 0 <= i < 3}", expr(), false, p_int32, &function0);

    constant N("N", Sdims(0), p_int32, true, nullptr, computation::root_dimension, &function0);
    constant D("D", Sdims(1), p_int32, true, nullptr, computation::root_dimension, &function0);
    constant M("M", Sdims(2), p_int32, true, nullptr, computation::root_dimension, &function0);

    var i("i"), j("j"), k("k");

    computation SM1("[N, D] -> {SM1[i, j] : 0 <= i < N and 0 <= j < D}", expr(), false,
                    p_int32, &function0);

    computation SM2("[D, M] -> {SM2[i, j] : 0 <= i < D and 0 <= j < M}", expr(), false,
                    p_int32, &function0);

    computation SMR("[N, D, M] -> {SMR[i, j, -1]: 0 <= i < N and 0 <= j < M}",
                    expr((int32_t) 0), true, p_int32, &function0);
    SMR.add_definitions("[N, D, M] -> {SMR[i, j, k]: 0 <= i < N and 0 <= j < M and 0 <= k < D}",
                        SMR(i, j, k-1) + SM1(i, k) * SM2(k, j), true, p_int32, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    SMR.before(SMR.get_update(1), k);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer Bdims("Bdims", {expr(3)}, p_int32, a_input, &function0);
    buffer BM1("BM1", {N, D}, p_int32, a_input, &function0);
    buffer BM2("BM2", {D, M}, p_int32, a_input, &function0);
    buffer BMR("BMR", {N, M}, p_int32, a_output, &function0);

    Sdims.set_access("{Sdims[i] -> Bdims[i]}");
    SM1.set_access("{SM1[i, j] -> BM1[i, j]}");
    SM2.set_access("{SM2[i, j] -> BM2[i, j]}");
    SMR.set_access("{SMR[i, j, k] -> BMR[i, j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&Bdims, &BM1, &BM2, &BMR});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
