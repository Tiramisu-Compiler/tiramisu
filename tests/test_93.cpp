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

#include "wrapper_test_93.h"

using namespace tiramisu;

/**
 * TODO: describe test
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();


    tiramisu::var i("i"), j("j"), k("k"), l("l"), m("m"), n("n"), o("o"), p("p");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::computation S4("[] -> {S4[i] : (1 <= i <= 3)}", expr(), false, p_int32, &function0);
    tiramisu::computation S6("[] -> {S6[i] : (1 <= i <= 2)}", expr(), false, p_int32, &function0);
    tiramisu::computation S8("[] -> {S8[i] : (1 <= i <= 2)}", expr(), false, p_int32, &function0);
    tiramisu::computation S0("[] -> {S0[i] : (i = 1)}", expr(), false, p_int32, &function0);
    tiramisu::computation S1("[] -> {S1[i] : (i = 1)}", expr(), false, p_int32, &function0);
    tiramisu::computation S2("[] -> {S2[i] : (i = 1)}", expr(), false, p_int32, &function0);
    tiramisu::computation S3("[] -> {S3[i] : (i = 1)}", expr(), false, p_int32, &function0);
    tiramisu::constant C0("C0", S4(expr((int32_t) 1)), p_int32, &function0);
    tiramisu::constant C1("C1", S4(expr((int32_t) 2)), p_int32, &function0);
    tiramisu::constant C2("C2", S4(expr((int32_t) 3)), p_int32, &function0);
    tiramisu::computation S5("[C1, C2, C0] -> {S5[i, j, k] : ((1 <= i <= C0) and (1 <= j <= C1) and (1 <= k <= C2))}", expr(), false, p_float64, &function0);
    tiramisu::constant C3("C3", S6(expr((int32_t) 1)), p_int32, &function0);
    tiramisu::constant C4("C4", S6(expr((int32_t) 2)), p_int32, &function0);
    tiramisu::computation S7("[C3, C4] -> {S7[i, j] : ((1 <= i <= C3) and (1 <= j <= C4))}", expr(), false, p_float64, &function0);
    tiramisu::constant C5("C5", S8(expr((int32_t) 1)), p_int32, &function0);
    tiramisu::constant C6("C6", S8(expr((int32_t) 2)), p_int32, &function0);
    tiramisu::computation S9("[C6, C5] -> {S9[i, j] : ((1 <= i <= C5) and (1 <= j <= C6))}", expr(), false, p_float64, &function0);
    tiramisu::constant C7("C7", S0(1), p_int32, &function0);
    tiramisu::constant C8("C8", S1(1), p_int32, &function0);
    tiramisu::computation S10("[C8, C7] -> {S10[i, j] : ((1 <= i <= C7) and (1 <= j <= C8))}", expr(), false, p_float64, &function0);
    tiramisu::computation S11("[] -> {S11[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S13("[] -> {S13[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S15("[] -> {S15[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S17("[] -> {S17[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S19("[] -> {S19[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S22("[] -> {S22[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S24("[] -> {S24[i] : ((1 <= i <= 1))}", expr(), false, p_float64, &function0);
    tiramisu::computation S12("[C8, C7] -> {S12[i, j] : ((1 <= i <= C7) and (1 <= j <= C8))}", expr((double) 0.0), true, p_float64, &function0);
    tiramisu::constant C9("C9", S2(1), p_int32, &function0);
    tiramisu::computation S14("[C8, C7, C9] -> {S14[i, j, k] : (((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9))}", expr((double) 0.0), true, p_float64, &function0);
    tiramisu::constant C10("C10", S3(1), p_int32, &function0);
    tiramisu::computation S16("[C8, C10, C7, C9] -> {S16[i, j, k, l] : ((((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9)) and (1 <= l <= C10))}", S5(i, k, l), true, p_float64, &function0);
    tiramisu::computation S18("[C8, C10, C7, C9] -> {S18[i, j, k, l] : ((((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9)) and (1 <= l <= C10))}", S9(l, j), true, p_float64, &function0);
    tiramisu::computation S20("[C8, C10, C7, C9] -> {S20[i, j, k, l] : ((((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9)) and (1 <= l <= C10))}", (S15(1) * S17(1)), true, p_float64, &function0);
    tiramisu::computation S21("[C8, C10, C7, C9] -> {S21[i, j, k, l] : ((((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9)) and (1 <= l <= C10))}", (S13(1) + S19(1)), true, p_float64, &function0);
    tiramisu::computation S23("[C8, C7, C9] -> {S23[i, j, k] : (((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9))}", S7(k, j), true, p_float64, &function0);
    tiramisu::computation S25("[C8, C7, C9] -> {S25[i, j, k] : (((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9))}", (S13(1) * S22(1)), true, p_float64, &function0);
    tiramisu::computation S26("[C8, C7, C9] -> {S26[i, j, k] : (((1 <= i <= C7) and (1 <= j <= C8)) and (1 <= k <= C9))}", (S11(1) + S24(1)), true, p_float64, &function0);
    tiramisu::computation S27("[C8, C7] -> {S27[i, j] : ((1 <= i <= C7) and (1 <= j <= C8))}", S11(1), true, p_float64, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    C1.after(C0, tiramisu::computation::root);
    C2.after(C1, tiramisu::computation::root);
    C3.after(C2, tiramisu::computation::root);
    C4.after(C3, tiramisu::computation::root);
    C5.after(C4, tiramisu::computation::root);
    C6.after(C5, tiramisu::computation::root);
    C7.after(C6, tiramisu::computation::root);
    C8.after(C7, tiramisu::computation::root);
    C9.after(C8, tiramisu::computation::root);
    C10.after(C9, tiramisu::computation::root);
    S12.after(C10, tiramisu::computation::root);
    S14.after(S12, j);
    S16.after(S14, k);
    S18.after(S16, l);
    S20.after(S18, l);
    S21.after(S20, l);
    S23.after(S21, k);
    S25.after(S23, k);
    S26.after(S25, k);
    S27.after(S26, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer BGS5("BGS5", {expr((int32_t) 3)}, p_int32, a_input, &function0);
    S4.set_access("{S4[i] -> BGS5[(i - 1)]}");
    tiramisu::buffer BGS10("BGS10", {expr((int32_t) 2)}, p_int32, a_input, &function0);
    S6.set_access("{S6[i] -> BGS10[(i - 1)]}");
    tiramisu::buffer BGS14("BGS14", {expr((int32_t) 2)}, p_int32, a_input, &function0);
    S8.set_access("{S8[i] -> BGS14[(i - 1)]}");
    tiramisu::buffer BSN2("BSN2", {expr((int32_t) 1)}, p_int32, a_input, &function0);
    S0.set_access("{S0[i] -> BSN2[0]}");
    tiramisu::buffer BSN3("BSN3", {expr((int32_t) 1)}, p_int32, a_input, &function0);
    S1.set_access("{S1[i] -> BSN3[0]}");
    tiramisu::buffer BSN4("BSN4", {expr((int32_t) 1)}, p_int32, a_input, &function0);
    S2.set_access("{S2[i] -> BSN4[0]}");
    tiramisu::buffer BSN5("BSN5", {expr((int32_t) 1)}, p_int32, a_input, &function0);
    S3.set_access("{S3[i] -> BSN5[0]}");
    tiramisu::buffer BSN6("BSN6", {C0, C1, C2}, p_float64, a_input, &function0);
    S5.set_access("{S5[i,j,k] -> BSN6[(i - 1),(j - 1),(k - 1)]}");
    tiramisu::buffer BSN7("BSN7", {C3, C4}, p_float64, a_input, &function0);
    S7.set_access("{S7[i,j] -> BSN7[(i - 1),(j - 1)]}");
    tiramisu::buffer BSN8("BSN8", {C5, C6}, p_float64, a_input, &function0);
    S9.set_access("{S9[i,j] -> BSN8[(i - 1),(j - 1)]}");
    tiramisu::buffer BSN9("BSN9", {C7, C8}, p_float64, a_output, &function0);
    S10.set_access("{S10[i,j] -> BSN9[(i - 1),(j - 1)]}");
    tiramisu::buffer BSN14("BSN14", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S11.set_access("{S11[i] -> BSN14[0]}");
    tiramisu::buffer BSN18("BSN18", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S13.set_access("{S13[i] -> BSN18[0]}");
    tiramisu::buffer BSSA26("BSSA26", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S15.set_access("{S15[i] -> BSSA26[0]}");
    tiramisu::buffer BSSA27("BSSA27", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S17.set_access("{S17[i] -> BSSA27[0]}");
    tiramisu::buffer BSSA29("BSSA29", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S19.set_access("{S19[i] -> BSSA29[0]}");
    tiramisu::buffer BSSA31("BSSA31", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S22.set_access("{S22[i] -> BSSA31[0]}");
    tiramisu::buffer BSSA33("BSSA33", {expr((int32_t) 1)}, p_float64, a_temporary, &function0);
    S24.set_access("{S24[i] -> BSSA33[0]}");
    S12.set_access("{S12[i,j] -> BSN14[0]}");
    S14.set_access("{S14[i,j,k] -> BSN18[0]}");
    S16.set_access("{S16[i,j,k,l] -> BSSA26[0]}");
    S18.set_access("{S18[i,j,k,l] -> BSSA27[0]}");
    S20.set_access("{S20[i,j,k,l] -> BSSA29[0]}");
    S21.set_access("{S21[i,j,k,l] -> BSN18[0]}");
    S23.set_access("{S23[i,j,k] -> BSSA31[0]}");
    S25.set_access("{S25[i,j,k] -> BSSA33[0]}");
    S26.set_access("{S26[i,j,k] -> BSN14[0]}");
    S27.set_access("{S27[i,j] -> BSN9[(i - 1),(j - 1)]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&BSN2, &BSN3, &BSN4, &BSN5, &BGS5, &BSN6, &BGS10, &BSN7, &BGS14, &BSN8, &BSN9});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
