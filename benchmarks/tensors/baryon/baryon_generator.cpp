#include <tiramisu/tiramisu.h>

#include <string.h>

#include "baryon_wrapper.h"

using namespace tiramisu;

/*
 * The goal is to generate code that implements the reference.
 * baryon_ref.cpp
 */
void generate_function(std::string name, int size)
{
    tiramisu::init(name);

    constant N("N", size);
    constant T("T", BT);
    constant a1("a1", 0);
    constant a2("a2", 0);
    constant a3("a3", 0);
    constant xp0("xp0", 0);
    constant K("K", BK);
    constant b0("b0", 0);
    constant b1("b1", 0);
    constant b2("b2", 0);

    var i1("i1", 0, N), i2("i2", 0, N), i3("i3", 0, N), k("k", 1, K), t("t", 0, T), k0("k", 0, 1);
    input fc1("fc1", {k}, p_int32);
    input fc2("fc2", {k}, p_int32);
    input fc3("fc3", {k}, p_int32);
    input S("S", {"xp0", "a1", "t", "i1", "i2", "i3", "d1"}, {1, 1, T, N, N, N, 1}, p_float32);
    input wp("wp", {"k", "b0", "b1", "b2"}, {K, 1, 1, 1}, p_float32);

    computation d1("d1", {t, i1, i2, i3, k}, fc1(k));
    computation d2("d2", {t, i1, i2, i3, k}, fc2(k));
    computation d3("d3", {t, i1, i2, i3, k}, fc3(k));

    computation Res0("Res0", {t, i1, i2, i3, k}, p_float32);
    Res0.set_expression(S(xp0, a1, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d3(0,0,0,0,0))
		      + S(xp0, a1, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d1(0,0,0,0,0))
		      + S(xp0, a1, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d2(0,0,0,0,0))
		      - S(xp0, a1, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d3(0,0,0,0,0))
		      - S(xp0, a1, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d1(0,0,0,0,0))
		      - S(xp0, a1, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d2(0,0,0,0,0)));

    computation Res1("Res1", {t, i1, i2, i3, k0}, expr((float) 0));
    computation Res1_update_0("Res1_update_0", {t, i1, i2, i3, k}, p_float32);
    Res1_update_0.set_expression(Res1(t, i1, i2, i3, k-1) + wp(k, b2, b1, b0) * Res0(t, i1, i2, i3, k));

    computation Res2("Res2", {t}, expr((float) 0));
    computation Res2_update_0("Res2_update_0", {t, i1, i2, i3}, p_float32);
    Res2_update_0.set_expression(Res2_update_0(t, i1, i2, i3) + /* exp(i(i3*px+i2*py+i1*pz)) */ Res1(t, i1, i2, i3, 0));

    global::get_implicit_function()->add_context_constraints("[N, M, K, T]->{:N=16}, T=16");

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_fc1("buf_fc1", {K}, p_int32, a_input);
    buffer buf_fc2("buf_fc2", {K}, p_int32, a_input);
    buffer buf_fc3("buf_fc3", {K}, p_int32, a_input);

    buffer buf_res0("buf_res0", {BZ}, p_float32, a_temporary);
    buf_res0.set_auto_allocate(false);
    computation *alloc_res0 = buf_res0.allocate_at(Res2, t);
    buffer buf_res1("buf_res1", {N}, p_float32, a_temporary);
    buf_res1.set_auto_allocate(false);
    computation *alloc_res1 = buf_res1.allocate_at(Res2, t);
    buffer buf_res2("buf_res2", {T}, p_float32, a_output);
    buffer buf_d1("buf_d1", {K}, p_int32, a_temporary);
    buf_d1.set_auto_allocate(false);
    computation *alloc_d1 = buf_d1.allocate_at(Res2, t);
    buffer buf_d2("buf_d2", {K}, p_int32, a_temporary);
    buf_d2.set_auto_allocate(false);
    computation *alloc_d2 = buf_d2.allocate_at(Res2, t);
    buffer buf_d3("buf_d3", {K}, p_int32, a_temporary);
    buf_d3.set_auto_allocate(false);
    computation *alloc_d3 = buf_d3.allocate_at(Res2, t);

    // S(d1, i3, i2, i1, t, a1, x’0)
    buffer buf_S("buf_S", {BARYON_P, BARYON_P, BARYON_P, N, N, N, BARYON_P1}, p_float32, a_input);

    buffer buf_wp("buf_wp", {BARYON_N, BARYON_P, BARYON_P, BARYON_P}, p_float32, a_input);

    fc1.store_in(&buf_fc1);
    fc2.store_in(&buf_fc2);
    fc3.store_in(&buf_fc3);
    d1.store_in(&buf_d1, {0});
    d2.store_in(&buf_d2, {0});
    d3.store_in(&buf_d3, {0});
    Res0.store_in(&buf_res0, {i3});
    Res1.store_in(&buf_res1, {i3});
    Res1_update_0.store_in(&buf_res1, {i3});
    Res2.store_in(&buf_res2, {t});
    Res2_update_0.store_in(&buf_res2, {t});
    S.store_in(&buf_S);
    wp.store_in(&buf_wp);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    Res2.then(*alloc_res1, t)
	.then(*alloc_res0, t)
	.then(*alloc_d1, t)
	.then(*alloc_d2, t)
	.then(*alloc_d3, t)
	.then(Res1, i3)
	.then(d1, i3)
	.then(d2, k)
	.then(d3, k)
	.then(Res0, k)
	.then(Res1_update_0, k)
	.then(Res2_update_0, i2);

    Res0.tag_vector_level(i3, BARYON_N);
    Res2.tag_parallel_level(t);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&buf_res2, &buf_S, &buf_wp, &buf_fc1, &buf_fc2, &buf_fc3},
		      "generated_" + std::string(TEST_NAME_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", BARYON_N);

    return 0;
}
