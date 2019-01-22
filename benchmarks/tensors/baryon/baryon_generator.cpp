#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "baryon_wrapper.h"

using namespace tiramisu;

/*
 * The goal is to generate code that implements the reference.
 * baryon_ref.cpp
 */
void generate_function(std::string name, int size)
{
    tiramisu::init(name);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::constant T("T", tiramisu::expr((int32_t) BT));
    tiramisu::constant a1("a1", tiramisu::expr((int32_t) 0));
    tiramisu::constant a2("a2", tiramisu::expr((int32_t) 0));
    tiramisu::constant a3("a3", tiramisu::expr((int32_t) 0));
    tiramisu::constant xp0("xp0", tiramisu::expr((int32_t) 0));
    tiramisu::constant K("K", tiramisu::expr((int32_t) BK));
    tiramisu::constant b0("b0", tiramisu::expr((int32_t) 0));
    tiramisu::constant b1("b1", tiramisu::expr((int32_t) 0));
    tiramisu::constant b2("b2", tiramisu::expr((int32_t) 0));

    tiramisu::var i1("i1", 0, N), i2("i2", 0, N), i3("i3", 0, N), k("k", 1, K), t("t", 0, T), k0("k", 0, 1);
    tiramisu::input fc1("fc1", {k}, p_int32);
    tiramisu::input fc2("fc2", {k}, p_int32);
    tiramisu::input fc3("fc3", {k}, p_int32);
    tiramisu::input S("S", {"xp0", "a1", "t", "i1", "i2", "i3", "d1"}, {1, 1, T, N, N, N, 1}, p_float32);
    tiramisu::input wp("wp", {"k", "b0", "b1", "b2"}, {K, 1, 1, 1}, p_float32);

    tiramisu::computation d1("d1", {t, i1, i2, i3, k}, fc1(k));
    tiramisu::computation d2("d2", {t, i1, i2, i3, k}, fc2(k));
    tiramisu::computation d3("d3", {t, i1, i2, i3, k}, fc3(k));

    tiramisu::computation Res0("Res0", {t, i1, i2, i3, k}, p_float32);
    Res0.set_expression(
			  S(xp0, a1, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d3(0,0,0,0,0))
			+ S(xp0, a1, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d1(0,0,0,0,0))
			+ S(xp0, a1, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d2(0,0,0,0,0))
		        - S(xp0, a1, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d3(0,0,0,0,0))
		        - S(xp0, a1, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d2(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d1(0,0,0,0,0))
		        - S(xp0, a1, t, i1, i2, i3, d1(0,0,0,0,0)) * S(xp0, a2, t, i1, i2, i3, d3(0,0,0,0,0)) * S(xp0, a3, t, i1, i2, i3, d2(0,0,0,0,0))
		);

    tiramisu::computation Res1("Res1", {t, i1, i2, i3, k0}, tiramisu::expr((float) 0));
    tiramisu::computation Res1_update_0("Res1_update_0", {t, i1, i2, i3, k}, p_float32);
    Res1_update_0.set_expression(Res1(t, i1, i2, i3, k-1) + wp(k, b2, b1, b0) * Res0(t, i1, i2, i3, k));

    tiramisu::computation Res2("Res2", {t}, tiramisu::expr((float) 0));
    tiramisu::computation Res2_update_0("Res2_update_0", {t, i1, i2, i3}, p_float32);
    Res2_update_0.set_expression(Res2_update_0(t, i1, i2, i3) + /* exp(i(i3*px+i2*py+i1*pz)) */ Res1(t, i1, i2, i3, 0));

    global::get_implicit_function()->add_context_constraints("[N, M, K, T]->{:N=16}, T=16");

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    tiramisu::buffer buf_fc1("buf_fc1", {K}, tiramisu::p_int32, a_input);
    tiramisu::buffer buf_fc2("buf_fc2", {K}, tiramisu::p_int32, a_input);
    tiramisu::buffer buf_fc3("buf_fc3", {K}, tiramisu::p_int32, a_input);

    tiramisu::buffer buf_res0("buf_res0", {BZ}, tiramisu::p_float32, a_temporary);
    buf_res0.set_auto_allocate(false);
    tiramisu::computation *alloc_res0 = buf_res0.allocate_at(Res2, t);
    tiramisu::buffer buf_res1("buf_res1", {N}, tiramisu::p_float32, a_temporary);
    buf_res1.set_auto_allocate(false);
    tiramisu::computation *alloc_res1 = buf_res1.allocate_at(Res2, t);
    tiramisu::buffer buf_res2("buf_res2", {T}, tiramisu::p_float32, a_output);
    tiramisu::buffer buf_d1("buf_d1", {K}, tiramisu::p_int32, a_temporary);
    buf_d1.set_auto_allocate(false);
    tiramisu::computation *alloc_d1 = buf_d1.allocate_at(Res2, t);
    tiramisu::buffer buf_d2("buf_d2", {K}, tiramisu::p_int32, a_temporary);
    buf_d2.set_auto_allocate(false);
    tiramisu::computation *alloc_d2 = buf_d2.allocate_at(Res2, t);
    tiramisu::buffer buf_d3("buf_d3", {K}, tiramisu::p_int32, a_temporary);
    buf_d3.set_auto_allocate(false);
    tiramisu::computation *alloc_d3 = buf_d3.allocate_at(Res2, t);

    // S(d1, i3, i2, i1, t, a1, x’0)
    tiramisu::buffer buf_S("buf_S", {tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), N, N, N, tiramisu::expr((int32_t) BARYON_P1)}, tiramisu::p_float32, a_input);

    tiramisu::buffer buf_wp("buf_wp", {tiramisu::expr((int32_t) BARYON_N), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P)}, tiramisu::p_float32, a_input);

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
