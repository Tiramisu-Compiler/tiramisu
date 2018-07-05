#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "baryon_wrapper.h"
#include "benchmarks.h"


using namespace tiramisu;

/**
 * Res2 = 0
   For x0 in 0 to BARYON_N
     For x1 in 0 to BARYON_N
       For x2 in 0 to BARYON_N
       {
          Res0 = S(c1, x0, x1, x2, t, a1, x’0)*S(c2, x0, x1, x2, t, a2, x’0)*S(c3, x0, x1, x2, t, a3, x’0)
                +S(c2, x0, x1, x2, t, a1, x’0)*S(c3, x0, x1, x2, t, a2, x’0)*S(c1, x0, x1, x2, t, a3, x’0)
                +S(c3, x0, x1, x2, t, a1, x’0)*S(c1, x0, x1, x2, t, a2, x’0)*S(c2, x0, x1, x2, t, a3, x’0)
                -S(c2, x0, x1, x2, t, a1, x’0)*S(c1, x0, x1, x2, t, a2, x’0)*S(c3, x0, x1, x2, t, a3, x’0)
                -S(c3, x0, x1, x2, t, a1, x’0)*S(c2, x0, x1, x2, t, a2, x’0)*S(c1, x0, x1, x2, t, a3, x’0)
                -S(c1, x0, x1, x2, t, a1, x’0)*S(c3, x0, x1, x2, t, a2, x’0)*S(c2, x0, x1, x2, t, a3, x’0)

         Res1 = 0
         For k = 1 to N(B(b0, b1, b2))
           Res1 += w’(c1, c2, c3, b0, b1, b2, k) * Res0;

         Res2 += exp(i(x0*px+x1*py+x2*pz)) * Res1;
       }

 - Questions:
 -------------
 - what is the size of the dimensions of S[] ?
 */

void generate_function(std::string name, int size)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N_CONST("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant c1("c1", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant c2("c2", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant c3("c3", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant t("t", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant a1("a1", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant a2("a2", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant a3("a3", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant xp0("xp0", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant KMAX("KMAX", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant b0("b0", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant b1("b1", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant b2("b2", tiramisu::expr((int32_t) 0), p_int32, true, NULL, 0, &function0);

    tiramisu::var x0("x0"), x1("x1"), x2("x2"), k("k");
    tiramisu::computation S("{S[i0, i1, i2, i3, i4, i5, i6]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::computation wp("{wp[c1, c2, c3, b1, b2, b3, k]}", tiramisu::expr(), false, p_float32, &function0);

    tiramisu::computation Res0("[N]->{Res0[x0, x1, x2]: 0<=x0<N and 0<=x1<N and 0<=x2<N}", tiramisu::expr(), true, p_float32, &function0);
    Res0.set_expression(
		  S(c1, x0, x1, x2, t, a1, xp0) * S(c2, x0, x1, x2, t, a2, xp0) * S(c3, x0, x1, x2, t, a3, xp0)
                + S(c2, x0, x1, x2, t, a1, xp0) * S(c3, x0, x1, x2, t, a2, xp0) * S(c1, x0, x1, x2, t, a3, xp0)
                + S(c3, x0, x1, x2, t, a1, xp0) * S(c1, x0, x1, x2, t, a2, xp0) * S(c2, x0, x1, x2, t, a3, xp0)
                - S(c2, x0, x1, x2, t, a1, xp0) * S(c1, x0, x1, x2, t, a2, xp0) * S(c3, x0, x1, x2, t, a3, xp0)
                - S(c3, x0, x1, x2, t, a1, xp0) * S(c2, x0, x1, x2, t, a2, xp0) * S(c1, x0, x1, x2, t, a3, xp0)
                - S(c1, x0, x1, x2, t, a1, xp0) * S(c3, x0, x1, x2, t, a2, xp0) * S(c2, x0, x1, x2, t, a3, xp0)
	    );
    tiramisu::computation Res1("[N]->{Res1[x0, x1, x2, k]: 0<=x0<N and 0<=x1<N and 0<=x2<N and k=0}", tiramisu::expr((float) 0), true, p_float32, &function0);
    tiramisu::computation Res1_update_0("[N, KMAX]->{Res1_update_0[x0, x1, x2, k]: 0<=x0<N and 0<=x1<N and 0<=x2<N and 1<=k<KMAX}", tiramisu::expr(), true, p_float32, &function0);
    Res1_update_0.set_expression(Res1(x0, x1, x2, k-1) + wp(c1, c2, c3, b0, b1, b2, k) * Res0(x0, x1, x2));

    tiramisu::computation Res2("[N]->{Res2[0]}", tiramisu::expr((float) 0), true, p_float32, &function0);
    tiramisu::computation Res2_update_0("[N]->{Res2_update_0[x0, x1, x2]: 0<=x0<N and 0<=x1<N and 0<=x2<N}", tiramisu::expr(), true, p_float32, &function0);
    Res2_update_0.set_expression(Res2_update_0(x0, x1, x2) + /* exp(i(x0*px+x1*py+x2*pz)) */ Res1_update_0(x0, x1, x2, KMAX));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    Res0.after(Res2, tiramisu::computation::root);
    Res1.after(Res0, x2);
    Res1_update_0.after(Res1, x2);
    Res2_update_0.after(Res1_update_0, x2);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf_res0("buf_res0", {tiramisu::expr((int32_t) 1)}, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_res1("buf_res1", {tiramisu::expr((int32_t) 1)}, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_res2("buf_res2", {tiramisu::expr((int32_t) 1)}, tiramisu::p_float32, a_output, &function0);
    // S(c1, x0, x1, x2, t, a1, x’0)
    tiramisu::buffer buf_S("buf_S", {tiramisu::expr((int32_t) BARYON_P), N_CONST, N_CONST, N_CONST, tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P)}, tiramisu::p_float32, a_input, &function0);
    // wp(c1, c2, c3, b0, b1, b2, k)
    tiramisu::buffer buf_wp("buf_wp", {tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), tiramisu::expr((int32_t) BARYON_P), KMAX}, tiramisu::p_float32, a_input, &function0);

    Res0.set_access("{Res0[x0,x1,x2]->buf_res0[0]}");
    Res1.set_access("{Res1[x0,x1,x2,k]->buf_res1[0]}");
    Res1_update_0.set_access("{Res1_update_0[x0,x1,x2,k]->buf_res1[0]}");
    Res2.set_access("{Res2[0]->buf_res2[0]}");
    Res2_update_0.set_access("{Res2_update_0[x0,x1,x2]->buf_res2[0]}");
    S.set_access("{S[c1,x0,x1,x2,t,a1,xp0]->buf_S[c1,x0,x1,x2,t,a1,xp0]}");
    wp.set_access("{wp[c1,c2,c3,b0,b1,b2,k]->buf_wp[c1,c2,c3,b0,b1,b2,k]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf_res2, &buf_S, &buf_wp});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_" + std::string(TEST_NAME_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", BARYON_N);

    return 0;
}
