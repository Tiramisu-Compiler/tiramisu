#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_dibaryon_hex_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

/*
 * The goal is to generate code that implements the reference.
 * baryon_ref.cpp
 */
void generate_function(std::string name)
{
    tiramisu::init(name);

    var nperm("nperm", 0, Nperms),
	b("b", 0, Nb),
	q("q", 0, Nq),
	q2("q2", 0, 2*Nq),
	to("to", 0, 2),
	on("on", 0, 1),
	wnum("wnum", 0, Nw2),
        t("t", 0, Nt),
	x("x", 0, Vsnk),
	m("m", 0, Nsrc),
	n("n", 0, NsnkHex),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);


    input C_r("C_r",      {m, n, t}, p_float64);
    input C_i("C_i",      {m, n, t}, p_float64);

    input B1_Blocal_re("B1_Blocal_re",      {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, p_float64);
    input B1_Blocal_im("B1_Blocal_im",      {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, p_float64);
 
    input B2_Blocal_re("B2_Blocal_re",      {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, p_float64);
    input B2_Blocal_im("B2_Blocal_im",      {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, p_float64);

    input  perms("perms", {nperm, q2}, p_int32);
    input  sigs("sigs", {nperm}, p_int32);
    input  overall_weight("overall_weight", {on}, p_int32);
    input  snk_color_weights("snk_color_weights", {to, wnum, q}, p_int32);
    input  snk_spin_weights("snk_spin_weights", {to, wnum, q}, p_int32);
    input  snk_weights("snk_weights", {wnum}, p_float64);
    input  hex_snk_psi_re("hex_snk_psi_re", {x, n}, p_float64);
    input  hex_snk_psi_im("hex_snk_psi_im", {x, n}, p_float64);

    computation snk_1("snk_1", {nperm, b}, perms(nperm, Nq*b+0) - 1, p_int32);
    computation snk_2("snk_2", {nperm, b}, perms(nperm, Nq*b+1) - 1, p_int32);
    computation snk_3("snk_3", {nperm, b}, perms(nperm, Nq*b+2) - 1, p_int32);

    computation snk_1_b("snk_1_b", {nperm, b}, (snk_1(nperm, b) - snk_1(nperm, b)%Nq)/Nq);
    computation snk_2_b("snk_2_b", {nperm, b}, (snk_2(nperm, b) - snk_2(nperm, b)%Nq)/Nq);
    computation snk_3_b("snk_3_b", {nperm, b}, (snk_3(nperm, b) - snk_3(nperm, b)%Nq)/Nq);
    computation snk_1_nq("snk_1_nq", {nperm, b}, snk_1(nperm, b)%Nq);
    computation snk_2_nq("snk_2_nq", {nperm, b}, snk_2(nperm, b)%Nq);
    computation snk_3_nq("snk_3_nq", {nperm, b}, snk_3(nperm, b)%Nq);

    computation iC1("iC1", {nperm, wnum}, snk_color_weights(snk_1_b(0, 0), wnum, snk_1_nq(0, 0))); //Original access: snk_1_b(nperm, 0) replaced by snk_1_b(0, 0)
    computation iS1("iS1", {nperm, wnum}, snk_spin_weights(snk_1_b(0, 0), wnum, snk_1_nq(0, 0)));
    computation jC1("jC1", {nperm, wnum}, snk_color_weights(snk_2_b(0, 0), wnum, snk_2_nq(0, 0)));
    computation jS1("jS1", {nperm, wnum}, snk_spin_weights(snk_2_b(0, 0), wnum, snk_2_nq(0, 0)));
    computation kC1("kC1", {nperm, wnum}, snk_color_weights(snk_3_b(0, 0), wnum, snk_3_nq(0, 0)));
    computation kS1("kS1", {nperm, wnum}, snk_spin_weights(snk_3_b(0, 0), wnum, snk_3_nq(0, 0)));
    computation iC2("iC2", {nperm, wnum}, snk_color_weights(snk_1_b(1, 1), wnum, snk_1_nq(1, 1)));
    computation iS2("iS2", {nperm, wnum}, snk_spin_weights(snk_1_b(1, 1), wnum, snk_1_nq(1, 1)));
    computation jC2("jC2", {nperm, wnum}, snk_color_weights(snk_2_b(1, 1), wnum, snk_2_nq(1, 1)));
    computation jS2("jS2", {nperm, wnum}, snk_spin_weights(snk_2_b(1, 1), wnum, snk_2_nq(1, 1)));
    computation kC2("kC2", {nperm, wnum}, snk_color_weights(snk_3_b(1, 1), wnum, snk_3_nq(1, 1)));
    computation kS2("kS2", {nperm, wnum}, snk_spin_weights(snk_3_b(1, 1), wnum, snk_3_nq(1, 1)));

    expr iC1e = iC1(0, 0); //Original access iC1(nperm, wnum) replaced by iC1(0, 0)
    expr iS1e = iS1(0, 0);
    expr jC1e = jC1(0, 0);
    expr jS1e = jS1(0, 0);
    expr kC1e = kC1(0, 0);
    expr kS1e = kS1(0, 0);
    expr iC2e = iC2(0, 0);
    expr iS2e = iS2(0, 0);
    expr jC2e = jC2(0, 0);
    expr jS2e = jS2(0, 0);
    expr kC2e = kC2(0, 0);
    expr kS2e = kS2(0, 0);

    complex_expr B1_Blocal(B1_Blocal_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e), B1_Blocal_im(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e));
    complex_expr B2_Blocal(B2_Blocal_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e), B2_Blocal_im(t, iC2e, iS2e, kC2e, kS2e, x, m, jC2e, jS2e));
    complex_expr B1_B2_Blocal = B1_Blocal * B2_Blocal;

    computation term_re("term_re", {nperm, wnum, t, x, m}, cast(p_float64, sigs(nperm) * overall_weight(0)) * snk_weights(wnum) * B1_B2_Blocal.get_real());
    computation term_im("term_im", {nperm, wnum, t, x, m}, cast(p_float64, sigs(nperm) * overall_weight(0)) * snk_weights(wnum) * B1_B2_Blocal.get_imag());

    complex_expr term(term_re(nperm, wnum, t, x, m), term_im(nperm, wnum, t, x, m));
    complex_expr hex_snk_psi(hex_snk_psi_re(x, n), hex_snk_psi_im(x, n));
    complex_expr term_hex = term * hex_snk_psi;

    computation C_update_r("C_update_r", {nperm, wnum, t, x, m, n}, C_r(m, n, t) + term_hex.get_real());
    computation C_update_i("C_update_i", {nperm, wnum, t, x, m, n}, C_i(m, n, t) + term_hex.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    snk_1.then(snk_2, b)
	 .then(snk_3, b)
	 .then(snk_1_b, b)
	 .then(snk_2_b, b)
	 .then(snk_3_b, b)
	 .then(snk_1_nq, b)
	 .then(snk_2_nq, b)
	 .then(snk_3_nq, b)
	 .then(iC1, nperm)
	 .then(iS1, wnum)
	 .then(jC1, wnum)
	 .then(jS1, wnum)
	 .then(kC1, wnum)
	 .then(kS1, wnum)
	 .then(iC2, wnum)
	 .then(iS2, wnum)
	 .then(jC2, wnum)
	 .then(jS2, wnum)
	 .then(kC2, wnum)
	 .then(kS2, wnum)
	 .then(term_re, wnum)
	 .then(term_im, m)
	 .then(C_update_r, m)
	 .then(C_update_i, n);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_snk_1("buf_snk_1", {Nb}, p_int32, a_temporary);
    buffer buf_snk_2("buf_snk_2", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3("buf_snk_3", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_b("buf_snk_1_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_2_b("buf_snk_2_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3_b("buf_snk_3_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_nq("buf_snk_1_nq", {Nb}, p_int32, a_temporary);
    buffer buf_snk_2_nq("buf_snk_2_nq", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3_nq("buf_snk_3_nq", {Nb}, p_int32, a_temporary);

    snk_1.store_in(&buf_snk_1, {b});
    snk_2.store_in(&buf_snk_2, {b});
    snk_3.store_in(&buf_snk_3, {b});
    snk_1_b.store_in(&buf_snk_1_b, {b});
    snk_2_b.store_in(&buf_snk_2_b, {b});
    snk_3_b.store_in(&buf_snk_3_b, {b});
    snk_1_nq.store_in(&buf_snk_1_nq, {b});
    snk_2_nq.store_in(&buf_snk_2_nq, {b});
    snk_3_nq.store_in(&buf_snk_3_nq, {b});

    buffer buf_C_r("buf_C_r", {Nsrc, NsnkHex, Nt}, p_float64, a_output);
    buffer buf_C_i("buf_C_i", {Nsrc, NsnkHex, Nt}, p_float64, a_output);

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);
    C_update_r.store_in(&buf_C_r, {m, n, t});
    C_update_i.store_in(&buf_C_i, {m, n, t});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	&buf_C_r, &buf_C_i,
        B1_Blocal_re.get_buffer(), B1_Blocal_im.get_buffer(), 
        B2_Blocal_re.get_buffer(), B2_Blocal_im.get_buffer(), 
	perms.get_buffer(), sigs.get_buffer(),
	overall_weight.get_buffer(),
	snk_color_weights.get_buffer(),
	snk_spin_weights.get_buffer(),
	snk_weights.get_buffer(),
	hex_snk_psi_re.get_buffer(),
	hex_snk_psi_im.get_buffer()},
        "generated_tiramisu_make_dibaryon_hex_correlator.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_make_dibaryon_hex_correlator");

    return 0;
}
