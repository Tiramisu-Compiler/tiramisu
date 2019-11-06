#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_dibaryon_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 0
#define PARALLEL 0

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
	x2("x2", 0, Vsnk),
	m("m", 0, Nsrc),
	n("n", 0, Nsnk),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);


    input C_r("C_r",      {m, n, t}, p_float64);
    input C_i("C_i",      {m, n, t}, p_float64);

    input  B1_Blocal_re("B1_Blocal_re",  {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, p_float64);
    input  B1_Blocal_im("B1_Blocal_im",  {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, p_float64);
    input B1_Bsingle_re("B1_Bsingle_re", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, p_float64);
    input B1_Bsingle_im("B1_Bsingle_im", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, p_float64);
    input B1_Bdouble_re("B1_Bdouble_re", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, p_float64);
    input B1_Bdouble_im("B1_Bdouble_im", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, p_float64);

    input B2_Blocal_re("B2_Blocal_re",   {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, p_float64);
    input B2_Blocal_im("B2_Blocal_im",   {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, p_float64);
    input B2_Bsingle_re("B2_Bsingle_re", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, p_float64);
    input B2_Bsingle_im("B2_Bsingle_im", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, p_float64);
    input B2_Bdouble_re("B2_Bdouble_re", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, p_float64);
    input B2_Bdouble_im("B2_Bdouble_im", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, p_float64);

    input  perms("perms", {nperm, q2}, p_int32);
    input  sigs("sigs", {nperm}, p_int32);
    input  overall_weight("overall_weight", {on}, p_float64);
    input  snk_color_weights("snk_color_weights", {to, wnum, q}, p_int32);
    input  snk_spin_weights("snk_spin_weights", {to, wnum, q}, p_int32);
    input  snk_weights("snk_weights", {wnum}, p_float64);
    input  snk_psi_re("snk_psi_re", {x, x2, n}, p_float64);
    input  snk_psi_im("snk_psi_im", {x, x2, n}, p_float64);

    computation snk_1("snk_1", {nperm, b}, perms(nperm, Nq*b+0) - 1, p_int32);
    computation snk_2("snk_2", {nperm, b}, perms(nperm, Nq*b+1) - 1, p_int32);
    computation snk_3("snk_3", {nperm, b}, perms(nperm, Nq*b+2) - 1, p_int32);

    computation snk_1_b("snk_1_b", {nperm, b}, (snk_1(nperm, b) - snk_1(nperm, b)%Nq)/Nq);
    computation snk_2_b("snk_2_b", {nperm, b}, (snk_2(nperm, b) - snk_2(nperm, b)%Nq)/Nq);
    computation snk_3_b("snk_3_b", {nperm, b}, (snk_3(nperm, b) - snk_3(nperm, b)%Nq)/Nq);
    computation snk_1_nq("snk_1_nq", {nperm, b}, snk_1(nperm, b)%Nq);
    computation snk_2_nq("snk_2_nq", {nperm, b}, snk_2(nperm, b)%Nq);
    computation snk_3_nq("snk_3_nq", {nperm, b}, snk_3(nperm, b)%Nq);

    // In all the following code, when an array is used for an indirect access, accesses to that
    // are not being treated correctly by Tiramisu. We set the first dimension in accesses in those
    // arrays to the value that we want, and only that value will be used in access later.
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

    computation term_re("term_re", {nperm, wnum, t, x, x2, m}, cast(p_float64, sigs(nperm)) * overall_weight(0) * snk_weights(wnum));
    computation term_im("term_im", {nperm, wnum, t, x, x2, m}, cast(p_float64, expr((double) 0)));

    computation new_term_re_0("new_term_re_0", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Blocal_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e) - term_im(nperm, wnum, t, x, x2, m) * B1_Blocal_im(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e));
    computation new_term_im_0("new_term_im_0", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Blocal_im(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e) + term_im(nperm, wnum, t, x, x2, m) * B1_Blocal_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e));
    new_term_re_0.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0);
    new_term_im_0.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0);

    computation new_term_re_1("new_term_re_1", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Blocal_re(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e) - term_im(nperm, wnum, t, x, x2, m) * B2_Blocal_im(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e));
    computation new_term_im_1("new_term_im_1", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Blocal_im(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e) + term_im(nperm, wnum, t, x, x2, m) * B2_Blocal_re(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e));
    new_term_re_1.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1);
    new_term_im_1.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1);

    computation new_term_re_2("new_term_re_2", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Bsingle_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2) - term_im(nperm, wnum, t, x, x2, m) * B1_Bsingle_im(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2));
    computation new_term_im_2("new_term_im_2", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Bsingle_im(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2) + term_im(nperm, wnum, t, x, x2, m) * B1_Bsingle_re(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2));
    new_term_re_2.add_predicate(snk_1_b(b, b) == 0 && snk_3_b(b, b) == 0);
    new_term_im_2.add_predicate(snk_1_b(b, b) == 0 && snk_3_b(b, b) == 0);

    computation new_term_re_3("new_term_re_3", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Bsingle_re(t, iC2e, iS2e, kC2e, kS2e, x, m, jC2e, jS2e, x2) - term_im(nperm, wnum, t, x, x2, m) * B2_Bsingle_im(t, iC2e, iS2e, kC2e, kS2e, x, m, jC2e, jS2e, x2));
    computation new_term_im_3("new_term_im_3", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Bsingle_im(t, iC2e, iS2e, kC2e, kS2e, x, m, jC2e, jS2e, x2) + term_im(nperm, wnum, t, x, x2, m) * B2_Bsingle_re(t, iC2e, iS2e, kC2e, kS2e, x, m, jC2e, jS2e, x2));
    new_term_re_3.add_predicate(snk_1_b(b, b) == 1 && snk_3_b(b, b) == 1);
    new_term_im_3.add_predicate(snk_1_b(b, b) == 1 && snk_3_b(b, b) == 1);

    computation new_term_re_4("new_term_re_4", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Bdouble_re(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2) - term_im(nperm, wnum, t, x, x2, m) * B1_Bdouble_im(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2));
    computation new_term_im_4("new_term_im_4", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B1_Bdouble_im(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2) + term_im(nperm, wnum, t, x, x2, m) * B1_Bdouble_re(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2));
    new_term_re_4.add_predicate((snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0) || (snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0));
    new_term_im_4.add_predicate((snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0) || (snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0));

    computation new_term_re_5("new_term_re_5", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Bdouble_re(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x2) - term_im(nperm, wnum, t, x, x2, m) * B2_Bdouble_im(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x2));
    computation new_term_im_5("new_term_im_5", {nperm, wnum, t, x, x2, m, b}, term_re(nperm, wnum, t, x, x2, m) * B2_Bdouble_im(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x2) + term_im(nperm, wnum, t, x, x2, m) * B2_Bdouble_re(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x2));
    new_term_re_5.add_predicate((snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1) || (snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1));
    new_term_im_5.add_predicate((snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1) || (snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1));

    computation term_re_1("term_re_1", {nperm, wnum, t, x, x2, m, b}, new_term_re_5(nperm, wnum, t, x, x2, m, b));
    computation term_im_1("term_im_1", {nperm, wnum, t, x, x2, m, b}, new_term_im_5(nperm, wnum, t, x, x2, m, b));
    complex_expr term(term_re_1(nperm, wnum, t, x, x2, m, Nb-1), term_im_1(nperm, wnum, t, x, x2, m, Nb-1));
    complex_expr snk_psi(snk_psi_re(x, x2, n), snk_psi_im(x, x2, n));
    complex_expr term_res = term * snk_psi;

    computation C_update_r("C_update_r", {nperm, wnum, t, x, x2, m, n}, C_r(m, n, t) + term_res.get_real());
    computation C_update_i("C_update_i", {nperm, wnum, t, x, x2, m, n}, C_i(m, n, t) + term_res.get_imag());

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
	 .then(new_term_re_0, m)
	 .then(new_term_im_0, b)
	 .then(new_term_re_1, b)
	 .then(new_term_im_1, b)
	 .then(new_term_re_2, b)
	 .then(new_term_im_2, b)
	 .then(new_term_re_3, b)
	 .then(new_term_im_3, b)
	 .then(new_term_re_4, b)
	 .then(new_term_im_4, b)
	 .then(new_term_re_5, b)
	 .then(new_term_im_5, b)
	 .then(term_re_1, b)
	 .then(term_im_1, b)
	 .then(C_update_r, m)
	 .then(C_update_i, n); 

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_snk_1("buf_snk_1", {Nb}, p_int32, a_output); //a_temporary);
    buffer buf_snk_2("buf_snk_2", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3("buf_snk_3", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_b("buf_snk_1_b", {Nb}, p_int32, a_output); //a_temporary);
    buffer buf_snk_2_b("buf_snk_2_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3_b("buf_snk_3_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_nq("buf_snk_1_nq", {Nb}, p_int32, a_output); //a_temporary);
    buffer buf_snk_2_nq("buf_snk_2_nq", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3_nq("buf_snk_3_nq", {Nb}, p_int32, a_temporary);

    buffer buf_iC1("buf_iC1", {1}, p_int32, a_temporary);
    buffer buf_iS1("buf_iS1", {1}, p_int32, a_temporary);
    buffer buf_jC1("buf_jC1", {1}, p_int32, a_temporary);
    buffer buf_jS1("buf_jS1", {1}, p_int32, a_temporary);
    buffer buf_kC1("buf_kC1", {1}, p_int32, a_temporary);
    buffer buf_kS1("buf_kS1", {1}, p_int32, a_temporary);
    buffer buf_iC2("buf_iC2", {1}, p_int32, a_temporary);
    buffer buf_iS2("buf_iS2", {1}, p_int32, a_temporary);
    buffer buf_jC2("buf_jC2", {1}, p_int32, a_temporary);
    buffer buf_jS2("buf_jS2", {1}, p_int32, a_temporary);
    buffer buf_kC2("buf_kC2", {1}, p_int32, a_temporary);
    buffer buf_kS2("buf_kS2", {1}, p_int32, a_temporary);

    iC1.store_in(&buf_iC1, {0});
    iS1.store_in(&buf_iS1, {0});
    jC1.store_in(&buf_jC1, {0});
    jS1.store_in(&buf_jS1, {0});
    kC1.store_in(&buf_kC1, {0});
    kS1.store_in(&buf_kS1, {0});
    iC2.store_in(&buf_iC2, {0});
    iS2.store_in(&buf_iS2, {0});
    jC2.store_in(&buf_jC2, {0});
    jS2.store_in(&buf_jS2, {0});
    kC2.store_in(&buf_kC2, {0});
    kS2.store_in(&buf_kS2, {0});

    snk_1.store_in(&buf_snk_1, {b});
    snk_2.store_in(&buf_snk_2, {b});
    snk_3.store_in(&buf_snk_3, {b});
    snk_1_b.store_in(&buf_snk_1_b, {b});
    snk_2_b.store_in(&buf_snk_2_b, {b});
    snk_3_b.store_in(&buf_snk_3_b, {b});
    snk_1_nq.store_in(&buf_snk_1_nq, {b});
    snk_2_nq.store_in(&buf_snk_2_nq, {b});
    snk_3_nq.store_in(&buf_snk_3_nq, {b});

    buffer buf_C_r("buf_C_r", {Nsrc, Nsnk, Nt}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Nsrc, Nsnk, Nt}, p_float64, a_input);

    buffer buf_snk_psi_re("buf_snk_psi_re", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);
    buffer buf_snk_psi_im("buf_snk_psi_im", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);

    buffer buf_new_term_r("buf_new_term_r", {1}, p_float64, a_output); // a_temporary);
    buffer buf_new_term_i("buf_new_term_i", {1}, p_float64, a_output); // a_temporary);

    buffer buf_term_r("buf_term_r", {1}, p_float64, a_output); //a_temporary);
    buffer buf_term_i("buf_term_i", {1}, p_float64, a_output); //a_temporary);

    term_re.store_in(&buf_term_r, {0});
    term_im.store_in(&buf_term_i, {0});
    term_re_1.store_in(&buf_term_r, {0});
    term_im_1.store_in(&buf_term_i, {0}); 

    new_term_re_0.store_in(&buf_new_term_r, {0});
    new_term_im_0.store_in(&buf_new_term_i, {0});
    new_term_re_1.store_in(&buf_new_term_r, {0});
    new_term_im_1.store_in(&buf_new_term_i, {0});
    new_term_re_2.store_in(&buf_new_term_r, {0});
    new_term_im_2.store_in(&buf_new_term_i, {0});
    new_term_re_3.store_in(&buf_new_term_r, {0});
    new_term_im_3.store_in(&buf_new_term_i, {0});
    new_term_re_4.store_in(&buf_new_term_r, {0});
    new_term_im_4.store_in(&buf_new_term_i, {0});
    new_term_re_5.store_in(&buf_new_term_r, {0});
    new_term_im_5.store_in(&buf_new_term_i, {0}); 

    snk_psi_re.store_in(&buf_snk_psi_re, {x, x2, n});
    snk_psi_im.store_in(&buf_snk_psi_im, {x, x2, n});

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
        B1_Bsingle_re.get_buffer(), B1_Bsingle_im.get_buffer(),
        B1_Bdouble_re.get_buffer(), B1_Bdouble_im.get_buffer(),
        B2_Blocal_re.get_buffer(), B2_Blocal_im.get_buffer(), 
        B2_Bsingle_re.get_buffer(), B2_Bsingle_im.get_buffer(),
        B2_Bdouble_re.get_buffer(), B2_Bdouble_im.get_buffer(),
	perms.get_buffer(), sigs.get_buffer(),
	overall_weight.get_buffer(),
	snk_color_weights.get_buffer(),
	snk_spin_weights.get_buffer(),
	snk_weights.get_buffer(),
	&buf_snk_psi_re,
	&buf_snk_psi_im,
	&buf_term_r,
	&buf_term_i,
	&buf_new_term_r,
	&buf_new_term_i,
	&buf_snk_1,
	&buf_snk_1_b,
	&buf_snk_1_nq
	},
        "generated_tiramisu_make_dibaryon_correlator.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_make_dibaryon_correlator");

    return 0;
}
