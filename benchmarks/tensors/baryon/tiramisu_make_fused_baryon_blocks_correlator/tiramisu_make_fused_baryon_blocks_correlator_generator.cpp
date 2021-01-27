#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_baryon_blocks_correlator_wrapper.h"
#include "../../utils/complex_util.h"
#include "../../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

void generate_function(std::string name)
{
    tiramisu::init(name);

   var r("r", 0, B1Nrows),
	rp("rp", 0, B1Nrows),
        nperm("nperm", 0, B1Nperms),
	q("q", 0, Nq),
	wnum("wnum", 0, Nw),
	wnumBlock("wnumBlock", 0, Nw),
        t("t", 0, Lt),
	x("x", 0, Vsnk),
	x_out("x_out", 0, Vsnk/sites_per_rank),
	x_in("x_in", 0, sites_per_rank),
        y("y", 0, Vsrc),
	m("m", 0, NsrcHex),
	n("n", 0, NsnkHex),
        tri("tri", 0, Nq),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);

   input C_r("C_r",      {t, x_out, rp, m, r, n}, p_float64);
   input C_i("C_i",      {t, x_out, rp, m, r, n}, p_float64);
   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_B1_r("src_psi_B1_r",    {y, m}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {y, m}, p_float64);
   input snk_psi_r("snk_psi_r", {x, n}, p_float64);
   input snk_psi_i("snk_psi_i", {x, n}, p_float64);
   input src_color_weights("src_color_weights", {rp, wnum, q}, p_int32);
   input src_spin_weights("src_spin_weights", {rp, wnum, q}, p_int32);
   input src_weights("src_weights", {rp, wnum}, p_float64);
   input snk_color_weights("snk_color_weights", {r, nperm, wnum, q}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, nperm, wnum, q}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);
   input src_spins("src_spins", {rp}, p_int32);
   input sigs("sigs", {nperm}, p_int32);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));

    /*
     * Computing B1_Blocal_r1
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B1_Blocal_r1_init(&B1_Blocal_r1_r_init, &B1_Blocal_r1_i_init);

    complex_expr B1_r1_prop_0 =  B1_prop(0, t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_2 =  B1_prop(2, t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_1 = B1_prop(1, t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr B1_r1_diquark = ( B1_r1_prop_0 * B1_r1_prop_2 ) *  src_weights(0, wnumBlock);

    /*complex_expr B1_r1_prop_0s =  B1_prop(2, t, iCprime, iSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_2s =  B1_prop(0, t, kCprime, kSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_0ps = B1_prop(2, t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_2ps = B1_prop(0, t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);

    complex_expr B1_r1_diquark = ( B1_r1_prop_0 * B1_r1_prop_2 - B1_r1_prop_0s * B1_r1_prop_2s ) *  src_weights(0, wnumBlock); */

    computation B1_Blocal_r1_r_props_init("B1_Blocal_r1_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_props_init("B1_Blocal_r1_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B1_Blocal_r1_r_diquark("B1_Blocal_r1_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_real());
    computation B1_Blocal_r1_i_diquark("B1_Blocal_r1_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_imag());

    complex_computation B1_Blocal_r1_diquark(&B1_Blocal_r1_r_diquark, &B1_Blocal_r1_i_diquark);

    complex_expr B1_r1_props = B1_r1_prop_1 * B1_Blocal_r1_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r1_r_props("B1_Blocal_r1_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r1_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r1_props.get_real());
    computation B1_Blocal_r1_i_props("B1_Blocal_r1_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r1_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r1_props.get_imag());

    complex_computation B1_Blocal_r1_props(&B1_Blocal_r1_r_props, &B1_Blocal_r1_i_props);

    complex_expr B1_r1 = src_psi_B1 * B1_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r1_r_update("B1_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_real());
    computation B1_Blocal_r1_i_update("B1_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_imag());

    /*
     * Computing B1_Blocal_r2
     */

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B1_Blocal_r2_init(&B1_Blocal_r2_r_init, &B1_Blocal_r2_i_init);

    complex_expr B1_r2_prop_0 =  B1_prop(0, t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_2 =  B1_prop(2, t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_0p = B1_prop(0, t, kCprime, kSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_2p = B1_prop(2, t, iCprime, iSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_1 = B1_prop(1, t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr B1_r2_diquark = ( B1_r2_prop_0 * B1_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation B1_Blocal_r2_r_props_init("B1_Blocal_r2_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r2_i_props_init("B1_Blocal_r2_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B1_Blocal_r2_r_diquark("B1_Blocal_r2_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_real());
    computation B1_Blocal_r2_i_diquark("B1_Blocal_r2_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_imag());

    complex_computation B1_Blocal_r2_diquark(&B1_Blocal_r2_r_diquark, &B1_Blocal_r2_i_diquark);

    complex_expr B1_r2_props = B1_r2_prop_1 * B1_Blocal_r2_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r2_r_props("B1_Blocal_r2_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_real());
    computation B1_Blocal_r2_i_props("B1_Blocal_r2_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_imag());

    complex_computation B1_Blocal_r2_props(&B1_Blocal_r2_r_props, &B1_Blocal_r2_i_props);

    complex_expr B1_r2 = src_psi_B1 * B1_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r2_r_update("B1_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_real());
    computation B1_Blocal_r2_i_update("B1_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_imag());

    /* Correlator */

    computation C_init_r("C_init_r", {t, x_out, rp, m, r, n}, expr((double) 0));
    computation C_init_i("C_init_i", {t, x_out, rp, m, r, n}, expr((double) 0));

    computation C_prop_init_r("C_prop_init_r", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    computation C_prop_init_i("C_prop_init_i", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    
    int b=0;
    /* r1, b = 0 */
    complex_computation new_term_0_r1_b1("new_term_0_r1_b1", {t, x_out, x_in, rp, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x_out, x_in, snk_color_weights(r, nperm, wnum, 0), snk_spin_weights(r, nperm, wnum, 0), snk_color_weights(r, nperm, wnum, 2), snk_spin_weights(r, nperm, wnum, 2), snk_color_weights(r, nperm, wnum, 1), snk_spin_weights(r, nperm, wnum, 1), m));
    new_term_0_r1_b1.add_predicate(src_spins(rp) == 1);
    /* r2, b = 0 */
    complex_computation new_term_0_r2_b1("new_term_0_r2_b1", {t, x_out, x_in, rp, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x_out, x_in, snk_color_weights(r, nperm, wnum, 0), snk_spin_weights(r, nperm, wnum, 0), snk_color_weights(r, nperm, wnum, 2), snk_spin_weights(r, nperm, wnum, 2), snk_color_weights(r, nperm, wnum, 1), snk_spin_weights(r, nperm, wnum, 1), m));
    new_term_0_r2_b1.add_predicate(src_spins(rp) == 2);

    complex_expr prefactor(cast(p_float64, snk_weights(r, wnum))*cast(p_float64, sigs(nperm)), 0.0);

    complex_expr term_res_b1 = new_term_0_r1_b1(t, x_out, x_in, rp, m, r, nperm, wnum);

    complex_expr snk_psi(snk_psi_r(x_out*sites_per_rank+x_in, n), snk_psi_i(x_out*sites_per_rank+x_in, n));

    complex_expr term_res = prefactor * term_res_b1;

    computation C_prop_update_r("C_prop_update_r", {t, x_out, x_in, rp, m, r, nperm, wnum}, C_prop_init_r(t, x_out, x_in, rp, m, r) + term_res.get_real());
    computation C_prop_update_i("C_prop_update_i", {t, x_out, x_in, rp, m, r, nperm, wnum}, C_prop_init_i(t, x_out, x_in, rp, m, r) + term_res.get_imag());

    complex_computation C_prop_update(&C_prop_update_r, &C_prop_update_i);

    complex_expr term = C_prop_update(t, x_out, x_in, rp, m, r, B1Nperms-1, Nw-1) * snk_psi;

    computation C_update_r("C_update_r", {t, x_out, x_in, rp, m, r, n}, C_init_r(t, x_out, rp, m, r, n) + term.get_real());
    computation C_update_i("C_update_i", {t, x_out, x_in, rp, m, r, n}, C_init_i(t, x_out, rp, m, r, n) + term.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, n)
    );

    // first the x only arrays
    handle = &(handle
        ->then(B1_Blocal_r1_r_init, t)
        .then(B1_Blocal_r1_i_init, jSprime)
        .then(B1_Blocal_r1_r_props_init, x_in)
        .then(B1_Blocal_r1_i_props_init, jSprime)
        .then(B1_Blocal_r1_r_diquark, y)
        .then(B1_Blocal_r1_i_diquark, wnumBlock)
        .then(B1_Blocal_r1_r_props, wnumBlock)
        .then(B1_Blocal_r1_i_props, jSprime)
        .then(B1_Blocal_r1_r_update, y)
        .then(B1_Blocal_r1_i_update, m)
        .then(B1_Blocal_r2_r_init, x_in)
        .then(B1_Blocal_r2_i_init, jSprime)
        .then(B1_Blocal_r2_r_props_init, x_in)
        .then(B1_Blocal_r2_i_props_init, jSprime)
        .then(B1_Blocal_r2_r_diquark, y)
        .then(B1_Blocal_r2_i_diquark, wnumBlock)
        .then(B1_Blocal_r2_r_props, wnumBlock)
        .then(B1_Blocal_r2_i_props, jSprime)
        .then(B1_Blocal_r2_r_update, y)
        .then(B1_Blocal_r2_i_update, m));

    handle = &(handle 
          ->then(C_prop_init_r, x_in) 
          .then(C_prop_init_i, r)
          .then( *(new_term_0_r1_b1.get_real()), r)
          .then( *(new_term_0_r1_b1.get_imag()), wnum)
          .then( *(new_term_0_r2_b1.get_real()), wnum)
          .then( *(new_term_0_r2_b1.get_imag()), wnum)
          .then(C_prop_update_r, wnum) 
          .then(C_prop_update_i, wnum)
          .then(C_update_r, r) 
          .then(C_update_i, n));

#if VECTORIZED

#endif

#if PARALLEL

    C_init_r.tag_distribute_level(t);

    B1_Blocal_r1_r_init.tag_distribute_level(t);
    B1_Blocal_r2_r_init.tag_distribute_level(t);

    C_prop_init_r.tag_distribute_level(t);

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float64, a_temporary);
    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_r_update.store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_update.store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_B1_Blocal_diquark_r1_r("buf_B1_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Blocal_diquark_r1_i("buf_B1_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    B1_Blocal_r1_r_diquark.store_in(&buf_B1_Blocal_diquark_r1_r, {0});
    B1_Blocal_r1_i_diquark.store_in(&buf_B1_Blocal_diquark_r1_i, {0});
    buffer buf_B1_Blocal_props_r1_r("buf_B1_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_props_r1_i("buf_B1_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    B1_Blocal_r1_r_props_init.store_in(&buf_B1_Blocal_props_r1_r, {jCprime, jSprime});
    B1_Blocal_r1_i_props_init.store_in(&buf_B1_Blocal_props_r1_i, {jCprime, jSprime});
    B1_Blocal_r1_r_props.store_in(&buf_B1_Blocal_props_r1_r, {jCprime, jSprime});
    B1_Blocal_r1_i_props.store_in(&buf_B1_Blocal_props_r1_i, {jCprime, jSprime});

    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float64, a_temporary);
    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_r_update.store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_update.store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_B1_Blocal_diquark_r2_r("buf_B1_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Blocal_diquark_r2_i("buf_B1_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    B1_Blocal_r2_r_diquark.store_in(&buf_B1_Blocal_diquark_r2_r, {0});
    B1_Blocal_r2_i_diquark.store_in(&buf_B1_Blocal_diquark_r2_i, {0});
    buffer buf_B1_Blocal_props_r2_r("buf_B1_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_props_r2_i("buf_B1_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    B1_Blocal_r2_r_props_init.store_in(&buf_B1_Blocal_props_r2_r, {jCprime, jSprime});
    B1_Blocal_r2_i_props_init.store_in(&buf_B1_Blocal_props_r2_i, {jCprime, jSprime});
    B1_Blocal_r2_r_props.store_in(&buf_B1_Blocal_props_r2_r, {jCprime, jSprime});
    B1_Blocal_r2_i_props.store_in(&buf_B1_Blocal_props_r2_i, {jCprime, jSprime});

    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, Vsnk/sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, Vsnk/sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float64, a_input);

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);

    buffer* buf_new_term_r_b1;//("buf_new_term_r_b1", {1}, p_float64, a_temporary);
    buffer* buf_new_term_i_b1;//("buf_new_term_i_b1", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_new_term_r_b1, buf_new_term_i_b1, {1}, "buf_new_term_b1");

    new_term_0_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_0_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0});

    new_term_0_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_0_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0});

    buffer buf_C_prop_r("buf_C_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_prop_i("buf_C_prop_i", {1}, p_float64, a_temporary);

    C_prop_init_r.store_in(&buf_C_prop_r, {0});
    C_prop_init_i.store_in(&buf_C_prop_i, {0});
    C_prop_update_r.store_in(&buf_C_prop_r, {0});
    C_prop_update_i.store_in(&buf_C_prop_i, {0});

    C_init_r.store_in(&buf_C_r, {t, x_out, rp, m, r, n});
    C_init_i.store_in(&buf_C_i, {t, x_out, rp, m, r, n});
    C_update_r.store_in(&buf_C_r, {t, x_out, rp, m, r, n});
    C_update_i.store_in(&buf_C_i, {t, x_out, rp, m, r, n});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	     &buf_C_r, &buf_C_i,
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        snk_psi_r.get_buffer(), snk_psi_i.get_buffer(),
	     src_color_weights.get_buffer(),
	     src_spin_weights.get_buffer(),
	     src_weights.get_buffer(),
	     src_spins.get_buffer(), 
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer(),
	     sigs.get_buffer()
        }, 
        "generated_tiramisu_make_fused_baryon_blocks_correlator.o");
}

int main(int argc, char **argv)
{

    generate_function("tiramisu_make_fused_baryon_blocks_correlator");

    return 0;
}
