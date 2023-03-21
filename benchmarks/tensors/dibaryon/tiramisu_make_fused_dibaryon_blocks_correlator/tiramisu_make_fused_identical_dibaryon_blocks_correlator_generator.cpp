#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "../../utils/complex_util.h"
#include "../../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

void generate_function(std::string name)
{
    tiramisu::init(name);

   int b;
   int NsrcTot = Nsrc+NsrcHex;
   int NsnkTot = Nsnk+NsnkHex;
   var nperm("nperm", 0, Nperms),
	r("r", 0, B2Nrows),
	rp("rp", 0, B2Nrows),
	q("q", 0, Nq),
	s("s", 0, 2),
	to("to", 0, 2),
	wnum("wnum", 0, Nw2),
	wnumHex("wnumHex", 0, Nw2Hex),
	wnumHexHex("wnumHexHex", 0, Nw2Hex),
	wnumBlock("wnumBlock", 0, Nw),
        t("t", 0, Lt),
	x("x", 0, Vsnk),
	x_out("x_out", 0, Vsnk/sites_per_rank),
	x_in("x_in", 0, sites_per_rank),
	x2("x2", 0, Vsnk),
        y("y", 0, Vsrc),
	y_out("y_out", 0, Vsrc/src_sites_per_rank),
	y_in("y_in", 0, src_sites_per_rank),
	m("m", 0, Nsrc),
	n("n", 0, Nsnk),
	ne("ne", 0, NEntangled),
	nue("nue", 0, Nsnk-NEntangled),
	mH("mH", 0, NsrcHex),
	nH("nH", 0, NsnkHex),
	mpmH("mpmH", 0, NsrcTot),
	npnH("npnH", 0, NsnkTot),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);

   input C_r("C_r",      {t, x_out, rp, mpmH, r, npnH}, p_float64);
   input C_i("C_i",      {t, x_out, rp, mpmH, r, npnH}, p_float64);

   input B1_prop_r("B1_prop_r",   {t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);

   input src_psi_B1_r("src_psi_B1_r",    {y, m}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {y, m}, p_float64);
   input src_psi_B2_r("src_psi_B2_r",    {y, m}, p_float64);
   input src_psi_B2_i("src_psi_B2_i",    {y, m}, p_float64);
   input snk_psi_B1_r("snk_psi_B1_r",    {x, n}, p_float64);
   input snk_psi_B1_i("snk_psi_B1_i",    {x, n}, p_float64);
   input snk_psi_B2_r("snk_psi_B2_r",    {x, n}, p_float64);
   input snk_psi_B2_i("snk_psi_B2_i",    {x, n}, p_float64);
   input hex_src_psi_r("hex_src_psi_r",    {y, mH}, p_float64);
   input hex_src_psi_i("hex_src_psi_i",    {y, mH}, p_float64);
   input hex_snk_psi_r("hex_snk_psi_r",    {x, nH}, p_float64);
   input hex_snk_psi_i("hex_snk_psi_i",    {x, nH}, p_float64);
   // computation is local in x1, only rank=x_out components of sink wavefunction stored
   input snk_psi_r("snk_psi_r", {x_in, x2, ne}, p_float64);
   input snk_psi_i("snk_psi_i", {x_in, x2, ne}, p_float64);
   //input snk_psi_r("snk_psi_r", {x, x2, ne}, p_float64);
   //input snk_psi_i("snk_psi_i", {x, x2, ne}, p_float64);

   input src_spins("src_spins", {rp, s, to}, p_int32);
   input src_spin_block_weights("src_spin_block_weights", {rp, s}, p_float64);
   input sigs("sigs", {nperm}, p_int32);
   input snk_b("snk_b", {nperm, q, to}, p_int32);
   input src_color_weights("src_color_weights", {r, wnumBlock, q}, p_int32);
   input src_spin_weights("src_spin_weights", {r, wnumBlock, q}, p_int32);
   input src_weights("src_weights", {r, wnumBlock}, p_float64);
   input snk_color_weights("snk_color_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);
   input hex_snk_color_weights("hex_snk_color_weights", {r, nperm, wnumHex, q, to}, p_int32);
   input hex_snk_spin_weights("hex_snk_spin_weights", {r, nperm, wnumHex, q, to}, p_int32);
   input hex_snk_weights("hex_snk_weights", {r, wnumHex}, p_float64);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));
    complex_expr src_psi_B2(src_psi_B2_r(y, m), src_psi_B2_i(y, m));

    complex_expr snk_psi_B1(snk_psi_B1_r(x, n), snk_psi_B1_i(x, n));
    complex_expr snk_psi_B2(snk_psi_B2_r(x, n), snk_psi_B2_i(x, n));
    complex_expr snk_psi_B1_x2(snk_psi_B1_r(x2, n), snk_psi_B1_i(x2, n));
    complex_expr snk_psi_B2_x2(snk_psi_B2_r(x2, n), snk_psi_B2_i(x2, n));

    complex_expr snk_psi_B1_ue(snk_psi_B1_r(x_out*sites_per_rank+x_in, NEntangled+nue), snk_psi_B1_i(x_out*sites_per_rank+x_in, NEntangled+nue));
    complex_expr snk_psi_B2_ue(snk_psi_B2_r(x_out*sites_per_rank+x_in, NEntangled+nue), snk_psi_B2_i(x_out*sites_per_rank+x_in, NEntangled+nue));
    complex_expr snk_psi_B1_x2_ue(snk_psi_B1_r(x2, NEntangled+nue), snk_psi_B1_i(x2, NEntangled+nue));
    complex_expr snk_psi_B2_x2_ue(snk_psi_B2_r(x2, NEntangled+nue), snk_psi_B2_i(x2, NEntangled+nue));

    complex_expr hex_src_psi(hex_src_psi_r(y_out*src_sites_per_rank+y_in, mH), hex_src_psi_i(y_out*src_sites_per_rank+y_in, mH));
    complex_expr hex_hex_src_psi(hex_src_psi_r(y, mH), hex_src_psi_i(y, mH));
    complex_expr hex_snk_psi(hex_snk_psi_r(x_out*sites_per_rank+x_in, nH), hex_snk_psi_i(x_out*sites_per_rank+x_in, nH));

    //complex_expr snk_psi(snk_psi_r(x_out*sites_per_rank+x_in, x2, ne), snk_psi_i(x_out*sites_per_rank+x_in, x2, ne));
    complex_expr snk_psi(snk_psi_r(x_in, x2, ne), snk_psi_i(x_in, x2, ne));

     /* Baryon blocks */

     // Computing B1_Blocal_r1, B1_Bsecond_r1, B1_Bfirst_r1

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bfirst_r1_r_init("B1_Bfirst_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bfirst_r1_i_init("B1_Bfirst_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bsecond_r1_r_init("B1_Bsecond_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bsecond_r1_i_init("B1_Bsecond_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bthird_r1_r_init("B1_Bthird_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bthird_r1_i_init("B1_Bthird_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B1_Blocal_r1_init(&B1_Blocal_r1_r_init, &B1_Blocal_r1_i_init);
    complex_computation B1_Bfirst_r1_init(&B1_Bfirst_r1_r_init, &B1_Bfirst_r1_i_init);
    complex_computation B1_Bsecond_r1_init(&B1_Bsecond_r1_r_init, &B1_Bsecond_r1_i_init);
    complex_computation B1_Bthird_r1_init(&B1_Bthird_r1_r_init, &B1_Bthird_r1_i_init);

    computation flip_B1_Blocal_r1_r_init("flip_B1_Blocal_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Blocal_r1_i_init("flip_B1_Blocal_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bfirst_r1_r_init("flip_B1_Bfirst_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bfirst_r1_i_init("flip_B1_Bfirst_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bsecond_r1_r_init("flip_B1_Bsecond_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bsecond_r1_i_init("flip_B1_Bsecond_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bthird_r1_r_init("flip_B1_Bthird_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bthird_r1_i_init("flip_B1_Bthird_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_B1_Blocal_r1_init(&flip_B1_Blocal_r1_r_init, &flip_B1_Blocal_r1_i_init);
    complex_computation flip_B1_Bfirst_r1_init(&flip_B1_Bfirst_r1_r_init, &flip_B1_Bfirst_r1_i_init);
    complex_computation flip_B1_Bsecond_r1_init(&flip_B1_Bsecond_r1_r_init, &flip_B1_Bsecond_r1_i_init);
    complex_computation flip_B1_Bthird_r1_init(&flip_B1_Bthird_r1_r_init, &flip_B1_Bthird_r1_i_init);

    complex_expr B1_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr first_B1_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x2, y);
    complex_expr second_B1_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x2, y);
    complex_expr third_B1_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x2, y);

    complex_expr B1_r1_diquark = ( B1_r1_prop_0 * B1_r1_prop_2 ) *  src_weights(0, wnumBlock);
    complex_expr first_B1_r1_diquark = ( first_B1_r1_prop_0 * B1_r1_prop_2 ) *  src_weights(0, wnumBlock);
    complex_expr third_B1_r1_diquark = ( B1_r1_prop_0 * third_B1_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation B1_Blocal_r1_r_props_init("B1_Blocal_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_props_init("B1_Blocal_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bfirst_r1_r_props_init("B1_Bfirst_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bfirst_r1_i_props_init("B1_Bfirst_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsecond_r1_r_props_init("B1_Bsecond_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsecond_r1_i_props_init("B1_Bsecond_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bthird_r1_r_props_init("B1_Bthird_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bthird_r1_i_props_init("B1_Bthird_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B1_Blocal_r1_r_diquark("B1_Blocal_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_real());
    computation B1_Blocal_r1_i_diquark("B1_Blocal_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_imag());
    computation B1_Bfirst_r1_r_diquark("B1_Bfirst_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B1_r1_diquark.get_real());
    computation B1_Bfirst_r1_i_diquark("B1_Bfirst_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B1_r1_diquark.get_imag());
    computation B1_Bthird_r1_r_diquark("B1_Bthird_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B1_r1_diquark.get_real());
    computation B1_Bthird_r1_i_diquark("B1_Bthird_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B1_r1_diquark.get_imag());

    complex_computation B1_Blocal_r1_diquark(&B1_Blocal_r1_r_diquark, &B1_Blocal_r1_i_diquark);
    complex_computation B1_Bfirst_r1_diquark(&B1_Bfirst_r1_r_diquark, &B1_Bfirst_r1_i_diquark);
    complex_computation B1_Bthird_r1_diquark(&B1_Bthird_r1_r_diquark, &B1_Bthird_r1_i_diquark);

    complex_expr B1_r1_props = B1_r1_prop_1 * B1_Blocal_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr first_B1_r1_props = B1_r1_prop_1 * B1_Bfirst_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr second_B1_r1_props = second_B1_r1_prop_1 * B1_Blocal_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr third_B1_r1_props = B1_r1_prop_1 * B1_Bthird_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r1_r_props("B1_Blocal_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r1_props.get_real());
    computation B1_Blocal_r1_i_props("B1_Blocal_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r1_props.get_imag());
    computation B1_Bfirst_r1_r_props("B1_Bfirst_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bfirst_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B1_r1_props.get_real());
     computation B1_Bfirst_r1_i_props("B1_Bfirst_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bfirst_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B1_r1_props.get_imag());
    computation B1_Bsecond_r1_r_props("B1_Bsecond_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bsecond_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B1_r1_props.get_real());
    computation B1_Bsecond_r1_i_props("B1_Bsecond_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bsecond_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B1_r1_props.get_imag());
    computation B1_Bthird_r1_r_props("B1_Bthird_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bthird_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B1_r1_props.get_real());
     computation B1_Bthird_r1_i_props("B1_Bthird_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bthird_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B1_r1_props.get_imag());

     complex_computation B1_Blocal_r1_props(&B1_Blocal_r1_r_props, &B1_Blocal_r1_i_props);
     complex_computation B1_Bfirst_r1_props(&B1_Bfirst_r1_r_props, &B1_Bfirst_r1_i_props);
     complex_computation B1_Bsecond_r1_props(&B1_Bsecond_r1_r_props, &B1_Bsecond_r1_i_props);
     complex_computation B1_Bthird_r1_props(&B1_Bthird_r1_r_props, &B1_Bthird_r1_i_props);

    complex_expr B1_r1 = src_psi_B1 * B1_Blocal_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr first_B1_r1 = src_psi_B1 * B1_Bfirst_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr second_B1_r1 = src_psi_B1 * B1_Bsecond_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr third_B1_r1 = src_psi_B1 * B1_Bthird_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r1_r_update("B1_Blocal_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_real());
    computation B1_Blocal_r1_i_update("B1_Blocal_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_imag());
    computation B1_Bfirst_r1_r_update("B1_Bfirst_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bfirst_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B1_r1.get_real());
    computation B1_Bfirst_r1_i_update("B1_Bfirst_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bfirst_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B1_r1.get_imag()); 
    computation B1_Bsecond_r1_r_update("B1_Bsecond_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bsecond_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B1_r1.get_real());
    computation B1_Bsecond_r1_i_update("B1_Bsecond_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bsecond_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B1_r1.get_imag());
    computation B1_Bthird_r1_r_update("B1_Bthird_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bthird_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B1_r1.get_real());
    computation B1_Bthird_r1_i_update("B1_Bthird_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bthird_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B1_r1.get_imag()); 

    complex_expr flip_B1_r1 = src_psi_B2 * B1_Blocal_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_first_B1_r1 = src_psi_B2 * B1_Bfirst_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_second_B1_r1 = src_psi_B2 * B1_Bsecond_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_third_B1_r1 = src_psi_B2 * B1_Bthird_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_B1_Blocal_r1_r_update("flip_B1_Blocal_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Blocal_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B1_r1.get_real());
    computation flip_B1_Blocal_r1_i_update("flip_B1_Blocal_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Blocal_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B1_r1.get_imag());
    computation flip_B1_Bfirst_r1_r_update("flip_B1_Bfirst_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bfirst_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B1_r1.get_real());
    computation flip_B1_Bfirst_r1_i_update("flip_B1_Bfirst_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bfirst_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B1_r1.get_imag()); 
    computation flip_B1_Bsecond_r1_r_update("flip_B1_Bsecond_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bsecond_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B1_r1.get_real());
    computation flip_B1_Bsecond_r1_i_update("flip_B1_Bsecond_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bsecond_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B1_r1.get_imag());
    computation flip_B1_Bthird_r1_r_update("flip_B1_Bthird_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bthird_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B1_r1.get_real());
    computation flip_B1_Bthird_r1_i_update("flip_B1_Bthird_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bthird_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B1_r1.get_imag()); 

     // Computing B1_Blocal_r2, B1_Bsecond_r2, B1_Bfirst_r2

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bfirst_r2_r_init("B1_Bfirst_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bfirst_r2_i_init("B1_Bfirst_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bsecond_r2_r_init("B1_Bsecond_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bsecond_r2_i_init("B1_Bsecond_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bthird_r2_r_init("B1_Bthird_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B1_Bthird_r2_i_init("B1_Bthird_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B1_Blocal_r2_init(&B1_Blocal_r2_r_init, &B1_Blocal_r2_i_init);
    complex_computation B1_Bfirst_r2_init(&B1_Bfirst_r2_r_init, &B1_Bfirst_r2_i_init);
    complex_computation B1_Bsecond_r2_init(&B1_Bsecond_r2_r_init, &B1_Bsecond_r2_i_init);
    complex_computation B1_Bthird_r2_init(&B1_Bthird_r2_r_init, &B1_Bthird_r2_i_init);

    computation flip_B1_Blocal_r2_r_init("flip_B1_Blocal_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Blocal_r2_i_init("flip_B1_Blocal_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bfirst_r2_r_init("flip_B1_Bfirst_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bfirst_r2_i_init("flip_B1_Bfirst_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bsecond_r2_r_init("flip_B1_Bsecond_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bsecond_r2_i_init("flip_B1_Bsecond_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bthird_r2_r_init("flip_B1_Bthird_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B1_Bthird_r2_i_init("flip_B1_Bthird_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_B1_Blocal_r2_init(&flip_B1_Blocal_r2_r_init, &flip_B1_Blocal_r2_i_init);
    complex_computation flip_B1_Bfirst_r2_init(&flip_B1_Bfirst_r2_r_init, &flip_B1_Bfirst_r2_i_init);
    complex_computation flip_B1_Bsecond_r2_init(&flip_B1_Bsecond_r2_r_init, &flip_B1_Bsecond_r2_i_init);
    complex_computation flip_B1_Bthird_r2_init(&flip_B1_Bthird_r2_r_init, &flip_B1_Bthird_r2_i_init);

    complex_expr B1_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);
    complex_expr first_B1_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x2, y);
    complex_expr second_B1_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x2, y);
    complex_expr third_B1_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x2, y);

    complex_expr B1_r2_diquark = ( B1_r2_prop_0 * B1_r2_prop_2 ) *  src_weights(1, wnumBlock);
    complex_expr first_B1_r2_diquark = ( first_B1_r2_prop_0 * B1_r2_prop_2 ) *  src_weights(1, wnumBlock);
    complex_expr third_B1_r2_diquark = ( B1_r2_prop_0 * third_B1_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation B1_Blocal_r2_r_props_init("B1_Blocal_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r2_i_props_init("B1_Blocal_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bfirst_r2_r_props_init("B1_Bfirst_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bfirst_r2_i_props_init("B1_Bfirst_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsecond_r2_r_props_init("B1_Bsecond_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsecond_r2_i_props_init("B1_Bsecond_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bthird_r2_r_props_init("B1_Bthird_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B1_Bthird_r2_i_props_init("B1_Bthird_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B1_Blocal_r2_r_diquark("B1_Blocal_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_real());
    computation B1_Blocal_r2_i_diquark("B1_Blocal_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_imag());
    computation B1_Bfirst_r2_r_diquark("B1_Bfirst_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B1_r2_diquark.get_real());
    computation B1_Bfirst_r2_i_diquark("B1_Bfirst_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B1_r2_diquark.get_imag());
    computation B1_Bthird_r2_r_diquark("B1_Bthird_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B1_r2_diquark.get_real());
    computation B1_Bthird_r2_i_diquark("B1_Bthird_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B1_r2_diquark.get_imag());

    complex_computation B1_Blocal_r2_diquark(&B1_Blocal_r2_r_diquark, &B1_Blocal_r2_i_diquark);
    complex_computation B1_Bfirst_r2_diquark(&B1_Bfirst_r2_r_diquark, &B1_Bfirst_r2_i_diquark);
    complex_computation B1_Bthird_r2_diquark(&B1_Bthird_r2_r_diquark, &B1_Bthird_r2_i_diquark);

    complex_expr B1_r2_props = B1_r2_prop_1 * B1_Blocal_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr first_B1_r2_props = B1_r2_prop_1 * B1_Bfirst_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr second_B1_r2_props = second_B1_r2_prop_1 * B1_Blocal_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr third_B1_r2_props = B1_r2_prop_1 * B1_Bthird_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r2_r_props("B1_Blocal_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_real());
    computation B1_Blocal_r2_i_props("B1_Blocal_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_imag());
    computation B1_Bfirst_r2_r_props("B1_Bfirst_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bfirst_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B1_r2_props.get_real());
    computation B1_Bfirst_r2_i_props("B1_Bfirst_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bfirst_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B1_r2_props.get_imag());
    computation B1_Bsecond_r2_r_props("B1_Bsecond_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bsecond_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B1_r2_props.get_real());
    computation B1_Bsecond_r2_i_props("B1_Bsecond_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bsecond_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B1_r2_props.get_imag());
    computation B1_Bthird_r2_r_props("B1_Bthird_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bthird_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B1_r2_props.get_real());
    computation B1_Bthird_r2_i_props("B1_Bthird_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Bthird_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B1_r2_props.get_imag());

    complex_computation B1_Blocal_r2_props(&B1_Blocal_r2_r_props, &B1_Blocal_r2_i_props);
    complex_computation B1_Bfirst_r2_props(&B1_Bfirst_r2_r_props, &B1_Bfirst_r2_i_props);
    complex_computation B1_Bsecond_r2_props(&B1_Bsecond_r2_r_props, &B1_Bsecond_r2_i_props);
    complex_computation B1_Bthird_r2_props(&B1_Bthird_r2_r_props, &B1_Bthird_r2_i_props);

    complex_expr B1_r2 = src_psi_B1 * B1_Blocal_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr first_B1_r2 = src_psi_B1 * B1_Bfirst_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr second_B1_r2 = src_psi_B1 * B1_Bsecond_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr third_B1_r2 = src_psi_B1 * B1_Bthird_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r2_r_update("B1_Blocal_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_real());
    computation B1_Blocal_r2_i_update("B1_Blocal_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_imag());
    computation B1_Bfirst_r2_r_update("B1_Bfirst_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bfirst_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B1_r2.get_real());
    computation B1_Bfirst_r2_i_update("B1_Bfirst_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bfirst_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B1_r2.get_imag());
    computation B1_Bsecond_r2_r_update("B1_Bsecond_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bsecond_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B1_r2.get_real());
    computation B1_Bsecond_r2_i_update("B1_Bsecond_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bsecond_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B1_r2.get_imag());
    computation B1_Bthird_r2_r_update("B1_Bthird_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bthird_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B1_r2.get_real());
    computation B1_Bthird_r2_i_update("B1_Bthird_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Bthird_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B1_r2.get_imag());

    complex_expr flip_B1_r2 = src_psi_B2 * B1_Blocal_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_first_B1_r2 = src_psi_B2 * B1_Bfirst_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_second_B1_r2 = src_psi_B2 * B1_Bsecond_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_third_B1_r2 = src_psi_B2 * B1_Bthird_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_B1_Blocal_r2_r_update("flip_B1_Blocal_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Blocal_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B1_r2.get_real());
    computation flip_B1_Blocal_r2_i_update("flip_B1_Blocal_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Blocal_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B1_r2.get_imag());
    computation flip_B1_Bfirst_r2_r_update("flip_B1_Bfirst_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bfirst_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B1_r2.get_real());
    computation flip_B1_Bfirst_r2_i_update("flip_B1_Bfirst_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bfirst_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B1_r2.get_imag()); 
    computation flip_B1_Bsecond_r2_r_update("flip_B1_Bsecond_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bsecond_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B1_r2.get_real());
    computation flip_B1_Bsecond_r2_i_update("flip_B1_Bsecond_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bsecond_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B1_r2.get_imag());
    computation flip_B1_Bthird_r2_r_update("flip_B1_Bthird_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bthird_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B1_r2.get_real());
    computation flip_B1_Bthird_r2_i_update("flip_B1_Bthird_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B1_Bthird_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B1_r2.get_imag()); 

     //Computing B2_Blocal_r1, B2_Bsecond_r1, B2_Bfirst_r1

    computation B2_Blocal_r1_r_init("B2_Blocal_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Blocal_r1_i_init("B2_Blocal_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bfirst_r1_r_init("B2_Bfirst_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bfirst_r1_i_init("B2_Bfirst_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bsecond_r1_r_init("B2_Bsecond_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bsecond_r1_i_init("B2_Bsecond_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bthird_r1_r_init("B2_Bthird_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bthird_r1_i_init("B2_Bthird_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B2_Blocal_r1_init(&B2_Blocal_r1_r_init, &B2_Blocal_r1_i_init);
    complex_computation B2_Bfirst_r1_init(&B2_Bfirst_r1_r_init, &B2_Bfirst_r1_i_init);
    complex_computation B2_Bsecond_r1_init(&B2_Bsecond_r1_r_init, &B2_Bsecond_r1_i_init);
    complex_computation B2_Bthird_r1_init(&B2_Bthird_r1_r_init, &B2_Bthird_r1_i_init);

    computation flip_B2_Blocal_r1_r_init("flip_B2_Blocal_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Blocal_r1_i_init("flip_B2_Blocal_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bfirst_r1_r_init("flip_B2_Bfirst_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bfirst_r1_i_init("flip_B2_Bfirst_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bsecond_r1_r_init("flip_B2_Bsecond_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bsecond_r1_i_init("flip_B2_Bsecond_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bthird_r1_r_init("flip_B2_Bthird_r1_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bthird_r1_i_init("flip_B2_Bthird_r1_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_B2_Blocal_r1_init(&flip_B2_Blocal_r1_r_init, &flip_B2_Blocal_r1_i_init);
    complex_computation flip_B2_Bfirst_r1_init(&flip_B2_Bfirst_r1_r_init, &flip_B2_Bfirst_r1_i_init);
    complex_computation flip_B2_Bsecond_r1_init(&flip_B2_Bsecond_r1_r_init, &flip_B2_Bsecond_r1_i_init);
    complex_computation flip_B2_Bthird_r1_init(&flip_B2_Bthird_r1_r_init, &flip_B2_Bthird_r1_i_init);

    complex_expr B2_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x2, y);
    complex_expr B2_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x2, y);
    complex_expr B2_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x2, y);
    complex_expr first_B2_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr second_B2_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);
    complex_expr third_B2_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);

    complex_expr B2_r1_diquark = ( B2_r1_prop_0 * B2_r1_prop_2 ) *  src_weights(0, wnumBlock);
    complex_expr first_B2_r1_diquark = ( first_B2_r1_prop_0 * B2_r1_prop_2 ) *  src_weights(0, wnumBlock);
    complex_expr third_B2_r1_diquark = ( B2_r1_prop_0 * third_B2_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation B2_Blocal_r1_r_props_init("B2_Blocal_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r1_i_props_init("B2_Blocal_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bfirst_r1_r_props_init("B2_Bfirst_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bfirst_r1_i_props_init("B2_Bfirst_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsecond_r1_r_props_init("B2_Bsecond_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsecond_r1_i_props_init("B2_Bsecond_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bthird_r1_r_props_init("B2_Bthird_r1_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bthird_r1_i_props_init("B2_Bthird_r1_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B2_Blocal_r1_r_diquark("B2_Blocal_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B2_r1_diquark.get_real());
    computation B2_Blocal_r1_i_diquark("B2_Blocal_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B2_r1_diquark.get_imag());
    computation B2_Bfirst_r1_r_diquark("B2_Bfirst_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B2_r1_diquark.get_real());
    computation B2_Bfirst_r1_i_diquark("B2_Bfirst_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B2_r1_diquark.get_imag());
    computation B2_Bthird_r1_r_diquark("B2_Bthird_r1_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B2_r1_diquark.get_real());
    computation B2_Bthird_r1_i_diquark("B2_Bthird_r1_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B2_r1_diquark.get_imag());

    complex_computation B2_Blocal_r1_diquark(&B2_Blocal_r1_r_diquark, &B2_Blocal_r1_i_diquark);
    complex_computation B2_Bfirst_r1_diquark(&B2_Bfirst_r1_r_diquark, &B2_Bfirst_r1_i_diquark);
    complex_computation B2_Bthird_r1_diquark(&B2_Bthird_r1_r_diquark, &B2_Bthird_r1_i_diquark);

    complex_expr B2_r1_props = B2_r1_prop_1 * B2_Blocal_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr first_B2_r1_props = B2_r1_prop_1 * B2_Bfirst_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr second_B2_r1_props = second_B2_r1_prop_1 * B2_Blocal_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr third_B2_r1_props = B2_r1_prop_1 * B2_Bthird_r1_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B2_Blocal_r1_r_props("B2_Blocal_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Blocal_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B2_r1_props.get_real());
    computation B2_Blocal_r1_i_props("B2_Blocal_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Blocal_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B2_r1_props.get_imag());
    computation B2_Bfirst_r1_r_props("B2_Bfirst_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bfirst_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B2_r1_props.get_real());
    computation B2_Bfirst_r1_i_props("B2_Bfirst_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bfirst_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B2_r1_props.get_imag());
    computation B2_Bsecond_r1_r_props("B2_Bsecond_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bsecond_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B2_r1_props.get_real());
    computation B2_Bsecond_r1_i_props("B2_Bsecond_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bsecond_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B2_r1_props.get_imag());
    computation B2_Bthird_r1_r_props("B2_Bthird_r1_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bthird_r1_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B2_r1_props.get_real());
    computation B2_Bthird_r1_i_props("B2_Bthird_r1_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bthird_r1_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B2_r1_props.get_imag());

    complex_computation B2_Blocal_r1_props(&B2_Blocal_r1_r_props, &B2_Blocal_r1_i_props);
    complex_computation B2_Bfirst_r1_props(&B2_Bfirst_r1_r_props, &B2_Bfirst_r1_i_props);
    complex_computation B2_Bsecond_r1_props(&B2_Bsecond_r1_r_props, &B2_Bsecond_r1_i_props);
    complex_computation B2_Bthird_r1_props(&B2_Bthird_r1_r_props, &B2_Bthird_r1_i_props);

    complex_expr B2_r1 = src_psi_B2 * B2_Blocal_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr first_B2_r1 = src_psi_B2 * B2_Bfirst_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr second_B2_r1 = src_psi_B2 * B2_Bsecond_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr third_B2_r1 = src_psi_B2 * B2_Bthird_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B2_Blocal_r1_r_update("B2_Blocal_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Blocal_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B2_r1.get_real());
    computation B2_Blocal_r1_i_update("B2_Blocal_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Blocal_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B2_r1.get_imag());
    computation B2_Bfirst_r1_r_update("B2_Bfirst_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bfirst_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B2_r1.get_real());
    computation B2_Bfirst_r1_i_update("B2_Bfirst_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bfirst_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B2_r1.get_imag()); 
    computation B2_Bsecond_r1_r_update("B2_Bsecond_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bsecond_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B2_r1.get_real());
    computation B2_Bsecond_r1_i_update("B2_Bsecond_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bsecond_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B2_r1.get_imag());
    computation B2_Bthird_r1_r_update("B2_Bthird_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bthird_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B2_r1.get_real());
    computation B2_Bthird_r1_i_update("B2_Bthird_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bthird_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B2_r1.get_imag()); 

    complex_expr flip_B2_r1 = src_psi_B1 * B2_Blocal_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_first_B2_r1 = src_psi_B1 * B2_Bfirst_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_second_B2_r1 = src_psi_B1 * B2_Bsecond_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_third_B2_r1 = src_psi_B1 * B2_Bthird_r1_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_B2_Blocal_r1_r_update("flip_B2_Blocal_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Blocal_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B2_r1.get_real());
    computation flip_B2_Blocal_r1_i_update("flip_B2_Blocal_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Blocal_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B2_r1.get_imag());
    computation flip_B2_Bfirst_r1_r_update("flip_B2_Bfirst_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bfirst_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B2_r1.get_real());
    computation flip_B2_Bfirst_r1_i_update("flip_B2_Bfirst_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bfirst_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B2_r1.get_imag()); 
    computation flip_B2_Bsecond_r1_r_update("flip_B2_Bsecond_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bsecond_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B2_r1.get_real());
    computation flip_B2_Bsecond_r1_i_update("flip_B2_Bsecond_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bsecond_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B2_r1.get_imag());
    computation flip_B2_Bthird_r1_r_update("flip_B2_Bthird_r1_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bthird_r1_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B2_r1.get_real());
    computation flip_B2_Bthird_r1_i_update("flip_B2_Bthird_r1_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bthird_r1_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B2_r1.get_imag()); 

     // Computing B2_Blocal_r2, B2_Bsecond_r2, B2_Bfirst_r2

    computation B2_Blocal_r2_r_init("B2_Blocal_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Blocal_r2_i_init("B2_Blocal_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bfirst_r2_r_init("B2_Bfirst_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bfirst_r2_i_init("B2_Bfirst_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bsecond_r2_r_init("B2_Bsecond_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bsecond_r2_i_init("B2_Bsecond_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bthird_r2_r_init("B2_Bthird_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation B2_Bthird_r2_i_init("B2_Bthird_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation B2_Blocal_r2_init(&B2_Blocal_r2_r_init, &B2_Blocal_r2_i_init);
    complex_computation B2_Bfirst_r2_init(&B2_Bfirst_r2_r_init, &B2_Bfirst_r2_i_init);
    complex_computation B2_Bsecond_r2_init(&B2_Bsecond_r2_r_init, &B2_Bsecond_r2_i_init);
    complex_computation B2_Bthird_r2_init(&B2_Bthird_r2_r_init, &B2_Bthird_r2_i_init);

    computation flip_B2_Blocal_r2_r_init("flip_B2_Blocal_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Blocal_r2_i_init("flip_B2_Blocal_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bfirst_r2_r_init("flip_B2_Bfirst_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bfirst_r2_i_init("flip_B2_Bfirst_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bsecond_r2_r_init("flip_B2_Bsecond_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bsecond_r2_i_init("flip_B2_Bsecond_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bthird_r2_r_init("flip_B2_Bthird_r2_r_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_B2_Bthird_r2_i_init("flip_B2_Bthird_r2_i_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_B2_Blocal_r2_init(&flip_B2_Blocal_r2_r_init, &flip_B2_Blocal_r2_i_init);
    complex_computation flip_B2_Bfirst_r2_init(&flip_B2_Bfirst_r2_r_init, &flip_B2_Bfirst_r2_i_init);
    complex_computation flip_B2_Bsecond_r2_init(&flip_B2_Bsecond_r2_r_init, &flip_B2_Bsecond_r2_i_init);
    complex_computation flip_B2_Bthird_r2_init(&flip_B2_Bthird_r2_r_init, &flip_B2_Bthird_r2_i_init);

    complex_expr B2_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x2, y);
    complex_expr B2_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x2, y);
    complex_expr B2_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x2, y);
    complex_expr first_B2_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr second_B2_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);
    complex_expr third_B2_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);

    complex_expr B2_r2_diquark = ( B2_r2_prop_0 * B2_r2_prop_2 ) *  src_weights(1, wnumBlock);
    complex_expr first_B2_r2_diquark = ( first_B2_r2_prop_0 * B2_r2_prop_2 ) *  src_weights(1, wnumBlock);
    complex_expr third_B2_r2_diquark = ( B2_r2_prop_0 * third_B2_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation B2_Blocal_r2_r_props_init("B2_Blocal_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r2_i_props_init("B2_Blocal_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bfirst_r2_r_props_init("B2_Bfirst_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bfirst_r2_i_props_init("B2_Bfirst_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsecond_r2_r_props_init("B2_Bsecond_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsecond_r2_i_props_init("B2_Bsecond_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bthird_r2_r_props_init("B2_Bthird_r2_r_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation B2_Bthird_r2_i_props_init("B2_Bthird_r2_i_props_init", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation B2_Blocal_r2_r_diquark("B2_Blocal_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B2_r2_diquark.get_real());
    computation B2_Blocal_r2_i_diquark("B2_Blocal_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B2_r2_diquark.get_imag());
    computation B2_Bfirst_r2_r_diquark("B2_Bfirst_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B2_r2_diquark.get_real());
    computation B2_Bfirst_r2_i_diquark("B2_Bfirst_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, first_B2_r2_diquark.get_imag());
    computation B2_Bthird_r2_r_diquark("B2_Bthird_r2_r_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B2_r2_diquark.get_real());
    computation B2_Bthird_r2_i_diquark("B2_Bthird_r2_i_diquark", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, third_B2_r2_diquark.get_imag());

    complex_computation B2_Blocal_r2_diquark(&B2_Blocal_r2_r_diquark, &B2_Blocal_r2_i_diquark);
    complex_computation B2_Bfirst_r2_diquark(&B2_Bfirst_r2_r_diquark, &B2_Bfirst_r2_i_diquark);
    complex_computation B2_Bthird_r2_diquark(&B2_Bthird_r2_r_diquark, &B2_Bthird_r2_i_diquark);

    complex_expr B2_r2_props = B2_r2_prop_1 * B2_Blocal_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr first_B2_r2_props = B2_r2_prop_1 * B2_Bfirst_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr second_B2_r2_props = second_B2_r2_prop_1 * B2_Blocal_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);
    complex_expr third_B2_r2_props = B2_r2_prop_1 * B2_Bthird_r2_diquark(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B2_Blocal_r2_r_props("B2_Blocal_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Blocal_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B2_r2_props.get_real());
    computation B2_Blocal_r2_i_props("B2_Blocal_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Blocal_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B2_r2_props.get_imag());
    computation B2_Bfirst_r2_r_props("B2_Bfirst_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bfirst_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B2_r2_props.get_real());
    computation B2_Bfirst_r2_i_props("B2_Bfirst_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bfirst_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + first_B2_r2_props.get_imag());
    computation B2_Bsecond_r2_r_props("B2_Bsecond_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bsecond_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B2_r2_props.get_real());
    computation B2_Bsecond_r2_i_props("B2_Bsecond_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bsecond_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + second_B2_r2_props.get_imag());
    computation B2_Bthird_r2_r_props("B2_Bthird_r2_r_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bthird_r2_r_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B2_r2_props.get_real());
    computation B2_Bthird_r2_i_props("B2_Bthird_r2_i_props", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B2_Bthird_r2_i_props_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + third_B2_r2_props.get_imag());

    complex_computation B2_Blocal_r2_props(&B2_Blocal_r2_r_props, &B2_Blocal_r2_i_props);
    complex_computation B2_Bfirst_r2_props(&B2_Bfirst_r2_r_props, &B2_Bfirst_r2_i_props);
    complex_computation B2_Bsecond_r2_props(&B2_Bsecond_r2_r_props, &B2_Bsecond_r2_i_props);
    complex_computation B2_Bthird_r2_props(&B2_Bthird_r2_r_props, &B2_Bthird_r2_i_props);

    complex_expr B2_r2 = src_psi_B2 * B2_Blocal_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr first_B2_r2 = src_psi_B2 * B2_Bfirst_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr second_B2_r2 = src_psi_B2 * B2_Bsecond_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr third_B2_r2 = src_psi_B2 * B2_Bthird_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B2_Blocal_r2_r_update("B2_Blocal_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Blocal_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B2_r2.get_real());
    computation B2_Blocal_r2_i_update("B2_Blocal_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Blocal_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B2_r2.get_imag());
    computation B2_Bfirst_r2_r_update("B2_Bfirst_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bfirst_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B2_r2.get_real());
    computation B2_Bfirst_r2_i_update("B2_Bfirst_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bfirst_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + first_B2_r2.get_imag());
    computation B2_Bsecond_r2_r_update("B2_Bsecond_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bsecond_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B2_r2.get_real());
    computation B2_Bsecond_r2_i_update("B2_Bsecond_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bsecond_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + second_B2_r2.get_imag());
    computation B2_Bthird_r2_r_update("B2_Bthird_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bthird_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B2_r2.get_real());
    computation B2_Bthird_r2_i_update("B2_Bthird_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B2_Bthird_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + third_B2_r2.get_imag());

    complex_expr flip_B2_r2 = src_psi_B1 * B2_Blocal_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_first_B2_r2 = src_psi_B1 * B2_Bfirst_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_second_B2_r2 = src_psi_B1 * B2_Bsecond_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);
    complex_expr flip_third_B2_r2 = src_psi_B1 * B2_Bthird_r2_props(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_B2_Blocal_r2_r_update("flip_B2_Blocal_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Blocal_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B2_r2.get_real());
    computation flip_B2_Blocal_r2_i_update("flip_B2_Blocal_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Blocal_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_B2_r2.get_imag());
    computation flip_B2_Bfirst_r2_r_update("flip_B2_Bfirst_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bfirst_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B2_r2.get_real());
    computation flip_B2_Bfirst_r2_i_update("flip_B2_Bfirst_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bfirst_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_first_B2_r2.get_imag()); 
    computation flip_B2_Bsecond_r2_r_update("flip_B2_Bsecond_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bsecond_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B2_r2.get_real());
    computation flip_B2_Bsecond_r2_i_update("flip_B2_Bsecond_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bsecond_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_second_B2_r2.get_imag());
    computation flip_B2_Bthird_r2_r_update("flip_B2_Bthird_r2_r_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bthird_r2_r_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B2_r2.get_real());
    computation flip_B2_Bthird_r2_i_update("flip_B2_Bthird_r2_i_update", {t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_B2_Bthird_r2_i_init(t, x_out, x_in, x2, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_third_B2_r2.get_imag()); 
    
     // Computing src_B1_Blocal_r1

    computation src_B1_Blocal_r1_r_init("src_B1_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation src_B1_Blocal_r1_i_init("src_B1_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation src_B1_Blocal_r1_init(&src_B1_Blocal_r1_r_init, &src_B1_Blocal_r1_i_init);

    computation flip_src_B1_Blocal_r1_r_init("flip_src_B1_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_src_B1_Blocal_r1_i_init("flip_src_B1_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_src_B1_Blocal_r1_init(&flip_src_B1_Blocal_r1_r_init, &flip_src_B1_Blocal_r1_i_init);

    complex_expr src_B1_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr src_B1_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr src_B1_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr src_B1_r1_diquark = ( src_B1_r1_prop_0 * src_B1_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation src_B1_Blocal_r1_r_props_init("src_B1_Blocal_r1_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation src_B1_Blocal_r1_i_props_init("src_B1_Blocal_r1_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation src_B1_Blocal_r1_r_diquark("src_B1_Blocal_r1_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B1_r1_diquark.get_real());
    computation src_B1_Blocal_r1_i_diquark("src_B1_Blocal_r1_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B1_r1_diquark.get_imag());

    complex_computation src_B1_Blocal_r1_diquark(&src_B1_Blocal_r1_r_diquark, &src_B1_Blocal_r1_i_diquark);

    complex_expr src_B1_r1_props = src_B1_r1_prop_1 * src_B1_Blocal_r1_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation src_B1_Blocal_r1_r_props("src_B1_Blocal_r1_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B1_Blocal_r1_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B1_r1_props.get_real());
    computation src_B1_Blocal_r1_i_props("src_B1_Blocal_r1_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B1_Blocal_r1_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B1_r1_props.get_imag());

    complex_computation src_B1_Blocal_r1_props(&src_B1_Blocal_r1_r_props, &src_B1_Blocal_r1_i_props);

    complex_expr src_B1_r1 = src_psi_B1 * src_B1_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation src_B1_Blocal_r1_r_update("src_B1_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B1_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B1_r1.get_real());
    computation src_B1_Blocal_r1_i_update("src_B1_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B1_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B1_r1.get_imag()); 

    complex_expr flip_src_B1_r1 = src_psi_B2 * src_B1_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_src_B1_Blocal_r1_r_update("flip_src_B1_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B1_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B1_r1.get_real());
    computation flip_src_B1_Blocal_r1_i_update("flip_src_B1_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B1_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B1_r1.get_imag());

     // Computing src_B1_Blocal_r2

    computation src_B1_Blocal_r2_r_init("src_B1_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation src_B1_Blocal_r2_i_init("src_B1_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation src_B1_Blocal_r2_init(&src_B1_Blocal_r2_r_init, &src_B1_Blocal_r2_i_init);

    computation flip_src_B1_Blocal_r2_r_init("flip_src_B1_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_src_B1_Blocal_r2_i_init("flip_src_B1_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_src_B1_Blocal_r2_init(&flip_src_B1_Blocal_r2_r_init, &flip_src_B1_Blocal_r2_i_init);

    complex_expr src_B1_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr src_B1_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr src_B1_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr src_B1_r2_diquark = ( src_B1_r2_prop_0 * src_B1_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation src_B1_Blocal_r2_r_props_init("src_B1_Blocal_r2_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation src_B1_Blocal_r2_i_props_init("src_B1_Blocal_r2_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation src_B1_Blocal_r2_r_diquark("src_B1_Blocal_r2_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B1_r2_diquark.get_real());
    computation src_B1_Blocal_r2_i_diquark("src_B1_Blocal_r2_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B1_r2_diquark.get_imag());

    complex_computation src_B1_Blocal_r2_diquark(&src_B1_Blocal_r2_r_diquark, &src_B1_Blocal_r2_i_diquark);

    complex_expr src_B1_r2_props = src_B1_r2_prop_1 * src_B1_Blocal_r2_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation src_B1_Blocal_r2_r_props("src_B1_Blocal_r2_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B1_Blocal_r2_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B1_r2_props.get_real());
    computation src_B1_Blocal_r2_i_props("src_B1_Blocal_r2_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B1_Blocal_r2_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B1_r2_props.get_imag());

    complex_computation src_B1_Blocal_r2_props(&src_B1_Blocal_r2_r_props, &src_B1_Blocal_r2_i_props);

    complex_expr src_B1_r2 = src_psi_B1 * src_B1_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation src_B1_Blocal_r2_r_update("src_B1_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B1_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B1_r2.get_real());
    computation src_B1_Blocal_r2_i_update("src_B1_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B1_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B1_r2.get_imag());

    complex_expr flip_src_B1_r2 = src_psi_B2 * src_B1_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_src_B1_Blocal_r2_r_update("flip_src_B1_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B1_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B1_r2.get_real());
    computation flip_src_B1_Blocal_r2_i_update("flip_src_B1_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B1_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B1_r2.get_imag());

     // Computing src_B2_Blocal_r1

    computation src_B2_Blocal_r1_r_init("src_B2_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation src_B2_Blocal_r1_i_init("src_B2_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation src_B2_Blocal_r1_init(&src_B2_Blocal_r1_r_init, &src_B2_Blocal_r1_i_init);

    computation flip_src_B2_Blocal_r1_r_init("flip_src_B2_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_src_B2_Blocal_r1_i_init("flip_src_B2_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_src_B2_Blocal_r1_init(&flip_src_B2_Blocal_r1_r_init, &flip_src_B2_Blocal_r1_i_init);

    complex_expr src_B2_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr src_B2_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr src_B2_r1_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr src_B2_r1_diquark = ( src_B2_r1_prop_0 * src_B2_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation src_B2_Blocal_r1_r_props_init("src_B2_Blocal_r1_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation src_B2_Blocal_r1_i_props_init("src_B2_Blocal_r1_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation src_B2_Blocal_r1_r_diquark("src_B2_Blocal_r1_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B2_r1_diquark.get_real());
    computation src_B2_Blocal_r1_i_diquark("src_B2_Blocal_r1_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B2_r1_diquark.get_imag());

    complex_computation src_B2_Blocal_r1_diquark(&src_B2_Blocal_r1_r_diquark, &src_B2_Blocal_r1_i_diquark);

    complex_expr src_B2_r1_props = src_B2_r1_prop_1 * src_B2_Blocal_r1_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation src_B2_Blocal_r1_r_props("src_B2_Blocal_r1_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B2_Blocal_r1_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B2_r1_props.get_real());
    computation src_B2_Blocal_r1_i_props("src_B2_Blocal_r1_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B2_Blocal_r1_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B2_r1_props.get_imag());

    complex_computation src_B2_Blocal_r1_props(&src_B2_Blocal_r1_r_props, &src_B2_Blocal_r1_i_props);

    complex_expr src_B2_r1 = src_psi_B2 * src_B2_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation src_B2_Blocal_r1_r_update("src_B2_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B2_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B2_r1.get_real());
    computation src_B2_Blocal_r1_i_update("src_B2_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B2_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B2_r1.get_imag());

    complex_expr flip_src_B2_r1 = src_psi_B1 * src_B2_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_src_B2_Blocal_r1_r_update("flip_src_B2_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B2_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B2_r1.get_real());
    computation flip_src_B2_Blocal_r1_i_update("flip_src_B2_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B2_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B2_r1.get_imag());

     // Computing src_B2_Blocal_r2

    computation src_B2_Blocal_r2_r_init("src_B2_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation src_B2_Blocal_r2_i_init("src_B2_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation src_B2_Blocal_r2_init(&src_B2_Blocal_r2_r_init, &src_B2_Blocal_r2_i_init);

    computation flip_src_B2_Blocal_r2_r_init("flip_src_B2_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));
    computation flip_src_B2_Blocal_r2_i_init("flip_src_B2_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation flip_src_B2_Blocal_r2_init(&flip_src_B2_Blocal_r2_r_init, &flip_src_B2_Blocal_r2_i_init);

    complex_expr src_B2_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr src_B2_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr src_B2_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr src_B2_r2_diquark = ( src_B2_r2_prop_0 * src_B2_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation src_B2_Blocal_r2_r_props_init("src_B2_Blocal_r2_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));
    computation src_B2_Blocal_r2_i_props_init("src_B2_Blocal_r2_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((double) 0));

    computation src_B2_Blocal_r2_r_diquark("src_B2_Blocal_r2_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B2_r2_diquark.get_real());
    computation src_B2_Blocal_r2_i_diquark("src_B2_Blocal_r2_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, src_B2_r2_diquark.get_imag());

    complex_computation src_B2_Blocal_r2_diquark(&src_B2_Blocal_r2_r_diquark, &src_B2_Blocal_r2_i_diquark);

    complex_expr src_B2_r2_props = src_B2_r2_prop_1 * src_B2_Blocal_r2_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation src_B2_Blocal_r2_r_props("src_B2_Blocal_r2_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B2_Blocal_r2_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B2_r2_props.get_real());
    computation src_B2_Blocal_r2_i_props("src_B2_Blocal_r2_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, src_B2_Blocal_r2_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + src_B2_r2_props.get_imag());

    complex_computation src_B2_Blocal_r2_props(&src_B2_Blocal_r2_r_props, &src_B2_Blocal_r2_i_props);

    complex_expr src_B2_r2 = src_psi_B2 * src_B2_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation src_B2_Blocal_r2_r_update("src_B2_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B2_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B2_r2.get_real());
    computation src_B2_Blocal_r2_i_update("src_B2_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, src_B2_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + src_B2_r2.get_imag());

    complex_expr flip_src_B2_r2 = src_psi_B1 * src_B2_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation flip_src_B2_Blocal_r2_r_update("flip_src_B2_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B2_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B2_r2.get_real());
    computation flip_src_B2_Blocal_r2_i_update("flip_src_B2_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, flip_src_B2_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + flip_src_B2_r2.get_imag()); 
    
     // Computing snk_B1_Blocal_r1

    computation snk_B1_Blocal_r1_r_init("snk_B1_Blocal_r1_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation snk_B1_Blocal_r1_i_init("snk_B1_Blocal_r1_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation snk_B1_Blocal_r1_init(&snk_B1_Blocal_r1_r_init, &snk_B1_Blocal_r1_i_init);

    computation flip_snk_B1_Blocal_r1_r_init("flip_snk_B1_Blocal_r1_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation flip_snk_B1_Blocal_r1_i_init("flip_snk_B1_Blocal_r1_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation flip_snk_B1_Blocal_r1_init(&flip_snk_B1_Blocal_r1_r_init, &flip_snk_B1_Blocal_r1_i_init);

    complex_expr snk_B1_r1_prop_0 =  B1_prop(t, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), iCprime, iSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B1_r1_prop_2 =  B1_prop(t, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), kCprime, kSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B1_r1_prop_1 = B1_prop(t, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), jCprime, jSprime, x, y_out*src_sites_per_rank+y_in);

    complex_expr snk_B1_r1_diquark = ( snk_B1_r1_prop_0 * snk_B1_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation snk_B1_Blocal_r1_r_props_init("snk_B1_Blocal_r1_r_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));
    computation snk_B1_Blocal_r1_i_props_init("snk_B1_Blocal_r1_i_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));

    computation snk_B1_Blocal_r1_r_diquark("snk_B1_Blocal_r1_r_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B1_r1_diquark.get_real());
    computation snk_B1_Blocal_r1_i_diquark("snk_B1_Blocal_r1_i_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B1_r1_diquark.get_imag());

    complex_computation snk_B1_Blocal_r1_diquark(&snk_B1_Blocal_r1_r_diquark, &snk_B1_Blocal_r1_i_diquark);

    complex_expr snk_B1_r1_props = snk_B1_r1_prop_1 * snk_B1_Blocal_r1_diquark(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock);

    computation snk_B1_Blocal_r1_r_props("snk_B1_Blocal_r1_r_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B1_Blocal_r1_r_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B1_r1_props.get_real());
    computation snk_B1_Blocal_r1_i_props("snk_B1_Blocal_r1_i_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B1_Blocal_r1_i_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B1_r1_props.get_imag());

    complex_computation snk_B1_Blocal_r1_props(&snk_B1_Blocal_r1_r_props, &snk_B1_Blocal_r1_i_props);

    complex_expr snk_B1_r1 = snk_psi_B1 * snk_B1_Blocal_r1_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation snk_B1_Blocal_r1_r_update("snk_B1_Blocal_r1_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B1_Blocal_r1_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B1_r1.get_real());
    computation snk_B1_Blocal_r1_i_update("snk_B1_Blocal_r1_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B1_Blocal_r1_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B1_r1.get_imag());

    complex_expr flip_snk_B1_r1 = snk_psi_B2 * snk_B1_Blocal_r1_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation flip_snk_B1_Blocal_r1_r_update("flip_snk_B1_Blocal_r1_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B1_Blocal_r1_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B1_r1.get_real());
    computation flip_snk_B1_Blocal_r1_i_update("flip_snk_B1_Blocal_r1_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B1_Blocal_r1_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B1_r1.get_imag());

     // Computing snk_B1_Blocal_r2

    computation snk_B1_Blocal_r2_r_init("snk_B1_Blocal_r2_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation snk_B1_Blocal_r2_i_init("snk_B1_Blocal_r2_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation snk_B1_Blocal_r2_init(&snk_B1_Blocal_r2_r_init, &snk_B1_Blocal_r2_i_init);

    computation flip_snk_B1_Blocal_r2_r_init("flip_snk_B1_Blocal_r2_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation flip_snk_B1_Blocal_r2_i_init("flip_snk_B1_Blocal_r2_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation flip_snk_B1_Blocal_r2_init(&flip_snk_B1_Blocal_r2_r_init, &flip_snk_B1_Blocal_r2_i_init);

    complex_expr snk_B1_r2_prop_0 =  B1_prop(t, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), iCprime, iSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B1_r2_prop_2 =  B1_prop(t, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), kCprime, kSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B1_r2_prop_1 = B1_prop(t, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), jCprime, jSprime, x, y_out*src_sites_per_rank+y_in);

    complex_expr snk_B1_r2_diquark = ( snk_B1_r2_prop_0 * snk_B1_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation snk_B1_Blocal_r2_r_props_init("snk_B1_Blocal_r2_r_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));
    computation snk_B1_Blocal_r2_i_props_init("snk_B1_Blocal_r2_i_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));

    computation snk_B1_Blocal_r2_r_diquark("snk_B1_Blocal_r2_r_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B1_r2_diquark.get_real());
    computation snk_B1_Blocal_r2_i_diquark("snk_B1_Blocal_r2_i_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B1_r2_diquark.get_imag());

    complex_computation snk_B1_Blocal_r2_diquark(&snk_B1_Blocal_r2_r_diquark, &snk_B1_Blocal_r2_i_diquark);

    complex_expr snk_B1_r2_props = snk_B1_r2_prop_1 * snk_B1_Blocal_r2_diquark(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock);

    computation snk_B1_Blocal_r2_r_props("snk_B1_Blocal_r2_r_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B1_Blocal_r2_r_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B1_r2_props.get_real());
    computation snk_B1_Blocal_r2_i_props("snk_B1_Blocal_r2_i_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B1_Blocal_r2_i_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B1_r2_props.get_imag());

    complex_computation snk_B1_Blocal_r2_props(&snk_B1_Blocal_r2_r_props, &snk_B1_Blocal_r2_i_props);

    complex_expr snk_B1_r2 = snk_psi_B1 * snk_B1_Blocal_r2_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation snk_B1_Blocal_r2_r_update("snk_B1_Blocal_r2_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B1_Blocal_r2_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B1_r2.get_real());
    computation snk_B1_Blocal_r2_i_update("snk_B1_Blocal_r2_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B1_Blocal_r2_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B1_r2.get_imag()); 

    complex_expr flip_snk_B1_r2 = snk_psi_B2 * snk_B1_Blocal_r2_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation flip_snk_B1_Blocal_r2_r_update("flip_snk_B1_Blocal_r2_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B1_Blocal_r2_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B1_r2.get_real());
    computation flip_snk_B1_Blocal_r2_i_update("flip_snk_B1_Blocal_r2_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B1_Blocal_r2_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B1_r2.get_imag());

     // Computing snk_B2_Blocal_r1

    computation snk_B2_Blocal_r1_r_init("snk_B2_Blocal_r1_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation snk_B2_Blocal_r1_i_init("snk_B2_Blocal_r1_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation snk_B2_Blocal_r1_init(&snk_B2_Blocal_r1_r_init, &snk_B2_Blocal_r1_i_init);

    computation flip_snk_B2_Blocal_r1_r_init("flip_snk_B2_Blocal_r1_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation flip_snk_B2_Blocal_r1_i_init("flip_snk_B2_Blocal_r1_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation flip_snk_B2_Blocal_r1_init(&flip_snk_B2_Blocal_r1_r_init, &flip_snk_B2_Blocal_r1_i_init);

    complex_expr snk_B2_r1_prop_0 =  B1_prop(t, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), iCprime, iSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B2_r1_prop_2 =  B1_prop(t, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), kCprime, kSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B2_r1_prop_1 = B1_prop(t, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), jCprime, jSprime, x, y_out*src_sites_per_rank+y_in);

    complex_expr snk_B2_r1_diquark = ( snk_B2_r1_prop_0 * snk_B2_r1_prop_2 ) *  src_weights(0, wnumBlock);

    computation snk_B2_Blocal_r1_r_props_init("snk_B2_Blocal_r1_r_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));
    computation snk_B2_Blocal_r1_i_props_init("snk_B2_Blocal_r1_i_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));

    computation snk_B2_Blocal_r1_r_diquark("snk_B2_Blocal_r1_r_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B2_r1_diquark.get_real());
    computation snk_B2_Blocal_r1_i_diquark("snk_B2_Blocal_r1_i_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B2_r1_diquark.get_imag());

    complex_computation snk_B2_Blocal_r1_diquark(&snk_B2_Blocal_r1_r_diquark, &snk_B2_Blocal_r1_i_diquark);

    complex_expr snk_B2_r1_props = snk_B2_r1_prop_1 * snk_B2_Blocal_r1_diquark(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock);

    computation snk_B2_Blocal_r1_r_props("snk_B2_Blocal_r1_r_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B2_Blocal_r1_r_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B2_r1_props.get_real());
    computation snk_B2_Blocal_r1_i_props("snk_B2_Blocal_r1_i_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B2_Blocal_r1_i_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B2_r1_props.get_imag());

    complex_computation snk_B2_Blocal_r1_props(&snk_B2_Blocal_r1_r_props, &snk_B2_Blocal_r1_i_props);

    complex_expr snk_B2_r1 = snk_psi_B2 * snk_B2_Blocal_r1_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation snk_B2_Blocal_r1_r_update("snk_B2_Blocal_r1_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B2_Blocal_r1_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B2_r1.get_real());
    computation snk_B2_Blocal_r1_i_update("snk_B2_Blocal_r1_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B2_Blocal_r1_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B2_r1.get_imag());

    complex_expr flip_snk_B2_r1 = snk_psi_B1 * snk_B2_Blocal_r1_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation flip_snk_B2_Blocal_r1_r_update("flip_snk_B2_Blocal_r1_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B2_Blocal_r1_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B2_r1.get_real());
    computation flip_snk_B2_Blocal_r1_i_update("flip_snk_B2_Blocal_r1_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B2_Blocal_r1_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B2_r1.get_imag());

     // Computing snk_B2_Blocal_r2

    computation snk_B2_Blocal_r2_r_init("snk_B2_Blocal_r2_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation snk_B2_Blocal_r2_i_init("snk_B2_Blocal_r2_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation snk_B2_Blocal_r2_init(&snk_B2_Blocal_r2_r_init, &snk_B2_Blocal_r2_i_init);

    computation flip_snk_B2_Blocal_r2_r_init("flip_snk_B2_Blocal_r2_r_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation flip_snk_B2_Blocal_r2_i_init("flip_snk_B2_Blocal_r2_i_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_computation flip_snk_B2_Blocal_r2_init(&flip_snk_B2_Blocal_r2_r_init, &flip_snk_B2_Blocal_r2_i_init);

    complex_expr snk_B2_r2_prop_0 =  B1_prop(t, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), iCprime, iSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B2_r2_prop_2 =  B1_prop(t, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), kCprime, kSprime, x, y_out*src_sites_per_rank+y_in);
    complex_expr snk_B2_r2_prop_1 = B1_prop(t, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), jCprime, jSprime, x, y_out*src_sites_per_rank+y_in);

    complex_expr snk_B2_r2_diquark = ( snk_B2_r2_prop_0 * snk_B2_r2_prop_2 ) *  src_weights(1, wnumBlock);

    computation snk_B2_Blocal_r2_r_props_init("snk_B2_Blocal_r2_r_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));
    computation snk_B2_Blocal_r2_i_props_init("snk_B2_Blocal_r2_i_props_init", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime}, expr((double) 0));

    computation snk_B2_Blocal_r2_r_diquark("snk_B2_Blocal_r2_r_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B2_r2_diquark.get_real());
    computation snk_B2_Blocal_r2_i_diquark("snk_B2_Blocal_r2_i_diquark", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock}, snk_B2_r2_diquark.get_imag());

    complex_computation snk_B2_Blocal_r2_diquark(&snk_B2_Blocal_r2_r_diquark, &snk_B2_Blocal_r2_i_diquark);

    complex_expr snk_B2_r2_props = snk_B2_r2_prop_1 * snk_B2_Blocal_r2_diquark(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock);

    computation snk_B2_Blocal_r2_r_props("snk_B2_Blocal_r2_r_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B2_Blocal_r2_r_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B2_r2_props.get_real());
    computation snk_B2_Blocal_r2_i_props("snk_B2_Blocal_r2_i_props", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, wnumBlock, jCprime, jSprime}, snk_B2_Blocal_r2_i_props_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime) + snk_B2_r2_props.get_imag());

    complex_computation snk_B2_Blocal_r2_props(&snk_B2_Blocal_r2_r_props, &snk_B2_Blocal_r2_i_props);

    complex_expr snk_B2_r2 = snk_psi_B2 * snk_B2_Blocal_r2_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation snk_B2_Blocal_r2_r_update("snk_B2_Blocal_r2_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B2_Blocal_r2_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B2_r2.get_real());
    computation snk_B2_Blocal_r2_i_update("snk_B2_Blocal_r2_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, snk_B2_Blocal_r2_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + snk_B2_r2.get_imag());

    complex_expr flip_snk_B2_r2 = snk_psi_B1 * snk_B2_Blocal_r2_props(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, Nw-1, jCprime, jSprime);

    computation flip_snk_B2_Blocal_r2_r_update("flip_snk_B2_Blocal_r2_r_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B2_Blocal_r2_r_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B2_r2.get_real());
    computation flip_snk_B2_Blocal_r2_i_update("flip_snk_B2_Blocal_r2_i_update", {t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, n}, flip_snk_B2_Blocal_r2_i_init(t, y_out, y_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) + flip_snk_B2_r2.get_imag());

    /* Correlators */

    computation C_init_r("C_init_r", {t, x_out, rp, mpmH, r, npnH}, expr((double) 0));
    computation C_init_i("C_init_i", {t, x_out, rp, mpmH, r, npnH}, expr((double) 0));

    // BB_BB
    computation C_BB_BB_prop_init_r("C_BB_BB_prop_init_r", {t, x_out, x_in, x2, rp, m, r}, expr((double) 0));
    computation C_BB_BB_prop_init_i("C_BB_BB_prop_init_i", {t, x_out, x_in, x2, rp, m, r}, expr((double) 0));
    
    b=0;
    // r1, b = 0 
    complex_computation BB_BB_new_term_0_r1_b1("BB_BB_new_term_0_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r1_b1("BB_BB_new_term_1_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_1_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r1_b1("BB_BB_new_term_2_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_2_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r1_b1("BB_BB_new_term_3_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_3_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r1_b1("BB_BB_new_term_4_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_4_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_5_r1_b1("BB_BB_new_term_5_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_5_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_6_r1_b1("BB_BB_new_term_6_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_6_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_7_r1_b1("BB_BB_new_term_7_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_7_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) &&  (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    // r2, b = 0 
    complex_computation BB_BB_new_term_0_r2_b1("BB_BB_new_term_0_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r2_b1("BB_BB_new_term_1_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_1_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r2_b1("BB_BB_new_term_2_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_2_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r2_b1("BB_BB_new_term_3_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_3_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r2_b1("BB_BB_new_term_4_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_4_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_5_r2_b1("BB_BB_new_term_5_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_5_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_6_r2_b1("BB_BB_new_term_6_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_6_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_7_r2_b1("BB_BB_new_term_7_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    BB_BB_new_term_7_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    b=1;
    // r1, b = 1 
    complex_computation BB_BB_new_term_0_r1_b2("BB_BB_new_term_0_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r1_b2("BB_BB_new_term_1_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_1_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r1_b2("BB_BB_new_term_2_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_2_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r1_b2("BB_BB_new_term_3_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_3_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r1_b2("BB_BB_new_term_4_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_4_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_5_r1_b2("BB_BB_new_term_5_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_5_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_6_r1_b2("BB_BB_new_term_6_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_6_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_7_r1_b2("BB_BB_new_term_7_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_7_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    // r2, b = 1
    complex_computation BB_BB_new_term_0_r2_b2("BB_BB_new_term_0_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r2_b2("BB_BB_new_term_1_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_1_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r2_b2("BB_BB_new_term_2_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_2_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r2_b2("BB_BB_new_term_3_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_3_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r2_b2("BB_BB_new_term_4_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_4_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_5_r2_b2("BB_BB_new_term_5_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_5_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_6_r2_b2("BB_BB_new_term_6_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B1_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_6_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_7_r2_b2("BB_BB_new_term_7_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, B2_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    BB_BB_new_term_7_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));

    complex_expr prefactor(cast(p_float64, sigs(nperm)) * snk_weights(r, wnum) * src_spin_block_weights(rp, s), 0.0);

    complex_expr BB_BB_term_res_b1 = BB_BB_new_term_0_r1_b1(t, x_out, x_in, x2, rp, m, r, s, nperm, wnum);
    complex_expr BB_BB_term_res_b2 = BB_BB_new_term_0_r1_b2(t, x_out, x_in, x2, rp, m, r, s, nperm, wnum);

    complex_expr BB_BB_term_res = prefactor * BB_BB_term_res_b1 * BB_BB_term_res_b2;

    b=0;
    // r1, b = 0 
    complex_computation flip_BB_BB_new_term_0_r1_b1("flip_BB_BB_new_term_0_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_1_r1_b1("flip_BB_BB_new_term_1_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_1_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_2_r1_b1("flip_BB_BB_new_term_2_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_2_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_3_r1_b1("flip_BB_BB_new_term_3_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_3_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_4_r1_b1("flip_BB_BB_new_term_4_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_4_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_5_r1_b1("flip_BB_BB_new_term_5_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_5_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_6_r1_b1("flip_BB_BB_new_term_6_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_6_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_7_r1_b1("flip_BB_BB_new_term_7_r1_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_7_r1_b1.add_predicate((src_spins(rp, s, 0) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    // r2, b = 0 
    complex_computation flip_BB_BB_new_term_0_r2_b1("flip_BB_BB_new_term_0_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_1_r2_b1("flip_BB_BB_new_term_1_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_1_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_2_r2_b1("flip_BB_BB_new_term_2_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_2_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_3_r2_b1("flip_BB_BB_new_term_3_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_3_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_4_r2_b1("flip_BB_BB_new_term_4_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_4_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_5_r2_b1("flip_BB_BB_new_term_5_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_5_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_6_r2_b1("flip_BB_BB_new_term_6_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_6_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_7_r2_b1("flip_BB_BB_new_term_7_r2_b1", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), m));
    flip_BB_BB_new_term_7_r2_b1.add_predicate((src_spins(rp, s, 0) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    b=1;
    // r1, b = 1 
    complex_computation flip_BB_BB_new_term_0_r1_b2("flip_BB_BB_new_term_0_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_1_r1_b2("flip_BB_BB_new_term_1_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Blocal_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_1_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_2_r1_b2("flip_BB_BB_new_term_2_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_2_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_3_r1_b2("flip_BB_BB_new_term_3_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bfirst_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_3_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_4_r1_b2("flip_BB_BB_new_term_4_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_4_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_5_r1_b2("flip_BB_BB_new_term_5_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bsecond_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_5_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_6_r1_b2("flip_BB_BB_new_term_6_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_6_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_7_r1_b2("flip_BB_BB_new_term_7_r1_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bthird_r1_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_7_r1_b2.add_predicate((src_spins(rp, s, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    // r2, b = 1 
    complex_computation flip_BB_BB_new_term_0_r2_b2("flip_BB_BB_new_term_0_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_1_r2_b2("flip_BB_BB_new_term_1_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Blocal_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_1_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_2_r2_b2("flip_BB_BB_new_term_2_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_2_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_3_r2_b2("flip_BB_BB_new_term_3_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bfirst_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_3_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_4_r2_b2("flip_BB_BB_new_term_4_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_4_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation flip_BB_BB_new_term_5_r2_b2("flip_BB_BB_new_term_5_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bsecond_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_5_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_6_r2_b2("flip_BB_BB_new_term_6_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B1_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_6_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation flip_BB_BB_new_term_7_r2_b2("flip_BB_BB_new_term_7_r2_b2", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, flip_B2_Bthird_r2_init(t, x_out, x_in, x2, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), m));
    flip_BB_BB_new_term_7_r2_b2.add_predicate((src_spins(rp, s, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));

    complex_expr flip_BB_BB_term_res_b1 = flip_BB_BB_new_term_0_r1_b1(t, x_out, x_in, x2, rp, m, r, s, nperm, wnum);
    complex_expr flip_BB_BB_term_res_b2 = flip_BB_BB_new_term_0_r1_b2(t, x_out, x_in, x2, rp, m, r, s, nperm, wnum);

    complex_expr flip_BB_BB_term_res = prefactor * flip_BB_BB_term_res_b1 * flip_BB_BB_term_res_b2;


    computation C_BB_BB_prop_update_r("C_BB_BB_prop_update_r", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, C_BB_BB_prop_init_r(t, x_out, x_in, x2, rp, m, r) + (BB_BB_term_res.get_real() + flip_BB_BB_term_res.get_real())/2.0 );
    computation C_BB_BB_prop_update_i("C_BB_BB_prop_update_i", {t, x_out, x_in, x2, rp, m, r, s, nperm, wnum}, C_BB_BB_prop_init_i(t, x_out, x_in, x2, rp, m, r) + (BB_BB_term_res.get_imag() + flip_BB_BB_term_res.get_imag())/2.0 );

    complex_computation C_BB_BB_prop_update(&C_BB_BB_prop_update_r, &C_BB_BB_prop_update_i);

    complex_expr BB_BB_term_s = (snk_psi_B1_ue * snk_psi_B2_x2_ue + snk_psi_B1_x2_ue * snk_psi_B2_ue) * C_BB_BB_prop_update(t, x_out, x_in, x2, rp, m, r, 1, Nperms-1, Nw2-1);
    complex_expr BB_BB_term_b = snk_psi * C_BB_BB_prop_update(t, x_out, x_in, x2, rp, m, r, 1, Nperms-1, Nw2-1);

    computation C_BB_BB_update_s_r("C_BB_BB_update_s_r", {t, x_out, x_in, x2, rp, m, r, nue}, C_init_r(t, x_out, rp, m, r, NEntangled+nue) + BB_BB_term_s.get_real());
    computation C_BB_BB_update_s_i("C_BB_BB_update_s_i", {t, x_out, x_in, x2, rp, m, r, nue}, C_init_i(t, x_out, rp, m, r, NEntangled+nue) + BB_BB_term_s.get_imag());

    computation C_BB_BB_update_b_r("C_BB_BB_update_b_r", {t, x_out, x_in, x2, rp, m, r, ne}, C_init_r(t, x_out, rp, m, r, ne) + BB_BB_term_b.get_real());
    computation C_BB_BB_update_b_i("C_BB_BB_update_b_i", {t, x_out, x_in, x2, rp, m, r, ne}, C_init_i(t, x_out, rp, m, r, ne) + BB_BB_term_b.get_imag());

    // BB_H
    computation C_BB_H_prop_init_r("C_BB_H_prop_init_r", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    computation C_BB_H_prop_init_i("C_BB_H_prop_init_i", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    
    complex_computation BB_H_new_term_0_r1_b1("BB_H_new_term_0_r1_b1", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, src_B1_Blocal_r1_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), m));
    BB_H_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1));
    complex_computation BB_H_new_term_0_r2_b1("BB_H_new_term_0_r2_b1", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, src_B1_Blocal_r2_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), m));
    BB_H_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2));

    complex_computation BB_H_new_term_0_r1_b2("BB_H_new_term_0_r1_b2", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, src_B2_Blocal_r1_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), m));
    BB_H_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1));
    complex_computation BB_H_new_term_0_r2_b2("BB_H_new_term_0_r2_b2", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, src_B2_Blocal_r2_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), m));
    BB_H_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2));

    complex_expr BB_H_term_res_b1 = BB_H_new_term_0_r1_b1(t, x_out, x_in, rp, m, r, s, nperm, wnumHex);
    complex_expr BB_H_term_res_b2 = BB_H_new_term_0_r1_b2(t, x_out, x_in, rp, m, r, s, nperm, wnumHex);

    complex_expr src_hex_prefactor(cast(p_float64, sigs(nperm)) * hex_snk_weights(r, wnumHex) * src_spin_block_weights(rp, s), 0.0);

    complex_expr BB_H_term_res = src_hex_prefactor * BB_H_term_res_b1 * BB_H_term_res_b2;
    
    complex_computation flip_BB_H_new_term_0_r1_b1("flip_BB_H_new_term_0_r1_b1", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, flip_src_B1_Blocal_r1_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), m));
    flip_BB_H_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1));
    complex_computation flip_BB_H_new_term_0_r2_b1("flip_BB_H_new_term_0_r2_b1", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, flip_src_B1_Blocal_r2_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), m));
    flip_BB_H_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2));

    complex_computation flip_BB_H_new_term_0_r1_b2("flip_BB_H_new_term_0_r1_b2", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, flip_src_B2_Blocal_r1_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), m));
    flip_BB_H_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1));
    complex_computation flip_BB_H_new_term_0_r2_b2("flip_BB_H_new_term_0_r2_b2", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, flip_src_B2_Blocal_r2_init(t, x_out, x_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), m));
    flip_BB_H_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2));

    complex_expr flip_BB_H_term_res_b1 = flip_BB_H_new_term_0_r1_b1(t, x_out, x_in, rp, m, r, s, nperm, wnumHex);
    complex_expr flip_BB_H_term_res_b2 = flip_BB_H_new_term_0_r1_b2(t, x_out, x_in, rp, m, r, s, nperm, wnumHex);

    complex_expr flip_BB_H_term_res = src_hex_prefactor * flip_BB_H_term_res_b1 * flip_BB_H_term_res_b2;

    computation C_BB_H_prop_update_r("C_BB_H_prop_update_r", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, C_BB_H_prop_init_r(t, x_out, x_in, rp, m, r) + (BB_H_term_res.get_real() + flip_BB_H_term_res.get_real())/2.0 );
    computation C_BB_H_prop_update_i("C_BB_H_prop_update_i", {t, x_out, x_in, rp, m, r, s, nperm, wnumHex}, C_BB_H_prop_init_i(t, x_out, x_in, rp, m, r) + (BB_H_term_res.get_imag() + flip_BB_H_term_res.get_imag())/2.0 );

    complex_computation C_BB_H_prop_update(&C_BB_H_prop_update_r, &C_BB_H_prop_update_i);

    complex_expr BB_H_term = hex_snk_psi * C_BB_H_prop_update(t, x_out, x_in, rp, m, r, 1, Nperms-1, Nw2Hex-1);

    computation C_BB_H_update_r("C_BB_H_update_r", {t, x_out, x_in, rp, m, r, nH}, C_init_r(t, x_out, rp, m, r, Nsnk+nH) + BB_H_term.get_real());
    computation C_BB_H_update_i("C_BB_H_update_i", {t, x_out, x_in, rp, m, r, nH}, C_init_i(t, x_out, rp, m, r, Nsnk+nH) + BB_H_term.get_imag()); 

    // H_BB
    computation C_H_BB_prop_init_r("C_H_BB_prop_init_r", {t, y_out, y_in, rp, n, r}, expr((double) 0));
    computation C_H_BB_prop_init_i("C_H_BB_prop_init_i", {t, y_out, y_in, rp, n, r}, expr((double) 0));
    
    complex_computation H_BB_new_term_0_r1_b1("H_BB_new_term_0_r1_b1", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, snk_B1_Blocal_r1_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), n));
    H_BB_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1));
    complex_computation H_BB_new_term_0_r2_b1("H_BB_new_term_0_r2_b1", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, snk_B1_Blocal_r2_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), n));
    H_BB_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2));

    complex_computation H_BB_new_term_0_r1_b2("H_BB_new_term_0_r1_b2", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, snk_B2_Blocal_r1_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), n));
    H_BB_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1));
    complex_computation H_BB_new_term_0_r2_b2("H_BB_new_term_0_r2_b2", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, snk_B2_Blocal_r2_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), n));
    H_BB_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2));

    complex_expr H_BB_term_res_b1 = H_BB_new_term_0_r1_b1(t, y_out, y_in, rp, n, r, s, nperm, wnumHex);
    complex_expr H_BB_term_res_b2 = H_BB_new_term_0_r1_b2(t, y_out, y_in, rp, n, r, s, nperm, wnumHex);

    complex_expr snk_hex_prefactor(cast(p_float64, sigs(nperm)) * hex_snk_weights(r, wnumHex) * src_spin_block_weights(rp, s), 0.0);

    complex_expr H_BB_term_res = snk_hex_prefactor * H_BB_term_res_b1 * H_BB_term_res_b2;

    complex_computation flip_H_BB_new_term_0_r1_b1("flip_H_BB_new_term_0_r1_b1", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, flip_snk_B1_Blocal_r1_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), n));
    flip_H_BB_new_term_0_r1_b1.add_predicate((src_spins(rp, s, 0) == 1));
    complex_computation flip_H_BB_new_term_0_r2_b1("flip_H_BB_new_term_0_r2_b1", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, flip_snk_B1_Blocal_r2_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 0), hex_snk_spin_weights(r, nperm, wnumHex, 0, 0), hex_snk_color_weights(r, nperm, wnumHex, 2, 0), hex_snk_spin_weights(r, nperm, wnumHex, 2, 0), hex_snk_color_weights(r, nperm, wnumHex, 1, 0), hex_snk_spin_weights(r, nperm, wnumHex, 1, 0), n));
    flip_H_BB_new_term_0_r2_b1.add_predicate((src_spins(rp, s, 0) == 2));

    complex_computation flip_H_BB_new_term_0_r1_b2("flip_H_BB_new_term_0_r1_b2", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, flip_snk_B2_Blocal_r1_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), n));
    flip_H_BB_new_term_0_r1_b2.add_predicate((src_spins(rp, s, 1) == 1));
    complex_computation flip_H_BB_new_term_0_r2_b2("flip_H_BB_new_term_0_r2_b2", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, flip_snk_B2_Blocal_r2_init(t, y_out, y_in, hex_snk_color_weights(r, nperm, wnumHex, 0, 1), hex_snk_spin_weights(r, nperm, wnumHex, 0, 1), hex_snk_color_weights(r, nperm, wnumHex, 2, 1), hex_snk_spin_weights(r, nperm, wnumHex, 2, 1), hex_snk_color_weights(r, nperm, wnumHex, 1, 1), hex_snk_spin_weights(r, nperm, wnumHex, 1, 1), n));
    flip_H_BB_new_term_0_r2_b2.add_predicate((src_spins(rp, s, 1) == 2));

    complex_expr flip_H_BB_term_res_b1 = flip_H_BB_new_term_0_r1_b1(t, y_out, y_in, rp, n, r, s, nperm, wnumHex);
    complex_expr flip_H_BB_term_res_b2 = flip_H_BB_new_term_0_r1_b2(t, y_out, y_in, rp, n, r, s, nperm, wnumHex);

    complex_expr flip_H_BB_term_res = snk_hex_prefactor * flip_H_BB_term_res_b1 * flip_H_BB_term_res_b2;

    computation C_H_BB_prop_update_r("C_H_BB_prop_update_r", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, C_H_BB_prop_init_r(t, y_out, y_in, rp, n, r) + (H_BB_term_res.get_real() + flip_H_BB_term_res.get_real())/2.0 );
    computation C_H_BB_prop_update_i("C_H_BB_prop_update_i", {t, y_out, y_in, rp, n, r, s, nperm, wnumHex}, C_H_BB_prop_init_i(t, y_out, y_in, rp, n, r) + (H_BB_term_res.get_imag() + flip_H_BB_term_res.get_imag())/2.0 );

    complex_computation C_H_BB_prop_update(&C_H_BB_prop_update_r, &C_H_BB_prop_update_i);

    complex_expr H_BB_term = hex_src_psi * C_H_BB_prop_update(t, y_out, y_in, rp, n, r, 1, Nperms-1, Nw2Hex-1);

    computation C_H_BB_update_r("C_H_BB_update_r", {t, y_out, y_in, rp, n, r, mH}, C_init_r(t, y_out, r, Nsrc+mH, rp, n) + H_BB_term.get_real());
    computation C_H_BB_update_i("C_H_BB_update_i", {t, y_out, y_in, rp, n, r, mH}, C_init_i(t, y_out, r, Nsrc+mH, rp, n) + H_BB_term.get_imag());  


    // H_H
    computation C_H_H_prop_init_r("C_H_H_prop_init_r", {t, x_out, x_in, rp, r, y}, expr((double) 0));
    computation C_H_H_prop_init_i("C_H_H_prop_init_i", {t, x_out, x_in, rp, r, y}, expr((double) 0));


    complex_expr H_H_B1_prop_0 =  B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,0,0), hex_snk_spin_weights(r,nperm,wnumHex,0,0), hex_snk_color_weights(rp,0,wnumHexHex,0,0), hex_snk_spin_weights(rp,0,wnumHexHex,0,0), x_out*sites_per_rank+x_in, y);
    complex_expr H_H_B1_prop_2 =  B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,2,0), hex_snk_spin_weights(r,nperm,wnumHex,2,0), hex_snk_color_weights(rp,0,wnumHexHex,2,0), hex_snk_spin_weights(rp,0,wnumHexHex,2,0), x_out*sites_per_rank+x_in, y);
    complex_expr H_H_B1_prop_1 = B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,1,0), hex_snk_spin_weights(r,nperm,wnumHex,1,0), hex_snk_color_weights(rp,0,wnumHexHex,1,0), hex_snk_spin_weights(rp,0,wnumHexHex,1,0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_H = H_H_B1_prop_0 * H_H_B1_prop_2 * H_H_B1_prop_1;

    complex_expr H_H_B2_prop_0 =  B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,0,1), hex_snk_spin_weights(r,nperm,wnumHex,0,1), hex_snk_color_weights(rp,0,wnumHexHex,0,1), hex_snk_spin_weights(rp,0,wnumHexHex,0,1), x_out*sites_per_rank+x_in, y);
    complex_expr H_H_B2_prop_2 =  B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,2,1), hex_snk_spin_weights(r,nperm,wnumHex,2,1), hex_snk_color_weights(rp,0,wnumHexHex,2,1), hex_snk_spin_weights(rp,0,wnumHexHex,2,1), x_out*sites_per_rank+x_in, y);
    complex_expr H_H_B2_prop_1 = B1_prop(t, hex_snk_color_weights(r,nperm,wnumHex,1,1), hex_snk_spin_weights(r,nperm,wnumHex,1,1), hex_snk_color_weights(rp,0,wnumHexHex,1,1), hex_snk_spin_weights(rp,0,wnumHexHex,1,1), x_out*sites_per_rank+x_in, y);
    complex_expr B2_H = H_H_B2_prop_0 * H_H_B2_prop_2 * H_H_B2_prop_1;

    complex_expr hex_hex_prefactor(cast(p_float64, sigs(nperm)) * hex_snk_weights(r, wnumHex) * hex_snk_weights(rp, wnumHexHex), 0.0);

    complex_expr H_H_term_res = hex_hex_prefactor * B1_H * B2_H;

    computation C_H_H_prop_update_r("C_H_H_prop_update_r", {t, x_out, x_in, rp, r, y, nperm, wnumHex, wnumHexHex}, C_H_H_prop_init_r(t, x_out, x_in, rp, r, y) + H_H_term_res.get_real());
    computation C_H_H_prop_update_i("C_H_H_prop_update_i", {t, x_out, x_in, rp, r, y, nperm, wnumHex, wnumHexHex}, C_H_H_prop_init_i(t, x_out, x_in, rp, r, y) + H_H_term_res.get_imag());

    complex_computation C_H_H_prop_update(&C_H_H_prop_update_r, &C_H_H_prop_update_i); 

    complex_expr H_H_term = hex_hex_src_psi * hex_snk_psi * C_H_H_prop_update(t, x_out, x_in, rp, r, y, Nperms-1, Nw2Hex-1, Nw2Hex-1);

    computation C_H_H_update_r("C_H_H_update_r", {t, x_out, x_in, rp, r, y, mH, nH}, C_init_r(t, x_out, rp, Nsrc+mH, r, Nsnk+nH) + H_H_term.get_real());
    computation C_H_H_update_i("C_H_H_update_i", {t, x_out, x_in, rp, r, y, mH, nH}, C_init_i(t, x_out, rp, Nsrc+mH, r, Nsnk+nH) + H_H_term.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, npnH)
          );

    // BB_BB
    handle = &(handle
          ->then(B1_Blocal_r1_r_init, t)
          .then(B1_Blocal_r1_i_init, m)
          .then(B1_Bfirst_r1_r_init, m)
          .then(B1_Bfirst_r1_i_init, m)
          .then(B1_Bsecond_r1_r_init, m)
          .then(B1_Bsecond_r1_i_init, m)
          .then(B1_Bthird_r1_r_init, m)
          .then(B1_Bthird_r1_i_init, m)
          .then(flip_B1_Blocal_r1_r_init, m)
          .then(flip_B1_Blocal_r1_i_init, m)
          .then(flip_B1_Bfirst_r1_r_init, m)
          .then(flip_B1_Bfirst_r1_i_init, m)
          .then(flip_B1_Bsecond_r1_r_init, m)
          .then(flip_B1_Bsecond_r1_i_init, m)
          .then(flip_B1_Bthird_r1_r_init, m)
          .then(flip_B1_Bthird_r1_i_init, m)
          .then(B1_Blocal_r1_r_props_init, x2)
          .then(B1_Blocal_r1_i_props_init, jSprime)
          .then(B1_Bfirst_r1_r_props_init, jSprime)
          .then(B1_Bfirst_r1_i_props_init, jSprime)
          .then(B1_Bsecond_r1_r_props_init, jSprime)
          .then(B1_Bsecond_r1_i_props_init, jSprime)
          .then(B1_Bthird_r1_r_props_init, jSprime)
          .then(B1_Bthird_r1_i_props_init, jSprime)
          .then(B1_Blocal_r1_r_diquark, y)
          .then(B1_Blocal_r1_i_diquark, wnumBlock)
          .then(B1_Bfirst_r1_r_diquark, wnumBlock)
          .then(B1_Bfirst_r1_i_diquark, wnumBlock)
          .then(B1_Bthird_r1_r_diquark, wnumBlock)
          .then(B1_Bthird_r1_i_diquark, wnumBlock)
          .then(B1_Blocal_r1_r_props, wnumBlock)
          .then(B1_Blocal_r1_i_props, jSprime)
          .then(B1_Bfirst_r1_r_props, jSprime)
          .then(B1_Bfirst_r1_i_props, jSprime)
          .then(B1_Bsecond_r1_r_props, jSprime)
          .then(B1_Bsecond_r1_i_props, jSprime)
          .then(B1_Bthird_r1_r_props, jSprime)
          .then(B1_Bthird_r1_i_props, jSprime)
          .then(B1_Blocal_r1_r_update, y)
          .then(B1_Blocal_r1_i_update, m)
          .then(B1_Bfirst_r1_r_update, m)
          .then(B1_Bfirst_r1_i_update, m)
          .then(B1_Bsecond_r1_r_update, m)
          .then(B1_Bsecond_r1_i_update, m)
          .then(B1_Bthird_r1_r_update, m)
          .then(B1_Bthird_r1_i_update, m)
          .then(flip_B1_Blocal_r1_r_update, m)
          .then(flip_B1_Blocal_r1_i_update, m)
          .then(flip_B1_Bfirst_r1_r_update, m)
          .then(flip_B1_Bfirst_r1_i_update, m)
          .then(flip_B1_Bsecond_r1_r_update, m)
          .then(flip_B1_Bsecond_r1_i_update, m)
          .then(flip_B1_Bthird_r1_r_update, m)
          .then(flip_B1_Bthird_r1_i_update, m)
          .then(B1_Blocal_r2_r_init, x2)
          .then(B1_Blocal_r2_i_init, m)
          .then(B1_Bfirst_r2_r_init, m)
          .then(B1_Bfirst_r2_i_init, m)
          .then(B1_Bsecond_r2_r_init, m)
          .then(B1_Bsecond_r2_i_init, m)
          .then(B1_Bthird_r2_r_init, m)
          .then(B1_Bthird_r2_i_init, m)
          .then(flip_B1_Blocal_r2_r_init, m)
          .then(flip_B1_Blocal_r2_i_init, m)
          .then(flip_B1_Bfirst_r2_r_init, m)
          .then(flip_B1_Bfirst_r2_i_init, m)
          .then(flip_B1_Bsecond_r2_r_init, m)
          .then(flip_B1_Bsecond_r2_i_init, m)
          .then(flip_B1_Bthird_r2_r_init, m)
          .then(flip_B1_Bthird_r2_i_init, m)
          .then(B1_Blocal_r2_r_props_init, x2)
          .then(B1_Blocal_r2_i_props_init, jSprime)
          .then(B1_Bfirst_r2_r_props_init, jSprime)
          .then(B1_Bfirst_r2_i_props_init, jSprime)
          .then(B1_Bsecond_r2_r_props_init, jSprime)
          .then(B1_Bsecond_r2_i_props_init, jSprime)
          .then(B1_Bthird_r2_r_props_init, jSprime)
          .then(B1_Bthird_r2_i_props_init, jSprime)
          .then(B1_Blocal_r2_r_diquark, y)
          .then(B1_Blocal_r2_i_diquark, wnumBlock)
          .then(B1_Bfirst_r2_r_diquark, wnumBlock)
          .then(B1_Bfirst_r2_i_diquark, wnumBlock)
          .then(B1_Bthird_r2_r_diquark, wnumBlock)
          .then(B1_Bthird_r2_i_diquark, wnumBlock)
          .then(B1_Blocal_r2_r_props, wnumBlock)
          .then(B1_Blocal_r2_i_props, jSprime)
          .then(B1_Bfirst_r2_r_props, jSprime)
          .then(B1_Bfirst_r2_i_props, jSprime)
          .then(B1_Bsecond_r2_r_props, jSprime)
          .then(B1_Bsecond_r2_i_props, jSprime)
          .then(B1_Bthird_r2_r_props, jSprime)
          .then(B1_Bthird_r2_i_props, jSprime)
          .then(B1_Blocal_r2_r_update, y)
          .then(B1_Blocal_r2_i_update, m)
          .then(B1_Bfirst_r2_r_update, m)
          .then(B1_Bfirst_r2_i_update, m)
          .then(B1_Bsecond_r2_r_update, m)
          .then(B1_Bsecond_r2_i_update, m)
          .then(B1_Bthird_r2_r_update, m)
          .then(B1_Bthird_r2_i_update, m)
          .then(flip_B1_Blocal_r2_r_update, m)
          .then(flip_B1_Blocal_r2_i_update, m)
          .then(flip_B1_Bfirst_r2_r_update, m)
          .then(flip_B1_Bfirst_r2_i_update, m)
          .then(flip_B1_Bsecond_r2_r_update, m)
          .then(flip_B1_Bsecond_r2_i_update, m)
          .then(flip_B1_Bthird_r2_r_update, m)
          .then(flip_B1_Bthird_r2_i_update, m)
          .then(B2_Blocal_r1_r_init, x2)
          .then(B2_Blocal_r1_i_init, m)
          .then(B2_Bfirst_r1_r_init, m)
          .then(B2_Bfirst_r1_i_init, m)
          .then(B2_Bsecond_r1_r_init, m)
          .then(B2_Bsecond_r1_i_init, m)
          .then(B2_Bthird_r1_r_init, m)
          .then(B2_Bthird_r1_i_init, m)
          .then(flip_B2_Blocal_r1_r_init, m)
          .then(flip_B2_Blocal_r1_i_init, m)
          .then(flip_B2_Bfirst_r1_r_init, m)
          .then(flip_B2_Bfirst_r1_i_init, m)
          .then(flip_B2_Bsecond_r1_r_init, m)
          .then(flip_B2_Bsecond_r1_i_init, m)
          .then(flip_B2_Bthird_r1_r_init, m)
          .then(flip_B2_Bthird_r1_i_init, m)
          .then(B2_Blocal_r1_r_props_init, x2)
          .then(B2_Blocal_r1_i_props_init, jSprime)
          .then(B2_Bfirst_r1_r_props_init, jSprime)
          .then(B2_Bfirst_r1_i_props_init, jSprime)
          .then(B2_Bsecond_r1_r_props_init, jSprime)
          .then(B2_Bsecond_r1_i_props_init, jSprime)
          .then(B2_Bthird_r1_r_props_init, jSprime)
          .then(B2_Bthird_r1_i_props_init, jSprime)
          .then(B2_Blocal_r1_r_diquark, y)
          .then(B2_Blocal_r1_i_diquark, wnumBlock)
          .then(B2_Bfirst_r1_r_diquark, wnumBlock)
          .then(B2_Bfirst_r1_i_diquark, wnumBlock)
          .then(B2_Bthird_r1_r_diquark, wnumBlock)
          .then(B2_Bthird_r1_i_diquark, wnumBlock)
          .then(B2_Blocal_r1_r_props, wnumBlock)
          .then(B2_Blocal_r1_i_props, jSprime)
          .then(B2_Bfirst_r1_r_props, jSprime)
          .then(B2_Bfirst_r1_i_props, jSprime)
          .then(B2_Bsecond_r1_r_props, jSprime)
          .then(B2_Bsecond_r1_i_props, jSprime)
          .then(B2_Bthird_r1_r_props, jSprime)
          .then(B2_Bthird_r1_i_props, jSprime)
          .then(B2_Blocal_r1_r_update, y)
          .then(B2_Blocal_r1_i_update, m)
          .then(B2_Bfirst_r1_r_update, m)
          .then(B2_Bfirst_r1_i_update, m)
          .then(B2_Bsecond_r1_r_update, m)
          .then(B2_Bsecond_r1_i_update, m)
          .then(B2_Bthird_r1_r_update, m)
          .then(B2_Bthird_r1_i_update, m)
          .then(flip_B2_Blocal_r1_r_update, m)
          .then(flip_B2_Blocal_r1_i_update, m)
          .then(flip_B2_Bfirst_r1_r_update, m)
          .then(flip_B2_Bfirst_r1_i_update, m)
          .then(flip_B2_Bsecond_r1_r_update, m)
          .then(flip_B2_Bsecond_r1_i_update, m)
          .then(flip_B2_Bthird_r1_r_update, m)
          .then(flip_B2_Bthird_r1_i_update, m)
          .then(B2_Blocal_r2_r_init, x2)
          .then(B2_Blocal_r2_i_init, m)
          .then(B2_Bfirst_r2_r_init, m)
          .then(B2_Bfirst_r2_i_init, m)
          .then(B2_Bsecond_r2_r_init, m)
          .then(B2_Bsecond_r2_i_init, m)
          .then(B2_Bthird_r2_r_init, m)
          .then(B2_Bthird_r2_i_init, m)
          .then(flip_B2_Blocal_r2_r_init, m)
          .then(flip_B2_Blocal_r2_i_init, m)
          .then(flip_B2_Bfirst_r2_r_init, m)
          .then(flip_B2_Bfirst_r2_i_init, m)
          .then(flip_B2_Bsecond_r2_r_init, m)
          .then(flip_B2_Bsecond_r2_i_init, m)
          .then(flip_B2_Bthird_r2_r_init, m)
          .then(flip_B2_Bthird_r2_i_init, m)
          .then(B2_Blocal_r2_r_props_init, x2)
          .then(B2_Blocal_r2_i_props_init, jSprime)
          .then(B2_Bfirst_r2_r_props_init, jSprime)
          .then(B2_Bfirst_r2_i_props_init, jSprime)
          .then(B2_Bsecond_r2_r_props_init, jSprime)
          .then(B2_Bsecond_r2_i_props_init, jSprime)
          .then(B2_Bthird_r2_r_props_init, jSprime)
          .then(B2_Bthird_r2_i_props_init, jSprime)
          .then(B2_Blocal_r2_r_diquark, y)
          .then(B2_Blocal_r2_i_diquark, wnumBlock)
          .then(B2_Bfirst_r2_r_diquark, wnumBlock)
          .then(B2_Bfirst_r2_i_diquark, wnumBlock)
          .then(B2_Bthird_r2_r_diquark, wnumBlock)
          .then(B2_Bthird_r2_i_diquark, wnumBlock)
          .then(B2_Blocal_r2_r_props, wnumBlock)
          .then(B2_Blocal_r2_i_props, jSprime)
          .then(B2_Bfirst_r2_r_props, jSprime)
          .then(B2_Bfirst_r2_i_props, jSprime)
          .then(B2_Bsecond_r2_r_props, jSprime)
          .then(B2_Bsecond_r2_i_props, jSprime)
          .then(B2_Bthird_r2_r_props, jSprime)
          .then(B2_Bthird_r2_i_props, jSprime)
          .then(B2_Blocal_r2_r_update, y)
          .then(B2_Blocal_r2_i_update, m)
          .then(B2_Bfirst_r2_r_update, m)
          .then(B2_Bfirst_r2_i_update, m) 
          .then(B2_Bsecond_r2_r_update, m)
          .then(B2_Bsecond_r2_i_update, m)
          .then(B2_Bthird_r2_r_update, m)
          .then(B2_Bthird_r2_i_update, m) 
          .then(flip_B2_Blocal_r2_r_update, m)
          .then(flip_B2_Blocal_r2_i_update, m)
          .then(flip_B2_Bfirst_r2_r_update, m)
          .then(flip_B2_Bfirst_r2_i_update, m) 
          .then(flip_B2_Bsecond_r2_r_update, m)
          .then(flip_B2_Bsecond_r2_i_update, m)
          .then(flip_B2_Bthird_r2_r_update, m)
          .then(flip_B2_Bthird_r2_i_update, m) 
          .then(C_BB_BB_prop_init_r, x2)
          .then(C_BB_BB_prop_init_i, r)
          .then( *(BB_BB_new_term_0_r1_b1.get_real()), r)
          .then( *(BB_BB_new_term_0_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_1_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_1_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_2_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_2_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_3_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_3_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_4_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_4_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_5_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_5_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_6_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_6_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_7_r1_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_7_r1_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_0_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_0_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_1_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_1_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_2_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_2_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_3_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_3_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_4_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_4_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_5_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_5_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_6_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_6_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_7_r2_b1.get_real()), wnum)
          .then( *(BB_BB_new_term_7_r2_b1.get_imag()), wnum)
          .then( *(BB_BB_new_term_0_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_0_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_1_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_1_r1_b2.get_imag()), wnum) 
          .then( *(BB_BB_new_term_2_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_2_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_3_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_3_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_4_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_4_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_5_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_5_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_6_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_6_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_7_r1_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_7_r1_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_0_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_0_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_1_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_1_r2_b2.get_imag()), wnum) 
          .then( *(BB_BB_new_term_2_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_2_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_3_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_3_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_4_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_4_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_5_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_5_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_6_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_6_r2_b2.get_imag()), wnum)
          .then( *(BB_BB_new_term_7_r2_b2.get_real()), wnum)
          .then( *(BB_BB_new_term_7_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_0_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_0_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_1_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_1_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_2_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_2_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_3_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_3_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_4_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_4_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_5_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_5_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_6_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_6_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_7_r1_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_7_r1_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_0_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_0_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_1_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_1_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_2_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_2_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_3_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_3_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_4_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_4_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_5_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_5_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_6_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_6_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_7_r2_b1.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_7_r2_b1.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_0_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_0_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_1_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_1_r1_b2.get_imag()), wnum) 
          .then( *(flip_BB_BB_new_term_2_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_2_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_3_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_3_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_4_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_4_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_5_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_5_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_6_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_6_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_7_r1_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_7_r1_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_0_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_0_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_1_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_1_r2_b2.get_imag()), wnum) 
          .then( *(flip_BB_BB_new_term_2_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_2_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_3_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_3_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_4_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_4_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_5_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_5_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_6_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_6_r2_b2.get_imag()), wnum)
          .then( *(flip_BB_BB_new_term_7_r2_b2.get_real()), wnum)
          .then( *(flip_BB_BB_new_term_7_r2_b2.get_imag()), wnum)
          .then(C_BB_BB_prop_update_r, wnum) 
          .then(C_BB_BB_prop_update_i, wnum)
          .then(C_BB_BB_update_b_r, r) 
          .then(C_BB_BB_update_b_i, ne)
          .then(C_BB_BB_update_s_r, r) 
          .then(C_BB_BB_update_s_i, nue)
          );

    // BB_H
    handle = &(handle
          ->then(src_B1_Blocal_r1_r_init, t)
          .then(src_B1_Blocal_r1_i_init, jSprime)
          .then(flip_src_B1_Blocal_r1_r_init, jSprime)
          .then(flip_src_B1_Blocal_r1_i_init, jSprime)
          .then(src_B1_Blocal_r1_r_props_init, x_in)
          .then(src_B1_Blocal_r1_i_props_init, jSprime)
          .then(src_B1_Blocal_r1_r_diquark, y)
          .then(src_B1_Blocal_r1_i_diquark, wnumBlock)
          .then(src_B1_Blocal_r1_r_props, wnumBlock)
          .then(src_B1_Blocal_r1_i_props, jSprime)
          .then(src_B1_Blocal_r1_r_update, y)
          .then(src_B1_Blocal_r1_i_update, m)
          .then(flip_src_B1_Blocal_r1_r_update, m)
          .then(flip_src_B1_Blocal_r1_i_update, m)
          .then(src_B1_Blocal_r2_r_init, x_in)
          .then(src_B1_Blocal_r2_i_init, jSprime)
          .then(flip_src_B1_Blocal_r2_r_init, jSprime)
          .then(flip_src_B1_Blocal_r2_i_init, jSprime)
          .then(src_B1_Blocal_r2_r_props_init, x_in)
          .then(src_B1_Blocal_r2_i_props_init, jSprime)
          .then(src_B1_Blocal_r2_r_diquark, y)
          .then(src_B1_Blocal_r2_i_diquark, wnumBlock)
          .then(src_B1_Blocal_r2_r_props, wnumBlock)
          .then(src_B1_Blocal_r2_i_props, jSprime)
          .then(src_B1_Blocal_r2_r_update, y)
          .then(src_B1_Blocal_r2_i_update, m)
          .then(flip_src_B1_Blocal_r2_r_update, m)
          .then(flip_src_B1_Blocal_r2_i_update, m)
          .then(src_B2_Blocal_r1_r_init, x_in)
          .then(src_B2_Blocal_r1_i_init, jSprime)
          .then(flip_src_B2_Blocal_r1_r_init, jSprime)
          .then(flip_src_B2_Blocal_r1_i_init, jSprime)
          .then(src_B2_Blocal_r1_r_props_init, x_in)
          .then(src_B2_Blocal_r1_i_props_init, jSprime)
          .then(src_B2_Blocal_r1_r_diquark, y)
          .then(src_B2_Blocal_r1_i_diquark, wnumBlock)
          .then(src_B2_Blocal_r1_r_props, wnumBlock)
          .then(src_B2_Blocal_r1_i_props, jSprime)
          .then(src_B2_Blocal_r1_r_update, y)
          .then(src_B2_Blocal_r1_i_update, m)
          .then(flip_src_B2_Blocal_r1_r_update, m)
          .then(flip_src_B2_Blocal_r1_i_update, m)
          .then(src_B2_Blocal_r2_r_init, x_in)
          .then(src_B2_Blocal_r2_i_init, jSprime)
          .then(flip_src_B2_Blocal_r2_r_init, jSprime)
          .then(flip_src_B2_Blocal_r2_i_init, jSprime)
          .then(src_B2_Blocal_r2_r_props_init, x_in)
          .then(src_B2_Blocal_r2_i_props_init, jSprime)
          .then(src_B2_Blocal_r2_r_diquark, y)
          .then(src_B2_Blocal_r2_i_diquark, wnumBlock)
          .then(src_B2_Blocal_r2_r_props, wnumBlock)
          .then(src_B2_Blocal_r2_i_props, jSprime)
          .then(src_B2_Blocal_r2_r_update, y)
          .then(src_B2_Blocal_r2_i_update, m)
          .then(flip_src_B2_Blocal_r2_r_update, m)
          .then(flip_src_B2_Blocal_r2_i_update, m) 
         .then(C_BB_H_prop_init_r, x_in)
          .then(C_BB_H_prop_init_i, r) 
          .then( *(BB_H_new_term_0_r1_b1.get_real()), r)
          .then( *(BB_H_new_term_0_r1_b1.get_imag()), wnumHex)
          .then( *(BB_H_new_term_0_r2_b1.get_real()), wnumHex)
          .then( *(BB_H_new_term_0_r2_b1.get_imag()), wnumHex)
          .then( *(BB_H_new_term_0_r1_b2.get_real()), wnumHex)
          .then( *(BB_H_new_term_0_r1_b2.get_imag()), wnumHex)
          .then( *(BB_H_new_term_0_r2_b2.get_real()), wnumHex)
          .then( *(BB_H_new_term_0_r2_b2.get_imag()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r1_b1.get_real()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r1_b1.get_imag()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r2_b1.get_real()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r2_b1.get_imag()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r1_b2.get_real()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r1_b2.get_imag()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r2_b2.get_real()), wnumHex)
          .then( *(flip_BB_H_new_term_0_r2_b2.get_imag()), wnumHex)
          .then(C_BB_H_prop_update_r, wnumHex) 
          .then(C_BB_H_prop_update_i, wnumHex) 
          .then(C_BB_H_update_r, r) 
          .then(C_BB_H_update_i, nH)
          );

    // H_BB
    handle = &(handle
          ->then( snk_B1_Blocal_r1_r_init, t)
          .then(snk_B1_Blocal_r1_i_init, jSprime)
          .then(flip_snk_B1_Blocal_r1_r_init, jSprime)
          .then(flip_snk_B1_Blocal_r1_i_init, jSprime)
          .then(snk_B1_Blocal_r1_r_props_init, y_in)
          .then(snk_B1_Blocal_r1_i_props_init, jSprime)
          .then(snk_B1_Blocal_r1_r_diquark, x)
          .then(snk_B1_Blocal_r1_i_diquark, wnumBlock)
          .then(snk_B1_Blocal_r1_r_props, wnumBlock)
          .then(snk_B1_Blocal_r1_i_props, jSprime)
          .then(snk_B1_Blocal_r1_r_update, x)
          .then(snk_B1_Blocal_r1_i_update, n)
          .then(flip_snk_B1_Blocal_r1_r_update, n)
          .then(flip_snk_B1_Blocal_r1_i_update, n)
          .then(snk_B1_Blocal_r2_r_init, y_in)
          .then(snk_B1_Blocal_r2_i_init, jSprime)
          .then(flip_snk_B1_Blocal_r2_r_init, jSprime)
          .then(flip_snk_B1_Blocal_r2_i_init, jSprime)
          .then(snk_B1_Blocal_r2_r_props_init, y_in)
          .then(snk_B1_Blocal_r2_i_props_init, jSprime)
          .then(snk_B1_Blocal_r2_r_diquark, x)
          .then(snk_B1_Blocal_r2_i_diquark, wnumBlock)
          .then(snk_B1_Blocal_r2_r_props, wnumBlock)
          .then(snk_B1_Blocal_r2_i_props, jSprime)
          .then(snk_B1_Blocal_r2_r_update, x)
          .then(snk_B1_Blocal_r2_i_update, n)
          .then(flip_snk_B1_Blocal_r2_r_update, n)
          .then(flip_snk_B1_Blocal_r2_i_update, n)
          .then(snk_B2_Blocal_r1_r_init, y_in)
          .then(snk_B2_Blocal_r1_i_init, jSprime)
          .then(flip_snk_B2_Blocal_r1_r_init, jSprime)
          .then(flip_snk_B2_Blocal_r1_i_init, jSprime)
          .then(snk_B2_Blocal_r1_r_props_init, y_in)
          .then(snk_B2_Blocal_r1_i_props_init, jSprime)
          .then(snk_B2_Blocal_r1_r_diquark, x)
          .then(snk_B2_Blocal_r1_i_diquark, wnumBlock)
          .then(snk_B2_Blocal_r1_r_props, wnumBlock)
          .then(snk_B2_Blocal_r1_i_props, jSprime)
          .then(snk_B2_Blocal_r1_r_update, x)
          .then(snk_B2_Blocal_r1_i_update, n)
          .then(flip_snk_B2_Blocal_r1_r_update, n)
          .then(flip_snk_B2_Blocal_r1_i_update, n)
          .then(snk_B2_Blocal_r2_r_init, y_in)
          .then(snk_B2_Blocal_r2_i_init, jSprime)
          .then(flip_snk_B2_Blocal_r2_r_init, jSprime)
          .then(flip_snk_B2_Blocal_r2_i_init, jSprime)
          .then(snk_B2_Blocal_r2_r_props_init, y_in)
          .then(snk_B2_Blocal_r2_i_props_init, jSprime)
          .then(snk_B2_Blocal_r2_r_diquark, x)
          .then(snk_B2_Blocal_r2_i_diquark, wnumBlock)
          .then(snk_B2_Blocal_r2_r_props, wnumBlock)
          .then(snk_B2_Blocal_r2_i_props, jSprime)
          .then(snk_B2_Blocal_r2_r_update, x)
          .then(flip_snk_B2_Blocal_r2_r_update, n)
          .then(flip_snk_B2_Blocal_r2_i_update, n)
          .then(snk_B2_Blocal_r2_i_update, n)
          .then(C_H_BB_prop_init_r, y_in)
          .then(C_H_BB_prop_init_i, r)
          .then( *(H_BB_new_term_0_r1_b1.get_real()), r)
          .then( *(H_BB_new_term_0_r1_b1.get_imag()), wnumHex)
          .then( *(H_BB_new_term_0_r2_b1.get_real()), wnumHex)
          .then( *(H_BB_new_term_0_r2_b1.get_imag()), wnumHex)
          .then( *(H_BB_new_term_0_r1_b2.get_real()), wnumHex)
          .then( *(H_BB_new_term_0_r1_b2.get_imag()), wnumHex)
          .then( *(H_BB_new_term_0_r2_b2.get_real()), wnumHex)
          .then( *(H_BB_new_term_0_r2_b2.get_imag()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r1_b1.get_real()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r1_b1.get_imag()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r2_b1.get_real()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r2_b1.get_imag()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r1_b2.get_real()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r1_b2.get_imag()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r2_b2.get_real()), wnumHex)
          .then( *(flip_H_BB_new_term_0_r2_b2.get_imag()), wnumHex)
          .then(C_H_BB_prop_update_r, wnumHex) 
          .then(C_H_BB_prop_update_i, wnumHex)
          .then(C_H_BB_update_r, r) 
          .then(C_H_BB_update_i, mH) 
          ); 

    // H_H
    handle = &(handle
          ->then(C_H_H_prop_init_r, t)
          .then(C_H_H_prop_init_i, y)
          .then(C_H_H_prop_update_r, y) 
          .then(C_H_H_prop_update_i, wnumHexHex)
          .then(C_H_H_update_r, y) 
          .then(C_H_H_update_i, nH) 
          ); 

#if VECTORIZED

    (BB_BB_new_term_0_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_H_new_term_0_r1_b1.get_real())->tag_vector_level(wnumHex, Nw2Hex);
    (H_BB_new_term_0_r1_b1.get_real())->tag_vector_level(wnumHex, Nw2Hex);
    C_H_H_prop_update_r.tag_vector_level(wnumHexHex, Nw2Hex);

#endif

#if PARALLEL

    B1_Blocal_r1_r_init.tag_distribute_level(t);
    src_B1_Blocal_r1_r_init.tag_distribute_level(t);
    snk_B1_Blocal_r1_r_init.tag_distribute_level(t);
    C_H_H_prop_init_r.tag_distribute_level(t);


#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_r1_r("buf_B1_Bfirst_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_r1_i("buf_B1_Bfirst_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_r1_r("buf_B1_Bsecond_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_r1_i("buf_B1_Bsecond_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bthird_r1_r("buf_B1_Bthird_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bthird_r1_i("buf_B1_Bthird_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r1_r_init.store_in(&buf_B1_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r1_i_init.store_in(&buf_B1_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r1_r_init.store_in(&buf_B1_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r1_i_init.store_in(&buf_B1_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r1_r_init.store_in(&buf_B1_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r1_i_init.store_in(&buf_B1_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_r_update.store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_update.store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r1_r_update.store_in(&buf_B1_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r1_i_update.store_in(&buf_B1_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r1_r_update.store_in(&buf_B1_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r1_i_update.store_in(&buf_B1_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r1_r_update.store_in(&buf_B1_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r1_i_update.store_in(&buf_B1_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_B1_Blocal_r1_r("buf_flip_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Blocal_r1_i("buf_flip_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bfirst_r1_r("buf_flip_B1_Bfirst_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bfirst_r1_i("buf_flip_B1_Bfirst_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bsecond_r1_r("buf_flip_B1_Bsecond_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bsecond_r1_i("buf_flip_B1_Bsecond_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bthird_r1_r("buf_flip_B1_Bthird_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bthird_r1_i("buf_flip_B1_Bthird_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_B1_Blocal_r1_r_init.store_in(&buf_flip_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r1_i_init.store_in(&buf_flip_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r1_r_init.store_in(&buf_flip_B1_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r1_i_init.store_in(&buf_flip_B1_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r1_r_init.store_in(&buf_flip_B1_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r1_i_init.store_in(&buf_flip_B1_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r1_r_init.store_in(&buf_flip_B1_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r1_i_init.store_in(&buf_flip_B1_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r1_r_update.store_in(&buf_flip_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r1_i_update.store_in(&buf_flip_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r1_r_update.store_in(&buf_flip_B1_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r1_i_update.store_in(&buf_flip_B1_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    flip_B1_Bsecond_r1_r_update.store_in(&buf_flip_B1_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r1_i_update.store_in(&buf_flip_B1_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r1_r_update.store_in(&buf_flip_B1_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r1_i_update.store_in(&buf_flip_B1_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    buffer buf_B1_Blocal_diquark_r1_r("buf_B1_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Blocal_diquark_r1_i("buf_B1_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_diquark_r1_r("buf_B1_Bfirst_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_diquark_r1_i("buf_B1_Bfirst_diquark_r1_i",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bthird_diquark_r1_r("buf_B1_Bthird_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bthird_diquark_r1_i("buf_B1_Bthird_diquark_r1_i",   {1}, p_float64, a_temporary);
    B1_Blocal_r1_r_diquark.store_in(&buf_B1_Blocal_diquark_r1_r, {0});
    B1_Blocal_r1_i_diquark.store_in(&buf_B1_Blocal_diquark_r1_i, {0});
    B1_Bfirst_r1_r_diquark.store_in(&buf_B1_Bfirst_diquark_r1_r, {0});
    B1_Bfirst_r1_i_diquark.store_in(&buf_B1_Bfirst_diquark_r1_i, {0}); 
    B1_Bthird_r1_r_diquark.store_in(&buf_B1_Bthird_diquark_r1_r, {0});
    B1_Bthird_r1_i_diquark.store_in(&buf_B1_Bthird_diquark_r1_i, {0}); 
    buffer buf_B1_Blocal_props_r1_r("buf_B1_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_props_r1_i("buf_B1_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_props_r1_r("buf_B1_Bfirst_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_props_r1_i("buf_B1_Bfirst_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_props_r1_r("buf_B1_Bsecond_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_props_r1_i("buf_B1_Bsecond_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bthird_props_r1_r("buf_B1_Bthird_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bthird_props_r1_i("buf_B1_Bthird_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    B1_Blocal_r1_r_props_init.store_in(&buf_B1_Blocal_props_r1_r, {jCprime, jSprime});
    B1_Blocal_r1_i_props_init.store_in(&buf_B1_Blocal_props_r1_i, {jCprime, jSprime});
    B1_Bfirst_r1_r_props_init.store_in(&buf_B1_Bfirst_props_r1_r, {jCprime, jSprime});
    B1_Bfirst_r1_i_props_init.store_in(&buf_B1_Bfirst_props_r1_i, {jCprime, jSprime});
    B1_Bsecond_r1_r_props_init.store_in(&buf_B1_Bsecond_props_r1_r, {jCprime, jSprime});
    B1_Bsecond_r1_i_props_init.store_in(&buf_B1_Bsecond_props_r1_i, {jCprime, jSprime});
    B1_Bthird_r1_r_props_init.store_in(&buf_B1_Bthird_props_r1_r, {jCprime, jSprime});
    B1_Bthird_r1_i_props_init.store_in(&buf_B1_Bthird_props_r1_i, {jCprime, jSprime});
    B1_Blocal_r1_r_props.store_in(&buf_B1_Blocal_props_r1_r, {jCprime, jSprime});
    B1_Blocal_r1_i_props.store_in(&buf_B1_Blocal_props_r1_i, {jCprime, jSprime});
    B1_Bfirst_r1_r_props.store_in(&buf_B1_Bfirst_props_r1_r, {jCprime, jSprime});
    B1_Bfirst_r1_i_props.store_in(&buf_B1_Bfirst_props_r1_i, {jCprime, jSprime}); 
    B1_Bsecond_r1_r_props.store_in(&buf_B1_Bsecond_props_r1_r, {jCprime, jSprime});
    B1_Bsecond_r1_i_props.store_in(&buf_B1_Bsecond_props_r1_i, {jCprime, jSprime});
    B1_Bthird_r1_r_props.store_in(&buf_B1_Bthird_props_r1_r, {jCprime, jSprime});
    B1_Bthird_r1_i_props.store_in(&buf_B1_Bthird_props_r1_i, {jCprime, jSprime}); 
    
    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_r2_r("buf_B1_Bfirst_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_r2_i("buf_B1_Bfirst_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_r2_r("buf_B1_Bsecond_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_r2_i("buf_B1_Bsecond_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bthird_r2_r("buf_B1_Bthird_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B1_Bthird_r2_i("buf_B1_Bthird_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r2_r_init.store_in(&buf_B1_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r2_i_init.store_in(&buf_B1_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r2_r_init.store_in(&buf_B1_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r2_i_init.store_in(&buf_B1_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r2_r_init.store_in(&buf_B1_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r2_i_init.store_in(&buf_B1_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_r_update.store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_update.store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r2_r_update.store_in(&buf_B1_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bfirst_r2_i_update.store_in(&buf_B1_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r2_r_update.store_in(&buf_B1_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bsecond_r2_i_update.store_in(&buf_B1_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r2_r_update.store_in(&buf_B1_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Bthird_r2_i_update.store_in(&buf_B1_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_B1_Blocal_r2_r("buf_flip_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Blocal_r2_i("buf_flip_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bfirst_r2_r("buf_flip_B1_Bfirst_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bfirst_r2_i("buf_flip_B1_Bfirst_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bsecond_r2_r("buf_flip_B1_Bsecond_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bsecond_r2_i("buf_flip_B1_Bsecond_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bthird_r2_r("buf_flip_B1_Bthird_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B1_Bthird_r2_i("buf_flip_B1_Bthird_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_B1_Blocal_r2_r_init.store_in(&buf_flip_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r2_i_init.store_in(&buf_flip_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r2_r_init.store_in(&buf_flip_B1_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r2_i_init.store_in(&buf_flip_B1_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r2_r_init.store_in(&buf_flip_B1_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r2_i_init.store_in(&buf_flip_B1_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r2_r_init.store_in(&buf_flip_B1_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r2_i_init.store_in(&buf_flip_B1_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r2_r_update.store_in(&buf_flip_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Blocal_r2_i_update.store_in(&buf_flip_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r2_r_update.store_in(&buf_flip_B1_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bfirst_r2_i_update.store_in(&buf_flip_B1_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    flip_B1_Bsecond_r2_r_update.store_in(&buf_flip_B1_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bsecond_r2_i_update.store_in(&buf_flip_B1_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r2_r_update.store_in(&buf_flip_B1_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B1_Bthird_r2_i_update.store_in(&buf_flip_B1_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    buffer buf_B1_Blocal_diquark_r2_r("buf_B1_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Blocal_diquark_r2_i("buf_B1_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_diquark_r2_r("buf_B1_Bfirst_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_diquark_r2_i("buf_B1_Bfirst_diquark_r2_i",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bthird_diquark_r2_r("buf_B1_Bthird_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B1_Bthird_diquark_r2_i("buf_B1_Bthird_diquark_r2_i",   {1}, p_float64, a_temporary);
    B1_Blocal_r2_r_diquark.store_in(&buf_B1_Blocal_diquark_r2_r, {0});
    B1_Blocal_r2_i_diquark.store_in(&buf_B1_Blocal_diquark_r2_i, {0});
    B1_Bfirst_r2_r_diquark.store_in(&buf_B1_Bfirst_diquark_r2_r, {0});
    B1_Bfirst_r2_i_diquark.store_in(&buf_B1_Bfirst_diquark_r2_i, {0}); 
    B1_Bthird_r2_r_diquark.store_in(&buf_B1_Bthird_diquark_r2_r, {0});
    B1_Bthird_r2_i_diquark.store_in(&buf_B1_Bthird_diquark_r2_i, {0}); 
    buffer buf_B1_Blocal_props_r2_r("buf_B1_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_props_r2_i("buf_B1_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_props_r2_r("buf_B1_Bfirst_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bfirst_props_r2_i("buf_B1_Bfirst_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_props_r2_r("buf_B1_Bsecond_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsecond_props_r2_i("buf_B1_Bsecond_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bthird_props_r2_r("buf_B1_Bthird_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bthird_props_r2_i("buf_B1_Bthird_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    B1_Blocal_r2_r_props_init.store_in(&buf_B1_Blocal_props_r2_r, {jCprime, jSprime});
    B1_Blocal_r2_i_props_init.store_in(&buf_B1_Blocal_props_r2_i, {jCprime, jSprime});
    B1_Bfirst_r2_r_props_init.store_in(&buf_B1_Bfirst_props_r2_r, {jCprime, jSprime});
    B1_Bfirst_r2_i_props_init.store_in(&buf_B1_Bfirst_props_r2_i, {jCprime, jSprime});
    B1_Bsecond_r2_r_props_init.store_in(&buf_B1_Bsecond_props_r2_r, {jCprime, jSprime});
    B1_Bsecond_r2_i_props_init.store_in(&buf_B1_Bsecond_props_r2_i, {jCprime, jSprime});
    B1_Bthird_r2_r_props_init.store_in(&buf_B1_Bthird_props_r2_r, {jCprime, jSprime});
    B1_Bthird_r2_i_props_init.store_in(&buf_B1_Bthird_props_r2_i, {jCprime, jSprime});
    B1_Blocal_r2_r_props.store_in(&buf_B1_Blocal_props_r2_r, {jCprime, jSprime});
    B1_Blocal_r2_i_props.store_in(&buf_B1_Blocal_props_r2_i, {jCprime, jSprime});
    B1_Bfirst_r2_r_props.store_in(&buf_B1_Bfirst_props_r2_r, {jCprime, jSprime});
    B1_Bfirst_r2_i_props.store_in(&buf_B1_Bfirst_props_r2_i, {jCprime, jSprime}); 
    B1_Bsecond_r2_r_props.store_in(&buf_B1_Bsecond_props_r2_r, {jCprime, jSprime});
    B1_Bsecond_r2_i_props.store_in(&buf_B1_Bsecond_props_r2_i, {jCprime, jSprime});
    B1_Bthird_r2_r_props.store_in(&buf_B1_Bthird_props_r2_r, {jCprime, jSprime});
    B1_Bthird_r2_i_props.store_in(&buf_B1_Bthird_props_r2_i, {jCprime, jSprime}); 
    
    buffer buf_B2_Blocal_r1_r("buf_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r1_i("buf_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_r1_r("buf_B2_Bfirst_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_r1_i("buf_B2_Bfirst_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_r1_r("buf_B2_Bsecond_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_r1_i("buf_B2_Bsecond_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bthird_r1_r("buf_B2_Bthird_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bthird_r1_i("buf_B2_Bthird_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    B2_Blocal_r1_r_init.store_in(&buf_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r1_i_init.store_in(&buf_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r1_r_init.store_in(&buf_B2_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r1_i_init.store_in(&buf_B2_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r1_r_init.store_in(&buf_B2_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r1_i_init.store_in(&buf_B2_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r1_r_init.store_in(&buf_B2_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r1_i_init.store_in(&buf_B2_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r1_r_update.store_in(&buf_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r1_i_update.store_in(&buf_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r1_r_update.store_in(&buf_B2_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r1_i_update.store_in(&buf_B2_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r1_r_update.store_in(&buf_B2_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r1_i_update.store_in(&buf_B2_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r1_r_update.store_in(&buf_B2_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r1_i_update.store_in(&buf_B2_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_B2_Blocal_r1_r("buf_flip_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Blocal_r1_i("buf_flip_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bfirst_r1_r("buf_flip_B2_Bfirst_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bfirst_r1_i("buf_flip_B2_Bfirst_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bsecond_r1_r("buf_flip_B2_Bsecond_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bsecond_r1_i("buf_flip_B2_Bsecond_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bthird_r1_r("buf_flip_B2_Bthird_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bthird_r1_i("buf_flip_B2_Bthird_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_B2_Blocal_r1_r_init.store_in(&buf_flip_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r1_i_init.store_in(&buf_flip_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r1_r_init.store_in(&buf_flip_B2_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r1_i_init.store_in(&buf_flip_B2_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r1_r_init.store_in(&buf_flip_B2_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r1_i_init.store_in(&buf_flip_B2_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r1_r_init.store_in(&buf_flip_B2_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r1_i_init.store_in(&buf_flip_B2_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r1_r_update.store_in(&buf_flip_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r1_i_update.store_in(&buf_flip_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r1_r_update.store_in(&buf_flip_B2_Bfirst_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r1_i_update.store_in(&buf_flip_B2_Bfirst_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    flip_B2_Bsecond_r1_r_update.store_in(&buf_flip_B2_Bsecond_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r1_i_update.store_in(&buf_flip_B2_Bsecond_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r1_r_update.store_in(&buf_flip_B2_Bthird_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r1_i_update.store_in(&buf_flip_B2_Bthird_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    buffer buf_B2_Blocal_diquark_r1_r("buf_B2_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Blocal_diquark_r1_i("buf_B2_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_diquark_r1_r("buf_B2_Bfirst_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_diquark_r1_i("buf_B2_Bfirst_diquark_r1_i",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bthird_diquark_r1_r("buf_B2_Bthird_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bthird_diquark_r1_i("buf_B2_Bthird_diquark_r1_i",   {1}, p_float64, a_temporary);
    B2_Blocal_r1_r_diquark.store_in(&buf_B2_Blocal_diquark_r1_r, {0});
    B2_Blocal_r1_i_diquark.store_in(&buf_B2_Blocal_diquark_r1_i, {0});
    B2_Bfirst_r1_r_diquark.store_in(&buf_B2_Bfirst_diquark_r1_r, {0});
    B2_Bfirst_r1_i_diquark.store_in(&buf_B2_Bfirst_diquark_r1_i, {0}); 
    B2_Bthird_r1_r_diquark.store_in(&buf_B2_Bthird_diquark_r1_r, {0});
    B2_Bthird_r1_i_diquark.store_in(&buf_B2_Bthird_diquark_r1_i, {0}); 
    buffer buf_B2_Blocal_props_r1_r("buf_B2_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_props_r1_i("buf_B2_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_props_r1_r("buf_B2_Bfirst_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_props_r1_i("buf_B2_Bfirst_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_props_r1_r("buf_B2_Bsecond_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_props_r1_i("buf_B2_Bsecond_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bthird_props_r1_r("buf_B2_Bthird_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bthird_props_r1_i("buf_B2_Bthird_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    B2_Blocal_r1_r_props_init.store_in(&buf_B2_Blocal_props_r1_r, {jCprime, jSprime});
    B2_Blocal_r1_i_props_init.store_in(&buf_B2_Blocal_props_r1_i, {jCprime, jSprime});
    B2_Bfirst_r1_r_props_init.store_in(&buf_B2_Bfirst_props_r1_r, {jCprime, jSprime});
    B2_Bfirst_r1_i_props_init.store_in(&buf_B2_Bfirst_props_r1_i, {jCprime, jSprime});
    B2_Bsecond_r1_r_props_init.store_in(&buf_B2_Bsecond_props_r1_r, {jCprime, jSprime});
    B2_Bsecond_r1_i_props_init.store_in(&buf_B2_Bsecond_props_r1_i, {jCprime, jSprime});
    B2_Bthird_r1_r_props_init.store_in(&buf_B2_Bthird_props_r1_r, {jCprime, jSprime});
    B2_Bthird_r1_i_props_init.store_in(&buf_B2_Bthird_props_r1_i, {jCprime, jSprime});
    B2_Blocal_r1_r_props.store_in(&buf_B2_Blocal_props_r1_r, {jCprime, jSprime});
    B2_Blocal_r1_i_props.store_in(&buf_B2_Blocal_props_r1_i, {jCprime, jSprime});
    B2_Bfirst_r1_r_props.store_in(&buf_B2_Bfirst_props_r1_r, {jCprime, jSprime});
    B2_Bfirst_r1_i_props.store_in(&buf_B2_Bfirst_props_r1_i, {jCprime, jSprime}); 
    B2_Bsecond_r1_r_props.store_in(&buf_B2_Bsecond_props_r1_r, {jCprime, jSprime});
    B2_Bsecond_r1_i_props.store_in(&buf_B2_Bsecond_props_r1_i, {jCprime, jSprime});
    B2_Bthird_r1_r_props.store_in(&buf_B2_Bthird_props_r1_r, {jCprime, jSprime});
    B2_Bthird_r1_i_props.store_in(&buf_B2_Bthird_props_r1_i, {jCprime, jSprime}); 
    
    buffer buf_B2_Blocal_r2_r("buf_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r2_i("buf_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_r2_r("buf_B2_Bfirst_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_r2_i("buf_B2_Bfirst_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_r2_r("buf_B2_Bsecond_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_r2_i("buf_B2_Bsecond_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bthird_r2_r("buf_B2_Bthird_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_B2_Bthird_r2_i("buf_B2_Bthird_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    B2_Blocal_r2_r_init.store_in(&buf_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r2_i_init.store_in(&buf_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r2_r_init.store_in(&buf_B2_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r2_i_init.store_in(&buf_B2_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r2_r_init.store_in(&buf_B2_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r2_i_init.store_in(&buf_B2_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r2_r_init.store_in(&buf_B2_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r2_i_init.store_in(&buf_B2_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r2_r_update.store_in(&buf_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Blocal_r2_i_update.store_in(&buf_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r2_r_update.store_in(&buf_B2_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bfirst_r2_i_update.store_in(&buf_B2_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r2_r_update.store_in(&buf_B2_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bsecond_r2_i_update.store_in(&buf_B2_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r2_r_update.store_in(&buf_B2_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B2_Bthird_r2_i_update.store_in(&buf_B2_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_B2_Blocal_r2_r("buf_flip_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Blocal_r2_i("buf_flip_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bfirst_r2_r("buf_flip_B2_Bfirst_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bfirst_r2_i("buf_flip_B2_Bfirst_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bsecond_r2_r("buf_flip_B2_Bsecond_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bsecond_r2_i("buf_flip_B2_Bsecond_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bthird_r2_r("buf_flip_B2_Bthird_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_B2_Bthird_r2_i("buf_flip_B2_Bthird_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_B2_Blocal_r2_r_init.store_in(&buf_flip_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r2_i_init.store_in(&buf_flip_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r2_r_init.store_in(&buf_flip_B2_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r2_i_init.store_in(&buf_flip_B2_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r2_r_init.store_in(&buf_flip_B2_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r2_i_init.store_in(&buf_flip_B2_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r2_r_init.store_in(&buf_flip_B2_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r2_i_init.store_in(&buf_flip_B2_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r2_r_update.store_in(&buf_flip_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Blocal_r2_i_update.store_in(&buf_flip_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r2_r_update.store_in(&buf_flip_B2_Bfirst_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bfirst_r2_i_update.store_in(&buf_flip_B2_Bfirst_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    flip_B2_Bsecond_r2_r_update.store_in(&buf_flip_B2_Bsecond_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bsecond_r2_i_update.store_in(&buf_flip_B2_Bsecond_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r2_r_update.store_in(&buf_flip_B2_Bthird_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_B2_Bthird_r2_i_update.store_in(&buf_flip_B2_Bthird_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}); 
    buffer buf_B2_Blocal_diquark_r2_r("buf_B2_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Blocal_diquark_r2_i("buf_B2_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_diquark_r2_r("buf_B2_Bfirst_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_diquark_r2_i("buf_B2_Bfirst_diquark_r2_i",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bthird_diquark_r2_r("buf_B2_Bthird_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_B2_Bthird_diquark_r2_i("buf_B2_Bthird_diquark_r2_i",   {1}, p_float64, a_temporary);
    B2_Blocal_r2_r_diquark.store_in(&buf_B2_Blocal_diquark_r2_r, {0});
    B2_Blocal_r2_i_diquark.store_in(&buf_B2_Blocal_diquark_r2_i, {0});
    B2_Bfirst_r2_r_diquark.store_in(&buf_B2_Bfirst_diquark_r2_r, {0});
    B2_Bfirst_r2_i_diquark.store_in(&buf_B2_Bfirst_diquark_r2_i, {0}); 
    B2_Bthird_r2_r_diquark.store_in(&buf_B2_Bthird_diquark_r2_r, {0});
    B2_Bthird_r2_i_diquark.store_in(&buf_B2_Bthird_diquark_r2_i, {0}); 
    buffer buf_B2_Blocal_props_r2_r("buf_B2_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_props_r2_i("buf_B2_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_props_r2_r("buf_B2_Bfirst_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bfirst_props_r2_i("buf_B2_Bfirst_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_props_r2_r("buf_B2_Bsecond_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsecond_props_r2_i("buf_B2_Bsecond_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bthird_props_r2_r("buf_B2_Bthird_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bthird_props_r2_i("buf_B2_Bthird_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    B2_Blocal_r2_r_props_init.store_in(&buf_B2_Blocal_props_r2_r, {jCprime, jSprime});
    B2_Blocal_r2_i_props_init.store_in(&buf_B2_Blocal_props_r2_i, {jCprime, jSprime});
    B2_Bfirst_r2_r_props_init.store_in(&buf_B2_Bfirst_props_r2_r, {jCprime, jSprime});
    B2_Bfirst_r2_i_props_init.store_in(&buf_B2_Bfirst_props_r2_i, {jCprime, jSprime});
    B2_Bsecond_r2_r_props_init.store_in(&buf_B2_Bsecond_props_r2_r, {jCprime, jSprime});
    B2_Bsecond_r2_i_props_init.store_in(&buf_B2_Bsecond_props_r2_i, {jCprime, jSprime});
    B2_Bthird_r2_r_props_init.store_in(&buf_B2_Bthird_props_r2_r, {jCprime, jSprime});
    B2_Bthird_r2_i_props_init.store_in(&buf_B2_Bthird_props_r2_i, {jCprime, jSprime});
    B2_Blocal_r2_r_props.store_in(&buf_B2_Blocal_props_r2_r, {jCprime, jSprime});
    B2_Blocal_r2_i_props.store_in(&buf_B2_Blocal_props_r2_i, {jCprime, jSprime});
    B2_Bfirst_r2_r_props.store_in(&buf_B2_Bfirst_props_r2_r, {jCprime, jSprime});
    B2_Bfirst_r2_i_props.store_in(&buf_B2_Bfirst_props_r2_i, {jCprime, jSprime}); 
    B2_Bsecond_r2_r_props.store_in(&buf_B2_Bsecond_props_r2_r, {jCprime, jSprime});
    B2_Bsecond_r2_i_props.store_in(&buf_B2_Bsecond_props_r2_i, {jCprime, jSprime});
    B2_Bthird_r2_r_props.store_in(&buf_B2_Bthird_props_r2_r, {jCprime, jSprime});
    B2_Bthird_r2_i_props.store_in(&buf_B2_Bthird_props_r2_i, {jCprime, jSprime}); 
    
    buffer buf_src_B1_Blocal_r1_r("buf_src_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_r1_i("buf_src_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    src_B1_Blocal_r1_r_init.store_in(&buf_src_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r1_i_init.store_in(&buf_src_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r1_r_update.store_in(&buf_src_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r1_i_update.store_in(&buf_src_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_src_B1_Blocal_r1_r("buf_flip_src_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_src_B1_Blocal_r1_i("buf_flip_src_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_src_B1_Blocal_r1_r_init.store_in(&buf_flip_src_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r1_i_init.store_in(&buf_flip_src_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r1_r_update.store_in(&buf_flip_src_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r1_i_update.store_in(&buf_flip_src_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_src_B1_Blocal_diquark_r1_r("buf_src_B1_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_diquark_r1_i("buf_src_B1_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    src_B1_Blocal_r1_r_diquark.store_in(&buf_src_B1_Blocal_diquark_r1_r, {0});
    src_B1_Blocal_r1_i_diquark.store_in(&buf_src_B1_Blocal_diquark_r1_i, {0});
    buffer buf_src_B1_Blocal_props_r1_r("buf_src_B1_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_props_r1_i("buf_src_B1_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    src_B1_Blocal_r1_r_props_init.store_in(&buf_src_B1_Blocal_props_r1_r, {jCprime, jSprime});
    src_B1_Blocal_r1_i_props_init.store_in(&buf_src_B1_Blocal_props_r1_i, {jCprime, jSprime});
    src_B1_Blocal_r1_r_props.store_in(&buf_src_B1_Blocal_props_r1_r, {jCprime, jSprime});
    src_B1_Blocal_r1_i_props.store_in(&buf_src_B1_Blocal_props_r1_i, {jCprime, jSprime});

    buffer buf_src_B1_Blocal_r2_r("buf_src_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_r2_i("buf_src_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    src_B1_Blocal_r2_r_init.store_in(&buf_src_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r2_i_init.store_in(&buf_src_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r2_r_update.store_in(&buf_src_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B1_Blocal_r2_i_update.store_in(&buf_src_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_src_B1_Blocal_r2_r("buf_flip_src_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_src_B1_Blocal_r2_i("buf_flip_src_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_src_B1_Blocal_r2_r_init.store_in(&buf_flip_src_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r2_i_init.store_in(&buf_flip_src_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r2_r_update.store_in(&buf_flip_src_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B1_Blocal_r2_i_update.store_in(&buf_flip_src_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_src_B1_Blocal_diquark_r2_r("buf_src_B1_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_diquark_r2_i("buf_src_B1_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    src_B1_Blocal_r2_r_diquark.store_in(&buf_src_B1_Blocal_diquark_r2_r, {0});
    src_B1_Blocal_r2_i_diquark.store_in(&buf_src_B1_Blocal_diquark_r2_i, {0});
    buffer buf_src_B1_Blocal_props_r2_r("buf_src_B1_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_src_B1_Blocal_props_r2_i("buf_src_B1_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    src_B1_Blocal_r2_r_props_init.store_in(&buf_src_B1_Blocal_props_r2_r, {jCprime, jSprime});
    src_B1_Blocal_r2_i_props_init.store_in(&buf_src_B1_Blocal_props_r2_i, {jCprime, jSprime});
    src_B1_Blocal_r2_r_props.store_in(&buf_src_B1_Blocal_props_r2_r, {jCprime, jSprime});
    src_B1_Blocal_r2_i_props.store_in(&buf_src_B1_Blocal_props_r2_i, {jCprime, jSprime});
    
    buffer buf_src_B2_Blocal_r1_r("buf_src_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_r1_i("buf_src_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    src_B2_Blocal_r1_r_init.store_in(&buf_src_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r1_i_init.store_in(&buf_src_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r1_r_update.store_in(&buf_src_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r1_i_update.store_in(&buf_src_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_src_B2_Blocal_r1_r("buf_flip_src_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_src_B2_Blocal_r1_i("buf_flip_src_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_src_B2_Blocal_r1_r_init.store_in(&buf_flip_src_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r1_i_init.store_in(&buf_flip_src_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r1_r_update.store_in(&buf_flip_src_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r1_i_update.store_in(&buf_flip_src_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_src_B2_Blocal_diquark_r1_r("buf_src_B2_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_diquark_r1_i("buf_src_B2_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    src_B2_Blocal_r1_r_diquark.store_in(&buf_src_B2_Blocal_diquark_r1_r, {0});
    src_B2_Blocal_r1_i_diquark.store_in(&buf_src_B2_Blocal_diquark_r1_i, {0});
    buffer buf_src_B2_Blocal_props_r1_r("buf_src_B2_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_props_r1_i("buf_src_B2_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    src_B2_Blocal_r1_r_props_init.store_in(&buf_src_B2_Blocal_props_r1_r, {jCprime, jSprime});
    src_B2_Blocal_r1_i_props_init.store_in(&buf_src_B2_Blocal_props_r1_i, {jCprime, jSprime});
    src_B2_Blocal_r1_r_props.store_in(&buf_src_B2_Blocal_props_r1_r, {jCprime, jSprime});
    src_B2_Blocal_r1_i_props.store_in(&buf_src_B2_Blocal_props_r1_i, {jCprime, jSprime});

    buffer buf_src_B2_Blocal_r2_r("buf_src_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_r2_i("buf_src_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    src_B2_Blocal_r2_r_init.store_in(&buf_src_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r2_i_init.store_in(&buf_src_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r2_r_update.store_in(&buf_src_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    src_B2_Blocal_r2_i_update.store_in(&buf_src_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_flip_src_B2_Blocal_r2_r("buf_flip_src_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    buffer buf_flip_src_B2_Blocal_r2_i("buf_flip_src_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_temporary);
    flip_src_B2_Blocal_r2_r_init.store_in(&buf_flip_src_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r2_i_init.store_in(&buf_flip_src_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r2_r_update.store_in(&buf_flip_src_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    flip_src_B2_Blocal_r2_i_update.store_in(&buf_flip_src_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    buffer buf_src_B2_Blocal_diquark_r2_r("buf_src_B2_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_diquark_r2_i("buf_src_B2_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    src_B2_Blocal_r2_r_diquark.store_in(&buf_src_B2_Blocal_diquark_r2_r, {0});
    src_B2_Blocal_r2_i_diquark.store_in(&buf_src_B2_Blocal_diquark_r2_i, {0});
    buffer buf_src_B2_Blocal_props_r2_r("buf_src_B2_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_src_B2_Blocal_props_r2_i("buf_src_B2_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    src_B2_Blocal_r2_r_props_init.store_in(&buf_src_B2_Blocal_props_r2_r, {jCprime, jSprime});
    src_B2_Blocal_r2_i_props_init.store_in(&buf_src_B2_Blocal_props_r2_i, {jCprime, jSprime});
    src_B2_Blocal_r2_r_props.store_in(&buf_src_B2_Blocal_props_r2_r, {jCprime, jSprime});
    src_B2_Blocal_r2_i_props.store_in(&buf_src_B2_Blocal_props_r2_i, {jCprime, jSprime}); 

    buffer buf_snk_B1_Blocal_r1_r("buf_snk_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_r1_i("buf_snk_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    snk_B1_Blocal_r1_r_init.store_in(&buf_snk_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r1_i_init.store_in(&buf_snk_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r1_r_update.store_in(&buf_snk_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r1_i_update.store_in(&buf_snk_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_flip_snk_B1_Blocal_r1_r("buf_flip_snk_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_flip_snk_B1_Blocal_r1_i("buf_flip_snk_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    flip_snk_B1_Blocal_r1_r_init.store_in(&buf_flip_snk_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r1_i_init.store_in(&buf_flip_snk_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r1_r_update.store_in(&buf_flip_snk_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r1_i_update.store_in(&buf_flip_snk_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_snk_B1_Blocal_diquark_r1_r("buf_snk_B1_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_diquark_r1_i("buf_snk_B1_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    snk_B1_Blocal_r1_r_diquark.store_in(&buf_snk_B1_Blocal_diquark_r1_r, {0});
    snk_B1_Blocal_r1_i_diquark.store_in(&buf_snk_B1_Blocal_diquark_r1_i, {0});
    buffer buf_snk_B1_Blocal_props_r1_r("buf_snk_B1_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_props_r1_i("buf_snk_B1_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    snk_B1_Blocal_r1_r_props_init.store_in(&buf_snk_B1_Blocal_props_r1_r, {jCprime, jSprime});
    snk_B1_Blocal_r1_i_props_init.store_in(&buf_snk_B1_Blocal_props_r1_i, {jCprime, jSprime});
    snk_B1_Blocal_r1_r_props.store_in(&buf_snk_B1_Blocal_props_r1_r, {jCprime, jSprime});
    snk_B1_Blocal_r1_i_props.store_in(&buf_snk_B1_Blocal_props_r1_i, {jCprime, jSprime});

    buffer buf_snk_B1_Blocal_r2_r("buf_snk_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_r2_i("buf_snk_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    snk_B1_Blocal_r2_r_init.store_in(&buf_snk_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r2_i_init.store_in(&buf_snk_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r2_r_update.store_in(&buf_snk_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B1_Blocal_r2_i_update.store_in(&buf_snk_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_flip_snk_B1_Blocal_r2_r("buf_flip_snk_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_flip_snk_B1_Blocal_r2_i("buf_flip_snk_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    flip_snk_B1_Blocal_r2_r_init.store_in(&buf_flip_snk_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r2_i_init.store_in(&buf_flip_snk_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r2_r_update.store_in(&buf_flip_snk_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B1_Blocal_r2_i_update.store_in(&buf_flip_snk_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_snk_B1_Blocal_diquark_r2_r("buf_snk_B1_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_diquark_r2_i("buf_snk_B1_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    snk_B1_Blocal_r2_r_diquark.store_in(&buf_snk_B1_Blocal_diquark_r2_r, {0});
    snk_B1_Blocal_r2_i_diquark.store_in(&buf_snk_B1_Blocal_diquark_r2_i, {0});
    buffer buf_snk_B1_Blocal_props_r2_r("buf_snk_B1_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_props_r2_i("buf_snk_B1_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    snk_B1_Blocal_r2_r_props_init.store_in(&buf_snk_B1_Blocal_props_r2_r, {jCprime, jSprime});
    snk_B1_Blocal_r2_i_props_init.store_in(&buf_snk_B1_Blocal_props_r2_i, {jCprime, jSprime});
    snk_B1_Blocal_r2_r_props.store_in(&buf_snk_B1_Blocal_props_r2_r, {jCprime, jSprime});
    snk_B1_Blocal_r2_i_props.store_in(&buf_snk_B1_Blocal_props_r2_i, {jCprime, jSprime});
    
    buffer buf_snk_B2_Blocal_r1_r("buf_snk_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_r1_i("buf_snk_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    snk_B2_Blocal_r1_r_init.store_in(&buf_snk_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r1_i_init.store_in(&buf_snk_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r1_r_update.store_in(&buf_snk_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r1_i_update.store_in(&buf_snk_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_flip_snk_B2_Blocal_r1_r("buf_flip_snk_B2_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_flip_snk_B2_Blocal_r1_i("buf_flip_snk_B2_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    flip_snk_B2_Blocal_r1_r_init.store_in(&buf_flip_snk_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r1_i_init.store_in(&buf_flip_snk_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r1_r_update.store_in(&buf_flip_snk_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r1_i_update.store_in(&buf_flip_snk_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_snk_B2_Blocal_diquark_r1_r("buf_snk_B2_Blocal_diquark_r1_r",   {1}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_diquark_r1_i("buf_snk_B2_Blocal_diquark_r1_i",   {1}, p_float64, a_temporary);
    snk_B2_Blocal_r1_r_diquark.store_in(&buf_snk_B2_Blocal_diquark_r1_r, {0});
    snk_B2_Blocal_r1_i_diquark.store_in(&buf_snk_B2_Blocal_diquark_r1_i, {0});
    buffer buf_snk_B2_Blocal_props_r1_r("buf_snk_B2_Blocal_props_r1_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_props_r1_i("buf_snk_B2_Blocal_props_r1_i",   {Nc, Ns}, p_float64, a_temporary);
    snk_B2_Blocal_r1_r_props_init.store_in(&buf_snk_B2_Blocal_props_r1_r, {jCprime, jSprime});
    snk_B2_Blocal_r1_i_props_init.store_in(&buf_snk_B2_Blocal_props_r1_i, {jCprime, jSprime});
    snk_B2_Blocal_r1_r_props.store_in(&buf_snk_B2_Blocal_props_r1_r, {jCprime, jSprime});
    snk_B2_Blocal_r1_i_props.store_in(&buf_snk_B2_Blocal_props_r1_i, {jCprime, jSprime});

    buffer buf_snk_B2_Blocal_r2_r("buf_snk_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_r2_i("buf_snk_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    snk_B2_Blocal_r2_r_init.store_in(&buf_snk_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r2_i_init.store_in(&buf_snk_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r2_r_update.store_in(&buf_snk_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    snk_B2_Blocal_r2_i_update.store_in(&buf_snk_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_flip_snk_B2_Blocal_r2_r("buf_flip_snk_B2_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    buffer buf_flip_snk_B2_Blocal_r2_i("buf_flip_snk_B2_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsnk}, p_float64, a_temporary);
    flip_snk_B2_Blocal_r2_r_init.store_in(&buf_flip_snk_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r2_i_init.store_in(&buf_flip_snk_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r2_r_update.store_in(&buf_flip_snk_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    flip_snk_B2_Blocal_r2_i_update.store_in(&buf_flip_snk_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    buffer buf_snk_B2_Blocal_diquark_r2_r("buf_snk_B2_Blocal_diquark_r2_r",   {1}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_diquark_r2_i("buf_snk_B2_Blocal_diquark_r2_i",   {1}, p_float64, a_temporary);
    snk_B2_Blocal_r2_r_diquark.store_in(&buf_snk_B2_Blocal_diquark_r2_r, {0});
    snk_B2_Blocal_r2_i_diquark.store_in(&buf_snk_B2_Blocal_diquark_r2_i, {0});
    buffer buf_snk_B2_Blocal_props_r2_r("buf_snk_B2_Blocal_props_r2_r",   {Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_props_r2_i("buf_snk_B2_Blocal_props_r2_i",   {Nc, Ns}, p_float64, a_temporary);
    snk_B2_Blocal_r2_r_props_init.store_in(&buf_snk_B2_Blocal_props_r2_r, {jCprime, jSprime});
    snk_B2_Blocal_r2_i_props_init.store_in(&buf_snk_B2_Blocal_props_r2_i, {jCprime, jSprime});
    snk_B2_Blocal_r2_r_props.store_in(&buf_snk_B2_Blocal_props_r2_r, {jCprime, jSprime});
    snk_B2_Blocal_r2_i_props.store_in(&buf_snk_B2_Blocal_props_r2_i, {jCprime, jSprime});


    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, Vsnk/sites_per_rank, B2Nrows, NsrcTot, B2Nrows, NsnkTot}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, Vsnk/sites_per_rank, B2Nrows, NsrcTot, B2Nrows, NsnkTot}, p_float64, a_input);

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);

    C_init_r.store_in(&buf_C_r, {t, x_out, rp, mpmH, r, npnH});
    C_init_i.store_in(&buf_C_i, {t, x_out, rp, mpmH, r, npnH});

    // BB_BB

    buffer* buf_BB_BB_new_term_r_b1;
    buffer* buf_BB_BB_new_term_i_b1;
    allocate_complex_buffers(buf_BB_BB_new_term_r_b1, buf_BB_BB_new_term_i_b1, {1}, "buf_BB_BB_new_term_b1");
    buffer* buf_BB_BB_new_term_r_b2;
    buffer* buf_BB_BB_new_term_i_b2;
    allocate_complex_buffers(buf_BB_BB_new_term_r_b2, buf_BB_BB_new_term_i_b2, {1}, "buf_BB_BB_new_term_b2"); 

    BB_BB_new_term_0_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_0_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_1_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_1_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 
    BB_BB_new_term_2_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_2_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_3_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_3_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_4_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_4_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_5_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_5_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 
    BB_BB_new_term_6_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_6_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_7_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_7_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 

    BB_BB_new_term_0_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_0_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_1_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_1_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_2_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_2_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_3_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_3_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_4_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_4_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_5_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_5_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_6_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_6_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_7_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_7_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 

    BB_BB_new_term_0_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_0_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_1_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_1_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 
    BB_BB_new_term_2_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_2_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_3_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_3_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_4_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_4_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_5_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_5_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 
    BB_BB_new_term_6_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_6_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0});
    BB_BB_new_term_7_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {0});
    BB_BB_new_term_7_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {0}); 

    BB_BB_new_term_0_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_0_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_1_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_1_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_2_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_2_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_3_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_3_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_4_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_4_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_5_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_5_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 
    BB_BB_new_term_6_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_6_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0});
    BB_BB_new_term_7_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {0});
    BB_BB_new_term_7_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {0}); 

    buffer* buf_flip_BB_BB_new_term_r_b1;
    buffer* buf_flip_BB_BB_new_term_i_b1;
    allocate_complex_buffers(buf_flip_BB_BB_new_term_r_b1, buf_flip_BB_BB_new_term_i_b1, {1}, "buf_flip_BB_BB_new_term_b1");
    buffer* buf_flip_BB_BB_new_term_r_b2;
    buffer* buf_flip_BB_BB_new_term_i_b2;
    allocate_complex_buffers(buf_flip_BB_BB_new_term_r_b2, buf_flip_BB_BB_new_term_i_b2, {1}, "buf_flip_BB_BB_new_term_b2"); 

    flip_BB_BB_new_term_0_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_0_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_1_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_1_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 
    flip_BB_BB_new_term_2_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_2_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_3_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_3_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_4_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_4_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_5_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_5_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 
    flip_BB_BB_new_term_6_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_6_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_7_r1_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_7_r1_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 

    flip_BB_BB_new_term_0_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_0_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_1_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_1_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_2_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_2_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_3_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_3_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_4_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_4_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_5_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_5_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_6_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_6_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_7_r1_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_7_r1_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 

    flip_BB_BB_new_term_0_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_0_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_1_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_1_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 
    flip_BB_BB_new_term_2_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_2_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_3_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_3_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_4_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_4_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_5_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_5_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 
    flip_BB_BB_new_term_6_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_6_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0});
    flip_BB_BB_new_term_7_r2_b1.get_real()->store_in(buf_flip_BB_BB_new_term_r_b1, {0});
    flip_BB_BB_new_term_7_r2_b1.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b1, {0}); 

    flip_BB_BB_new_term_0_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_0_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_1_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_1_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_2_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_2_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_3_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_3_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_4_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_4_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_5_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_5_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 
    flip_BB_BB_new_term_6_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_6_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0});
    flip_BB_BB_new_term_7_r2_b2.get_real()->store_in(buf_flip_BB_BB_new_term_r_b2, {0});
    flip_BB_BB_new_term_7_r2_b2.get_imag()->store_in(buf_flip_BB_BB_new_term_i_b2, {0}); 

    buffer buf_C_BB_BB_prop_r("buf_C_BB_BB_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_BB_BB_prop_i("buf_C_BB_BB_prop_i", {1}, p_float64, a_temporary);

    C_BB_BB_prop_init_r.store_in(&buf_C_BB_BB_prop_r, {0});
    C_BB_BB_prop_init_i.store_in(&buf_C_BB_BB_prop_i, {0});
    C_BB_BB_prop_update_r.store_in(&buf_C_BB_BB_prop_r, {0});
    C_BB_BB_prop_update_i.store_in(&buf_C_BB_BB_prop_i, {0});

    C_BB_BB_update_b_r.store_in(&buf_C_r, {t, x_out, rp, m, r, ne});
    C_BB_BB_update_b_i.store_in(&buf_C_i, {t, x_out, rp, m, r, ne});
    C_BB_BB_update_s_r.store_in(&buf_C_r, {t, x_out, rp, m, r, NEntangled+nue});
    C_BB_BB_update_s_i.store_in(&buf_C_i, {t, x_out, rp, m, r, NEntangled+nue});

    // BB_H

    buffer* buf_BB_H_new_term_r_b1;
    buffer* buf_BB_H_new_term_i_b1;
    allocate_complex_buffers(buf_BB_H_new_term_r_b1, buf_BB_H_new_term_i_b1, {1}, "buf_BB_H_new_term_b1");

    BB_H_new_term_0_r1_b1.get_real()->store_in(buf_BB_H_new_term_r_b1, {0});
    BB_H_new_term_0_r1_b1.get_imag()->store_in(buf_BB_H_new_term_i_b1, {0});

    BB_H_new_term_0_r2_b1.get_real()->store_in(buf_BB_H_new_term_r_b1, {0});
    BB_H_new_term_0_r2_b1.get_imag()->store_in(buf_BB_H_new_term_i_b1, {0});

    buffer* buf_BB_H_new_term_r_b2;
    buffer* buf_BB_H_new_term_i_b2;
    allocate_complex_buffers(buf_BB_H_new_term_r_b2, buf_BB_H_new_term_i_b2, {1}, "buf_BB_H_new_term_b2");

    BB_H_new_term_0_r1_b2.get_real()->store_in(buf_BB_H_new_term_r_b2, {0});
    BB_H_new_term_0_r1_b2.get_imag()->store_in(buf_BB_H_new_term_i_b2, {0});

    BB_H_new_term_0_r2_b2.get_real()->store_in(buf_BB_H_new_term_r_b2, {0});
    BB_H_new_term_0_r2_b2.get_imag()->store_in(buf_BB_H_new_term_i_b2, {0});

    buffer* buf_flip_BB_H_new_term_r_b1;
    buffer* buf_flip_BB_H_new_term_i_b1;
    allocate_complex_buffers(buf_flip_BB_H_new_term_r_b1, buf_flip_BB_H_new_term_i_b1, {1}, "buf_flip_BB_H_new_term_b1");

    flip_BB_H_new_term_0_r1_b1.get_real()->store_in(buf_flip_BB_H_new_term_r_b1, {0});
    flip_BB_H_new_term_0_r1_b1.get_imag()->store_in(buf_flip_BB_H_new_term_i_b1, {0});

    flip_BB_H_new_term_0_r2_b1.get_real()->store_in(buf_flip_BB_H_new_term_r_b1, {0});
    flip_BB_H_new_term_0_r2_b1.get_imag()->store_in(buf_flip_BB_H_new_term_i_b1, {0});

    buffer* buf_flip_BB_H_new_term_r_b2;
    buffer* buf_flip_BB_H_new_term_i_b2;
    allocate_complex_buffers(buf_flip_BB_H_new_term_r_b2, buf_flip_BB_H_new_term_i_b2, {1}, "buf_flip_BB_H_new_term_b2");

    flip_BB_H_new_term_0_r1_b2.get_real()->store_in(buf_flip_BB_H_new_term_r_b2, {0});
    flip_BB_H_new_term_0_r1_b2.get_imag()->store_in(buf_flip_BB_H_new_term_i_b2, {0});

    flip_BB_H_new_term_0_r2_b2.get_real()->store_in(buf_flip_BB_H_new_term_r_b2, {0});
    flip_BB_H_new_term_0_r2_b2.get_imag()->store_in(buf_flip_BB_H_new_term_i_b2, {0});


    buffer buf_C_BB_H_prop_r("buf_C_BB_H_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_BB_H_prop_i("buf_C_BB_H_prop_i", {1}, p_float64, a_temporary);

    C_BB_H_prop_init_r.store_in(&buf_C_BB_H_prop_r, {0});
    C_BB_H_prop_init_i.store_in(&buf_C_BB_H_prop_i, {0});
    C_BB_H_prop_update_r.store_in(&buf_C_BB_H_prop_r, {0});
    C_BB_H_prop_update_i.store_in(&buf_C_BB_H_prop_i, {0}); 

    C_BB_H_update_r.store_in(&buf_C_r, {t, x_out, rp, m, r, Nsnk+nH});
    C_BB_H_update_i.store_in(&buf_C_i, {t, x_out, rp, m, r, Nsnk+nH});

    // H_BB

    buffer* buf_H_BB_new_term_r_b1;
    buffer* buf_H_BB_new_term_i_b1;
    allocate_complex_buffers(buf_H_BB_new_term_r_b1, buf_H_BB_new_term_i_b1, {1}, "buf_H_BB_new_term_b1");

    H_BB_new_term_0_r1_b1.get_real()->store_in(buf_H_BB_new_term_r_b1, {0});
    H_BB_new_term_0_r1_b1.get_imag()->store_in(buf_H_BB_new_term_i_b1, {0});

    H_BB_new_term_0_r2_b1.get_real()->store_in(buf_H_BB_new_term_r_b1, {0});
    H_BB_new_term_0_r2_b1.get_imag()->store_in(buf_H_BB_new_term_i_b1, {0});

    buffer* buf_H_BB_new_term_r_b2;
    buffer* buf_H_BB_new_term_i_b2;
    allocate_complex_buffers(buf_H_BB_new_term_r_b2, buf_H_BB_new_term_i_b2, {1}, "buf_H_BB_new_term_b2");

    H_BB_new_term_0_r1_b2.get_real()->store_in(buf_H_BB_new_term_r_b2, {0});
    H_BB_new_term_0_r1_b2.get_imag()->store_in(buf_H_BB_new_term_i_b2, {0});

    H_BB_new_term_0_r2_b2.get_real()->store_in(buf_H_BB_new_term_r_b2, {0});
    H_BB_new_term_0_r2_b2.get_imag()->store_in(buf_H_BB_new_term_i_b2, {0});

    buffer* buf_flip_H_BB_new_term_r_b1;
    buffer* buf_flip_H_BB_new_term_i_b1;
    allocate_complex_buffers(buf_flip_H_BB_new_term_r_b1, buf_flip_H_BB_new_term_i_b1, {1}, "buf_flip_H_BB_new_term_b1");

    flip_H_BB_new_term_0_r1_b1.get_real()->store_in(buf_flip_H_BB_new_term_r_b1, {0});
    flip_H_BB_new_term_0_r1_b1.get_imag()->store_in(buf_flip_H_BB_new_term_i_b1, {0});

    flip_H_BB_new_term_0_r2_b1.get_real()->store_in(buf_flip_H_BB_new_term_r_b1, {0});
    flip_H_BB_new_term_0_r2_b1.get_imag()->store_in(buf_flip_H_BB_new_term_i_b1, {0});

    buffer* buf_flip_H_BB_new_term_r_b2;
    buffer* buf_flip_H_BB_new_term_i_b2;
    allocate_complex_buffers(buf_flip_H_BB_new_term_r_b2, buf_flip_H_BB_new_term_i_b2, {1}, "buf_flip_H_BB_new_term_b2");

    flip_H_BB_new_term_0_r1_b2.get_real()->store_in(buf_flip_H_BB_new_term_r_b2, {0});
    flip_H_BB_new_term_0_r1_b2.get_imag()->store_in(buf_flip_H_BB_new_term_i_b2, {0});

    flip_H_BB_new_term_0_r2_b2.get_real()->store_in(buf_flip_H_BB_new_term_r_b2, {0});
    flip_H_BB_new_term_0_r2_b2.get_imag()->store_in(buf_flip_H_BB_new_term_i_b2, {0});

    buffer buf_C_H_BB_prop_r("buf_C_H_BB_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_H_BB_prop_i("buf_C_H_BB_prop_i", {1}, p_float64, a_temporary);

    C_H_BB_prop_init_r.store_in(&buf_C_H_BB_prop_r, {0});
    C_H_BB_prop_init_i.store_in(&buf_C_H_BB_prop_i, {0});
    C_H_BB_prop_update_r.store_in(&buf_C_H_BB_prop_r, {0});
    C_H_BB_prop_update_i.store_in(&buf_C_H_BB_prop_i, {0});

    C_H_BB_update_r.store_in(&buf_C_r, {t, y_out, r, Nsrc+mH, rp, n});
    C_H_BB_update_i.store_in(&buf_C_i, {t, y_out, r, Nsrc+mH, rp, n});

    // H_H

    buffer buf_C_H_H_prop_r("buf_C_H_H_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_H_H_prop_i("buf_C_H_H_prop_i", {1}, p_float64, a_temporary);

    C_H_H_prop_init_r.store_in(&buf_C_H_H_prop_r, {0});
    C_H_H_prop_init_i.store_in(&buf_C_H_H_prop_i, {0});
    C_H_H_prop_update_r.store_in(&buf_C_H_H_prop_r, {0});
    C_H_H_prop_update_i.store_in(&buf_C_H_H_prop_i, {0});

    C_H_H_update_r.store_in(&buf_C_r, {t, x_out, rp, Nsrc+mH, r, Nsnk+nH});
    C_H_H_update_i.store_in(&buf_C_i, {t, x_out, rp, Nsrc+mH, r, Nsnk+nH});  

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	     &buf_C_r, &buf_C_i,
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        src_psi_B2_r.get_buffer(), src_psi_B2_i.get_buffer(),
        snk_psi_B1_r.get_buffer(), snk_psi_B1_i.get_buffer(), 
        snk_psi_B2_r.get_buffer(), snk_psi_B2_i.get_buffer(),
        hex_src_psi_r.get_buffer(), hex_src_psi_i.get_buffer(),
        hex_snk_psi_r.get_buffer(), hex_snk_psi_i.get_buffer(), 
        snk_psi_r.get_buffer(), snk_psi_i.get_buffer(), 
	     src_spins.get_buffer(), 
	     src_spin_block_weights.get_buffer(), 
        sigs.get_buffer(),
	     src_color_weights.get_buffer(),
	     src_spin_weights.get_buffer(),
	     src_weights.get_buffer(),
	     snk_b.get_buffer(),
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer(),
	     hex_snk_color_weights.get_buffer(),
	     hex_snk_spin_weights.get_buffer(),
	     hex_snk_weights.get_buffer()
        }, 
        "generated_tiramisu_make_fused_identical_dibaryon_blocks_correlator.o");
}

int main(int argc, char **argv)
{
	generate_function("tiramisu_make_fused_identical_dibaryon_blocks_correlator");

    return 0;
}
