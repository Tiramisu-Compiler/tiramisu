#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

// Used to remember relevant (sub)computation of Q and its user computations (B1_Blocal_r1 and B1_Bsingle_r1)
struct Q2UserEdge {
      computation *q_r, *q_i,
                  *bs_r, *bs_i,
                  *bl_r, *bl_i;
};
struct snkQ2UserEdge {
      computation *q_r, *q_i,
                  *bl_r, *bl_i;
};

struct O2UserEdge {
      computation *o_r, *o_i,
                  *bd_r, *bd_i;
};

// Similar to Q2UserEdge, used to record (sub)computation of P and the corresponding use in B1_Bdouble_r1
struct P2UserEdge {
      computation *p_r, *p_i,
                  *bd_r, *bd_i;
};

/*
 * The goal is to generate code that implements the reference.
 * baryon_ref.cpp
 */
void generate_function(std::string name)
{
    tiramisu::init(name);

   int Nr = 6;
   int b;
   int NsrcTot = Nsrc+NsrcHex;
   int NsnkTot = Nsnk+NsnkHex;
   var nperm("nperm", 0, Nperms),
	r("r", 0, Nr),
	q("q", 0, Nq),
	to("to", 0, 2),
	wnum("wnum", 0, Nw2),
        t("t", 0, Lt),
	x("x", 0, Vsnk),
	x2("x2", 0, Vsnk),
        y("y", 0, Vsrc),
	m("m", 0, Nsrc),
	n("n", 0, Nsnk),
	mH("mH", 0, NsrcHex),
	nH("nH", 0, NsnkHex),
	mpmH("mpmH", 0, NsrcTot),
	npnH("npnH", 0, NsnkTot),
        tri("tri", 0, Nq),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);

   input C_r("C_r",      {r, mpmH, npnH, t}, p_float64);
   input C_i("C_i",      {r, mpmH, npnH, t}, p_float64);

   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_r("B2_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_i("B2_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);

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
   input snk_psi_r("snk_psi_r", {x, x2, n}, p_float64);
   input snk_psi_i("snk_psi_i", {x, x2, n}, p_float64);

   input snk_blocks("snk_blocks", {r, to}, p_int32);
   input sigs("sigs", {nperm}, p_int32);
   input snk_b("snk_b", {nperm, q, to}, p_int32);
   input snk_color_weights("snk_color_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);
    complex_computation B2_prop(&B2_prop_r, &B2_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));
    complex_expr src_psi_B2(src_psi_B2_r(y, m), src_psi_B2_i(y, m));

    complex_expr snk_psi_B1(snk_psi_B1_r(x, n), snk_psi_B1_i(x, n));
    complex_expr snk_psi_B2(snk_psi_B2_r(x, n), snk_psi_B2_i(x, n));

    complex_expr hex_src_psi(hex_src_psi_r(y, mH), hex_src_psi_i(y, mH));
    complex_expr hex_snk_psi(hex_snk_psi_r(x, nH), hex_snk_psi_i(x, nH));

    complex_expr snk_psi(snk_psi_r(x, x2, n), snk_psi_i(x, x2, n));

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsingle_r1_r_init("B1_Bsingle_r1_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsingle_r1_i_init("B1_Bsingle_r1_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bdouble_r1_r_init("B1_Bdouble_r1_r_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));
    computation B1_Bdouble_r1_i_init("B1_Bdouble_r1_i_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));

    complex_computation B1_Bsingle_r1_init(&B1_Bsingle_r1_r_init, &B1_Bsingle_r1_i_init);
    complex_computation B1_Blocal_r1_init(&B1_Blocal_r1_r_init, &B1_Blocal_r1_i_init);
    complex_computation B1_Bdouble_r1_init(&B1_Bdouble_r1_r_init, &B1_Bdouble_r1_i_init);
    std::vector<std::pair<computation *, computation *>> B1_Bsingle_r1_updates;
    std::vector<std::pair<computation *, computation *>> B1_Blocal_r1_updates;
    std::vector<std::pair<computation *, computation *>> B1_Bdouble_r1_o_updates;
    std::vector<std::pair<computation *, computation *>> B1_Bdouble_r1_p_updates;

    complex_expr B1_Q_exprs_r1[Nc][Ns];
    complex_expr B1_O_exprs_r1[Nc][Ns];
    complex_expr B1_P_exprs_r1[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];

      complex_expr B1_prop_0 =  B1_prop(0, t, iCprime, iSprime, ic, is, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, kCprime, kSprime, kc, ks, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, kCprime, kSprime, ic, is, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, iCprime, iSprime, kc, ks, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, jCprime, jSprime, jc, js, x, y);
      
      B1_Q_exprs_r1[jc][js] += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * w;

      B1_O_exprs_r1[ic][is] += B1_prop_1 * B1_prop_2 * w;

      B1_P_exprs_r1[kc][ks] += B1_prop_0p * B1_prop_1 * w;
    }

    // DEFINE computation of Q, and its user -- B1_Blocal_r1 and B1_Bsingle_r1
    std::vector<Q2UserEdge> B1_q2userEdges_r1;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (B1_Q_exprs_r1[jc][js].is_zero())
          continue;

        complex_computation q_computation(
            str_fmt("B1_q_r1_%d_%d", jc, js),
            {t, x, iCprime, iSprime, kCprime, kSprime, y},
            B1_Q_exprs_r1[jc][js]);

        complex_expr q = q_computation(t, x, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r1_init(t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B1_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B1_Bsingle_r1_init(t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B1;
        complex_computation bsingle_update(
            str_fmt("B1_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // predicate
            (iCprime != kCprime || iSprime != kSprime),
            // definition
            bsingle_update_def);
        B1_Bsingle_r1_updates.push_back(bsingle_update);


        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *bsingle_r = bsingle_update.get_real();
        auto *bsingle_i = bsingle_update.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, bsingle_r, bsingle_i, blocal_r, blocal_i};
        B1_q2userEdges_r1.push_back(edge);
      }
    }

    // DEFINE computation of O and its user update on B1_Bdouble_r1
    std::vector<O2UserEdge> B1_o2userEdges_r1;
    for (int ic = 0; ic < Nc; ic++) {
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r1[ic][is].is_zero())
          continue;

        complex_computation o_computation(
            // name
            str_fmt("B1_o_r1_%d_%d", ic, is),
            // iterators
            {t, x, jCprime, jSprime, kCprime, kSprime, y},
            B1_O_exprs_r1[ic][is]);

        complex_expr o = o_computation(t, x, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) +
          o * B1_prop(0, t, iCprime, iSprime, ic, is, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);

        B1_Bdouble_r1_o_updates.push_back(bdouble_update);

        computation *o_real = o_computation.get_real();
        computation *o_imag = o_computation.get_imag();
        O2UserEdge edge {o_real, o_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B1_o2userEdges_r1.push_back(edge);
      }
    }

    // DEFINE computation of P and its user update on B1_Bdouble_r1
    std::vector<P2UserEdge> B1_p2userEdges_r1;
    for (int kc = 0; kc < Nc; kc++) {
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r1[kc][ks].is_zero())
          continue;
        complex_computation p_computation(
            // name
            str_fmt("B1_p_r1_%d_%d", kc, ks),
            // iterators
            {t, x, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B1_P_exprs_r1[kc][ks]);

        complex_expr p = p_computation(t, x, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) -
          p * B1_prop(2, t, iCprime, iSprime, kc, ks, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);
        B1_Bdouble_r1_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B1_p2userEdges_r1.push_back(edge);
      }
    }
    

    /*
     * Computing B2_Blocal_r1, B2_Bsingle_r1, B2_Bdouble_r1.
     */

    computation B2_Blocal_r1_r_init("B2_Blocal_r1_r_init", {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r1_i_init("B2_Blocal_r1_i_init", {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsingle_r1_r_init("B2_Bsingle_r1_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsingle_r1_i_init("B2_Bsingle_r1_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bdouble_r1_r_init("B2_Bdouble_r1_r_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));
    computation B2_Bdouble_r1_i_init("B2_Bdouble_r1_i_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));

    complex_computation B2_Bsingle_r1_init(&B2_Bsingle_r1_r_init, &B2_Bsingle_r1_i_init);
    complex_computation B2_Blocal_r1_init(&B2_Blocal_r1_r_init, &B2_Blocal_r1_i_init);
    complex_computation B2_Bdouble_r1_init(&B2_Bdouble_r1_r_init, &B2_Bdouble_r1_i_init);

    std::vector<std::pair<computation *, computation *>> B2_Bsingle_r1_updates;
    std::vector<std::pair<computation *, computation *>> B2_Blocal_r1_updates;
    std::vector<std::pair<computation *, computation *>> B2_Bdouble_r1_o_updates;
    std::vector<std::pair<computation *, computation *>> B2_Bdouble_r1_p_updates;

    complex_expr B2_Q_exprs_r1[Nc][Ns];
    complex_expr B2_O_exprs_r1[Nc][Ns];
    complex_expr B2_P_exprs_r1[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];

      complex_expr B2_prop_0 =  B2_prop(0, t, iCprime, iSprime, ic, is, x2, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, kCprime, kSprime, kc, ks, x2, y);
      complex_expr B2_prop_0p = B2_prop(0, t, kCprime, kSprime, ic, is, x2, y);
      complex_expr B2_prop_2p = B2_prop(2, t, iCprime, iSprime, kc, ks, x2, y);
      complex_expr B2_prop_1 = B2_prop(1, t, jCprime, jSprime, jc, js, x2, y);
      
      B2_Q_exprs_r1[jc][js] += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * w;

      B2_O_exprs_r1[ic][is] += B2_prop_1 * B2_prop_2 * w;

      B2_P_exprs_r1[kc][ks] += B2_prop_0p * B2_prop_1 * w;
    }

    // DEFINE computation of Q, and its user -- B2_Blocal_r1 and B2_Bsingle_r1
    std::vector<Q2UserEdge> B2_q2userEdges_r1;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (B2_Q_exprs_r1[jc][js].is_zero())
          continue;

        complex_computation q_computation(
            str_fmt("B2_q_r1_%d_%d", jc, js),
            {t, x2, iCprime, iSprime, kCprime, kSprime, y},
            B2_Q_exprs_r1[jc][js]);

        complex_expr q = q_computation(t, x2, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B2_Blocal_r1_init(t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("B2_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B2_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B2_Bsingle_r1_init(t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B2;
        complex_computation bsingle_update(
            str_fmt("B2_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // predicate
            (iCprime != kCprime || iSprime != kSprime),
            // definition
            bsingle_update_def);
        B2_Bsingle_r1_updates.push_back(bsingle_update);


        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *bsingle_r = bsingle_update.get_real();
        auto *bsingle_i = bsingle_update.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, bsingle_r, bsingle_i, blocal_r, blocal_i};
        B2_q2userEdges_r1.push_back(edge);
      }
    }

    // DEFINE computation of O and its user update on B2_Bdouble_r1
    std::vector<O2UserEdge> B2_o2userEdges_r1;
    for (int ic = 0; ic < Nc; ic++) {
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r1[ic][is].is_zero())
          continue;

        complex_computation o_computation(
            // name
            str_fmt("B2_o_r1_%d_%d", ic, is),
            // iterators
            {t, x2, jCprime, jSprime, kCprime, kSprime, y},
            B2_O_exprs_r1[ic][is]);

        complex_expr o = o_computation(t, x2, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r1_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) +
          o * B2_prop(0, t, iCprime, iSprime, ic, is, x, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);

        B2_Bdouble_r1_o_updates.push_back(bdouble_update);

        computation *o_real = o_computation.get_real();
        computation *o_imag = o_computation.get_imag();
        O2UserEdge edge {o_real, o_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B2_o2userEdges_r1.push_back(edge);
      }
    }

    // DEFINE computation of P and its user update on B2_Bdouble_r1
    std::vector<P2UserEdge> B2_p2userEdges_r1;
    for (int kc = 0; kc < Nc; kc++) {
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r1[kc][ks].is_zero())
          continue;
        complex_computation p_computation(
            // name
            str_fmt("B2_p_r1_%d_%d", kc, ks),
            // iterators
            {t, x2, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B2_P_exprs_r1[kc][ks]);

        complex_expr p = p_computation(t, x2, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r1_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) -
          p * B2_prop(2, t, iCprime, iSprime, kc, ks, x, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);
        B2_Bdouble_r1_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B2_p2userEdges_r1.push_back(edge);
      }
    }

    /*
     * Computing B1_Blocal_r2, B1_Bsingle_r2, B1_Bdouble_r2.
     */

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsingle_r2_r_init("B1_Bsingle_r2_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bsingle_r2_i_init("B1_Bsingle_r2_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Bdouble_r2_r_init("B1_Bdouble_r2_r_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));
    computation B1_Bdouble_r2_i_init("B1_Bdouble_r2_i_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));

    complex_computation B1_Bsingle_r2_init(&B1_Bsingle_r2_r_init, &B1_Bsingle_r2_i_init);
    complex_computation B1_Blocal_r2_init(&B1_Blocal_r2_r_init, &B1_Blocal_r2_i_init);
    complex_computation B1_Bdouble_r2_init(&B1_Bdouble_r2_r_init, &B1_Bdouble_r2_i_init);
    std::vector<std::pair<computation *, computation *>> B1_Bsingle_r2_updates;
    std::vector<std::pair<computation *, computation *>> B1_Blocal_r2_updates;
    std::vector<std::pair<computation *, computation *>> B1_Bdouble_r2_o_updates;
    std::vector<std::pair<computation *, computation *>> B1_Bdouble_r2_p_updates;

    complex_expr B1_Q_exprs_r2[Nc][Ns];
    complex_expr B1_O_exprs_r2[Nc][Ns];
    complex_expr B1_P_exprs_r2[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];

      complex_expr B1_prop_0 =  B1_prop(0, t, iCprime, iSprime, ic, is, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, kCprime, kSprime, kc, ks, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, kCprime, kSprime, ic, is, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, iCprime, iSprime, kc, ks, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, jCprime, jSprime, jc, js, x, y);
      
      B1_Q_exprs_r2[jc][js] += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * w;

      B1_O_exprs_r2[ic][is] += B1_prop_1 * B1_prop_2 * w;

      B1_P_exprs_r2[kc][ks] += B1_prop_0p * B1_prop_1 * w;
    }

    // DEFINE computation of Q, and its user -- B1_Blocal_r2 and B1_Bsingle_r2
    std::vector<Q2UserEdge> B1_q2userEdges_r2;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (B1_Q_exprs_r2[jc][js].is_zero())
          continue;

        complex_computation q_computation(
            str_fmt("B1_q_r2_%d_%d", jc, js),
            {t, x, iCprime, iSprime, kCprime, kSprime, y},
            B1_Q_exprs_r2[jc][js]);

        complex_expr q = q_computation(t, x, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r2_init(t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B1_Blocal_r2_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B1_Bsingle_r2_init(t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B1;
        complex_computation bsingle_update(
            str_fmt("B1_bsingle_update_r2_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // predicate
            (iCprime != kCprime || iSprime != kSprime),
            // definition
            bsingle_update_def);
        B1_Bsingle_r2_updates.push_back(bsingle_update);


        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *bsingle_r = bsingle_update.get_real();
        auto *bsingle_i = bsingle_update.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, bsingle_r, bsingle_i, blocal_r, blocal_i};
        B1_q2userEdges_r2.push_back(edge);
      }
    }

    // DEFINE computation of O and its user update on B1_Bdouble_r2
    std::vector<O2UserEdge> B1_o2userEdges_r2;
    for (int ic = 0; ic < Nc; ic++) {
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r2[ic][is].is_zero())
          continue;

        complex_computation o_computation(
            // name
            str_fmt("B1_o_r2_%d_%d", ic, is),
            // iterators
            {t, x, jCprime, jSprime, kCprime, kSprime, y},
            B1_O_exprs_r2[ic][is]);

        complex_expr o = o_computation(t, x, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r2_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) +
          o * B1_prop(0, t, iCprime, iSprime, ic, is, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_o_update_r2_%d_%d", ic, is),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);

        B1_Bdouble_r2_o_updates.push_back(bdouble_update);

        computation *o_real = o_computation.get_real();
        computation *o_imag = o_computation.get_imag();
        O2UserEdge edge {o_real, o_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B1_o2userEdges_r2.push_back(edge);
      }
    }

    // DEFINE computation of P and its user update on B1_Bdouble_r2
    std::vector<P2UserEdge> B1_p2userEdges_r2;
    for (int kc = 0; kc < Nc; kc++) {
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r2[kc][ks].is_zero())
          continue;
        complex_computation p_computation(
            // name
            str_fmt("B1_p_r2_%d_%d", kc, ks),
            // iterators
            {t, x, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B1_P_exprs_r2[kc][ks]);

        complex_expr p = p_computation(t, x, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r2_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) -
          p * B1_prop(2, t, iCprime, iSprime, kc, ks, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_p_update_r2_%d_%d", kc, ks),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);
        B1_Bdouble_r2_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B1_p2userEdges_r2.push_back(edge);
      }
    }

    /*
     * Computing B2_Blocal_r2, B2_Bsingle_r2, B2_Bdouble_r2.
     */

    computation B2_Blocal_r2_r_init("B2_Blocal_r2_r_init", {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r2_i_init("B2_Blocal_r2_i_init", {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsingle_r2_r_init("B2_Bsingle_r2_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bsingle_r2_i_init("B2_Bsingle_r2_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Bdouble_r2_r_init("B2_Bdouble_r2_r_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));
    computation B2_Bdouble_r2_i_init("B2_Bdouble_r2_i_init", {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime}, expr((double) 0));

    complex_computation B2_Bsingle_r2_init(&B2_Bsingle_r2_r_init, &B2_Bsingle_r2_i_init);
    complex_computation B2_Blocal_r2_init(&B2_Blocal_r2_r_init, &B2_Blocal_r2_i_init);
    complex_computation B2_Bdouble_r2_init(&B2_Bdouble_r2_r_init, &B2_Bdouble_r2_i_init);

    std::vector<std::pair<computation *, computation *>> B2_Bsingle_r2_updates;
    std::vector<std::pair<computation *, computation *>> B2_Blocal_r2_updates;
    std::vector<std::pair<computation *, computation *>> B2_Bdouble_r2_o_updates;
    std::vector<std::pair<computation *, computation *>> B2_Bdouble_r2_p_updates;

    complex_expr B2_Q_exprs_r2[Nc][Ns];
    complex_expr B2_O_exprs_r2[Nc][Ns];
    complex_expr B2_P_exprs_r2[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];

      complex_expr B2_prop_0 =  B2_prop(0, t, iCprime, iSprime, ic, is, x2, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, kCprime, kSprime, kc, ks, x2, y);
      complex_expr B2_prop_0p = B2_prop(0, t, kCprime, kSprime, ic, is, x2, y);
      complex_expr B2_prop_2p = B2_prop(2, t, iCprime, iSprime, kc, ks, x2, y);
      complex_expr B2_prop_1 = B2_prop(1, t, jCprime, jSprime, jc, js, x2, y);
      
      B2_Q_exprs_r2[jc][js] += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * w;

      B2_O_exprs_r2[ic][is] += B2_prop_1 * B2_prop_2 * w;

      B2_P_exprs_r2[kc][ks] += B2_prop_0p * B2_prop_1 * w;
    }

    // DEFINE computation of Q, and its user -- B2_Blocal_r2 and B2_Bsingle_r2
    std::vector<Q2UserEdge> B2_q2userEdges_r2;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (B2_Q_exprs_r2[jc][js].is_zero())
          continue;

        complex_computation q_computation(
            str_fmt("B2_q_r2_%d_%d", jc, js),
            {t, x2, iCprime, iSprime, kCprime, kSprime, y},
            B2_Q_exprs_r2[jc][js]);

        complex_expr q = q_computation(t, x2, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B2_Blocal_r2_init(t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("B2_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        //blocal_update.add_predicate(x==0);
        B2_Blocal_r2_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B2_Bsingle_r2_init(t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B2;
        complex_computation bsingle_update(
            str_fmt("B2_bsingle_update_r2_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // predicate
            (iCprime != kCprime || iSprime != kSprime),
            // definition
            bsingle_update_def);
        B2_Bsingle_r2_updates.push_back(bsingle_update);


        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *bsingle_r = bsingle_update.get_real();
        auto *bsingle_i = bsingle_update.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, bsingle_r, bsingle_i, blocal_r, blocal_i};
        B2_q2userEdges_r2.push_back(edge);
      }
    }

    // DEFINE computation of O and its user update on B2_Bdouble_r2
    std::vector<O2UserEdge> B2_o2userEdges_r2;
    for (int ic = 0; ic < Nc; ic++) {
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r2[ic][is].is_zero())
          continue;

        complex_computation o_computation(
            // name
            str_fmt("B2_o_r2_%d_%d", ic, is),
            // iterators
            {t, x2, jCprime, jSprime, kCprime, kSprime, y},
            B2_O_exprs_r2[ic][is]);
        //o_computation.add_predicate(m==0);
        //o_computation.add_predicate((x==0)&&(m==0));

        complex_expr o = o_computation(t, x2, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r2_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) +
          o * B2_prop(0, t, iCprime, iSprime, ic, is, x, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_o_update_r2_%d_%d", ic, is),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);

        B2_Bdouble_r2_o_updates.push_back(bdouble_update);

        computation *o_real = o_computation.get_real();
        computation *o_imag = o_computation.get_imag();
        O2UserEdge edge {o_real, o_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B2_o2userEdges_r2.push_back(edge);
      }
    }

    // DEFINE computation of P and its user update on B2_Bdouble_r2
    std::vector<P2UserEdge> B2_p2userEdges_r2;
    for (int kc = 0; kc < Nc; kc++) {
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r2[kc][ks].is_zero())
          continue;
        complex_computation p_computation(
            // name
            str_fmt("B2_p_r2_%d_%d", kc, ks),
            // iterators
            {t, x2, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B2_P_exprs_r2[kc][ks]);

        complex_expr p = p_computation(t, x2, jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r2_init(t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime) -
          p * B2_prop(2, t, iCprime, iSprime, kc, ks, x, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_p_update_r2_%d_%d", kc, ks),
            // iterator
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y},
            // definition
            bdouble_update_def);
        B2_Bdouble_r2_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B2_p2userEdges_r2.push_back(edge);
      }
    }

    /*
     * Computing snk_B1_Blocal_r1
     */
    computation snk_B1_Blocal_r1_r_init("snk_B1_Blocal_r1_r_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation snk_B1_Blocal_r1_i_init("snk_B1_Blocal_r1_i_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    complex_computation snk_B1_Blocal_r1_init(&snk_B1_Blocal_r1_r_init, &snk_B1_Blocal_r1_i_init);
    std::vector<std::pair<computation *, computation *>> snk_B1_Blocal_r1_updates;
    complex_expr snk_B1_Q_exprs_r1[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];
      complex_expr B1_prop_0 =  B1_prop(0, t, ic, is, iCprime, iSprime, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, kc, ks, kCprime, kSprime, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, ic, is, kCprime, kSprime, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, kc, ks, iCprime, iSprime, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, jc, js, jCprime, jSprime, x, y);
      snk_B1_Q_exprs_r1[jc][js] += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * w;
    }
    // DEFINE computation of Q, and its user -- snk_B1_Blocal_r1 and snk_B1_Bsingle_r1
    std::vector<snkQ2UserEdge> snk_B1_q2userEdges_r1;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (snk_B1_Q_exprs_r1[jc][js].is_zero())
          continue;
        complex_computation q_computation(
            str_fmt("snk_B1_q_r1_%d_%d", jc, js),
            {t, y, iCprime, iSprime, kCprime, kSprime, x},
            snk_B1_Q_exprs_r1[jc][js]);
        complex_expr q = q_computation(t, y, iCprime, iSprime, kCprime, kSprime, x);
        // define local block
        complex_expr blocal_update_def = 
          snk_B1_Blocal_r1_init(t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jc, js, jCprime, jSprime, x, y) * snk_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("snk_B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x},
            // definition
            blocal_update_def);
        snk_B1_Blocal_r1_updates.push_back(blocal_update);
        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        snkQ2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        snk_B1_q2userEdges_r1.push_back(edge);
      }
    }
    /*
     * Computing snk_B2_Blocal_r1
     */
    computation snk_B2_Blocal_r1_r_init("snk_B2_Blocal_r1_r_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation snk_B2_Blocal_r1_i_init("snk_B2_Blocal_r1_i_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    complex_computation snk_B2_Blocal_r1_init(&snk_B2_Blocal_r1_r_init, &snk_B2_Blocal_r1_i_init);
    std::vector<std::pair<computation *, computation *>> snk_B2_Blocal_r1_updates;
    complex_expr snk_B2_Q_exprs_r1[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];
      complex_expr B2_prop_0 =  B2_prop(0, t, ic, is, iCprime, iSprime, x, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, kc, ks, kCprime, kSprime, x, y);
      complex_expr B2_prop_0p = B2_prop(0, t, ic, is, kCprime, kSprime, x, y);
      complex_expr B2_prop_2p = B2_prop(2, t, kc, ks, iCprime, iSprime, x, y);
      complex_expr B2_prop_1 = B2_prop(1, t, jc, js, jCprime, jSprime, x, y);
      snk_B2_Q_exprs_r1[jc][js] += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * w;
    }
    // DEFINE computation of Q, and its user -- snk_B2_Blocal_r1 and snk_B2_Bsingle_r1
    std::vector<snkQ2UserEdge> snk_B2_q2userEdges_r1;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (snk_B2_Q_exprs_r1[jc][js].is_zero())
          continue;
        complex_computation q_computation(
            str_fmt("snk_B2_q_r1_%d_%d", jc, js),
            {t, y, iCprime, iSprime, kCprime, kSprime, x},
            snk_B2_Q_exprs_r1[jc][js]);
        complex_expr q = q_computation(t, y, iCprime, iSprime, kCprime, kSprime, x);
        // define local block
        complex_expr blocal_update_def = 
          snk_B2_Blocal_r1_init(t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jc, js, jCprime, jSprime, x, y) * snk_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("snk_B2_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x},
            // definition
            blocal_update_def);
        snk_B2_Blocal_r1_updates.push_back(blocal_update);
        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        snkQ2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        snk_B2_q2userEdges_r1.push_back(edge);
      }
    }
    /*
     * Computing snk_B1_Blocal_r2
     */
    computation snk_B1_Blocal_r2_r_init("snk_B1_Blocal_r2_r_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation snk_B1_Blocal_r2_i_init("snk_B1_Blocal_r2_i_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    complex_computation snk_B1_Blocal_r2_init(&snk_B1_Blocal_r2_r_init, &snk_B1_Blocal_r2_i_init);
    std::vector<std::pair<computation *, computation *>> snk_B1_Blocal_r2_updates;
    complex_expr snk_B1_Q_exprs_r2[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];
      complex_expr B1_prop_0 =  B1_prop(0, t, ic, is, iCprime, iSprime, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, kc, ks, kCprime, kSprime, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, ic, is, kCprime, kSprime, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, kc, ks, iCprime, iSprime, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, jc, js, jCprime, jSprime, x, y);
      snk_B1_Q_exprs_r2[jc][js] += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * w;
    }
    // DEFINE computation of Q, and its user -- snk_B1_Blocal_r2 and snk_B1_Bsingle_r2
    std::vector<snkQ2UserEdge> snk_B1_q2userEdges_r2;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (snk_B1_Q_exprs_r2[jc][js].is_zero())
          continue;
        complex_computation q_computation(
            str_fmt("snk_B1_q_r2_%d_%d", jc, js),
            {t, y, iCprime, iSprime, kCprime, kSprime, x},
            snk_B1_Q_exprs_r2[jc][js]);
        complex_expr q = q_computation(t, y, iCprime, iSprime, kCprime, kSprime, x);
        // define local block
        complex_expr blocal_update_def = 
          snk_B1_Blocal_r2_init(t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jc, js, jCprime, jSprime, x, y) * snk_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("snk_B1_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x},
            // definition
            blocal_update_def);
        snk_B1_Blocal_r2_updates.push_back(blocal_update);
        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        snkQ2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        snk_B1_q2userEdges_r2.push_back(edge);
      }
    }
    /*
     * Computing snk_B2_Blocal_r2
     */
    computation snk_B2_Blocal_r2_r_init("snk_B2_Blocal_r2_r_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation snk_B2_Blocal_r2_i_init("snk_B2_Blocal_r2_i_init", {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    complex_computation snk_B2_Blocal_r2_init(&snk_B2_Blocal_r2_r_init, &snk_B2_Blocal_r2_i_init);
    std::vector<std::pair<computation *, computation *>> snk_B2_Blocal_r2_updates;
    complex_expr snk_B2_Q_exprs_r2[Nc][Ns];
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];
      complex_expr B2_prop_0 =  B2_prop(0, t, ic, is, iCprime, iSprime, x, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, kc, ks, kCprime, kSprime, x, y);
      complex_expr B2_prop_0p = B2_prop(0, t, ic, is, kCprime, kSprime, x, y);
      complex_expr B2_prop_2p = B2_prop(2, t, kc, ks, iCprime, iSprime, x, y);
      complex_expr B2_prop_1 = B2_prop(1, t, jc, js, jCprime, jSprime, x, y);
      snk_B2_Q_exprs_r2[jc][js] += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * w;
    }
    // DEFINE computation of Q, and its user -- snk_B2_Blocal_r2 and snk_B2_Bsingle_r2
    std::vector<snkQ2UserEdge> snk_B2_q2userEdges_r2;
    for (int jc = 0; jc < Nc; jc++) {
      for (int js = 0; js < Ns; js++) {
        if (snk_B2_Q_exprs_r2[jc][js].is_zero())
          continue;
        complex_computation q_computation(
            str_fmt("snk_B2_q_r2_%d_%d", jc, js),
            {t, y, iCprime, iSprime, kCprime, kSprime, x},
            snk_B2_Q_exprs_r2[jc][js]);
        complex_expr q = q_computation(t, y, iCprime, iSprime, kCprime, kSprime, x);
        // define local block
        complex_expr blocal_update_def = 
          snk_B2_Blocal_r2_init(t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jc, js, jCprime, jSprime, x, y) * snk_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("snk_B2_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x},
            // definition
            blocal_update_def);
        snk_B2_Blocal_r2_updates.push_back(blocal_update);
        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        snkQ2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        snk_B2_q2userEdges_r2.push_back(edge);
      }
    }

    /* Correlator */

    computation C_init_r("C_init_r", {t, mpmH, r, npnH}, expr((double) 0));
    computation C_init_i("C_init_i", {t, mpmH, r, npnH}, expr((double) 0));

    // BB_BB
    computation C_BB_BB_par_init_r("C_BB_BB_par_init_r", {t, x, m, r, n}, expr((double) 0));
    computation C_BB_BB_par_init_i("C_BB_BB_par_init_i", {t, x, m, r, n}, expr((double) 0));
    
    b=0;
    /* r1, b = 0 */
    complex_computation BB_BB_new_term_0_r1_b1("BB_BB_new_term_0_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_0_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r1_b1("BB_BB_new_term_1_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r1_init(t, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_1_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r1_b1("BB_BB_new_term_2_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_2_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r1_b1("BB_BB_new_term_3_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r1_init(t, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), x));
    BB_BB_new_term_3_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r1_b1("BB_BB_new_term_4_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    BB_BB_new_term_4_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation BB_BB_new_term_5_r1_b1("BB_BB_new_term_5_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    BB_BB_new_term_5_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    /* r2, b = 0 */
    complex_computation BB_BB_new_term_0_r2_b1("BB_BB_new_term_0_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_0_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r2_b1("BB_BB_new_term_1_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r2_init(t, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_1_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r2_b1("BB_BB_new_term_2_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_2_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r2_b1("BB_BB_new_term_3_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_3_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r2_b1("BB_BB_new_term_4_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    BB_BB_new_term_4_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation BB_BB_new_term_5_r2_b1("BB_BB_new_term_5_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    BB_BB_new_term_5_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    b=1;
    /* r1, b = 1 */
    complex_computation BB_BB_new_term_0_r1_b2("BB_BB_new_term_0_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_0_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r1_b2("BB_BB_new_term_1_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r1_init(t, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_1_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r1_b2("BB_BB_new_term_2_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_2_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r1_b2("BB_BB_new_term_3_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_3_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r1_b2("BB_BB_new_term_4_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    BB_BB_new_term_4_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation BB_BB_new_term_5_r1_b2("BB_BB_new_term_5_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    BB_BB_new_term_5_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    /* r2, b = 1 */
    complex_computation BB_BB_new_term_0_r2_b2("BB_BB_new_term_0_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_0_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_1_r2_b2("BB_BB_new_term_1_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r2_init(t, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_1_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_2_r2_b2("BB_BB_new_term_2_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BB_BB_new_term_2_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation BB_BB_new_term_3_r2_b2("BB_BB_new_term_3_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BB_BB_new_term_3_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation BB_BB_new_term_4_r2_b2("BB_BB_new_term_4_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    BB_BB_new_term_4_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation BB_BB_new_term_5_r2_b2("BB_BB_new_term_5_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    BB_BB_new_term_5_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));


    complex_expr prefactor(cast(p_float64, sigs(nperm)) * snk_weights(r, wnum), 0.0);

    complex_expr BB_BB_term_res_b1 = BB_BB_new_term_0_r1_b1(t, x, x2, m, r, nperm, wnum);
    complex_expr BB_BB_term_res_b2 = BB_BB_new_term_0_r1_b2(t, x, x2, m, r, nperm, wnum);

    complex_expr BB_BB_term_res = prefactor * BB_BB_term_res_b1 * BB_BB_term_res_b2 * snk_psi;

    computation C_BB_BB_par_update_r("C_BB_BB_par_update_r", {t, x, x2, m, r, nperm, wnum, n}, C_BB_BB_par_init_r(t, x, m, r, n) + BB_BB_term_res.get_real());
    computation C_BB_BB_par_update_i("C_BB_BB_par_update_i", {t, x, x2, m, r, nperm, wnum, n}, C_BB_BB_par_init_i(t, x, m, r, n) + BB_BB_term_res.get_imag());

    computation C_BB_BB_update_r("C_BB_BB_update_r", {t, x, m, r, n}, C_init_r(t, m, r, n) + C_BB_BB_par_init_r(t, x, m, r, n));
    computation C_BB_BB_update_i("C_BB_BB_update_i", {t, x, m, r, n}, C_init_i(t, m, r, n) + C_BB_BB_par_init_i(t, x, m, r, n));

    // BB_H
    computation C_BB_H_par_init_r("C_BB_H_par_init_r", {t, x, m, r, nH}, expr((double) 0));
    computation C_BB_H_par_init_i("C_BB_H_par_init_i", {t, x, m, r, nH}, expr((double) 0));
    
    complex_computation BBH_new_term_0_r1_b1("BBH_new_term_0_r1_b1", {t, x, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BBH_new_term_0_r1_b1.add_predicate((snk_blocks(r, 0) == 1));
    complex_computation BBH_new_term_0_r2_b1("BBH_new_term_0_r2_b1", {t, x, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    BBH_new_term_0_r2_b1.add_predicate((snk_blocks(r, 0) == 2));

    complex_computation BBH_new_term_0_r1_b2("BBH_new_term_0_r1_b2", {t, x, m, r, nperm, wnum}, B2_Blocal_r1_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BBH_new_term_0_r1_b2.add_predicate((snk_blocks(r, 1) == 1));
    complex_computation BBH_new_term_0_r2_b2("BBH_new_term_0_r2_b2", {t, x, m, r, nperm, wnum}, B2_Blocal_r2_init(t, x, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    BBH_new_term_0_r2_b2.add_predicate((snk_blocks(r, 1) == 2));

    complex_expr BBH_term_res_b1 = BBH_new_term_0_r1_b1(t, x, m, r, nperm, wnum);
    complex_expr BBH_term_res_b2 = BBH_new_term_0_r1_b2(t, x, m, r, nperm, wnum);

    complex_expr BBH_term_res = prefactor * BBH_term_res_b1 * BBH_term_res_b2 * hex_snk_psi;

    computation C_BB_H_par_update_r("C_BB_H_par_update_r", {t, x, m, r, nperm, wnum, nH}, C_BB_H_par_init_r(t, x, m, r, nH) + BBH_term_res.get_real());
    computation C_BB_H_par_update_i("C_BB_H_par_update_i", {t, x, m, r, nperm, wnum, nH}, C_BB_H_par_init_i(t, x, m, r, nH) + BBH_term_res.get_imag());

    computation C_BB_H_update_r("C_BB_H_update_r", {t, x, m, r, npnH}, C_init_r(t, m, r, npnH) + C_BB_H_par_init_r(t, x, m, r, npnH-Nsnk));
    C_BB_H_update_r.add_predicate(npnH >= Nsnk);
    computation C_BB_H_update_i("C_BB_H_update_i", {t, x, m, r, npnH}, C_init_i(t, m, r, npnH) + C_BB_H_par_init_i(t, x, m, r, npnH-Nsnk));
    C_BB_H_update_i.add_predicate(npnH >= Nsnk);

    // H_BB
    computation C_H_BB_par_init_r("C_H_BB_par_init_r", {t, y, n, r, mH}, expr((double) 0));
    computation C_H_BB_par_init_i("C_H_BB_par_init_i", {t, y, n, r, mH}, expr((double) 0));
    
    complex_computation HBB_new_term_0_r1_b1("HBB_new_term_0_r1_b1", {t, y, n, r, nperm, wnum}, snk_B1_Blocal_r1_init(t, y, n, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    HBB_new_term_0_r1_b1.add_predicate((snk_blocks(r, 0) == 1));
    complex_computation HBB_new_term_0_r2_b1("HBB_new_term_0_r2_b1", {t, y, n, r, nperm, wnum}, snk_B1_Blocal_r2_init(t, y, n, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    HBB_new_term_0_r2_b1.add_predicate((snk_blocks(r, 0) == 2));

    complex_computation HBB_new_term_0_r1_b2("HBB_new_term_0_r1_b2", {t, y, n, r, nperm, wnum}, snk_B2_Blocal_r1_init(t, y, n, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    HBB_new_term_0_r1_b2.add_predicate((snk_blocks(r, 1) == 1));
    complex_computation HBB_new_term_0_r2_b2("HBB_new_term_0_r2_b2", {t, y, n, r, nperm, wnum}, snk_B2_Blocal_r2_init(t, y, n, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    HBB_new_term_0_r2_b2.add_predicate((snk_blocks(r, 1) == 2));

    complex_expr HBB_term_res_b1 = HBB_new_term_0_r1_b1(t, y, n, r, nperm, wnum);
    complex_expr HBB_term_res_b2 = HBB_new_term_0_r1_b2(t, y, n, r, nperm, wnum);

    complex_expr HBB_term_res = prefactor * HBB_term_res_b1 * HBB_term_res_b2 * hex_src_psi;

    computation C_H_BB_par_update_r("C_H_BB_par_update_r", {t, y, n, r, nperm, wnum, mH}, C_H_BB_par_init_r(t, y, n, r, mH) + HBB_term_res.get_real());
    computation C_H_BB_par_update_i("C_H_BB_par_update_i", {t, y, n, r, nperm, wnum, mH}, C_H_BB_par_init_i(t, y, n, r, mH) + HBB_term_res.get_imag());

    computation C_H_BB_update_r("C_H_BB_update_r", {t, y, mpmH, r, n}, C_init_r(t, mpmH, r, n) + C_H_BB_par_init_r(t, y, n, r, mpmH-Nsrc));
    C_H_BB_update_r.add_predicate(mpmH >= Nsrc);
    computation C_H_BB_update_i("C_H_BB_update_i", {t, y, mpmH, r, n}, C_init_i(t, mpmH, r, n) + C_H_BB_par_init_i(t, y, n, r, mpmH-Nsrc));
    C_H_BB_update_i.add_predicate(mpmH >= Nsrc);


    // H_H
    computation C_H_H_par_init_r("C_H_H_par_init_r", {t, x, r, mH, nH}, expr((double) 0));
    computation C_H_H_par_init_i("C_H_H_par_init_i", {t, x, r, mH, nH}, expr((double) 0));

    complex_expr B1_H_r1;
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];
      complex_expr B1_prop_0 =  B1_prop(0, t, snk_color_weights(r,nperm,wnum,0,0), snk_spin_weights(r,nperm,wnum,0,0), ic, is, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, snk_color_weights(r,nperm,wnum,2,0), snk_spin_weights(r,nperm,wnum,2,0), kc, ks, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, snk_color_weights(r,nperm,wnum,2,0), snk_spin_weights(r,nperm,wnum,2,0), ic, is, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, snk_color_weights(r,nperm,wnum,0,0), snk_spin_weights(r,nperm,wnum,0,0), kc, ks, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, snk_color_weights(r,nperm,wnum,1,0), snk_spin_weights(r,nperm,wnum,1,0), jc, js, x, y);
      B1_H_r1 += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * B1_prop_1 * w;
    }
    complex_expr B2_H_r1;
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r1_P[ii][0];
      int is = src_spin_weights_r1_P[ii][0];
      int jc = src_color_weights_r1_P[ii][1];
      int js = src_spin_weights_r1_P[ii][1];
      int kc = src_color_weights_r1_P[ii][2];
      int ks = src_spin_weights_r1_P[ii][2];
      double w = src_weights_r1_P[ii];
      complex_expr B2_prop_0 =  B2_prop(0, t, snk_color_weights(r,nperm,wnum,0,1), snk_spin_weights(r,nperm,wnum,0,1), ic, is, x, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, snk_color_weights(r,nperm,wnum,2,1), snk_spin_weights(r,nperm,wnum,2,1), kc, ks, x, y);
      complex_expr B2_prop_0p = B2_prop(0, t, snk_color_weights(r,nperm,wnum,2,1), snk_spin_weights(r,nperm,wnum,2,1), ic, is, x, y);
      complex_expr B2_prop_2p = B2_prop(2, t, snk_color_weights(r,nperm,wnum,0,1), snk_spin_weights(r,nperm,wnum,0,1), kc, ks, x, y);
      complex_expr B2_prop_1 = B2_prop(1, t, snk_color_weights(r,nperm,wnum,1,1), snk_spin_weights(r,nperm,wnum,1,1), jc, js, x, y);
      B2_H_r1 += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * B2_prop_1 * w;
    }
    complex_expr B1_H_r2;
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];
      complex_expr B1_prop_0 =  B1_prop(0, t, snk_color_weights(r,nperm,wnum,0,0), snk_spin_weights(r,nperm,wnum,0,0), ic, is, x, y);
      complex_expr B1_prop_2 =  B1_prop(2, t, snk_color_weights(r,nperm,wnum,2,0), snk_spin_weights(r,nperm,wnum,2,0), kc, ks, x, y);
      complex_expr B1_prop_0p = B1_prop(0, t, snk_color_weights(r,nperm,wnum,2,0), snk_spin_weights(r,nperm,wnum,2,0), ic, is, x, y);
      complex_expr B1_prop_2p = B1_prop(2, t, snk_color_weights(r,nperm,wnum,0,0), snk_spin_weights(r,nperm,wnum,0,0), kc, ks, x, y);
      complex_expr B1_prop_1 = B1_prop(1, t, snk_color_weights(r,nperm,wnum,1,0), snk_spin_weights(r,nperm,wnum,1,0), jc, js, x, y);
      B1_H_r2 += (B1_prop_0 * B1_prop_2 - B1_prop_0p * B1_prop_2p) * B1_prop_1 * w;
    }
    complex_expr B2_H_r2;
    for (int ii = 0; ii < Nw; ii++) {
      int ic = src_color_weights_r2_P[ii][0];
      int is = src_spin_weights_r2_P[ii][0];
      int jc = src_color_weights_r2_P[ii][1];
      int js = src_spin_weights_r2_P[ii][1];
      int kc = src_color_weights_r2_P[ii][2];
      int ks = src_spin_weights_r2_P[ii][2];
      double w = src_weights_r2_P[ii];
      complex_expr B2_prop_0 =  B2_prop(0, t, snk_color_weights(r,nperm,wnum,0,1), snk_spin_weights(r,nperm,wnum,0,1), ic, is, x, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, snk_color_weights(r,nperm,wnum,2,1), snk_spin_weights(r,nperm,wnum,2,1), kc, ks, x, y);
      complex_expr B2_prop_0p = B2_prop(0, t, snk_color_weights(r,nperm,wnum,2,1), snk_spin_weights(r,nperm,wnum,2,1), ic, is, x, y);
      complex_expr B2_prop_2p = B2_prop(2, t, snk_color_weights(r,nperm,wnum,0,1), snk_spin_weights(r,nperm,wnum,0,1), kc, ks, x, y);
      complex_expr B2_prop_1 = B2_prop(1, t, snk_color_weights(r,nperm,wnum,1,1), snk_spin_weights(r,nperm,wnum,1,1), jc, js, x, y);
      B2_H_r2 += (B2_prop_0 * B2_prop_2 - B2_prop_0p * B2_prop_2p) * B2_prop_1 * w;
    }

    complex_computation HH_new_term_0_r1_b1("HH_new_term_0_r1_b1", {t, x, y, r, nperm, wnum}, B1_H_r1);
    HH_new_term_0_r1_b1.add_predicate((snk_blocks(r, 0) == 1));
    complex_computation HH_new_term_0_r2_b1("HH_new_term_0_r2_b1", {t, x, y, r, nperm, wnum}, B1_H_r2);
    HH_new_term_0_r2_b1.add_predicate((snk_blocks(r, 0) == 2));

    complex_computation HH_new_term_0_r1_b2("HH_new_term_0_r1_b2", {t, x, y, r, nperm, wnum}, B2_H_r1);
    HH_new_term_0_r1_b2.add_predicate((snk_blocks(r, 1) == 1));
    complex_computation HH_new_term_0_r2_b2("HH_new_term_0_r2_b2", {t, x, y, r, nperm, wnum}, B2_H_r2);
    HH_new_term_0_r2_b2.add_predicate((snk_blocks(r, 1) == 2));

    complex_expr HH_term_res = prefactor * hex_src_psi * hex_snk_psi * HH_new_term_0_r1_b1(t, x, y, r, nperm, wnum) * HH_new_term_0_r1_b2(t, x, y, r, nperm, wnum);

    computation C_H_H_par_update_r("C_H_H_par_update_r", {t, x, y, r, nperm, wnum, mH, nH}, C_H_H_par_init_r(t, x, r, mH, nH) + HH_term_res.get_real());
    computation C_H_H_par_update_i("C_H_H_par_update_i", {t, x, y, r, nperm, wnum, mH, nH}, C_H_H_par_init_i(t, x, r, mH, nH) + HH_term_res.get_imag());

    computation C_H_H_update_r("C_H_H_update_r", {t, x, mpmH, r, npnH}, C_init_r(t, mpmH, r, npnH) + C_H_H_par_init_r(t, x, r, mpmH-Nsrc, npnH-Nsnk));
    C_H_H_update_r.add_predicate((npnH >= Nsnk) && (mpmH >= Nsrc));
    computation C_H_H_update_i("C_H_H_update_i", {t, x, mpmH, r, npnH}, C_init_i(t, mpmH, r, npnH) + C_H_H_par_init_i(t, x, r, mpmH-Nsrc, npnH-Nsnk));
    C_H_H_update_i.add_predicate((npnH >= Nsnk) && (mpmH >= Nsrc));


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, npnH)
          .then(C_BB_BB_par_init_r, t)
          .then(C_BB_BB_par_init_i, n)
          .then(C_BB_H_par_init_r, t)
          .then(C_BB_H_par_init_i, nH)
          .then(C_H_BB_par_init_r, t)
          .then(C_H_BB_par_init_i, mH)
          .then(C_H_H_par_init_r, t)
          .then(C_H_H_par_init_i, nH)
    );

    // first the y only arrays
    handle = &(handle
        ->then(snk_B1_Blocal_r1_r_init, t)
        .then(snk_B1_Blocal_r1_i_init, jSprime)
        .then(snk_B1_Blocal_r2_r_init, n)
        .then(snk_B1_Blocal_r2_i_init, jSprime)
        .then(snk_B2_Blocal_r1_r_init, n)
        .then(snk_B2_Blocal_r1_i_init, jSprime)
        .then(snk_B2_Blocal_r2_r_init, n)
        .then(snk_B2_Blocal_r2_i_init, jSprime));
    // schedule snk_B1_Blocal_r1
    for (int i = 0; i < snk_B1_q2userEdges_r1.size(); i++)
    {
      auto edge = snk_B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, y)
          .then(*edge.q_i, x)
          .then(*edge.bl_r, y)
          .then(*edge.bl_i, x)
	  );
    }
     // schedule snk_B1_Blocal_r2
    for (int i = 0; i < snk_B1_q2userEdges_r2.size(); i++)
    {
      auto edge = snk_B1_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, y)
          .then(*edge.q_i, x)
          .then(*edge.bl_r, y)
          .then(*edge.bl_i, x)
	  );
    }
    // schedule snk_B2_Blocal_r1
    for (int i = 0; i < snk_B2_q2userEdges_r1.size(); i++)
    {
      auto edge = snk_B2_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, y)
          .then(*edge.q_i, x)
          .then(*edge.bl_r, y)
          .then(*edge.bl_i, x)
	  );
    }
    // schedule snk_B2_Blocal_r2
    for (int i = 0; i < snk_B2_q2userEdges_r2.size(); i++)
    {
      auto edge = snk_B2_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, y)
          .then(*edge.q_i, x)
          .then(*edge.bl_r, y)
          .then(*edge.bl_i, x)
	  );
    }

    
    // then the x only arrays
    handle = &(handle
        ->then(B1_Blocal_r1_r_init, t)
        .then(B1_Blocal_r1_i_init, jSprime)
        .then(B1_Blocal_r2_r_init, m)
        .then(B1_Blocal_r2_i_init, jSprime));
    // schedule B1_Blocal_r1
    for (int i = 0; i < B1_q2userEdges_r1.size(); i++)
    {
      auto edge = B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, x)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x)
          .then(*edge.bl_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r1
    for (int i = 0; i < B1_o2userEdges_r1.size(); i++)
    {
      auto edge  = B1_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.o_r, x)
          .then(*edge.o_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r1
    for (int i = 0; i < B1_p2userEdges_r1.size(); i++)
    {
      auto edge  = B1_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
	  );
    }
    // schedule B1_Blocal_r2
    for (int i = 0; i < B1_q2userEdges_r2.size(); i++)
    {
      auto edge = B1_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, x)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x)
          .then(*edge.bl_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r2
    for (int i = 0; i < B1_o2userEdges_r2.size(); i++)
    {
      auto edge  = B1_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.o_r, x)
          .then(*edge.o_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r2
    for (int i = 0; i < B1_p2userEdges_r2.size(); i++)
    {
      auto edge  = B1_p2userEdges_r2[i];

      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
	  );
    }

    // then the x2 only arrays
    handle = &(handle
        ->then(B2_Blocal_r1_r_init, t)
        .then(B2_Blocal_r1_i_init, jSprime)
        .then(B2_Blocal_r2_r_init, m)
        .then(B2_Blocal_r2_i_init, jSprime));

    // schedule B2_Blocal_r1 
    for (int i = 0; i < B2_q2userEdges_r1.size(); i++)
    {
      auto edge = B2_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, x2)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x2)
          .then(*edge.bl_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r1
    for (int i = 0; i < B2_o2userEdges_r1.size(); i++)
    {
      auto edge  = B2_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.o_r, x2)
          .then(*edge.o_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r1
    for (int i = 0; i < B2_p2userEdges_r1.size(); i++)
    {
      auto edge  = B2_p2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
	  );
    }
    // schedule B2_Blocal_r2
    for (int i = 0; i < B2_q2userEdges_r2.size(); i++)
    {
      auto edge = B2_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, x2)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x2)
          .then(*edge.bl_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r2
    for (int i = 0; i < B2_o2userEdges_r2.size(); i++)
    {
      auto edge  = B2_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.o_r, x2)
          .then(*edge.o_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r2
    for (int i = 0; i < B2_p2userEdges_r2.size(); i++)
    {
      auto edge  = B2_p2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
	  );
    }

    // then (x1, x2) arrays
    handle = &(handle
        ->then(B1_Bsingle_r1_r_init, t)
        .then(B1_Bsingle_r1_i_init, jSprime)
        .then(B1_Bdouble_r1_r_init, m)
        .then(B1_Bdouble_r1_i_init, iSprime)
	);
    // schedule B1_Bsingle_r1
    for (int i = 0; i < B1_q2userEdges_r1.size(); i++)
    {
      auto edge = B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.bs_r, m)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r1
    for (int i = 0; i < B1_o2userEdges_r1.size(); i++)
    {
      auto edge  = B1_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r1
    for (int i = 0; i < B1_p2userEdges_r1.size(); i++)
    {
      auto edge  = B1_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B2_Bsingle_r1_r_init, m)
        .then(B2_Bsingle_r1_i_init, jSprime)
        .then(B2_Bdouble_r1_r_init, m)
        .then(B2_Bdouble_r1_i_init, iSprime)
	);

    // schedule B2_Bsingle_r1
    for (int i = 0; i < B2_q2userEdges_r1.size(); i++)
    {
      auto edge = B2_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.bs_r, m)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r1
    for (int i = 0; i < B2_o2userEdges_r1.size(); i++)
    {
      auto edge  = B2_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r1
    for (int i = 0; i < B2_p2userEdges_r1.size(); i++)
    {
      auto edge  = B2_p2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B1_Bsingle_r2_r_init, m)
        .then(B1_Bsingle_r2_i_init, jSprime)
        .then(B1_Bdouble_r2_r_init, m)
        .then(B1_Bdouble_r2_i_init, iSprime)
	);
    // schedule B1_Bsingle_r2
    for (int i = 0; i < B1_q2userEdges_r2.size(); i++)
    {
      auto edge = B1_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.bs_r, m)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r2
    for (int i = 0; i < B1_o2userEdges_r2.size(); i++)
    {
      auto edge  = B1_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r2
    for (int i = 0; i < B1_p2userEdges_r2.size(); i++)
    {
      auto edge  = B1_p2userEdges_r2[i];

      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B2_Bsingle_r2_r_init, m)
        .then(B2_Bsingle_r2_i_init, jSprime)
        .then(B2_Bdouble_r2_r_init, m)
        .then(B2_Bdouble_r2_i_init, iSprime)
	);

    // schedule B2_Bsingle_r2
    for (int i = 0; i < B2_q2userEdges_r2.size(); i++)
    {
      auto edge = B2_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.bs_r, m)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r2
    for (int i = 0; i < B2_o2userEdges_r2.size(); i++)
    {
      auto edge  = B2_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.bd_r, m)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r2
    for (int i = 0; i < B2_p2userEdges_r2.size(); i++)
    {
      auto edge  = B2_p2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
          ->then( *(BB_BB_new_term_0_r1_b1.get_real()), m)
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
          .then(C_BB_BB_par_update_r, wnum) 
          .then(C_BB_BB_par_update_i, n));
      
      handle = &(handle
          ->then( *(BBH_new_term_0_r1_b1.get_real()), x)
          .then( *(BBH_new_term_0_r1_b1.get_imag()), wnum)
          .then( *(BBH_new_term_0_r2_b1.get_real()), wnum)
          .then( *(BBH_new_term_0_r2_b1.get_imag()), wnum)
          .then( *(BBH_new_term_0_r1_b2.get_real()), wnum)
          .then( *(BBH_new_term_0_r1_b2.get_imag()), wnum)
          .then( *(BBH_new_term_0_r2_b2.get_real()), wnum)
          .then( *(BBH_new_term_0_r2_b2.get_imag()), wnum)
          .then(C_BB_H_par_update_r, wnum) 
          .then(C_BB_H_par_update_i, wnum));

      handle = &(handle
          ->then( *(HBB_new_term_0_r1_b1.get_real()), t)
          .then( *(HBB_new_term_0_r1_b1.get_imag()), wnum)
          .then( *(HBB_new_term_0_r2_b1.get_real()), wnum)
          .then( *(HBB_new_term_0_r2_b1.get_imag()), wnum)
          .then( *(HBB_new_term_0_r1_b2.get_real()), wnum)
          .then( *(HBB_new_term_0_r1_b2.get_imag()), wnum)
          .then( *(HBB_new_term_0_r2_b2.get_real()), wnum)
          .then( *(HBB_new_term_0_r2_b2.get_imag()), wnum)
          .then(C_H_BB_par_update_r, wnum) 
          .then(C_H_BB_par_update_i, wnum));

      handle = &(handle
          ->then( *(HH_new_term_0_r1_b1.get_real()), x)
          .then( *(HH_new_term_0_r1_b1.get_imag()), wnum)
          .then( *(HH_new_term_0_r2_b1.get_real()), wnum)
          .then( *(HH_new_term_0_r2_b1.get_imag()), wnum)
          .then( *(HH_new_term_0_r1_b2.get_real()), wnum)
          .then( *(HH_new_term_0_r1_b2.get_imag()), wnum)
          .then( *(HH_new_term_0_r2_b2.get_real()), wnum)
          .then( *(HH_new_term_0_r2_b2.get_imag()), wnum)
          .then(C_H_H_par_update_r, wnum) 
          .then(C_H_H_par_update_i, wnum));

      handle = &(handle
          ->then(C_BB_BB_update_r, t) 
          .then(C_BB_BB_update_i, n)
          .then(C_BB_H_update_r, t) 
          .then(C_BB_H_update_i, npnH)
          .then(C_H_BB_update_r, t) 
          .then(C_H_BB_update_i, n)
          .then(C_H_H_update_r, t) 
          .then(C_H_H_update_i, npnH)
	  );

#if VECTORIZED

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_vector_level(y, Vsrc);
//      edge.bs_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
//      edge.bl_r->tag_vector_level(jSprime, Ns); // Disabled due to a an error
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : snk_B1_q2userEdges_r1) {
      edge.q_r->tag_vector_level(x, Vsnk);
//      edge.bl_r->tag_vector_level(jSprime, Ns); // Disabled due to a an error
    }
    
    for (auto edge : B2_q2userEdges_r1) {
      edge.q_r->tag_vector_level(y, Vsrc);
//      edge.bs_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
//      edge.bl_r->tag_vector_level(jSprime, Ns); // Disabled due to a an error
    }
    for (auto edge : B2_o2userEdges_r1) {
      edge.o_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : B2_p2userEdges_r1) {
      edge.p_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }

    for (auto edge : B1_q2userEdges_r2) {
      edge.q_r->tag_vector_level(y, Vsrc);
//      edge.bs_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
//      edge.bl_r->tag_vector_level(jSprime, Ns); // Disabled due to a an error
    }
    for (auto edge : B1_o2userEdges_r2) {
      edge.o_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : B1_p2userEdges_r2) {
      edge.p_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : snk_B1_q2userEdges_r2) {
      edge.q_r->tag_vector_level(x, Vsnk);
    }

    for (auto edge : B2_q2userEdges_r2) {
      edge.q_r->tag_vector_level(y, Vsrc);
//      edge.bs_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
//      edge.bl_r->tag_vector_level(jSprime, Ns); // Disabled due to a an error
    }
    for (auto edge : B2_o2userEdges_r2) {
      edge.o_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    }
    for (auto edge : B2_p2userEdges_r2) {
      edge.p_r->tag_vector_level(y, Vsrc);
//      edge.bd_r->tag_vector_level(x2, Vsnk); // Disabled due to a an error
    } 

/*    (BB_BB_new_term_0_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_0_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);

    (BB_BB_new_term_0_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_0_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);

    (BB_BB_new_term_0_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_0_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r1_b2.get_real())->ta_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);

    (BB_BB_new_term_0_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_0_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_1_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_2_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_3_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_4_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (BB_BB_new_term_5_r2_b2.get_imag())->tag_vector_level(wnum, Nw2); */

    C_BB_BB_par_update_r.tag_vector_level(n, Nsnk);
    C_BB_BB_par_update_i.tag_vector_level(n, Nsnk);  
/*    C_BB_BB_update_r.tag_vector_level(n, Nsnk);
    C_BB_BB_update_i.tag_vector_level(n, Nsnk);  

    C_BB_H_par_update_r.tag_vector_level(nH, NsnkHex);
    C_BB_H_par_update_i.tag_vector_level(nH, NsnkHex);  
    C_BB_H_update_r.tag_vector_level(npnH, NsnkTot);
    C_BB_H_update_i.tag_vector_level(npnH, NsnkTot);  

    C_H_BB_par_update_r.tag_vector_level(mH, NsrcHex);
    C_H_BB_par_update_i.tag_vector_level(mH, NsrcHex);  
    C_H_BB_update_r.tag_vector_level(n, Nsnk);
    C_H_BB_update_i.tag_vector_level(n, Nsnk);   

    C_H_H_par_update_r.tag_vector_level(nH, NsnkHex);
    C_H_H_par_update_i.tag_vector_level(nH, NsnkHex);  
    C_H_H_update_r.tag_vector_level(npnH, NsnkTot);
    C_H_H_update_i.tag_vector_level(npnH, NsnkTot); */

#endif

#if PARALLEL

//   C_init_r.tag_parallel_level(mpmH);
//    C_init_i.tag_parallel_level(mpmH);

    C_BB_BB_par_init_r.tag_parallel_level(t);
    C_BB_BB_par_init_i.tag_parallel_level(t);

    C_BB_H_par_init_r.tag_parallel_level(t);
    C_BB_H_par_init_i.tag_parallel_level(t);

    C_H_BB_par_init_r.tag_parallel_level(t);
    C_H_BB_par_init_i.tag_parallel_level(t);

    C_H_H_par_init_r.tag_parallel_level(t);
    C_H_H_par_init_i.tag_parallel_level(t);

    B1_Blocal_r1_r_init.tag_parallel_level(t);
    B1_Blocal_r1_i_init.tag_parallel_level(t);
    B1_Bsingle_r1_r_init.tag_parallel_level(t);
    B1_Bsingle_r1_i_init.tag_parallel_level(t);
    B1_Bdouble_r1_r_init.tag_parallel_level(t);
    B1_Bdouble_r1_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bs_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->tag_parallel_level(t);
      edge.o_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->tag_parallel_level(t);
      edge.p_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }

    B2_Blocal_r1_r_init.tag_parallel_level(t);
    B2_Blocal_r1_i_init.tag_parallel_level(t);
    B2_Bsingle_r1_r_init.tag_parallel_level(t);
    B2_Bsingle_r1_i_init.tag_parallel_level(t);
    B2_Bdouble_r1_r_init.tag_parallel_level(t);
    B2_Bdouble_r1_i_init.tag_parallel_level(t);

    for (auto edge : B2_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bs_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    } 
    for (auto edge : B2_o2userEdges_r1) {
      edge.o_r->tag_parallel_level(t);
      edge.o_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }
    for (auto edge : B2_p2userEdges_r1) {
      edge.p_r->tag_parallel_level(t);
      edge.p_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }

    B1_Blocal_r2_r_init.tag_parallel_level(t);
    B1_Blocal_r2_i_init.tag_parallel_level(t);
    B1_Bsingle_r2_r_init.tag_parallel_level(t);
    B1_Bsingle_r2_i_init.tag_parallel_level(t);
    B1_Bdouble_r2_r_init.tag_parallel_level(t);
    B1_Bdouble_r2_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bs_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }
    for (auto edge : B1_o2userEdges_r2) {
      edge.o_r->tag_parallel_level(t);
      edge.o_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }
    for (auto edge : B1_p2userEdges_r2) {
      edge.p_r->tag_parallel_level(t);
      edge.p_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }

    B2_Blocal_r2_r_init.tag_parallel_level(t);
    B2_Blocal_r2_i_init.tag_parallel_level(t);
    B2_Bsingle_r2_r_init.tag_parallel_level(t);
    B2_Bsingle_r2_i_init.tag_parallel_level(t);
    B2_Bdouble_r2_r_init.tag_parallel_level(t);
    B2_Bdouble_r2_i_init.tag_parallel_level(t);

    for (auto edge : B2_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bs_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }
    for (auto edge : B2_o2userEdges_r2) {
      edge.o_r->tag_parallel_level(t);
      edge.o_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }
    for (auto edge : B2_p2userEdges_r2) {
      edge.p_r->tag_parallel_level(t);
      edge.p_i->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
      edge.bd_i->tag_parallel_level(t);
    }

    snk_B1_Blocal_r1_r_init.tag_parallel_level(t);
    snk_B1_Blocal_r1_i_init.tag_parallel_level(t);
    for (auto edge : snk_B1_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }
    snk_B2_Blocal_r1_r_init.tag_parallel_level(t);
    snk_B2_Blocal_r1_i_init.tag_parallel_level(t);
    for (auto edge : snk_B2_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    } 
    snk_B1_Blocal_r2_r_init.tag_parallel_level(t);
    snk_B1_Blocal_r2_i_init.tag_parallel_level(t);
    for (auto edge : snk_B1_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }
    snk_B2_Blocal_r2_r_init.tag_parallel_level(t);
    snk_B2_Blocal_r2_i_init.tag_parallel_level(t);
    for (auto edge : snk_B2_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }

    (BB_BB_new_term_0_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_0_r1_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_1_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_1_r1_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_2_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_2_r1_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_3_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_3_r1_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_4_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_4_r1_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_5_r1_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_5_r1_b1.get_imag())->tag_parallel_level(t);

    (BB_BB_new_term_0_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_0_r2_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_1_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_1_r2_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_2_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_2_r2_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_3_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_3_r2_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_4_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_4_r2_b1.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_5_r2_b1.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_5_r2_b1.get_imag())->tag_parallel_level(t);

    (BB_BB_new_term_0_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_0_r1_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_1_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_1_r1_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_2_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_2_r1_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_3_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_3_r1_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_4_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_4_r1_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_5_r1_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_5_r1_b2.get_imag())->tag_parallel_level(t);

    (BB_BB_new_term_0_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_0_r2_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_1_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_1_r2_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_2_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_2_r2_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_3_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_3_r2_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_4_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_4_r2_b2.get_imag())->tag_parallel_level(t);
    (BB_BB_new_term_5_r2_b2.get_real())->tag_parallel_level(t);
    (BB_BB_new_term_5_r2_b2.get_imag())->tag_parallel_level(t);

    C_BB_BB_par_update_r.tag_parallel_level(t);
    C_BB_BB_par_update_i.tag_parallel_level(t);  

//    C_BB_BB_update_r.tag_parallel_level(m);
//    C_BB_BB_update_i.tag_parallel_level(m);  

    (BBH_new_term_0_r1_b1.get_real())->tag_parallel_level(t);
    (BBH_new_term_0_r1_b1.get_imag())->tag_parallel_level(t);

    (BBH_new_term_0_r2_b1.get_real())->tag_parallel_level(t);
    (BBH_new_term_0_r2_b1.get_imag())->tag_parallel_level(t);

    (BBH_new_term_0_r1_b2.get_real())->tag_parallel_level(t);
    (BBH_new_term_0_r1_b2.get_imag())->tag_parallel_level(t);

    (BBH_new_term_0_r2_b2.get_real())->tag_parallel_level(t);
    (BBH_new_term_0_r2_b2.get_imag())->tag_parallel_level(t);

    C_BB_H_par_update_r.tag_parallel_level(t);
    C_BB_H_par_update_i.tag_parallel_level(t);  

//    C_BB_H_update_r.tag_parallel_level(m);
//    C_BB_H_update_i.tag_parallel_level(m);  

    (HBB_new_term_0_r1_b1.get_real())->tag_parallel_level(t);
    (HBB_new_term_0_r1_b1.get_imag())->tag_parallel_level(t);

    (HBB_new_term_0_r2_b1.get_real())->tag_parallel_level(t);
    (HBB_new_term_0_r2_b1.get_imag())->tag_parallel_level(t);

    (HBB_new_term_0_r1_b2.get_real())->tag_parallel_level(t);
    (HBB_new_term_0_r1_b2.get_imag())->tag_parallel_level(t);

    (HBB_new_term_0_r2_b2.get_real())->tag_parallel_level(t);
    (HBB_new_term_0_r2_b2.get_imag())->tag_parallel_level(t);

    C_H_BB_par_update_r.tag_parallel_level(t);
    C_H_BB_par_update_i.tag_parallel_level(t);  

//    C_H_BB_update_r.tag_parallel_level(mpmH);
//    C_H_BB_update_i.tag_parallel_level(mpmH); 

    (HH_new_term_0_r1_b1.get_real())->tag_parallel_level(t);
    (HH_new_term_0_r1_b1.get_imag())->tag_parallel_level(t);

    (HH_new_term_0_r2_b1.get_real())->tag_parallel_level(t);
    (HH_new_term_0_r2_b1.get_imag())->tag_parallel_level(t);

    (HH_new_term_0_r1_b2.get_real())->tag_parallel_level(t);
    (HH_new_term_0_r1_b2.get_imag())->tag_parallel_level(t);

    (HH_new_term_0_r2_b2.get_real())->tag_parallel_level(t);
    (HH_new_term_0_r2_b2.get_imag())->tag_parallel_level(t);

    C_H_H_par_update_r.tag_parallel_level(t);
    C_H_H_par_update_i.tag_parallel_level(t);  

//    C_H_H_update_r.tag_parallel_level(mpmH);
//    C_H_H_update_i.tag_parallel_level(mpmH); 

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_r("buf_B1_Bsingle_r1_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_i("buf_B1_Bsingle_r1_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_r("buf_B1_Bdouble_r1_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_i("buf_B1_Bdouble_r1_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r1_r_init.store_in(&buf_B1_Bsingle_r1_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r1_i_init.store_in(&buf_B1_Bsingle_r1_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bdouble_r1_r_init.store_in(&buf_B1_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B1_Bdouble_r1_i_init.store_in(&buf_B1_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

    buffer *B1_q_r1_r_buf;
    buffer *B1_q_r1_i_buf;
    buffer *B1_o_r1_r_buf;
    buffer *B1_o_r1_i_buf;
    buffer *B1_p_r1_r_buf;
    buffer *B1_p_r1_i_buf;

    int B1_r1_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_Q_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_q_r1_r_buf, B1_q_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_q_r1_%d_%d", ic, is));
        (B1_q2userEdges_r1[B1_r1_q_index]).q_r->store_in(B1_q_r1_r_buf, {x, iCprime, iSprime, kCprime, kSprime, y});
        (B1_q2userEdges_r1[B1_r1_q_index]).q_i->store_in(B1_q_r1_i_buf, {x, iCprime, iSprime, kCprime, kSprime, y});
        B1_r1_q_index++;
        }
    int B1_r1_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_o_r1_r_buf, B1_o_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_o_r1_%d_%d", ic, is));
        (B1_o2userEdges_r1[B1_r1_o_index]).o_r->store_in(B1_o_r1_r_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        (B1_o2userEdges_r1[B1_r1_o_index]).o_i->store_in(B1_o_r1_i_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        B1_r1_o_index++;
        }
    int B1_r1_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r1[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B1_p_r1_r_buf, B1_p_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_p_r1_%d_%d", kc, ks));
        (B1_p2userEdges_r1[B1_r1_p_index]).p_r->store_in(B1_p_r1_r_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        (B1_p2userEdges_r1[B1_r1_p_index]).p_i->store_in(B1_p_r1_i_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        B1_r1_p_index++;
        }
    for (auto computations: B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r1_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r1_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r1_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Bsingle_r1_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B1_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B1_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    

    buffer buf_B2_Blocal_r1_r("buf_B2_Blocal_r1_r", {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r1_i("buf_B2_Blocal_r1_i", {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_r("buf_B2_Bsingle_r1_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_i("buf_B2_Bsingle_r1_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_r("buf_B2_Bdouble_r1_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_i("buf_B2_Bdouble_r1_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B2_Blocal_r1_r_init.store_in(&buf_B2_Blocal_r1_r, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Blocal_r1_i_init.store_in(&buf_B2_Blocal_r1_i, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bsingle_r1_r_init.store_in(&buf_B2_Bsingle_r1_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Bsingle_r1_i_init.store_in(&buf_B2_Bsingle_r1_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bdouble_r1_r_init.store_in(&buf_B2_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B2_Bdouble_r1_i_init.store_in(&buf_B2_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

    buffer *B2_q_r1_r_buf;
    buffer *B2_q_r1_i_buf;
    buffer *B2_o_r1_r_buf;
    buffer *B2_o_r1_i_buf;
    buffer *B2_p_r1_r_buf;
    buffer *B2_p_r1_i_buf;
    
    int B2_r1_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_Q_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_q_r1_r_buf, B2_q_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_q_r1_%d_%d", ic, is));
        (B2_q2userEdges_r1[B2_r1_q_index]).q_r->store_in(B2_q_r1_r_buf, {x2, iCprime, iSprime, kCprime, kSprime, y});
        (B2_q2userEdges_r1[B2_r1_q_index]).q_i->store_in(B2_q_r1_i_buf, {x2, iCprime, iSprime, kCprime, kSprime, y});
        B2_r1_q_index++;
        }
    int B2_r1_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_o_r1_r_buf, B2_o_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_o_r1_%d_%d", ic, is));
        (B2_o2userEdges_r1[B2_r1_o_index]).o_r->store_in(B2_o_r1_r_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        (B2_o2userEdges_r1[B2_r1_o_index]).o_i->store_in(B2_o_r1_i_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        B2_r1_o_index++;
        }
    int B2_r1_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r1[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B2_p_r1_r_buf, B2_p_r1_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_p_r1_%d_%d", kc, ks));
        (B2_p2userEdges_r1[B2_r1_p_index]).p_r->store_in(B2_p_r1_r_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        (B2_p2userEdges_r1[B2_r1_p_index]).p_i->store_in(B2_p_r1_i_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        B2_r1_p_index++;
        }

    for (auto computations: B2_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Blocal_r1_r, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Blocal_r1_i, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B2_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bsingle_r1_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Bsingle_r1_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B2_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B2_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r1_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r2_r("buf_B1_Bsingle_r2_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r2_i("buf_B1_Bsingle_r2_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r2_r("buf_B1_Bdouble_r2_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r2_i("buf_B1_Bdouble_r2_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r2_r_init.store_in(&buf_B1_Bsingle_r2_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r2_i_init.store_in(&buf_B1_Bsingle_r2_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bdouble_r2_r_init.store_in(&buf_B1_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B1_Bdouble_r2_i_init.store_in(&buf_B1_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

    buffer *B1_q_r2_r_buf;
    buffer *B1_q_r2_i_buf;
    buffer *B1_o_r2_r_buf;
    buffer *B1_o_r2_i_buf;
    buffer *B1_p_r2_r_buf;
    buffer *B1_p_r2_i_buf;

    int B1_r2_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_Q_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_q_r2_r_buf, B1_q_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_q_r2_%d_%d", ic, is));
        (B1_q2userEdges_r2[B1_r2_q_index]).q_r->store_in(B1_q_r2_r_buf, {x, iCprime, iSprime, kCprime, kSprime, y});
        (B1_q2userEdges_r2[B1_r2_q_index]).q_i->store_in(B1_q_r2_i_buf, {x, iCprime, iSprime, kCprime, kSprime, y});
        B1_r2_q_index++;
        }
    int B1_r2_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_o_r2_r_buf, B1_o_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_o_r2_%d_%d", ic, is));
        (B1_o2userEdges_r2[B1_r2_o_index]).o_r->store_in(B1_o_r2_r_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        (B1_o2userEdges_r2[B1_r2_o_index]).o_i->store_in(B1_o_r2_i_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        B1_r2_o_index++;
        }
    int B1_r2_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r2[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B1_p_r2_r_buf, B1_p_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_p_r2_%d_%d", kc, ks));
        (B1_p2userEdges_r2[B1_r2_p_index]).p_r->store_in(B1_p_r2_r_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        (B1_p2userEdges_r2[B1_r2_p_index]).p_i->store_in(B1_p_r2_i_buf, {x, jCprime, jSprime, kCprime, kSprime, y});
        B1_r2_p_index++;
        }
    for (auto computations: B1_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r2_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r2_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r2_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Bsingle_r2_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B1_Bdouble_r2_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B1_Bdouble_r2_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    buffer buf_B2_Blocal_r2_r("buf_B2_Blocal_r2_r", {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r2_i("buf_B2_Blocal_r2_i", {Vsnk, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r2_r("buf_B2_Bsingle_r2_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r2_i("buf_B2_Bsingle_r2_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r2_r("buf_B2_Bdouble_r2_r", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r2_i("buf_B2_Bdouble_r2_i", {Vsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B2_Blocal_r2_r_init.store_in(&buf_B2_Blocal_r2_r, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Blocal_r2_i_init.store_in(&buf_B2_Blocal_r2_i, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bsingle_r2_r_init.store_in(&buf_B2_Bsingle_r2_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Bsingle_r2_i_init.store_in(&buf_B2_Bsingle_r2_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bdouble_r2_r_init.store_in(&buf_B2_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B2_Bdouble_r2_i_init.store_in(&buf_B2_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

    buffer *B2_q_r2_r_buf;
    buffer *B2_q_r2_i_buf;
    buffer *B2_o_r2_r_buf;
    buffer *B2_o_r2_i_buf;
    buffer *B2_p_r2_r_buf;
    buffer *B2_p_r2_i_buf;
    
    int B2_r2_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_Q_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_q_r2_r_buf, B2_q_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_q_r2_%d_%d", ic, is));
        (B2_q2userEdges_r2[B2_r2_q_index]).q_r->store_in(B2_q_r2_r_buf, {x2, iCprime, iSprime, kCprime, kSprime, y});
        (B2_q2userEdges_r2[B2_r2_q_index]).q_i->store_in(B2_q_r2_i_buf, {x2, iCprime, iSprime, kCprime, kSprime, y});
        B2_r2_q_index++;
        }
    int B2_r2_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_o_r2_r_buf, B2_o_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_o_r2_%d_%d", ic, is));
        (B2_o2userEdges_r2[B2_r2_o_index]).o_r->store_in(B2_o_r2_r_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        (B2_o2userEdges_r2[B2_r2_o_index]).o_i->store_in(B2_o_r2_i_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        B2_r2_o_index++;
        }
    int B2_r2_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r2[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B2_p_r2_r_buf, B2_p_r2_i_buf, {Vsnk, Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_p_r2_%d_%d", kc, ks));
        (B2_p2userEdges_r2[B2_r2_p_index]).p_r->store_in(B2_p_r2_r_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        (B2_p2userEdges_r2[B2_r2_p_index]).p_i->store_in(B2_p_r2_i_buf, {x2, jCprime, jSprime, kCprime, kSprime, y});
        B2_r2_p_index++;
        }

    for (auto computations: B2_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Blocal_r2_r, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Blocal_r2_i, {x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B2_Bsingle_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bsingle_r2_r, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Bsingle_r2_i, {x, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B2_Bdouble_r2_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B2_Bdouble_r2_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r2_r, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r2_i, {x, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    buffer buf_snk_B1_Blocal_r1_r("buf_snk_B1_Blocal_r1_r",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_r1_i("buf_snk_B1_Blocal_r1_i",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    snk_B1_Blocal_r1_r_init.store_in(&buf_snk_B1_Blocal_r1_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    snk_B1_Blocal_r1_i_init.store_in(&buf_snk_B1_Blocal_r1_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    buffer *snk_B1_q_r1_r_buf;
    buffer *snk_B1_q_r1_i_buf;
    int snk_B1_r1_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (snk_B1_Q_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(snk_B1_q_r1_r_buf, snk_B1_q_r1_i_buf, {Vsrc, Nc, Ns, Nc, Ns, Vsnk}, str_fmt("buf_snk_B1_q_r1_%d_%d", ic, is));
        (snk_B1_q2userEdges_r1[snk_B1_r1_q_index]).q_r->store_in(snk_B1_q_r1_r_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        (snk_B1_q2userEdges_r1[snk_B1_r1_q_index]).q_i->store_in(snk_B1_q_r1_i_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        snk_B1_r1_q_index++;
        }
    for (auto computations: snk_B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_snk_B1_Blocal_r1_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_snk_B1_Blocal_r1_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    buffer buf_snk_B2_Blocal_r1_r("buf_snk_B2_Blocal_r1_r",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_r1_i("buf_snk_B2_Blocal_r1_i",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    snk_B2_Blocal_r1_r_init.store_in(&buf_snk_B2_Blocal_r1_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    snk_B2_Blocal_r1_i_init.store_in(&buf_snk_B2_Blocal_r1_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    buffer *snk_B2_q_r1_r_buf;
    buffer *snk_B2_q_r1_i_buf;
    int snk_B2_r1_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (snk_B2_Q_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(snk_B2_q_r1_r_buf, snk_B2_q_r1_i_buf, {Vsrc, Nc, Ns, Nc, Ns, Vsnk}, str_fmt("buf_snk_B2_q_r1_%d_%d", ic, is));
        (snk_B2_q2userEdges_r1[snk_B2_r1_q_index]).q_r->store_in(snk_B2_q_r1_r_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        (snk_B2_q2userEdges_r1[snk_B2_r1_q_index]).q_i->store_in(snk_B2_q_r1_i_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        snk_B2_r1_q_index++;
        }
    for (auto computations: snk_B2_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_snk_B2_Blocal_r1_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_snk_B2_Blocal_r1_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    buffer buf_snk_B1_Blocal_r2_r("buf_snk_B1_Blocal_r2_r",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B1_Blocal_r2_i("buf_snk_B1_Blocal_r2_i",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    snk_B1_Blocal_r2_r_init.store_in(&buf_snk_B1_Blocal_r2_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    snk_B1_Blocal_r2_i_init.store_in(&buf_snk_B1_Blocal_r2_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    buffer *snk_B1_q_r2_r_buf;
    buffer *snk_B1_q_r2_i_buf;
    int snk_B1_r2_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (snk_B1_Q_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(snk_B1_q_r2_r_buf, snk_B1_q_r2_i_buf, {Vsrc, Nc, Ns, Nc, Ns, Vsnk}, str_fmt("buf_snk_B1_q_r2_%d_%d", ic, is));
        (snk_B1_q2userEdges_r2[snk_B1_r2_q_index]).q_r->store_in(snk_B1_q_r2_r_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        (snk_B1_q2userEdges_r2[snk_B1_r2_q_index]).q_i->store_in(snk_B1_q_r2_i_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        snk_B1_r2_q_index++;
        }
    for (auto computations: snk_B1_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_snk_B1_Blocal_r2_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_snk_B1_Blocal_r2_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    buffer buf_snk_B2_Blocal_r2_r("buf_snk_B2_Blocal_r2_r",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_snk_B2_Blocal_r2_i("buf_snk_B2_Blocal_r2_i",   {Vsrc, Nsnk, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    snk_B2_Blocal_r2_r_init.store_in(&buf_snk_B2_Blocal_r2_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    snk_B2_Blocal_r2_i_init.store_in(&buf_snk_B2_Blocal_r2_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    buffer *snk_B2_q_r2_r_buf;
    buffer *snk_B2_q_r2_i_buf;
    int snk_B2_r2_q_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (snk_B2_Q_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(snk_B2_q_r2_r_buf, snk_B2_q_r2_i_buf, {Vsrc, Nc, Ns, Nc, Ns, Vsnk}, str_fmt("buf_snk_B2_q_r2_%d_%d", ic, is));
        (snk_B2_q2userEdges_r2[snk_B2_r2_q_index]).q_r->store_in(snk_B2_q_r2_r_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        (snk_B2_q2userEdges_r2[snk_B2_r2_q_index]).q_i->store_in(snk_B2_q_r2_i_buf, {y, iCprime, iSprime, kCprime, kSprime, x});
        snk_B2_r2_q_index++;
        }
    for (auto computations: snk_B2_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_snk_B2_Blocal_r2_r, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_snk_B2_Blocal_r2_i, {y, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }

    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, NsrcTot, Nr, NsnkTot}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, NsrcTot, Nr, NsnkTot}, p_float64, a_input);

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);

    C_init_r.store_in(&buf_C_r, {t, mpmH, r, npnH});
    C_init_i.store_in(&buf_C_i, {t, mpmH, r, npnH});

    buffer buf_C_BB_BB_par_r("buf_C_BB_BB_par_r", {Vsnk, Nsrc, Nr, Nsnk}, p_float64, a_temporary);
    buffer buf_C_BB_BB_par_i("buf_C_BB_BB_par_i", {Vsnk, Nsrc, Nr, Nsnk}, p_float64, a_temporary);

    buffer* buf_BB_BB_new_term_r_b1;//("buf_BB_BB_new_term_r_b1", {1}, p_float64, a_temporary);
    buffer* buf_BB_BB_new_term_i_b1;//("buf_BB_BB_new_term_i_b1", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_BB_BB_new_term_r_b1, buf_BB_BB_new_term_i_b1, {Vsnk}, "buf_BB_BB_new_term_b1");
    buffer* buf_BB_BB_new_term_r_b2;//("buf_BB_BB_new_term_r_b2", {1}, p_float64, a_temporary);
    buffer* buf_BB_BB_new_term_i_b2;//("buf_BB_BB_new_term_i_b2", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_BB_BB_new_term_r_b2, buf_BB_BB_new_term_i_b2, {Vsnk}, "buf_BB_BB_new_term_b2"); 

    BB_BB_new_term_0_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_0_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_1_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_1_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x}); 
    BB_BB_new_term_2_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_2_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_3_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_3_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_4_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_4_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_5_r1_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_5_r1_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x}); 

    BB_BB_new_term_0_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_0_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_1_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_1_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 
    BB_BB_new_term_2_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_2_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_3_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_3_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 
    BB_BB_new_term_4_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_4_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_5_r1_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_5_r1_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 

    BB_BB_new_term_0_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_0_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_1_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_1_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x}); 
    BB_BB_new_term_2_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_2_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_3_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_3_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_4_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_4_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x});
    BB_BB_new_term_5_r2_b1.get_real()->store_in(buf_BB_BB_new_term_r_b1, {x});
    BB_BB_new_term_5_r2_b1.get_imag()->store_in(buf_BB_BB_new_term_i_b1, {x}); 

    BB_BB_new_term_0_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_0_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_1_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_1_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 
    BB_BB_new_term_2_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_2_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_3_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_3_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 
    BB_BB_new_term_4_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_4_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x});
    BB_BB_new_term_5_r2_b2.get_real()->store_in(buf_BB_BB_new_term_r_b2, {x});
    BB_BB_new_term_5_r2_b2.get_imag()->store_in(buf_BB_BB_new_term_i_b2, {x}); 

    C_BB_BB_par_init_r.store_in(&buf_C_BB_BB_par_r, {x, m, r, n});
    C_BB_BB_par_init_i.store_in(&buf_C_BB_BB_par_i, {x, m, r, n});
    C_BB_BB_par_update_r.store_in(&buf_C_BB_BB_par_r, {x, m, r, n});
    C_BB_BB_par_update_i.store_in(&buf_C_BB_BB_par_i, {x, m, r, n});

    C_BB_BB_update_r.store_in(&buf_C_r, {t, m, r, n});
    C_BB_BB_update_i.store_in(&buf_C_i, {t, m, r, n});

    buffer buf_C_BB_H_par_r("buf_C_BB_H_par_r", {Vsnk, Nsrc, Nr, NsnkHex}, p_float64, a_temporary);
    buffer buf_C_BB_H_par_i("buf_C_BB_H_par_i", {Vsnk, Nsrc, Nr, NsnkHex}, p_float64, a_temporary);

    buffer* buf_BBH_new_term_r_b1;
    buffer* buf_BBH_new_term_i_b1;
    allocate_complex_buffers(buf_BBH_new_term_r_b1, buf_BBH_new_term_i_b1, {Vsnk}, "buf_BBH_new_term_b1");
    buffer* buf_BBH_new_term_r_b2;
    buffer* buf_BBH_new_term_i_b2;
    allocate_complex_buffers(buf_BBH_new_term_r_b2, buf_BBH_new_term_i_b2, {Vsnk}, "buf_BBH_new_term_b2");

    BBH_new_term_0_r1_b1.get_real()->store_in(buf_BBH_new_term_r_b1, {x});
    BBH_new_term_0_r1_b1.get_imag()->store_in(buf_BBH_new_term_i_b1, {x});

    BBH_new_term_0_r2_b1.get_real()->store_in(buf_BBH_new_term_r_b1, {x});
    BBH_new_term_0_r2_b1.get_imag()->store_in(buf_BBH_new_term_i_b1, {x});

    BBH_new_term_0_r1_b2.get_real()->store_in(buf_BBH_new_term_r_b2, {x});
    BBH_new_term_0_r1_b2.get_imag()->store_in(buf_BBH_new_term_i_b2, {x});

    BBH_new_term_0_r2_b2.get_real()->store_in(buf_BBH_new_term_r_b2, {x});
    BBH_new_term_0_r2_b2.get_imag()->store_in(buf_BBH_new_term_i_b2, {x});

    C_BB_H_par_init_r.store_in(&buf_C_BB_H_par_r, {x, m, r, nH});
    C_BB_H_par_init_i.store_in(&buf_C_BB_H_par_i, {x, m, r, nH});
    C_BB_H_par_update_r.store_in(&buf_C_BB_H_par_r, {x, m, r, nH});
    C_BB_H_par_update_i.store_in(&buf_C_BB_H_par_i, {x, m, r, nH});

    C_BB_H_update_r.store_in(&buf_C_r, {t, m, r, npnH});
    C_BB_H_update_i.store_in(&buf_C_i, {t, m, r, npnH});

    buffer buf_C_H_BB_par_r("buf_C_H_BB_par_r", {Vsrc, Nsnk, Nr, NsrcHex}, p_float64, a_temporary);
    buffer buf_C_H_BB_par_i("buf_C_H_BB_par_i", {Vsrc, Nsnk, Nr, NsrcHex}, p_float64, a_temporary);

    buffer* buf_HBB_new_term_r_b1;
    buffer* buf_HBB_new_term_i_b1;
    allocate_complex_buffers(buf_HBB_new_term_r_b1, buf_HBB_new_term_i_b1, {Vsrc}, "buf_HBB_new_term_b1");
    buffer* buf_HBB_new_term_r_b2;
    buffer* buf_HBB_new_term_i_b2;
    allocate_complex_buffers(buf_HBB_new_term_r_b2, buf_HBB_new_term_i_b2, {Vsrc}, "buf_HBB_new_term_b2");

    HBB_new_term_0_r1_b1.get_real()->store_in(buf_HBB_new_term_r_b1, {y});
    HBB_new_term_0_r1_b1.get_imag()->store_in(buf_HBB_new_term_i_b1, {y});

    HBB_new_term_0_r2_b1.get_real()->store_in(buf_HBB_new_term_r_b1, {y});
    HBB_new_term_0_r2_b1.get_imag()->store_in(buf_HBB_new_term_i_b1, {y});

    HBB_new_term_0_r1_b2.get_real()->store_in(buf_HBB_new_term_r_b2, {y});
    HBB_new_term_0_r1_b2.get_imag()->store_in(buf_HBB_new_term_i_b2, {y});

    HBB_new_term_0_r2_b2.get_real()->store_in(buf_HBB_new_term_r_b2, {y});
    HBB_new_term_0_r2_b2.get_imag()->store_in(buf_HBB_new_term_i_b2, {y});

    C_H_BB_par_init_r.store_in(&buf_C_H_BB_par_r, {y, n, r, mH});
    C_H_BB_par_init_i.store_in(&buf_C_H_BB_par_i, {y, n, r, mH});
    C_H_BB_par_update_r.store_in(&buf_C_H_BB_par_r, {y, n, r, mH});
    C_H_BB_par_update_i.store_in(&buf_C_H_BB_par_i, {y, n, r, mH});
    C_H_BB_update_r.store_in(&buf_C_r, {t, mpmH, r, n});
    C_H_BB_update_i.store_in(&buf_C_i, {t, mpmH, r, n});

    buffer buf_C_H_H_par_r("buf_C_H_H_par_r", {Vsnk, Nr, NsrcHex, NsnkHex}, p_float64, a_temporary);
    buffer buf_C_H_H_par_i("buf_C_H_H_par_i", {Vsnk, Nr, NsrcHex, NsnkHex}, p_float64, a_temporary);

    buffer* buf_HH_new_term_r_b1;
    buffer* buf_HH_new_term_i_b1;
    allocate_complex_buffers(buf_HH_new_term_r_b1, buf_HH_new_term_i_b1, {Vsnk}, "buf_HH_new_term_b1");
    buffer* buf_HH_new_term_r_b2;
    buffer* buf_HH_new_term_i_b2;
    allocate_complex_buffers(buf_HH_new_term_r_b2, buf_HH_new_term_i_b2, {Vsnk}, "buf_HH_new_term_b2");

    HH_new_term_0_r1_b1.get_real()->store_in(buf_HH_new_term_r_b1, {x});
    HH_new_term_0_r1_b1.get_imag()->store_in(buf_HH_new_term_i_b1, {x});

    HH_new_term_0_r2_b1.get_real()->store_in(buf_HH_new_term_r_b1, {x});
    HH_new_term_0_r2_b1.get_imag()->store_in(buf_HH_new_term_i_b1, {x});

    HH_new_term_0_r1_b2.get_real()->store_in(buf_HH_new_term_r_b2, {x});
    HH_new_term_0_r1_b2.get_imag()->store_in(buf_HH_new_term_i_b2, {x});

    HH_new_term_0_r2_b2.get_real()->store_in(buf_HH_new_term_r_b2, {x});
    HH_new_term_0_r2_b2.get_imag()->store_in(buf_HH_new_term_i_b2, {x});

    C_H_H_par_init_r.store_in(&buf_C_H_H_par_r, {x, r, mH, nH});
    C_H_H_par_init_i.store_in(&buf_C_H_H_par_i, {x, r, mH, nH});
    C_H_H_par_update_r.store_in(&buf_C_H_H_par_r, {x, r, mH, nH});
    C_H_H_par_update_i.store_in(&buf_C_H_H_par_i, {x, r, mH, nH});
    C_H_H_update_r.store_in(&buf_C_r, {t, mpmH, r, npnH});
    C_H_H_update_i.store_in(&buf_C_i, {t, mpmH, r, npnH});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	     &buf_C_r, &buf_C_i,
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        B2_prop_r.get_buffer(), B2_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        src_psi_B2_r.get_buffer(), src_psi_B2_i.get_buffer(),
        snk_psi_B1_r.get_buffer(), snk_psi_B1_i.get_buffer(), 
        snk_psi_B2_r.get_buffer(), snk_psi_B2_i.get_buffer(),
        hex_src_psi_r.get_buffer(), hex_src_psi_i.get_buffer(),
        hex_snk_psi_r.get_buffer(), hex_snk_psi_i.get_buffer(), 
        snk_psi_r.get_buffer(), snk_psi_i.get_buffer(), 
	     snk_blocks.get_buffer(), 
        sigs.get_buffer(),
	     snk_b.get_buffer(),
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer()
        }, 
        "generated_tiramisu_make_fused_dibaryon_blocks_correlator.o");
}

int main(int argc, char **argv)
{
	generate_function("tiramisu_make_fused_dibaryon_blocks_correlator");

    return 0;
}
