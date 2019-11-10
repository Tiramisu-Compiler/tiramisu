#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 0
#define PARALLEL 0

// Used to remember relevant (sub)computation of Q and its user computations (B1_Blocal_r1 and B1_Bsingle_r1)
struct Q2UserEdge {
      computation *q_r, *q_i,
                  *bs_r, *bs_i,
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

   int Nr = 1;
   var nperm("nperm", 0, Nperms),
	r("r", 0, Nr),
	b("b", 0, Nb),
	q("q", 0, Nq),
	q2("q2", 0, 2*Nq),
	to("to", 0, 2),
	on("on", 0, 1),
	wnum("wnum", 0, Nw2),
   t("t", 0, Lt),
	x("x", 0, Vsnk),
	x2("x2", 0, Vsnk),
   y("y", 0, Vsrc),
	m("m", 0, Nsrc),
	n("n", 0, Nsnk),
   tri("tri", 0, Nq),
   iCprime("iCprime", 0, Nc),
   iSprime("iSprime", 0, Ns),
   jCprime("jCprime", 0, Nc),
   jSprime("jSprime", 0, Ns),
   kCprime("kCprime", 0, Nc),
   kSprime("kSprime", 0, Ns);
	//x("x", 0, Vsnk),
	//x2("x2", 0, Vsnk),
	//m("m", 0, Nsrc),

   input C_r("C_r",      {m, n, t}, p_float64);
   input C_i("C_i",      {m, n, t}, p_float64);
   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_r("B2_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_i("B2_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_B1_r("src_psi_B1_r",    {m, y}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {m, y}, p_float64);
   input src_psi_B2_r("src_psi_B2_r",    {m, y}, p_float64);
   input src_psi_B2_i("src_psi_B2_i",    {m, y}, p_float64);
   input perms("perms", {nperm, q2}, p_int32);
   input sigs("sigs", {nperm}, p_int32);
   input overall_weight("overall_weight", {on}, p_float64);
   input snk_color_weights("snk_color_weights", {to, wnum, q}, p_int32);
   input snk_spin_weights("snk_spin_weights", {to, wnum, q}, p_int32);
   input snk_weights("snk_weights", {wnum}, p_float64);
   input snk_psi_re("snk_psi_re", {x, x2, n}, p_float64);
   input snk_psi_im("snk_psi_im", {x, x2, n}, p_float64);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);
    complex_computation B2_prop(&B2_prop_r, &B2_prop_i);

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, expr((double) 0));

    complex_expr src_psi_B1(src_psi_B1_r(m, y), src_psi_B1_i(m, y));

    computation B1_Bsingle_r1_r_init("B1_Bsingle_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, expr((double) 0));
    computation B1_Bsingle_r1_i_init("B1_Bsingle_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, expr((double) 0));
    computation B1_Bdouble_r1_r_init("B1_Bdouble_r1_r_init", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, expr((double) 0));
    computation B1_Bdouble_r1_i_init("B1_Bdouble_r1_i_init", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, expr((double) 0));

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
            {t, iCprime, iSprime, kCprime, kSprime, x, y},
            B1_Q_exprs_r1[jc][js]);

        complex_expr q = q_computation(t, iCprime, iSprime, kCprime, kSprime, x, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B1_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B1_Bsingle_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B1;
        complex_computation bsingle_update(
            str_fmt("B1_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2, y},
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
            {t, jCprime, jSprime, kCprime, kSprime, x, y},
            B1_O_exprs_r1[ic][is]);

        complex_expr o = o_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2) +
          o * B1_prop(0, t, iCprime, iSprime, ic, is, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2, y},
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
            {t, jCprime, jSprime, kCprime, kSprime, x, y},
            // definition
            B1_P_exprs_r1[kc][ks]);
        complex_expr p = p_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2) -
          p * B1_prop(2, t, iCprime, iSprime, kc, ks, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2, y},
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

    computation B2_Blocal_r1_r_init("B2_Blocal_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r1_i_init("B2_Blocal_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime}, expr((double) 0));

    complex_expr src_psi_B2(src_psi_B2_r(m, y), src_psi_B2_i(m, y));

    computation B2_Bsingle_r1_r_init("B2_Bsingle_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, expr((double) 0));
    computation B2_Bsingle_r1_i_init("B2_Bsingle_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2}, expr((double) 0));
    computation B2_Bdouble_r1_r_init("B2_Bdouble_r1_r_init", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, expr((double) 0));
    computation B2_Bdouble_r1_i_init("B2_Bdouble_r1_i_init", {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2}, expr((double) 0));

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

      complex_expr B2_prop_0 =  B2_prop(0, t, iCprime, iSprime, ic, is, x, y);
      complex_expr B2_prop_2 =  B2_prop(2, t, kCprime, kSprime, kc, ks, x, y);
      complex_expr B2_prop_0p = B2_prop(0, t, kCprime, kSprime, ic, is, x, y);
      complex_expr B2_prop_2p = B2_prop(2, t, iCprime, iSprime, kc, ks, x, y);
      complex_expr B2_prop_1 = B2_prop(1, t, jCprime, jSprime, jc, js, x, y);
      
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
            {t, iCprime, iSprime, kCprime, kSprime, x, y},
            B2_Q_exprs_r1[jc][js]);

        complex_expr q = q_computation(t, iCprime, iSprime, kCprime, kSprime, x, y);

        // define local block
        complex_expr blocal_update_def = 
          B2_Blocal_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("B2_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B2_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B2_Bsingle_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B2;
        complex_computation bsingle_update(
            str_fmt("B2_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2, y},
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
            {t, jCprime, jSprime, kCprime, kSprime, x, y},
            B2_O_exprs_r1[ic][is]);

        complex_expr o = o_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2) +
          o * B2_prop(0, t, iCprime, iSprime, ic, is, x2, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2, y},
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
            {t, jCprime, jSprime, kCprime, kSprime, x, y},
            // definition
            B2_P_exprs_r1[kc][ks]);
        complex_expr p = p_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

        complex_expr bdouble_update_def =
          B2_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2) -
          p * B2_prop(2, t, iCprime, iSprime, kc, ks, x2, y) * src_psi_B2;
        complex_computation bdouble_update(
            // name
            str_fmt("B2_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2, y},
            // definition
            bdouble_update_def);
        B2_Bdouble_r1_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B2_p2userEdges_r1.push_back(edge);
      }
    }

    computation *B1_Blocal_r1_real_update;
    computation *B1_Blocal_r1_imag_update;
    std::tie(B1_Blocal_r1_real_update, B1_Blocal_r1_imag_update) = B1_Blocal_r1_updates.back();
    computation *B2_Blocal_r1_real_update;
    computation *B2_Blocal_r1_imag_update;
    std::tie(B2_Blocal_r1_real_update, B2_Blocal_r1_imag_update) = B2_Blocal_r1_updates.back();
    computation *B1_Bsingle_r1_real_update;
    computation *B1_Bsingle_r1_imag_update;
    std::tie(B1_Bsingle_r1_real_update, B1_Bsingle_r1_imag_update) = B1_Bsingle_r1_updates.back();
    computation *B2_Bsingle_r1_real_update;
    computation *B2_Bsingle_r1_imag_update;
    std::tie(B2_Bsingle_r1_real_update, B2_Bsingle_r1_imag_update) = B2_Bsingle_r1_updates.back();
    computation *B1_Bdouble_r1_real_update;
    computation *B1_Bdouble_r1_imag_update;
    std::tie(B1_Bdouble_r1_real_update, B1_Bdouble_r1_imag_update) = B1_Bdouble_r1_p_updates.back();
    computation *B2_Bdouble_r1_real_update;
    computation *B2_Bdouble_r1_imag_update;
    std::tie(B2_Bdouble_r1_real_update, B2_Bdouble_r1_imag_update) = B2_Bdouble_r1_p_updates.back();

    /* Correlator */

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

    computation term_re("term_re", {t, nperm, wnum, x, x2, m}, cast(p_float64, sigs(nperm)) * overall_weight(0) * snk_weights(wnum));
    computation term_im("term_im", {t, nperm, wnum, x, x2, m}, cast(p_float64, expr((double) 0)));

    computation new_term_re_0("new_term_re_0", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Blocal_r1_real_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B1_Blocal_r1_imag_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, Vsrc-1)) ;
    computation new_term_im_0("new_term_im_0", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Blocal_r1_imag_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B1_Blocal_r1_real_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, Vsrc-1));
    //computation new_term_re_0("new_term_re_0", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0) ;
    //computation new_term_im_0("new_term_im_0", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_0.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0);
    new_term_im_0.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0);

    //computation new_term_re_1("new_term_re_1", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Blocal_r1_real_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B2_Blocal_r1_imag_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, Vsrc-1));
    //computation new_term_im_1("new_term_im_1", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Blocal_r1_imag_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B2_Blocal_r1_real_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, Vsrc-1));
    computation new_term_re_1("new_term_re_1", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0);
    computation new_term_im_1("new_term_im_1", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_1.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1);
    new_term_im_1.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1);

    //computation new_term_re_2("new_term_re_2", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Bsingle_r1_real_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B1_Bsingle_r1_imag_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2, Vsrc-1));
    //computation new_term_im_2("new_term_im_2", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Bsingle_r1_imag_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B1_Bsingle_r1_real_update)(t, iC1e, iS1e, kC1e, kS1e, x, m, jC1e, jS1e, x2, Vsrc-1));
    computation new_term_re_2("new_term_re_2", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0);
    computation new_term_im_2("new_term_im_2", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_2.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b,b) == 1 && snk_3_b(b, b) == 0);
    new_term_im_2.add_predicate(snk_1_b(b, b) == 0 && snk_2_b(b,b) == 1 && snk_3_b(b, b) == 0);

    //computation new_term_re_3("new_term_re_3", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Bsingle_r1_real_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, x, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B2_Bsingle_r1_imag_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, x, Vsrc-1));
    //computation new_term_im_3("new_term_im_3", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Bsingle_r1_imag_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, x, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B2_Bsingle_r1_real_update)(t, iC2e, iS2e, kC2e, kS2e, x2, m, jC2e, jS2e, x, Vsrc-1));
    computation new_term_re_3("new_term_re_3", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0);
    computation new_term_im_3("new_term_im_3", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_3.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b,b) == 0 && snk_3_b(b, b) == 1);
    new_term_im_3.add_predicate(snk_1_b(b, b) == 1 && snk_2_b(b,b) == 0 && snk_3_b(b, b) == 1);

    //computation new_term_re_4("new_term_re_4", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Bdouble_r1_real_update)(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B1_Bdouble_r1_imag_update)(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2, Vsrc-1));
    //computation new_term_im_4("new_term_im_4", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B1_Bdouble_r1_imag_update)(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B1_Bdouble_r1_real_update)(t, jC1e, jS1e, kC1e, kS1e, x, m, iC1e, iS1e, x2, Vsrc-1));
    computation new_term_re_4("new_term_re_4", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0);
    computation new_term_im_4("new_term_im_4", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_4.add_predicate((snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b,b) == 1) || (snk_1_b(b,b) == 1 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0));
    new_term_im_4.add_predicate((snk_1_b(b, b) == 0 && snk_2_b(b, b) == 0 && snk_3_b(b,b) == 1) || (snk_1_b(b,b) == 1 && snk_2_b(b, b) == 0 && snk_3_b(b, b) == 0));

    //computation new_term_re_5("new_term_re_5", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Bdouble_r1_real_update)(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x, Vsrc-1) - term_im(t, nperm, wnum, x, x2, m) * (*B2_Bdouble_r1_imag_update)(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x, Vsrc-1));
    //computation new_term_im_5("new_term_im_5", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * (*B2_Bdouble_r1_imag_update)(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x, Vsrc-1) + term_im(t, nperm, wnum, x, x2, m) * (*B2_Bdouble_r1_real_update)(t, jC2e, jS2e, kC2e, kS2e, x2, m, iC2e, iS2e, x, Vsrc-1));
    computation new_term_re_5("new_term_re_5", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 1.0 - term_im(t, nperm, wnum, x, x2, m) * 0.0);
    computation new_term_im_5("new_term_im_5", {t, nperm, wnum, x, x2, m, b}, term_re(t, nperm, wnum, x, x2, m) * 0.0 + term_im(t, nperm, wnum, x, x2, m) * 1.0);
    new_term_re_5.add_predicate((snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b,b) == 0) || (snk_1_b(b,b) == 0 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1));
    new_term_im_5.add_predicate((snk_1_b(b, b) == 1 && snk_2_b(b, b) == 1 && snk_3_b(b,b) == 0) || (snk_1_b(b,b) == 0 && snk_2_b(b, b) == 1 && snk_3_b(b, b) == 1));

    computation term_re_1("term_re_1", {t, nperm, wnum, x, x2, m, b}, new_term_re_5(t, nperm, wnum, x, x2, m, b));
    computation term_im_1("term_im_1", {t, nperm, wnum, x, x2, m, b}, new_term_im_5(t, nperm, wnum, x, x2, m, b));
    complex_expr term(term_re_1(t, nperm, wnum, x, x2, m, Nb-1), term_im_1(t, nperm, wnum, x, x2, m, Nb-1));
    complex_expr snk_psi(snk_psi_re(x, x2, n), snk_psi_im(x, x2, n));
    complex_expr term_res = term * snk_psi;

    computation C_update_r("C_update_r", {t, x, x2, m, n, nperm, wnum}, C_r(m, n, t) + term_res.get_real());
    computation C_update_i("C_update_i", {t, x, x2, m, n, nperm, wnum}, C_i(m, n, t) + term_res.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation *handle = &(
        B1_Blocal_r1_r_init
        .then(B1_Blocal_r1_i_init, jSprime)
        .then(B1_Bsingle_r1_r_init, jSprime)
        .then(B1_Bsingle_r1_i_init, x2)
        .then(B1_Bdouble_r1_r_init, x2)
        .then(B1_Bdouble_r1_i_init, x2)
	);

    // schedule B1_Blocal_r1 and B1_Bsingle_r1
    for (int i = 0; i < B1_q2userEdges_r1.size(); i++)
    {
      auto edge = B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, x)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, jCprime)
          .then(*edge.bs_i, y)
	  );
    }

    // schedule O update of B1_Bdouble_r1
    for (int i = 0; i < B1_o2userEdges_r1.size(); i++)
    {
      auto edge  = B1_o2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.o_r, x)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y)
	  );
    }

    // schedule P update of B1_Bdouble_r1
    for (int i = 0; i < B1_p2userEdges_r1.size(); i++)
    {
      auto edge  = B1_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.p_r, x)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B2_Blocal_r1_r_init, jSprime)
        .then(B2_Blocal_r1_i_init, jSprime)
        .then(B2_Bsingle_r1_r_init, jSprime)
        .then(B2_Bsingle_r1_i_init, x2)
        .then(B2_Bdouble_r1_r_init, x2)
        .then(B2_Bdouble_r1_i_init, x2)
	);

    // schedule B2_Blocal_r1 and B2_Bsingle_r1
    for (int i = 0; i < B2_q2userEdges_r1.size(); i++)
    {
      auto edge = B2_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, x)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, jCprime)
          .then(*edge.bs_i, y)
	  );
    }

    // schedule O update of B2_Bdouble_r1
    for (int i = 0; i < B2_o2userEdges_r1.size(); i++)
    {
      auto edge  = B2_o2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.o_r, x)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y)
	  );
    }

    // schedule P update of B2_Bdouble_r1
    for (int i = 0; i < B2_p2userEdges_r1.size(); i++)
    {
      auto edge  = B2_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.p_r, x)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y)
	  );
    }

    handle->then(term_re, t)
          .then(term_im, m)
          .then(snk_1, nperm)
          .then(snk_2, b)
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
	       .then(new_term_re_0, wnum)
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
          .then(C_update_r, wnum)
          .then(C_update_i, wnum);

#if VECTORIZED
    B1_Blocal_r1_r_init.tag_vector_level(jSprime, Ns);
    B1_Blocal_r1_i_init.tag_vector_level(jSprime, Ns);
    B1_Bsingle_r1_r_init.tag_vector_level(x2, Vsnk);
    B1_Bsingle_r1_i_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r1_r_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r1_i_init.tag_vector_level(x2, Vsnk);

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

    B2_Blocal_r1_r_init.tag_vector_level(jSprime, Ns);
    B2_Blocal_r1_i_init.tag_vector_level(jSprime, Ns);
    B2_Bsingle_r1_r_init.tag_vector_level(x2, Vsnk);
    B2_Bsingle_r1_i_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r1_r_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r1_i_init.tag_vector_level(x2, Vsnk);

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
#endif

#if PARALLEL
    B1_Blocal_r1_r_init.tag_parallel_level(t);
    B1_Blocal_r1_i_init.tag_parallel_level(t);
    B1_Bsingle_r1_r_init.tag_parallel_level(t);
    B1_Bsingle_r1_i_init.tag_parallel_level(t);
    B1_Bdouble_r1_r_init.tag_parallel_level(t);
    B1_Bdouble_r1_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }

    B2_Blocal_r1_r_init.tag_parallel_level(t);
    B2_Blocal_r1_i_init.tag_parallel_level(t);
    B2_Bsingle_r1_r_init.tag_parallel_level(t);
    B2_Bsingle_r1_i_init.tag_parallel_level(t);
    B2_Bdouble_r1_r_init.tag_parallel_level(t);
    B2_Bdouble_r1_i_init.tag_parallel_level(t);

    for (auto edge : B2_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
    }
    for (auto edge : B2_o2userEdges_r1) {
      edge.o_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
    for (auto edge : B2_p2userEdges_r1) {
      edge.p_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_r("buf_B1_Bsingle_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_i("buf_B1_Bsingle_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_r("buf_B1_Bdouble_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_i("buf_B1_Bdouble_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);

    buffer buf_B2_Blocal_r1_r("buf_B2_Blocal_r1_r",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r1_i("buf_B2_Blocal_r1_i",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_r("buf_B2_Bsingle_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_i("buf_B2_Bsingle_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_r("buf_B2_Bdouble_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_i("buf_B2_Bdouble_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_temporary);

    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r);
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i);

    B1_Bsingle_r1_r_init.store_in(&buf_B1_Bsingle_r1_r);
    B1_Bsingle_r1_i_init.store_in(&buf_B1_Bsingle_r1_i);

    B1_Bdouble_r1_r_init.store_in(&buf_B1_Bdouble_r1_r);
    B1_Bdouble_r1_i_init.store_in(&buf_B1_Bdouble_r1_i);

    buffer *B1_q_r1_r_buf;
    buffer *B1_q_r1_i_buf;
    buffer *B1_o_r1_r_buf;
    buffer *B1_o_r1_i_buf;
    buffer *B1_p_r1_r_buf;
    buffer *B1_p_r1_i_buf;

    allocate_complex_buffers(B1_q_r1_r_buf, B1_q_r1_i_buf, { Lt, Vsrc }, "buf_B1_q_r1");
    allocate_complex_buffers(B1_o_r1_r_buf, B1_o_r1_i_buf, { Lt, Vsrc }, "buf_B1_o_r1");
    allocate_complex_buffers(B1_p_r1_r_buf, B1_p_r1_i_buf, { Lt, Vsrc }, "buf_B1_p_r1");

    B2_Blocal_r1_r_init.store_in(&buf_B2_Blocal_r1_r);
    B2_Blocal_r1_i_init.store_in(&buf_B2_Blocal_r1_i);

    B2_Bsingle_r1_r_init.store_in(&buf_B2_Bsingle_r1_r);
    B2_Bsingle_r1_i_init.store_in(&buf_B2_Bsingle_r1_i);

    B2_Bdouble_r1_r_init.store_in(&buf_B2_Bdouble_r1_r);
    B2_Bdouble_r1_i_init.store_in(&buf_B2_Bdouble_r1_i);

    buffer *B2_q_r1_r_buf;
    buffer *B2_q_r1_i_buf;
    buffer *B2_o_r1_r_buf;
    buffer *B2_o_r1_i_buf;
    buffer *B2_p_r1_r_buf;
    buffer *B2_p_r1_i_buf;

    allocate_complex_buffers(B2_q_r1_r_buf, B2_q_r1_i_buf, { Lt, Vsrc }, "buf_B2_q_r1");
    allocate_complex_buffers(B2_o_r1_r_buf, B2_o_r1_i_buf, { Lt, Vsrc }, "buf_B2_o_r1");
    allocate_complex_buffers(B2_p_r1_r_buf, B2_p_r1_i_buf, { Lt, Vsrc }, "buf_B2_p_r1");

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->store_in(B1_q_r1_r_buf, {t, y});
      edge.q_i->store_in(B1_q_r1_i_buf, {t, y});
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->store_in(B1_o_r1_r_buf, {t, y});
      edge.o_i->store_in(B1_o_r1_i_buf, {t, y});
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->store_in(B1_p_r1_r_buf, {t, y});
      edge.p_i->store_in(B1_p_r1_i_buf, {t, y});
    }

    for (auto computations: B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2});
      imag->store_in(&buf_B1_Bsingle_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2});
    }
    for (auto computations : B1_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
      imag->store_in(&buf_B1_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
    }
    for (auto computations : B1_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
      imag->store_in(&buf_B1_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
    }

    for (auto edge : B2_q2userEdges_r1) {
      edge.q_r->store_in(B2_q_r1_r_buf, {t, y});
      edge.q_i->store_in(B2_q_r1_i_buf, {t, y});
    }
    for (auto edge : B2_o2userEdges_r1) {
      edge.o_r->store_in(B2_o_r1_r_buf, {t, y});
      edge.o_i->store_in(B2_o_r1_i_buf, {t, y});
    }
    for (auto edge : B2_p2userEdges_r1) {
      edge.p_r->store_in(B2_p_r1_r_buf, {t, y});
      edge.p_i->store_in(B2_p_r1_i_buf, {t, y});
    }

    for (auto computations: B2_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Blocal_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime});
      imag->store_in(&buf_B2_Blocal_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime});
    }
    for (auto computations: B2_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bsingle_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2});
      imag->store_in(&buf_B2_Bsingle_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, m, jCprime, jSprime, x2});
    }
    for (auto computations : B2_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
      imag->store_in(&buf_B2_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
    }
    for (auto computations : B2_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
      imag->store_in(&buf_B2_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, m, iCprime, iSprime, x2});
    }

    buffer buf_snk_1("buf_snk_1", {Nb}, p_int32, a_temporary);
    buffer buf_snk_2("buf_snk_2", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3("buf_snk_3", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_b("buf_snk_1_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_2_b("buf_snk_2_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_3_b("buf_snk_3_b", {Nb}, p_int32, a_temporary);
    buffer buf_snk_1_nq("buf_snk_1_nq", {Nb}, p_int32, a_temporary);
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

    buffer buf_C_r("buf_C_r", {Nsrc, Nsnk, Lt}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Nsrc, Nsnk, Lt}, p_float64, a_input);

    buffer buf_snk_psi_re("buf_snk_psi_re", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);
    buffer buf_snk_psi_im("buf_snk_psi_im", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);

    buffer buf_new_term_r("buf_new_term_r", {1}, p_float64, a_temporary);
    buffer buf_new_term_i("buf_new_term_i", {1}, p_float64, a_temporary);

    buffer buf_term_r("buf_term_r", {1}, p_float64, a_temporary);
    buffer buf_term_i("buf_term_i", {1}, p_float64, a_temporary);

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
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        B2_prop_r.get_buffer(), B2_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        src_psi_B2_r.get_buffer(), src_psi_B2_i.get_buffer(),
	     perms.get_buffer(), sigs.get_buffer(),
	     overall_weight.get_buffer(),
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer(),
	     &buf_snk_psi_re,
	     &buf_snk_psi_im
        }, 
        "generated_tiramisu_make_fused_dibaryon_blocks_correlator.o");
}

int main(int argc, char **argv)
{
	generate_function("tiramisu_make_fused_dibaryon_blocks_correlator");

    return 0;
}
