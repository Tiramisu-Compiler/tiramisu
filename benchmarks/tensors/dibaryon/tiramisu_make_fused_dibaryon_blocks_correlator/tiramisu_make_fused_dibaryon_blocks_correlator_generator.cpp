#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
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

   int Nr = 6;
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
   tri("tri", 0, Nq),
   iCprime("iCprime", 0, Nc),
   iSprime("iSprime", 0, Ns),
   jCprime("jCprime", 0, Nc),
   jSprime("jSprime", 0, Ns),
   kCprime("kCprime", 0, Nc),
   kSprime("kSprime", 0, Ns);

   input C_r("C_r",      {r, m, n, t}, p_float64);
   input C_i("C_i",      {r, m, n, t}, p_float64);
   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_r("B2_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B2_prop_i("B2_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_B1_r("src_psi_B1_r",    {y, m}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {y, m}, p_float64);
   input src_psi_B2_r("src_psi_B2_r",    {y, m}, p_float64);
   input src_psi_B2_i("src_psi_B2_i",    {y, m}, p_float64);
   input snk_blocks("snk_blocks", {r, to}, p_int32);
   input sigs("sigs", {nperm}, p_int32);
   input snk_b("snk_b", {nperm, q, to}, p_int32);
   input snk_color_weights("snk_color_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, nperm, wnum, q, to}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);
   input snk_psi_re("snk_psi_re", {x, x2, n}, p_float64);
   input snk_psi_im("snk_psi_im", {x, x2, n}, p_float64);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);
    complex_computation B2_prop(&B2_prop_r, &B2_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));
    complex_expr src_psi_B2(src_psi_B2_r(y, m), src_psi_B2_i(y, m));

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    //B1_Blocal_r1_r_init.add_predicate(x2==0);
    //B1_Blocal_r1_i_init.add_predicate(x2==0);
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
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, y},
            B1_Q_exprs_r1[jc][js]);
        //q_computation.add_predicate((x2==0)&&(m==0));

        complex_expr q = q_computation(t, x, 0, 0, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r1_init(t, x, 0, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        //blocal_update.add_predicate(x2==0);
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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            B1_O_exprs_r1[ic][is]);
        //o_computation.add_predicate((x2==0)&&(m==0));

        complex_expr o = o_computation(t, x, 0, 0, jCprime, jSprime, kCprime, kSprime, y);

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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B1_P_exprs_r1[kc][ks]);
        //p_computation.add_predicate((x2==0)&&(m==0));

        complex_expr p = p_computation(t, x, 0, 0, jCprime, jSprime, kCprime, kSprime, y);

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

    computation B2_Blocal_r1_r_init("B2_Blocal_r1_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r1_i_init("B2_Blocal_r1_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    //B2_Blocal_r1_r_init.add_predicate(x==0);
    //B2_Blocal_r1_i_init.add_predicate(x==0);
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
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, y},
            B2_Q_exprs_r1[jc][js]);
        //q_computation.add_predicate(m==0);
        //q_computation.add_predicate((x==0)&&(m==0));

        complex_expr q = q_computation(t, x, x2, 0, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B2_Blocal_r1_init(t, 0, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("B2_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        //blocal_update.add_predicate(x==0);
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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            B2_O_exprs_r1[ic][is]);
        //o_computation.add_predicate(m==0);
        //o_computation.add_predicate((x==0)&&(m==0));

        complex_expr o = o_computation(t, x, x2, 0, jCprime, jSprime, kCprime, kSprime, y);

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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B2_P_exprs_r1[kc][ks]);
        //p_computation.add_predicate(m==0);
        //p_computation.add_predicate((x==0)&&(m==0));

        complex_expr p = p_computation(t, 0, x2, 0, jCprime, jSprime, kCprime, kSprime, y);

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

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    //B1_Blocal_r2_r_init.add_predicate(x2==0);
    //B1_Blocal_r2_i_init.add_predicate(x2==0);
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
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, y},
            B1_Q_exprs_r2[jc][js]);
        //q_computation.add_predicate((x2==0)&&(m==0));

        complex_expr q = q_computation(t, x, 0, 0, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r2_init(t, x, 0, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        //blocal_update.add_predicate(x2==0);
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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            B1_O_exprs_r2[ic][is]);
        //o_computation.add_predicate((x2==0)&&(m==0));

        complex_expr o = o_computation(t, x, 0, 0, jCprime, jSprime, kCprime, kSprime, y);

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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B1_P_exprs_r2[kc][ks]);
        //p_computation.add_predicate((x2==0)&&(m==0));

        complex_expr p = p_computation(t, x, 0, 0, jCprime, jSprime, kCprime, kSprime, y);

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

    computation B2_Blocal_r2_r_init("B2_Blocal_r2_r_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B2_Blocal_r2_i_init("B2_Blocal_r2_i_init", {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    //B2_Blocal_r2_r_init.add_predicate(x==0);
    //B2_Blocal_r2_i_init.add_predicate(x==0);
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
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, y},
            B2_Q_exprs_r2[jc][js]);
        //q_computation.add_predicate(m==0);
        //q_computation.add_predicate((x==0)&&(m==0));

        complex_expr q = q_computation(t, x, x2, 0, iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B2_Blocal_r2_init(t, 0, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime) +
          q * B2_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B2;
        complex_computation blocal_update(
            // name
            str_fmt("B2_blocal_update_r2_%d_%d", jc, js),
            // iterator
            {t, x, x2, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y},
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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            B2_O_exprs_r2[ic][is]);
        //o_computation.add_predicate(m==0);
        //o_computation.add_predicate((x==0)&&(m==0));

        complex_expr o = o_computation(t, x, x2, 0, jCprime, jSprime, kCprime, kSprime, y);

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
            {t, x, x2, m, jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B2_P_exprs_r2[kc][ks]);
        //p_computation.add_predicate(m==0);
        //p_computation.add_predicate((x==0)&&(m==0));

        complex_expr p = p_computation(t, x, x2, 0, jCprime, jSprime, kCprime, kSprime, y);

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

    /* Correlator */

    computation C_init_r("C_init_r", {t, m, r, n}, expr((double) 0));
    computation C_init_i("C_init_i", {t, m, r, n}, expr((double) 0));
    
    int b=0;
    /* r1, b = 0 */
    complex_computation new_term_0_r1_b1("new_term_0_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x, 0, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_0_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_1_r1_b1("new_term_1_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r1_init(t, 0, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_1_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_2_r1_b1("new_term_2_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_2_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_3_r1_b1("new_term_3_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r1_init(t, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), x));
    new_term_3_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_4_r1_b1("new_term_4_r1_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    new_term_4_r1_b1.add_predicate((snk_blocks(r, 0) == 1) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation new_term_5_r1_b1("new_term_5_r1_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    new_term_5_r1_b1.add_predicate((snk_blocks(r, 1) == 1) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    /* r2, b = 0 */
    complex_computation new_term_0_r2_b1("new_term_0_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x, 0, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_0_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_1_r2_b1("new_term_1_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r2_init(t, 0, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_1_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_2_r2_b1("new_term_2_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_2_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_3_r2_b1("new_term_3_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_3_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_4_r2_b1("new_term_4_r2_b1", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    new_term_4_r2_b1.add_predicate((snk_blocks(r, 0) == 2) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation new_term_5_r2_b1("new_term_5_r2_b1", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    new_term_5_r2_b1.add_predicate((snk_blocks(r, 1) == 2) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    b=1;
    /* r1, b = 1 */
    complex_computation new_term_0_r1_b2("new_term_0_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x, 0, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_0_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_1_r1_b2("new_term_1_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r1_init(t, 0, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_1_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_2_r1_b2("new_term_2_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_2_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_3_r1_b2("new_term_3_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_3_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_4_r1_b2("new_term_4_r1_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    new_term_4_r1_b2.add_predicate((snk_blocks(r, 0) == 1) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation new_term_5_r1_b2("new_term_5_r1_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r1_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    new_term_5_r1_b2.add_predicate((snk_blocks(r, 1) == 1) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));
    /* r2, b = 1 */
    complex_computation new_term_0_r2_b2("new_term_0_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x, 0, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_0_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_1_r2_b2("new_term_1_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Blocal_r2_init(t, 0, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_1_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_2_r2_b2("new_term_2_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0)));
    new_term_2_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0));
    complex_computation new_term_3_r2_b2("new_term_3_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bsingle_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1)));
    new_term_3_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1));
    complex_computation new_term_4_r2_b2("new_term_4_r2_b2", {t, x, x2, m, r, nperm, wnum}, B1_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 0), snk_spin_weights(r, nperm, wnum, 1, 0), snk_color_weights(r, nperm, wnum, 2, 0), snk_spin_weights(r, nperm, wnum, 2, 0), snk_color_weights(r, nperm, wnum, 0, 0), snk_spin_weights(r, nperm, wnum, 0, 0)));
    new_term_4_r2_b2.add_predicate((snk_blocks(r, 0) == 2) && ((snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 0) || (snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 0 && snk_b(nperm, 2, b) == 1)));
    complex_computation new_term_5_r2_b2("new_term_5_r2_b2", {t, x, x2, m, r, nperm, wnum}, B2_Bdouble_r2_init(t, x, x2, m, snk_color_weights(r, nperm, wnum, 1, 1), snk_spin_weights(r, nperm, wnum, 1, 1), snk_color_weights(r, nperm, wnum, 2, 1), snk_spin_weights(r, nperm, wnum, 2, 1), snk_color_weights(r, nperm, wnum, 0, 1), snk_spin_weights(r, nperm, wnum, 0, 1)));
    new_term_5_r2_b2.add_predicate((snk_blocks(r, 1) == 2) && ((snk_b(nperm, 0, b) == 0 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 1) || (snk_b(nperm, 0, b) == 1 && snk_b(nperm, 1, b) == 1 && snk_b(nperm, 2, b) == 0)));


    complex_expr prefactor(cast(p_float64, sigs(nperm)) * snk_weights(r, wnum), 0.0);

    complex_expr term_res_b1 = new_term_0_r1_b1(t, x, x2, m, r, nperm, wnum);
    complex_expr term_res_b2 = new_term_0_r1_b2(t, x, x2, m, r, nperm, wnum);

    complex_expr snk_psi(snk_psi_re(x, x2, n), snk_psi_im(x, x2, n));

    complex_expr term_res = prefactor * term_res_b1 * term_res_b2 * snk_psi;

    computation C_update_r("C_update_r", {t, x, x2, m, r, nperm, wnum, n}, C_init_r(t, m, r, n) + term_res.get_real());
    computation C_update_i("C_update_i", {t, x, x2, m, r, nperm, wnum, n}, C_init_i(t, m, r, n) + term_res.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, n)
    );

    handle = &(handle
        ->then(B1_Blocal_r1_r_init, t)
        .then(B1_Blocal_r1_i_init, jSprime)
        .then(B1_Bsingle_r1_r_init, m)
        .then(B1_Bsingle_r1_i_init, jSprime)
        .then(B1_Bdouble_r1_r_init, m)
        .then(B1_Bdouble_r1_i_init, iSprime)
	);
    // schedule B1_Blocal_r1 and B1_Bsingle_r1
    for (int i = 0; i < B1_q2userEdges_r1.size(); i++)
    {
      auto edge = B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, m)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, kSprime)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, kSprime)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r1
    for (int i = 0; i < B1_o2userEdges_r1.size(); i++)
    {
      auto edge  = B1_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.o_r, m)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r1
    for (int i = 0; i < B1_p2userEdges_r1.size(); i++)
    {
      auto edge  = B1_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }
    
    handle = &(handle
        ->then(B2_Blocal_r1_r_init, m)
        .then(B2_Blocal_r1_i_init, jSprime)
        .then(B2_Bsingle_r1_r_init, m)
        .then(B2_Bsingle_r1_i_init, jSprime)
        .then(B2_Bdouble_r1_r_init, m)
        .then(B2_Bdouble_r1_i_init, iSprime)
	);
    // schedule B2_Blocal_r1 and B2_Bsingle_r1
    for (int i = 0; i < B2_q2userEdges_r1.size(); i++)
    {
      auto edge = B2_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, m)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, kSprime)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, kSprime)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r1
    for (int i = 0; i < B2_o2userEdges_r1.size(); i++)
    {
      auto edge  = B2_o2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.o_r, m)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r1
    for (int i = 0; i < B2_p2userEdges_r1.size(); i++)
    {
      auto edge  = B2_p2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B1_Blocal_r2_r_init, m)
        .then(B1_Blocal_r2_i_init, jSprime)
        .then(B1_Bsingle_r2_r_init, m)
        .then(B1_Bsingle_r2_i_init, jSprime)
        .then(B1_Bdouble_r2_r_init, m)
        .then(B1_Bdouble_r2_i_init, iSprime)
	);
    // schedule B1_Blocal_r2 and B1_Bsingle_r2
    for (int i = 0; i < B1_q2userEdges_r2.size(); i++)
    {
      auto edge = B1_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, m)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, kSprime)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, kSprime)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B1_Bdouble_r2
    for (int i = 0; i < B1_o2userEdges_r2.size(); i++)
    {
      auto edge  = B1_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.o_r, m)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B1_Bdouble_r2
    for (int i = 0; i < B1_p2userEdges_r2.size(); i++)
    {
      auto edge  = B1_p2userEdges_r2[i];

      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }

    handle = &(handle
        ->then(B2_Blocal_r2_r_init, m)
        .then(B2_Blocal_r2_i_init, jSprime)
        .then(B2_Bsingle_r2_r_init, m)
        .then(B2_Bsingle_r2_i_init, jSprime)
        .then(B2_Bdouble_r2_r_init, m)
        .then(B2_Bdouble_r2_i_init, iSprime)
	);
    // schedule B2_Blocal_r2 and B2_Bsingle_r2
    for (int i = 0; i < B2_q2userEdges_r2.size(); i++)
    {
      auto edge = B2_q2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.q_r, m)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, kSprime)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, kSprime)
          .then(*edge.bs_i, y)
	  );
    }
    // schedule O update of B2_Bdouble_r2
    for (int i = 0; i < B2_o2userEdges_r2.size(); i++)
    {
      auto edge  = B2_o2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.o_r, m)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }
    // schedule P update of B2_Bdouble_r2
    for (int i = 0; i < B2_p2userEdges_r2.size(); i++)
    {
      auto edge  = B2_p2userEdges_r2[i];
      handle = &(handle
          ->then(*edge.p_r, kSprime)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, kSprime)
          .then(*edge.bd_i, y)
	  );
    }

    handle->then( *(new_term_0_r1_b1.get_real()), m)
          .then( *(new_term_0_r1_b1.get_imag()), wnum)
          .then( *(new_term_1_r1_b1.get_real()), wnum)
          .then( *(new_term_1_r1_b1.get_imag()), wnum)
          .then( *(new_term_2_r1_b1.get_real()), wnum)
          .then( *(new_term_2_r1_b1.get_imag()), wnum)
          .then( *(new_term_3_r1_b1.get_real()), wnum)
          .then( *(new_term_3_r1_b1.get_imag()), wnum)
          .then( *(new_term_4_r1_b1.get_real()), wnum)
          .then( *(new_term_4_r1_b1.get_imag()), wnum)
          .then( *(new_term_5_r1_b1.get_real()), wnum)
          .then( *(new_term_5_r1_b1.get_imag()), wnum)
          .then( *(new_term_0_r2_b1.get_real()), wnum)
          .then( *(new_term_0_r2_b1.get_imag()), wnum)
          .then( *(new_term_1_r2_b1.get_real()), wnum)
          .then( *(new_term_1_r2_b1.get_imag()), wnum)
          .then( *(new_term_2_r2_b1.get_real()), wnum)
          .then( *(new_term_2_r2_b1.get_imag()), wnum)
          .then( *(new_term_3_r2_b1.get_real()), wnum)
          .then( *(new_term_3_r2_b1.get_imag()), wnum)
          .then( *(new_term_4_r2_b1.get_real()), wnum)
          .then( *(new_term_4_r2_b1.get_imag()), wnum)
          .then( *(new_term_5_r2_b1.get_real()), wnum)
          .then( *(new_term_5_r2_b1.get_imag()), wnum)
          .then( *(new_term_0_r1_b2.get_real()), wnum)
          .then( *(new_term_0_r1_b2.get_imag()), wnum)
          .then( *(new_term_1_r1_b2.get_real()), wnum)
          .then( *(new_term_1_r1_b2.get_imag()), wnum) 
          .then( *(new_term_2_r1_b2.get_real()), wnum)
          .then( *(new_term_2_r1_b2.get_imag()), wnum)
          .then( *(new_term_3_r1_b2.get_real()), wnum)
          .then( *(new_term_3_r1_b2.get_imag()), wnum)
          .then( *(new_term_4_r1_b2.get_real()), wnum)
          .then( *(new_term_4_r1_b2.get_imag()), wnum)
          .then( *(new_term_5_r1_b2.get_real()), wnum)
          .then( *(new_term_5_r1_b2.get_imag()), wnum)
          .then( *(new_term_0_r2_b2.get_real()), wnum)
          .then( *(new_term_0_r2_b2.get_imag()), wnum)
          .then( *(new_term_1_r2_b2.get_real()), wnum)
          .then( *(new_term_1_r2_b2.get_imag()), wnum) 
          .then( *(new_term_2_r2_b2.get_real()), wnum)
          .then( *(new_term_2_r2_b2.get_imag()), wnum)
          .then( *(new_term_3_r2_b2.get_real()), wnum)
          .then( *(new_term_3_r2_b2.get_imag()), wnum)
          .then( *(new_term_4_r2_b2.get_real()), wnum)
          .then( *(new_term_4_r2_b2.get_imag()), wnum)
          .then( *(new_term_5_r2_b2.get_real()), wnum)
          .then( *(new_term_5_r2_b2.get_imag()), wnum)
          .then(C_update_r, wnum) 
          .then(C_update_i, n);

#if VECTORIZED
/*    B1_Blocal_r1_r_init.tag_vector_level(jSprime, Ns);
    B1_Blocal_r1_i_init.tag_vector_level(jSprime, Ns);
    B1_Bsingle_r1_r_init.tag_vector_level(x2, Vsnk);
    B1_Bsingle_r1_i_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r1_r_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r1_i_init.tag_vector_level(x2, Vsnk); */

    /*

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
    */

    /*
    B2_Blocal_r1_r_init.tag_vector_level(jSprime, Ns);
    B2_Blocal_r1_i_init.tag_vector_level(jSprime, Ns);
    B2_Bsingle_r1_r_init.tag_vector_level(x2, Vsnk);
    B2_Bsingle_r1_i_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r1_r_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r1_i_init.tag_vector_level(x2, Vsnk); */

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

/*    B1_Blocal_r2_r_init.tag_vector_level(jSprime, Ns);
    B1_Blocal_r2_i_init.tag_vector_level(jSprime, Ns);
    B1_Bsingle_r2_r_init.tag_vector_level(x2, Vsnk);
    B1_Bsingle_r2_i_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r2_r_init.tag_vector_level(x2, Vsnk);
    B1_Bdouble_r2_i_init.tag_vector_level(x2, Vsnk); */

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

/*    B2_Blocal_r2_r_init.tag_vector_level(jSprime, Ns);
    B2_Blocal_r2_i_init.tag_vector_level(jSprime, Ns);
    B2_Bsingle_r2_r_init.tag_vector_level(x2, Vsnk);
    B2_Bsingle_r2_i_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r2_r_init.tag_vector_level(x2, Vsnk);
    B2_Bdouble_r2_i_init.tag_vector_level(x2, Vsnk); */

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

/*    (new_term_0_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_0_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_1_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_1_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_2_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_2_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_3_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_3_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_4_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_4_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_5_r1_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_5_r1_b1.get_imag())->tag_vector_level(wnum, Nw2);

    (new_term_0_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_0_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_1_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_1_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_2_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_2_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_3_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_3_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_4_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_4_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_5_r2_b1.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_5_r2_b1.get_imag())->tag_vector_level(wnum, Nw2);

    (new_term_0_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_0_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_1_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_1_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_2_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_2_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_3_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_3_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_4_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_4_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_5_r1_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_5_r1_b2.get_imag())->tag_vector_level(wnum, Nw2);

    (new_term_0_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_0_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_1_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_1_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_2_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_2_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_3_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_3_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_4_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_4_r2_b2.get_imag())->tag_vector_level(wnum, Nw2);
    (new_term_5_r2_b2.get_real())->tag_vector_level(wnum, Nw2);
    (new_term_5_r2_b2.get_imag())->tag_vector_level(wnum, Nw2); */

    C_update_r.tag_vector_level(n, Nsnk);
    //C_update_i.tag_vector_level(n, Nsnk);  
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

    B1_Blocal_r2_r_init.tag_parallel_level(t);
    B1_Blocal_r2_i_init.tag_parallel_level(t);
    B1_Bsingle_r2_r_init.tag_parallel_level(t);
    B1_Bsingle_r2_i_init.tag_parallel_level(t);
    B1_Bdouble_r2_r_init.tag_parallel_level(t);
    B1_Bdouble_r2_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
    }
    for (auto edge : B1_o2userEdges_r2) {
      edge.o_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
    for (auto edge : B1_p2userEdges_r2) {
      edge.p_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }

    B2_Blocal_r2_r_init.tag_parallel_level(t);
    B2_Blocal_r2_i_init.tag_parallel_level(t);
    B2_Bsingle_r2_r_init.tag_parallel_level(t);
    B2_Bsingle_r2_i_init.tag_parallel_level(t);
    B2_Bdouble_r2_r_init.tag_parallel_level(t);
    B2_Bdouble_r2_i_init.tag_parallel_level(t);

    for (auto edge : B2_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
    }
    for (auto edge : B2_o2userEdges_r2) {
      edge.o_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
    for (auto edge : B2_p2userEdges_r2) {
      edge.p_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_r("buf_B1_Bsingle_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r1_i("buf_B1_Bsingle_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_r("buf_B1_Bdouble_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r1_i("buf_B1_Bdouble_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r1_r_init.store_in(&buf_B1_Bsingle_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r1_i_init.store_in(&buf_B1_Bsingle_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bdouble_r1_r_init.store_in(&buf_B1_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B1_Bdouble_r1_i_init.store_in(&buf_B1_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

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
        allocate_complex_buffers(B1_q_r1_r_buf, B1_q_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_q_r1_%d_%d", ic, is));
        (B1_q2userEdges_r1[B1_r1_q_index]).q_r->store_in(B1_q_r1_r_buf, {iCprime, iSprime, kCprime, kSprime, y});
        (B1_q2userEdges_r1[B1_r1_q_index]).q_i->store_in(B1_q_r1_i_buf, {iCprime, iSprime, kCprime, kSprime, y});
        B1_r1_q_index++;
        }
    int B1_r1_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_o_r1_r_buf, B1_o_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_o_r1_%d_%d", ic, is));
        (B1_o2userEdges_r1[B1_r1_o_index]).o_r->store_in(B1_o_r1_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B1_o2userEdges_r1[B1_r1_o_index]).o_i->store_in(B1_o_r1_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B1_r1_o_index++;
        }
    int B1_r1_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r1[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B1_p_r1_r_buf, B1_p_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_p_r1_%d_%d", kc, ks));
        (B1_p2userEdges_r1[B1_r1_p_index]).p_r->store_in(B1_p_r1_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B1_p2userEdges_r1[B1_r1_p_index]).p_i->store_in(B1_p_r1_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B1_r1_p_index++;
        }
    for (auto computations: B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Bsingle_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B1_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B1_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    

    buffer buf_B2_Blocal_r1_r("buf_B2_Blocal_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r1_i("buf_B2_Blocal_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_r("buf_B2_Bsingle_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r1_i("buf_B2_Bsingle_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_r("buf_B2_Bdouble_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r1_i("buf_B2_Bdouble_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B2_Blocal_r1_r_init.store_in(&buf_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Blocal_r1_i_init.store_in(&buf_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bsingle_r1_r_init.store_in(&buf_B2_Bsingle_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Bsingle_r1_i_init.store_in(&buf_B2_Bsingle_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bdouble_r1_r_init.store_in(&buf_B2_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B2_Bdouble_r1_i_init.store_in(&buf_B2_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

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
        allocate_complex_buffers(B2_q_r1_r_buf, B2_q_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_q_r1_%d_%d", ic, is));
        (B2_q2userEdges_r1[B2_r1_q_index]).q_r->store_in(B2_q_r1_r_buf, {iCprime, iSprime, kCprime, kSprime, y});
        (B2_q2userEdges_r1[B2_r1_q_index]).q_i->store_in(B2_q_r1_i_buf, {iCprime, iSprime, kCprime, kSprime, y});
        B2_r1_q_index++;
        }
    int B2_r1_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r1[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_o_r1_r_buf, B2_o_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_o_r1_%d_%d", ic, is));
        (B2_o2userEdges_r1[B2_r1_o_index]).o_r->store_in(B2_o_r1_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B2_o2userEdges_r1[B2_r1_o_index]).o_i->store_in(B2_o_r1_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B2_r1_o_index++;
        }
    int B2_r1_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r1[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B2_p_r1_r_buf, B2_p_r1_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_p_r1_%d_%d", kc, ks));
        (B2_p2userEdges_r1[B2_r1_p_index]).p_r->store_in(B2_p_r1_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B2_p2userEdges_r1[B2_r1_p_index]).p_i->store_in(B2_p_r1_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B2_r1_p_index++;
        }

    for (auto computations: B2_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B2_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bsingle_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Bsingle_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B2_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B2_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r2_r("buf_B1_Bsingle_r2_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bsingle_r2_i("buf_B1_Bsingle_r2_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r2_r("buf_B1_Bdouble_r2_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Bdouble_r2_i("buf_B1_Bdouble_r2_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r2_r_init.store_in(&buf_B1_Bsingle_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bsingle_r2_i_init.store_in(&buf_B1_Bsingle_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Bdouble_r2_r_init.store_in(&buf_B1_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B1_Bdouble_r2_i_init.store_in(&buf_B1_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

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
        allocate_complex_buffers(B1_q_r2_r_buf, B1_q_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_q_r2_%d_%d", ic, is));
        (B1_q2userEdges_r2[B1_r2_q_index]).q_r->store_in(B1_q_r2_r_buf, {iCprime, iSprime, kCprime, kSprime, y});
        (B1_q2userEdges_r2[B1_r2_q_index]).q_i->store_in(B1_q_r2_i_buf, {iCprime, iSprime, kCprime, kSprime, y});
        B1_r2_q_index++;
        }
    int B1_r2_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B1_O_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B1_o_r2_r_buf, B1_o_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_o_r2_%d_%d", ic, is));
        (B1_o2userEdges_r2[B1_r2_o_index]).o_r->store_in(B1_o_r2_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B1_o2userEdges_r2[B1_r2_o_index]).o_i->store_in(B1_o_r2_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B1_r2_o_index++;
        }
    int B1_r2_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B1_P_exprs_r2[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B1_p_r2_r_buf, B1_p_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B1_p_r2_%d_%d", kc, ks));
        (B1_p2userEdges_r2[B1_r2_p_index]).p_r->store_in(B1_p_r2_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B1_p2userEdges_r2[B1_r2_p_index]).p_i->store_in(B1_p_r2_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B1_r2_p_index++;
        }
    for (auto computations: B1_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Bsingle_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B1_Bdouble_r2_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B1_Bdouble_r2_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B1_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    buffer buf_B2_Blocal_r2_r("buf_B2_Blocal_r2_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Blocal_r2_i("buf_B2_Blocal_r2_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r2_r("buf_B2_Bsingle_r2_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bsingle_r2_i("buf_B2_Bsingle_r2_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r2_r("buf_B2_Bdouble_r2_r", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B2_Bdouble_r2_i("buf_B2_Bdouble_r2_i", {Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B2_Blocal_r2_r_init.store_in(&buf_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Blocal_r2_i_init.store_in(&buf_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bsingle_r2_r_init.store_in(&buf_B2_Bsingle_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B2_Bsingle_r2_i_init.store_in(&buf_B2_Bsingle_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    B2_Bdouble_r2_r_init.store_in(&buf_B2_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    B2_Bdouble_r2_i_init.store_in(&buf_B2_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});

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
        allocate_complex_buffers(B2_q_r2_r_buf, B2_q_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_q_r2_%d_%d", ic, is));
        (B2_q2userEdges_r2[B2_r2_q_index]).q_r->store_in(B2_q_r2_r_buf, {iCprime, iSprime, kCprime, kSprime, y});
        (B2_q2userEdges_r2[B2_r2_q_index]).q_i->store_in(B2_q_r2_i_buf, {iCprime, iSprime, kCprime, kSprime, y});
        B2_r2_q_index++;
        }
    int B2_r2_o_index=0;
    for (int ic = 0; ic < Nc; ic++)
      for (int is = 0; is < Ns; is++) {
        if (B2_O_exprs_r2[ic][is].is_zero()) 
          continue;
        allocate_complex_buffers(B2_o_r2_r_buf, B2_o_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_o_r2_%d_%d", ic, is));
        (B2_o2userEdges_r2[B2_r2_o_index]).o_r->store_in(B2_o_r2_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B2_o2userEdges_r2[B2_r2_o_index]).o_i->store_in(B2_o_r2_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B2_r2_o_index++;
        }
    int B2_r2_p_index=0;
    for (int kc = 0; kc < Nc; kc++)
      for (int ks = 0; ks < Ns; ks++) {
        if (B2_P_exprs_r2[kc][ks].is_zero())
          continue;
        allocate_complex_buffers(B2_p_r2_r_buf, B2_p_r2_i_buf, {Nc, Ns, Nc, Ns, Vsrc}, str_fmt("buf_B2_p_r2_%d_%d", kc, ks));
        (B2_p2userEdges_r2[B2_r2_p_index]).p_r->store_in(B2_p_r2_r_buf, {jCprime, jSprime, kCprime, kSprime, y});
        (B2_p2userEdges_r2[B2_r2_p_index]).p_i->store_in(B2_p_r2_i_buf, {jCprime, jSprime, kCprime, kSprime, y});
        B2_r2_p_index++;
        }

    for (auto computations: B2_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Blocal_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Blocal_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations: B2_Bsingle_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bsingle_r2_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B2_Bsingle_r2_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }
    for (auto computations : B2_Bdouble_r2_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }
    for (auto computations : B2_Bdouble_r2_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B2_Bdouble_r2_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
      imag->store_in(&buf_B2_Bdouble_r2_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime});
    }

    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, Nsrc, Nr, Nsnk}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, Nsrc, Nr, Nsnk}, p_float64, a_input);

    buffer buf_snk_psi_re("buf_snk_psi_re", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);
    buffer buf_snk_psi_im("buf_snk_psi_im", {Vsnk, Vsnk, Nsnk}, p_float64, a_input);

    buffer* buf_new_term_r_b1;//("buf_new_term_r_b1", {1}, p_float64, a_temporary);
    buffer* buf_new_term_i_b1;//("buf_new_term_i_b1", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_new_term_r_b1, buf_new_term_i_b1, {1}, "buf_new_term_b1");
    buffer* buf_new_term_r_b2;//("buf_new_term_r_b2", {1}, p_float64, a_temporary);
    buffer* buf_new_term_i_b2;//("buf_new_term_i_b2", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_new_term_r_b2, buf_new_term_i_b2, {1}, "buf_new_term_b2"); 

    new_term_0_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_0_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_1_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_1_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0}); 
    new_term_2_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_2_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_3_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_3_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_4_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_4_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_5_r1_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_5_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {0}); 

    new_term_0_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_0_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_1_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_1_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 
    new_term_2_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_2_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_3_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_3_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 
    new_term_4_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_4_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_5_r1_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_5_r1_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 

    new_term_0_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_0_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_1_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_1_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0}); 
    new_term_2_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_2_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_3_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_3_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_4_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_4_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0});
    new_term_5_r2_b1.get_real()->store_in(buf_new_term_r_b1, {0});
    new_term_5_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {0}); 

    new_term_0_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_0_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_1_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_1_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 
    new_term_2_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_2_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_3_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_3_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 
    new_term_4_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_4_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0});
    new_term_5_r2_b2.get_real()->store_in(buf_new_term_r_b2, {0});
    new_term_5_r2_b2.get_imag()->store_in(buf_new_term_i_b2, {0}); 

    snk_psi_re.store_in(&buf_snk_psi_re, {x, x2, n});
    snk_psi_im.store_in(&buf_snk_psi_im, {x, x2, n});

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);
    C_init_r.store_in(&buf_C_r, {t, m, r, n});
    C_init_i.store_in(&buf_C_i, {t, m, r, n});
    C_update_r.store_in(&buf_C_r, {t, m, r, n});
    C_update_i.store_in(&buf_C_i, {t, m, r, n});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	     &buf_C_r, &buf_C_i,
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        B2_prop_r.get_buffer(), B2_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        src_psi_B2_r.get_buffer(), src_psi_B2_i.get_buffer(),
	     snk_blocks.get_buffer(), 
        sigs.get_buffer(),
	     snk_b.get_buffer(),
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
