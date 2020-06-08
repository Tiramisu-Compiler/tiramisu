#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_fused_baryon_blocks_correlator_wrapper.h"
#include "../utils/complex_util.h"
#include "../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

// Used to remember relevant (sub)computation of Q and its user computations (B1_Blocal_r1 and B1_Bsingle_r1)
struct Q2UserEdge {
      computation *q_r, *q_i,
                  *bl_r, *bl_i;
};

/*
 * The goal is to generate code that implements the reference.
 * baryon_ref.cpp
 */
void generate_function(std::string name)
{
    tiramisu::init(name);

   int Nr = 2;

   var r("r", 0, Nr),
	q("q", 0, Nq),
	wnum("wnum", 0, Nw),
        t("t", 0, Lt),
	x("x", 0, Vsnk),
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

   input C_r("C_r",      {r, m, n, t}, p_float64);
   input C_i("C_i",      {r, m, n, t}, p_float64);
   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_B1_r("src_psi_B1_r",    {y, m}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {y, m}, p_float64);
   input snk_psi_r("snk_psi_r", {x, n}, p_float64);
   input snk_psi_i("snk_psi_i", {x, n}, p_float64);
   input snk_color_weights("snk_color_weights", {r, wnum, q}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, wnum, q}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);
   input snk_blocks("snk_blocks", {r}, p_int32);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));

    complex_computation B1_Blocal_r1_init(&B1_Blocal_r1_r_init, &B1_Blocal_r1_i_init);
    std::vector<std::pair<computation *, computation *>> B1_Blocal_r1_updates;

    complex_expr B1_Q_exprs_r1[Nc][Ns];
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
    }

    // DEFINE computation of Q, and its user -- B1_Blocal_r1
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

        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        B1_q2userEdges_r1.push_back(edge);
      }
    }

    /*
     * Computing B1_Blocal_r2, B1_Bsingle_r2, B1_Bdouble_r2.
     */

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime}, expr((double) 0));

    complex_computation B1_Blocal_r2_init(&B1_Blocal_r2_r_init, &B1_Blocal_r2_i_init);
    std::vector<std::pair<computation *, computation *>> B1_Blocal_r2_updates;

    complex_expr B1_Q_exprs_r2[Nc][Ns];
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

        // FIXME: remove these
        auto *q_real = q_computation.get_real();
        auto *q_imag = q_computation.get_imag();
        auto *blocal_r = blocal_update.get_real();
        auto *blocal_i = blocal_update.get_imag();
        Q2UserEdge edge {q_real, q_imag, blocal_r, blocal_i};
        B1_q2userEdges_r2.push_back(edge);
      }
    }

    /* Correlator */

    computation C_par_init_r("C_par_init_r", {t, x, m, r, n}, expr((double) 0));
    computation C_par_init_i("C_par_init_i", {t, x, m, r, n}, expr((double) 0));
    computation C_init_r("C_init_r", {t, m, r, n}, expr((double) 0));
    computation C_init_i("C_init_i", {t, m, r, n}, expr((double) 0));
    
    int b=0;
    /* r1, b = 0 */
    complex_computation new_term_0_r1_b1("new_term_0_r1_b1", {t, x,  m, r, wnum}, B1_Blocal_r1_init(t, x, m, snk_color_weights(r, wnum, 0), snk_spin_weights(r, wnum, 0), snk_color_weights(r, wnum, 2), snk_spin_weights(r, wnum, 2), snk_color_weights(r, wnum, 1), snk_spin_weights(r, wnum, 1)));
    new_term_0_r1_b1.add_predicate(snk_blocks(r) == 1);
    /* r2, b = 0 */
    complex_computation new_term_0_r2_b1("new_term_0_r2_b1", {t, x,  m, r, wnum}, B1_Blocal_r2_init(t, x, m, snk_color_weights(r, wnum, 0), snk_spin_weights(r, wnum, 0), snk_color_weights(r, wnum, 2), snk_spin_weights(r, wnum, 2), snk_color_weights(r, wnum, 1), snk_spin_weights(r, wnum, 1)));
    new_term_0_r2_b1.add_predicate(snk_blocks(r) == 2);

    complex_expr prefactor(cast(p_float64, snk_weights(r, wnum)), 0.0);

    complex_expr term_res_b1 = new_term_0_r1_b1(t, x, m, r, wnum);

    complex_expr snk_psi(snk_psi_r(x, n), snk_psi_i(x, n));

    complex_expr term_res = prefactor * term_res_b1 * snk_psi;

    computation C_par_update_r("C_par_update_r", {t, x, m, r, wnum, n}, C_par_init_r(t, x, m, r, n) + term_res.get_real());
    computation C_par_update_i("C_par_update_i", {t, x, m, r, wnum, n}, C_par_init_i(t, x, m, r, n) + term_res.get_imag());


    computation C_update_r("C_update_r", {t, x, m, r, n}, C_init_r(t, m, r, n) + C_par_init_r(t, x, m, r, n));
    computation C_update_i("C_update_i", {t, x, m, r, n}, C_init_i(t, m, r, n) + C_par_init_i(t, x, m, r, n));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, n)
          .then(C_par_init_r, t)
          .then(C_par_init_i, n)
    );

    // first the x only arrays
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

    handle->then( *(new_term_0_r1_b1.get_real()), m)
          .then( *(new_term_0_r1_b1.get_imag()), wnum)
          .then( *(new_term_0_r2_b1.get_real()), wnum)
          .then( *(new_term_0_r2_b1.get_imag()), wnum)
          .then(C_par_update_r, wnum) 
          .then(C_par_update_i, wnum)
          .then(C_update_r, t) 
          .then(C_update_i, n);

#if VECTORIZED


    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_vector_level(y, Vsrc);
    }
    for (auto edge : B1_q2userEdges_r2) {
      edge.q_r->tag_vector_level(y, Vsrc);
    }

    C_par_update_r.tag_vector_level(n, NsnkHex);
    C_par_update_i.tag_vector_level(n, NsnkHex);  

    C_update_r.tag_vector_level(n, NsnkHex);
    C_update_i.tag_vector_level(n, NsnkHex);  
#endif

#if PARALLEL

    C_par_init_r.tag_parallel_level(t);
    C_par_init_i.tag_parallel_level(t);
    C_init_r.tag_parallel_level(m);
    C_init_i.tag_parallel_level(m);

    B1_Blocal_r1_r_init.tag_parallel_level(t);
    B1_Blocal_r1_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }

    B1_Blocal_r2_r_init.tag_parallel_level(t);
    B1_Blocal_r2_i_init.tag_parallel_level(t);

    for (auto edge : B1_q2userEdges_r2) {
      edge.q_r->tag_parallel_level(t);
      edge.q_i->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
      edge.bl_i->tag_parallel_level(t);
    }

    (new_term_0_r1_b1.get_real())->tag_parallel_level(t);
    (new_term_0_r1_b1.get_imag())->tag_parallel_level(t);

    (new_term_0_r2_b1.get_real())->tag_parallel_level(t);
    (new_term_0_r2_b1.get_imag())->tag_parallel_level(t);

    C_par_update_r.tag_parallel_level(t);
    C_par_update_i.tag_parallel_level(t);  

    C_update_r.tag_parallel_level(m);
    C_update_i.tag_parallel_level(m);  

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Vsnk, NsrcHex, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Vsnk, NsrcHex, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    buffer *B1_q_r1_r_buf;
    buffer *B1_q_r1_i_buf;

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
    for (auto computations: B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r1_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r1_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }

    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Vsnk, NsrcHex, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Vsnk, NsrcHex, Nc, Ns, Nc, Ns, Nc, Ns}, p_float64, a_temporary);

    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});

    buffer *B1_q_r2_r_buf;
    buffer *B1_q_r2_i_buf;

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
    for (auto computations: B1_Blocal_r2_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r2_r, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r2_i, {x, m, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime});
    }

    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, NsrcHex, Nr, NsnkHex}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, NsrcHex, Nr, NsnkHex}, p_float64, a_input);

    buffer buf_C_par_r("buf_C_par_r", {Vsnk, NsrcHex, Nr, NsnkHex}, p_float64, a_temporary);
    buffer buf_C_par_i("buf_C_par_i", {Vsnk, NsrcHex, Nr, NsnkHex}, p_float64, a_temporary);

    buffer buf_snk_psi_r("buf_snk_psi_r", {Vsnk, Vsnk, NsnkHex}, p_float64, a_input);
    buffer buf_snk_psi_i("buf_snk_psi_i", {Vsnk, Vsnk, NsnkHex}, p_float64, a_input);

    buffer* buf_new_term_r_b1;//("buf_new_term_r_b1", {1}, p_float64, a_temporary);
    buffer* buf_new_term_i_b1;//("buf_new_term_i_b1", {1}, p_float64, a_temporary);
    allocate_complex_buffers(buf_new_term_r_b1, buf_new_term_i_b1, {Vsnk}, "buf_new_term_b1");

    new_term_0_r1_b1.get_real()->store_in(buf_new_term_r_b1, {x});
    new_term_0_r1_b1.get_imag()->store_in(buf_new_term_i_b1, {x});

    new_term_0_r2_b1.get_real()->store_in(buf_new_term_r_b1, {x});
    new_term_0_r2_b1.get_imag()->store_in(buf_new_term_i_b1, {x});

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);

    C_par_init_r.store_in(&buf_C_par_r, {x, m, r, n});
    C_par_init_i.store_in(&buf_C_par_i, {x, m, r, n});
    C_par_update_r.store_in(&buf_C_par_r, {x, m, r, n});
    C_par_update_i.store_in(&buf_C_par_i, {x, m, r, n});

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
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        snk_psi_r.get_buffer(), snk_psi_i.get_buffer(),
	     snk_blocks.get_buffer(), 
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer()
        }, 
        "generated_tiramisu_make_fused_baryon_blocks_correlator.o");
}

int main(int argc, char **argv)
{
	generate_function("tiramisu_make_fused_baryon_blocks_correlator");

    return 0;
}
