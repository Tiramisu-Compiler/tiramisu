#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_local_single_double_block_wrapper.h"
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

    var n("n", 0, Nsrc),
    iCprime("iCprime", 0, Nc),
    iSprime("iSprime", 0, Ns),
    jCprime("jCprime", 0, Nc),
    jSprime("jSprime", 0, Ns),
    kCprime("kCprime", 0, Nc),
    kSprime("kSprime", 0, Ns),
    lCprime("lCprime", 0, Nc),
    lSprime("lSprime", 0, Ns),
    y("y", 0, Vsrc),
    tri("tri", 0, Nq);

    input B1_Blocal_r1_r("B1_Blocal_r1_r",      {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, p_float64);
    input B1_Blocal_r1_i("B1_Blocal_r1_i",      {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, p_float64);
    input   B1_prop_r("B1_prop_r",   {tri, iCprime, iSprime, jCprime, jSprime, y}, p_float64);
    input   B1_prop_i("B1_prop_i",   {tri, iCprime, iSprime, jCprime, jSprime, y}, p_float64);
    input   B2_prop_r("B2_prop_r",   {tri, iCprime, iSprime, jCprime, jSprime, y}, p_float64);
    input   B2_prop_i("B2_prop_i",   {tri, iCprime, iSprime, jCprime, jSprime, y}, p_float64);
    input    src_psi_B1_r("src_psi_B1_r",    {y, n}, p_float64);
    input    src_psi_B1_i("src_psi_B1_i",    {y, n}, p_float64);

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);
    complex_computation B2_prop(&B2_prop_r, &B2_prop_i);

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));

    complex_expr src_psi_B1(src_psi_B1_r(y, n), src_psi_B1_i(y, n));

    computation B1_Bsingle_r1_r_init("B1_Bsingle_r1_r_init", {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation B1_Bsingle_r1_i_init("B1_Bsingle_r1_i_init", {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n}, expr((double) 0));
    computation B1_Bdouble_r1_r_init("B1_Bdouble_r1_r_init", {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n}, expr((double) 0));
    computation B1_Bdouble_r1_i_init("B1_Bdouble_r1_i_init", {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n}, expr((double) 0));

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
      int ic = src_color_weights_r1[ii][0];
      int is = src_spin_weights_r1[ii][0];
      int jc = src_color_weights_r1[ii][1];
      int js = src_spin_weights_r1[ii][1];
      int kc = src_color_weights_r1[ii][2];
      int ks = src_spin_weights_r1[ii][2];
      double w = src_weights_r1[ii];

      complex_expr B1_prop_0 =  B1_prop(0, iCprime, iSprime, ic, is, y);
      complex_expr B1_prop_2 =  B1_prop(2, kCprime, kSprime, kc, ks, y);
      complex_expr B1_prop_0p = B1_prop(0, kCprime, kSprime, ic, is, y);
      complex_expr B1_prop_2p = B1_prop(2, iCprime, iSprime, kc, ks, y);
      complex_expr B1_prop_1 = B1_prop(1, jCprime, jSprime, jc, js, y);
      
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
            {iCprime, iSprime, kCprime, kSprime, y},
            B1_Q_exprs_r1[jc][js]);

        complex_expr q = q_computation(iCprime, iSprime, kCprime, kSprime, y);

        // define local block
        complex_expr blocal_update_def = 
          B1_Blocal_r1_init(iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) +
          q * B1_prop(1, jCprime, jSprime, jc, js, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y, n},
            // definition
            blocal_update_def);
        B1_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B1_Bsingle_r1_init(iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n) +
          q * B2_prop(1, jCprime, jSprime, jc, js, y) * src_psi_B1;
        complex_computation bsingle_update(
            str_fmt("B1_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, y, n},
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
            {jCprime, jSprime, kCprime, kSprime, y},
            B1_O_exprs_r1[ic][is]);

        complex_expr o = o_computation(jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n) +
          o * B2_prop(0, iCprime, iSprime, ic, is, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y, n},
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
            {jCprime, jSprime, kCprime, kSprime, y},
            // definition
            B1_P_exprs_r1[kc][ks]);
        complex_expr p = p_computation(jCprime, jSprime, kCprime, kSprime, y);

        complex_expr bdouble_update_def =
          B1_Bdouble_r1_init(jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n) -
          p * B2_prop(2, iCprime, iSprime, kc, ks, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, y, n},
            // definition
            bdouble_update_def);
        B1_Bdouble_r1_p_updates.push_back(bdouble_update);

        computation *p_real = p_computation.get_real();
        computation *p_imag = p_computation.get_imag();
        P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
        B1_p2userEdges_r1.push_back(edge);
      }
    }

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation *handle = &(
        B1_Blocal_r1_r_init
        .then(B1_Blocal_r1_i_init, n)
        .then(B1_Bsingle_r1_r_init, n)
        .then(B1_Bsingle_r1_i_init, n)
        .then(B1_Bdouble_r1_r_init, n)
        .then(B1_Bdouble_r1_i_init, n)
	);


    // schedule B1_Blocal_r1 and B1_Bsingle_r1
    for (int i = 0; i < B1_q2userEdges_r1.size(); i++)
    {
      auto edge = B1_q2userEdges_r1[i];
      handle = &(handle
          ->then(*edge.q_r, y)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, n)
          .then(*edge.bl_i, n)
          .then(*edge.bs_r, n)
          .then(*edge.bs_i, n)
	  );
    }

    // schedule O update of B1_Bdouble_r1
    for (int i = 0; i < B1_o2userEdges_r1.size(); i++)
    {
      auto edge  = B1_o2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.o_r, y)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, n)
          .then(*edge.bd_i, n)
	  );
    }

    // schedule P update of B1_Bdouble_r1
    for (int i = 0; i < B1_p2userEdges_r1.size(); i++)
    {
      auto edge  = B1_p2userEdges_r1[i];

      handle = &(handle
          ->then(*edge.p_r, y)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, n)
          .then(*edge.bd_i, n)
	  );
    }



#if VECTORIZED
    B1_Blocal_r1_r_init.tag_vector_level(n, Nsrc);
    B1_Blocal_r1_i_init.tag_vector_level(n, Nsrc);
    B1_Bsingle_r1_r_init.tag_vector_level(n, Nsrc);
    B1_Bsingle_r1_i_init.tag_vector_level(n, Nsrc);
    B1_Bdouble_r1_r_init.tag_vector_level(n, Nsrc);
    B1_Bdouble_r1_i_init.tag_vector_level(n, Nsrc);

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_vector_level(jSprime, Ns);
      edge.bs_r->tag_vector_level(n, Nsrc);
      edge.bl_r->tag_vector_level(n, Nsrc);
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->tag_vector_level(jSprime, Ns);
      edge.bd_r->tag_vector_level(n, Nsrc);
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->tag_vector_level(jSprime, Ns);
      edge.bd_r->tag_vector_level(n, Nsrc);
    }
#endif

#if PARALLEL
    B1_Blocal_r1_r_init.tag_parallel_level(y);
    B1_Blocal_r1_i_init.tag_parallel_level(y);
    B1_Bsingle_r1_r_init.tag_parallel_level(y);
    B1_Bsingle_r1_i_init.tag_parallel_level(y);
    B1_Bdouble_r1_r_init.tag_parallel_level(y);
    B1_Bdouble_r1_i_init.tag_parallel_level(y);

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->tag_parallel_level(y);
      edge.bs_r->tag_parallel_level(y);
      edge.bl_r->tag_parallel_level(y);
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->tag_parallel_level(y);
      edge.bd_r->tag_parallel_level(y);
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->tag_parallel_level(y);
      edge.bd_r->tag_parallel_level(y);
    }
#endif


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);
    buffer buf_B1_Bsingle_r1_r("buf_B1_Bsingle_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);
    buffer buf_B1_Bsingle_r1_i("buf_B1_Bsingle_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);
    buffer buf_B1_Bdouble_r1_r("buf_B1_Bdouble_r1_r", {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);
    buffer buf_B1_Bdouble_r1_i("buf_B1_Bdouble_r1_i", {Nc, Ns, Nc, Ns, Nc, Ns, Nsrc}, p_float64, a_output);

    B1_Blocal_r1_r.store_in(&buf_B1_Blocal_r1_r);
    B1_Blocal_r1_i.store_in(&buf_B1_Blocal_r1_i);
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

// ????? 
    allocate_complex_buffers(B1_q_r1_r_buf, B1_q_r1_i_buf, { Vsrc }, "buf_B1_q_r1");
    allocate_complex_buffers(B1_o_r1_r_buf, B1_o_r1_i_buf, { Vsrc }, "buf_B1_o_r1");
    allocate_complex_buffers(B1_p_r1_r_buf, B1_p_r1_i_buf, { Vsrc }, "buf_B1_p_r1");

    for (auto edge : B1_q2userEdges_r1) {
      edge.q_r->store_in(B1_q_r1_r_buf, {y});
      edge.q_i->store_in(B1_q_r1_i_buf, {y});
    }
    for (auto edge : B1_o2userEdges_r1) {
      edge.o_r->store_in(B1_o_r1_r_buf, {y});
      edge.o_i->store_in(B1_o_r1_i_buf, {y});
    }
    for (auto edge : B1_p2userEdges_r1) {
      edge.p_r->store_in(B1_p_r1_r_buf, {y});
      edge.p_i->store_in(B1_p_r1_i_buf, {y});
    }
// ????? 


    for (auto computations: B1_Blocal_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Blocal_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
      imag->store_in(&buf_B1_Blocal_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    }
    for (auto computations: B1_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r1_r, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
      imag->store_in(&buf_B1_Bsingle_r1_i, {iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, n});
    }
    for (auto computations : B1_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n});
      imag->store_in(&buf_B1_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n});
    }
    for (auto computations : B1_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n});
      imag->store_in(&buf_B1_Bdouble_r1_i, {jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, n});
    }

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        &buf_B1_Blocal_r1_r, &buf_B1_Blocal_r1_i, 
        B1_prop_r.get_buffer(), B1_prop_i.get_buffer(),
        src_psi_B1_r.get_buffer(), src_psi_B1_i.get_buffer(), 
        &buf_B1_Bsingle_r1_r, &buf_B1_Bsingle_r1_i,
        B1_Bdouble_r1_r_init.get_buffer(), B1_Bdouble_r1_i_init.get_buffer()},
        "generated_tiramisu_make_local_single_double_block.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_make_local_single_double_block");

    return 0;
}
