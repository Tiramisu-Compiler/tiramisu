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
void generate_function(std::string name, std::string object_file_name, int src_color_weights_r1[Nw][Nq], int src_spin_weights_r1[Nw][Nq], double src_weights_r1[Nw])
{
    tiramisu::init(name);

   var nperm("nperm", 0, Nperms),
	r("r", 0, Nr),
	b("b", 0, Nb),
	q("q", 0, Nq),
	q2("q2", 0, 2*Nq),
	to("to", 0, 2),
	on("on", 0, 1),
	wnum("wnum", 0, Nw2),
   t("t", 0, Nt),
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

   input C_r("C_r",      {m, n, t}, p_float64);
   input C_i("C_i",      {m, n, t}, p_float64);
   input B1_prop_r("B1_prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input B1_prop_i("B1_prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_B1_r("src_psi_B1_r",    {n, y}, p_float64);
   input src_psi_B1_i("src_psi_B1_i",    {n, y}, p_float64);

    /*
     * Computing B1_Blocal_r1, B1_Bsingle_r1, B1_Bdouble_r1.
     */

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, expr((double) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime}, expr((double) 0));

    complex_expr src_psi_B1(src_psi_B1_r(n, y), src_psi_B1_i(n, y));

    computation B1_Bsingle_r1_r_init("B1_Bsingle_r1_r_init", {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2}, expr((double) 0));
    computation B1_Bsingle_r1_i_init("B1_Bsingle_r1_i_init", {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2}, expr((double) 0));
    computation B1_Bdouble_r1_r_init("B1_Bdouble_r1_r_init", {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2}, expr((double) 0));
    computation B1_Bdouble_r1_i_init("B1_Bdouble_r1_i_init", {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2}, expr((double) 0));

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
          B1_Blocal_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x, y) * src_psi_B1;
        complex_computation blocal_update(
            // name
            str_fmt("B1_blocal_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, y},
            // definition
            blocal_update_def);
        B1_Blocal_r1_updates.push_back(blocal_update);

        // define single block
        complex_expr bsingle_update_def =
          B1_Bsingle_r1_init(t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2) +
          q * B1_prop(1, t, jCprime, jSprime, jc, js, x2, y) * src_psi_B1;
        complex_computation bsingle_update(
            str_fmt("B1_bsingle_update_r1_%d_%d", jc, js),
            // iterator
            {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2, y},
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
          B1_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2) +
          o * B1_prop(0, t, iCprime, iSprime, ic, is, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_o_update_r1_%d_%d", ic, is),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2, y},
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
          B1_Bdouble_r1_init(t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2) -
          p * B1_prop(2, t, iCprime, iSprime, kc, ks, x2, y) * src_psi_B1;
        complex_computation bdouble_update(
            // name
            str_fmt("B1_bdouble_p_update_r1_%d_%d", kc, ks),
            // iterator
            {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2, y},
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
#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_output);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns}, p_float64, a_output);
    buffer buf_B1_Bsingle_r1_r("buf_B1_Bsingle_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_B1_Bsingle_r1_i("buf_B1_Bsingle_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_B1_Bdouble_r1_r("buf_B1_Bdouble_r1_r", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_B1_Bdouble_r1_i("buf_B1_Bdouble_r1_i", {Lt, Nc, Ns, Nc, Ns, Vsnk, Nsrc, Nc, Ns, Vsnk}, p_float64, a_output);

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

    allocate_complex_buffers(B1_q_r1_r_buf, B1_q_r1_i_buf, { Lt, Vsrc }, "buf_B1_q_r1");
    allocate_complex_buffers(B1_o_r1_r_buf, B1_o_r1_i_buf, { Lt, Vsrc }, "buf_B1_o_r1");
    allocate_complex_buffers(B1_p_r1_r_buf, B1_p_r1_i_buf, { Lt, Vsrc }, "buf_B1_p_r1");

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
      real->store_in(&buf_B1_Blocal_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime});
      imag->store_in(&buf_B1_Blocal_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime});
    }
    for (auto computations: B1_Bsingle_r1_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bsingle_r1_r, {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2});
      imag->store_in(&buf_B1_Bsingle_r1_i, {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2});
    }
    for (auto computations : B1_Bdouble_r1_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2});
      imag->store_in(&buf_B1_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2});
    }
    for (auto computations : B1_Bdouble_r1_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_B1_Bdouble_r1_r, {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2});
      imag->store_in(&buf_B1_Bdouble_r1_i, {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2});
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
        object_file_name);
}

int main(int argc, char **argv)
{
	generate_function("tiramisu_make_fused_dibaryon_blocks_correlator", "generated_tiramisu_make_fused_dibaryon_blocks_correlator.o", src_color_weights_r1_P, src_spin_weights_r1_P, src_weights_r1_P);
/*    if (R1)
	generate_function("tiramisu_make_local_single_double_block_r1", "generated_tiramisu_make_local_single_double_block.o", src_color_weights_r1_P, src_spin_weights_r1_P, src_weights_r1_P);
    else
	generate_function("tiramisu_make_local_single_double_block_r2", "generated_tiramisu_make_local_single_double_block.o", src_color_weights_r2_P, src_spin_weights_r2_P, src_weights_r2_P);
	generate_function("tiramisu_make_local_single_double_block_r1", "generated_tiramisu_make_local_single_double_block_r1.o", src_color_weights_r1_P, src_spin_weights_r1_P, src_weights_r1_P);
	generate_function("tiramisu_make_local_single_double_block_r2", "generated_tiramisu_make_local_single_double_block_r2.o", src_color_weights_r2_P, src_spin_weights_r2_P, src_weights_r2_P); */

    return 0;
}
