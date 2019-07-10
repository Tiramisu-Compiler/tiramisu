#include <tiramisu/tiramisu.h>
#include <string.h>
#include "baryon_wrapper.h"
#include "complex_util.h"
#include "util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

typedef buffer *BufferPtrTy;
/**
  * Helper function to allocate buffer for complex tensor
  */
void allocate_complex_buffers(
    BufferPtrTy &real_buff, BufferPtrTy &imag_buff, 
    std::vector<expr> dims, std::string name)
{
  real_buff = new buffer(
      // name
      str_fmt("%s_r", name.c_str()),
      // dimensions
      dims,
      // type
      tiramisu::p_float64, 
      // usage/source
      a_temporary);
  imag_buff = new buffer(
      // name
      str_fmt("%s_i", name.c_str()),
      // dimensions
      dims,
      // type
      tiramisu::p_float64, 
      // usage/source
      a_temporary);
}

// if Key is not in the map insert val
// otherwise add val to existing entry
template <typename K>
void insert_or_add(std::map<K, complex_expr> &map, K key, complex_expr val)
{
  auto it = map.find(key);
  if (it == map.end()) {
    map[key] = val;
    return;
  }
  it->second = it->second + val;
}

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
	x("x", 0, Vsnk),
	x2("x2", 0, Vsnk),
	t("t", 0, Lt),
	y("y", 0, Vsrc),
	tri("tri", 0, Nq);

    input Blocal_r("Blocal_r", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, p_float64);
    input Blocal_i("Blocal_i", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, p_float64);
    input   prop_r("prop_r",   {t, tri, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
    input   prop_i("prop_i",   {t, tri, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
    input    psi_r("psi_r",    {n, y}, p_float64);
    input    psi_i("psi_i",    {n, y}, p_float64);

    complex_computation prop(&prop_r, &prop_i);

    computation Blocal_r_init("Blocal_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, expr((double) 0));
    computation Blocal_i_init("Blocal_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, expr((double) 0));

    std::map<std::string, expr> prop_loads;
    // mapping <jc,js> -> Q[..., jc, js ...] in the reference impl.
    std::map<std::pair<int,int>, complex_expr> Q_exprs;
    // mapping <ic,is> -> O[..., ic, is ...] in the reference impl.
    std::map<std::pair<int,int>, complex_expr> O_exprs;
    // mapping <kc,ks> -> O[..., kc, ks ...] in the reference impl.
    std::map<std::pair<int,int>, complex_expr> P_exprs;
    // FIRST: build the ``unrolled'' expressions of Q, O, and P
    for (int ii = 0; ii < Nw; ii++) {
      int ic = test_color_weights[ii][0];
      int is = test_spin_weights[ii][0];
      int jc = test_color_weights[ii][1];
      int js = test_spin_weights[ii][1];
      int kc = test_color_weights[ii][2];
      int ks = test_spin_weights[ii][2];
      double w = test_weights[ii];

      complex_expr prop_0 = prop(t, 0, iCprime, iSprime, ic, is, x, y);
      complex_expr prop_2 = prop(t, 2, kCprime, kSprime, kc, ks, x, y);
      complex_expr prop_0p = prop(t, 0, kCprime, kSprime, ic, is, x, y);
      complex_expr prop_2p = prop(t, 2, iCprime, iSprime, kc, ks, x, y);
      
      complex_expr q = (prop_0 * prop_2 - prop_0p * prop_2p) * w;
      insert_or_add(Q_exprs, {jc, js}, q);

      complex_expr prop_1 = prop(t, 1, jCprime, jSprime, jc, js, x, y);
      complex_expr o = prop_1 * prop_2 * w;
      insert_or_add(O_exprs, {ic, is}, o);

      complex_expr p = prop_0p * prop_1 * w;
      insert_or_add(P_exprs, {kc, ks}, p);
    }

    // DEFINE computation of Q
    std::map<std::pair<int,int>, std::pair<computation *, computation *>> Q;
    for (auto &jAndExpr : Q_exprs) {
      int jc, js;
      std::tie(jc, js) = jAndExpr.first;
      Q[{jc,js}] = complex_computation(
          str_fmt("q_%d_%d", jc, js),
          { t, iCprime, iSprime, kCprime, kSprime, x, y },
          jAndExpr.second);
    }
    // DEFINE computation of O
    std::map<std::pair<int,int>, std::pair<computation *, computation *>> O;
    for (auto &iAndExpr : O_exprs) {
      int ic, is;
      std::tie(ic, is) = iAndExpr.first;
      O[{ic,is}] = complex_computation(
          // name
          str_fmt("o_%d_%d", ic, is),
          // iterators
          { t, jCprime, jSprime, kCprime, kSprime, x, y },
          iAndExpr.second);
    }
    // DEFINE computation of P
    std::map<std::pair<int,int>, std::pair<computation *, computation *>> P;
    for (auto &kAndExpr : P_exprs) {
      int kc, ks;
      std::tie(kc, ks) = kAndExpr.first;
      P[{kc,ks}] = complex_computation(
          // name
          str_fmt("p_%d_%d", kc, ks),
          // iterators
          { t, jCprime, jSprime, kCprime, kSprime, x, y },
          // definition
          kAndExpr.second);
    }
    complex_expr psi(psi_r(n, y), psi_i(n, y));

    computation Bsingle_r_init("Bsingle_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));
    computation Bsingle_i_init("Bsingle_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));
    computation Bdouble_r_init("Bdouble_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));
    computation Bdouble_i_init("Bdouble_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));

    complex_computation Bsingle_init(&Bsingle_r_init, &Bsingle_i_init);
    complex_computation Blocal_init(&Blocal_r_init, &Blocal_i_init);
    complex_computation Bdouble_init(&Bdouble_r_init, &Bdouble_i_init);

    std::vector<std::pair<computation *, computation *>> Bsingle_updates;
    std::vector<std::pair<computation *, computation *>> Blocal_updates;
    std::vector<std::pair<computation *, computation *>> Bdouble_o_updates;
    std::vector<std::pair<computation *, computation *>> Bdouble_p_updates;

    // Used to remember relevant (sub)computation of Q and its user computations (Blocal and Bsingle)
    struct Q2UserEdge {
      computation *q_r, *q_i,
                  *bs_r, *bs_i,
                  *bl_r, *bl_i;
    };
    std::vector<Q2UserEdge> q2userEdges;
    
    // Define Bsingle and Blocal computation
    for (auto &jAndComp : Q) {
      int jc, js;
      std::tie(jc, js) = jAndComp.first;
      complex_computation q_computation = jAndComp.second;

      // NOTE
      // iterators of Q is { t, iCprime, iSprime, kCprime, kSprime, x, y },
      //
      complex_expr q = q_computation(t, iCprime, iSprime, kCprime, kSprime, x, y);

      // define single block
      complex_expr bsingle_update_def =
        Bsingle_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) +
        q * prop(t, 1, jCprime, jSprime, jc, js, x2, y) * psi;
      complex_computation bsingle_update(
          str_fmt("bsingle_update_%d_%d", jc, js),
          // iterator
          {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, x2, y},
	  // predicate
	  (iCprime != kCprime || iSprime != kSprime),
          // definition
          bsingle_update_def);
      Bsingle_updates.push_back(bsingle_update);

      // define local block
      complex_expr blocal_update_def = 
        Blocal_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x) +
        q * prop(t, 1, jCprime, jSprime, jc, js, x, y) * psi;
      complex_computation blocal_update(
          // name
          str_fmt("blocal_update_%d_%d", jc, js),
          // iterator
          {t, iCprime, iSprime, kCprime, kSprime, x, n, jCprime, jSprime, y},
          // definition
          blocal_update_def);
      Blocal_updates.push_back(blocal_update);

      // FIXME: remove these
      auto *q_real = jAndComp.second.first;
      auto *q_imag = jAndComp.second.second;
      auto *bsingle_r = bsingle_update.get_real();
      auto *bsingle_i = bsingle_update.get_imag();
      auto *blocal_r = blocal_update.get_real();
      auto *blocal_i = blocal_update.get_imag();
      Q2UserEdge edge {q_real, q_imag, bsingle_r, bsingle_i, blocal_r, blocal_i};
      q2userEdges.push_back(edge);
    }

    // Similar to Q2UserEdge, used to record (sub)computation of P and the corresponding use in Bdouble
    struct P2UserEdge {
      computation *p_r, *p_i,
                  *bd_r, *bd_i;
    };
    std::vector<P2UserEdge> p2userEdges;

    // Define Update of Bdouble using P
    for (auto &kAndComp : P) {
      int kc, ks;
      std::tie(kc, ks) = kAndComp.first;

      complex_computation p_computation = kAndComp.second;
      // NOTE
      // iterators of P = { t, iCprime, iSprime, kCprime, kSprime, x, y },
      //
      complex_expr p = p_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

      complex_expr bdouble_update_def =
        Bdouble_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) -
        p * prop(t, 2, iCprime, iSprime, kc, ks, x2, y) * psi;
      complex_computation bdouble_update(
          // name
          str_fmt("bdouble_p_update_%d_%d", kc, ks),
          // iterator
          {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2, y},
          // definition
          bdouble_update_def);
      Bdouble_p_updates.push_back(bdouble_update);

      computation *p_real;
      computation *p_imag;
      std::tie(p_real, p_imag) = kAndComp.second;
      P2UserEdge edge {p_real, p_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
      p2userEdges.push_back(edge);
    }

    // Similar to P2UserEdge
    struct O2UserEdge {
      computation *o_r, *o_i,
                  *bd_r, *bd_i;
    };
    std::vector<O2UserEdge> o2userEdges;

    // Define Update of Bdouble using O
    for (auto &iAndComp : O) {
      int ic, is;
      std::tie(ic, is) = iAndComp.first;

      complex_computation o_computation = iAndComp.second;
      // 
      // NOTE
      // iterators of O { t, iCprime, iSprime, kCprime, kSprime, x, y },
      //
      complex_expr o = o_computation(t, jCprime, jSprime, kCprime, kSprime, x, y);

      complex_expr bdouble_update_def =
        Bdouble_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) +
        o * prop(t, 0, iCprime, iSprime, ic, is, x2, y) * psi;
      complex_computation bdouble_update(
          // name
          str_fmt("bdouble_o_update_%d_%d", ic, is),
          // iterator
          {t, jCprime, jSprime, kCprime, kSprime, x, n, iCprime, iSprime, x2, y},
          // definition
          bdouble_update_def);

      Bdouble_o_updates.push_back(bdouble_update);

      computation *o_real;
      computation *o_imag;
      std::tie(o_real, o_imag) = iAndComp.second;
      O2UserEdge edge {o_real, o_imag, bdouble_update.get_real(), bdouble_update.get_imag()};
      o2userEdges.push_back(edge);
    }

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation *handle = &(
        Blocal_r_init
        .then(Blocal_i_init, computation::root)
        .then(Bsingle_r_init, computation::root)
        .then(Bsingle_i_init, computation::root)
        .then(Bdouble_r_init, computation::root)
        .then(Bdouble_i_init, computation::root));

    bool first_comp = true; // Used to order the first computation edge.q_r in
			    // a way different from the rest.

    // schedule Blocal and Bsingle
    for (auto edge : q2userEdges) {
      handle = &(handle
          ->then(*edge.q_r, first_comp?computation::root:x)
          .then(*edge.q_i, y)
          .then(*edge.bl_r, x)
          .then(*edge.bl_i, y)
          .then(*edge.bs_r, jSprime)
          .then(*edge.bs_i, y));
      first_comp = false;
    }

    // schedule O update of Bdouble
    for (auto edge : o2userEdges) {
      handle = &(handle
          ->then(*edge.o_r, x)
          .then(*edge.o_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y));
    }

    // schedule P update of Bdouble
    for (auto edge : p2userEdges) {
      handle = &(handle
          ->then(*edge.p_r, x)
          .then(*edge.p_i, y)
          .then(*edge.bd_r, x)
          .then(*edge.bd_i, y));
    }

#if VECTORIZED
    Blocal_r_init.tag_vector_level(x, Vsnk);
    Blocal_i_init.tag_vector_level(x, Vsnk);
    Bsingle_r_init.tag_vector_level(x2, Vsnk);
    Bsingle_i_init.tag_vector_level(x2, Vsnk);
    Bdouble_r_init.tag_vector_level(x2, Vsnk);
    Bdouble_i_init.tag_vector_level(x2, Vsnk);

    for (auto edge : q2userEdges) {
      edge.q_r->tag_vector_level(y, Vsrc);
      edge.bs_r->tag_vector_level(x2, Vsnk);
    }
    for (auto edge : o2userEdges) {
      edge.o_r->tag_vector_level(y, Vsrc);
      edge.bd_r->tag_vector_level(x2, Vsnk);
    }
    for (auto edge : p2userEdges) {
      edge.p_r->tag_vector_level(y, Vsrc);
      edge.bd_r->tag_vector_level(x2, Vsnk);
    }
#endif

#if PARALLEL
    Blocal_r_init.tag_parallel_level(t);
    Blocal_i_init.tag_parallel_level(t);
    Bsingle_r_init.tag_parallel_level(t);
    Bsingle_i_init.tag_parallel_level(t);
    Bdouble_r_init.tag_parallel_level(t);
    Bdouble_i_init.tag_parallel_level(t);

    for (auto edge : q2userEdges) {
      edge.q_r->tag_parallel_level(t);
      edge.bs_r->tag_parallel_level(t);
      edge.bl_r->tag_parallel_level(t);
    }
    for (auto edge : o2userEdges) {
      edge.o_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
    for (auto edge : p2userEdges) {
      edge.p_r->tag_parallel_level(t);
      edge.bd_r->tag_parallel_level(t);
    }
#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_Blocal_r("buf_Blocal_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_Blocal_i("buf_Blocal_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_Bsingle_r("buf_Bsingle_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bsingle_i("buf_Bsingle_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bdouble_r("buf_Bdouble_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bdouble_i("buf_Bdouble_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);

    Blocal_r.store_in(&buf_Blocal_r);
    Blocal_i.store_in(&buf_Blocal_i);
    Blocal_r_init.store_in(&buf_Blocal_r);
    Blocal_i_init.store_in(&buf_Blocal_i);

    Bsingle_r_init.store_in(&buf_Bsingle_r);
    Bsingle_i_init.store_in(&buf_Bsingle_i);

    Bdouble_r_init.store_in(&buf_Bdouble_r);
    Bdouble_i_init.store_in(&buf_Bdouble_i);

    buffer *q_r_buf;
    buffer *q_i_buf;
    buffer *o_r_buf;
    buffer *o_i_buf;
    buffer *p_r_buf;
    buffer *p_i_buf;

    allocate_complex_buffers(q_r_buf, q_i_buf, { Lt, Vsnk }, "buf_q");
    allocate_complex_buffers(o_r_buf, o_i_buf, { Lt, Vsnk }, "buf_o");
    allocate_complex_buffers(p_r_buf, p_i_buf, { Lt, Vsnk }, "buf_p");

    for (auto edge : q2userEdges) {
      edge.q_r->store_in(q_r_buf, {t, y});
      edge.q_i->store_in(q_i_buf, {t, y});
    }
    for (auto edge : o2userEdges) {
      edge.o_r->store_in(o_r_buf, {t, y});
      edge.o_i->store_in(o_i_buf, {t, y});
    }
    for (auto edge : p2userEdges) {
      edge.p_r->store_in(p_r_buf, {t, y});
      edge.p_i->store_in(p_i_buf, {t, y});
    }

    for (auto computations: Blocal_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_Blocal_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x});
      imag->store_in(&buf_Blocal_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x});
    }
    for (auto computations: Bsingle_updates) {
      computation *real;
      computation *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_Bsingle_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
      imag->store_in(&buf_Bsingle_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    }
    for (auto computations : Bdouble_o_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_Bdouble_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
      imag->store_in(&buf_Bdouble_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    }
    for (auto computations : Bdouble_p_updates) {
      computation *real, *imag;
      std::tie(real, imag) = computations;
      real->store_in(&buf_Bdouble_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
      imag->store_in(&buf_Bdouble_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    }

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        &buf_Blocal_r, &buf_Blocal_i, 
        prop_r.get_buffer(), prop_i.get_buffer(),
        psi_r.get_buffer(), psi_i.get_buffer(), 
        &buf_Bsingle_r, &buf_Bsingle_i,
        Bdouble_r_init.get_buffer(),
        Bdouble_i_init.get_buffer()},
        "generated_baryon.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
