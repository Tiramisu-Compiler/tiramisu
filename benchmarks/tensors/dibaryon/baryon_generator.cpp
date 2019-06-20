#include <tiramisu/tiramisu.h>
#include <string.h>
#include "baryon_wrapper.h"

#define FUSE 1
#define PARALLEL 0

using namespace tiramisu;

/**
  * Multiply the two complex numbers p1 and p2 and return the real part.
  */
expr mul_r(std::pair<expr, expr> p1, std::pair<expr, expr> p2)
{
    expr e1 = (p1.first * p2.first);
    expr e2 = (p1.second * p2.second);
    return (e1 - e2);
}

/**
  * Multiply the two complex numbers p1 and p2 and return the imaginary part.
  */
expr mul_i(std::pair<expr, expr> p1, std::pair<expr, expr> p2)
{
    expr e1 = (p1.first * p2.second);
    expr e2 = (p1.second * p2.first);
    return (e1 + e2);
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
	wnum("wnum", 0, Nw),
	y("y", 0, Vsrc),
	tri("tri", 0, Nq);

    input Blocal_r("Blocal_r", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, p_float64);
    input Blocal_i("Blocal_i", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, p_float64);
    input   prop_r("prop_r",   {t, tri, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
    input   prop_i("prop_i",   {t, tri, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
    input  weights("weights",  {wnum}, p_float64);
    input    psi_r("psi_r",    {n, y}, p_float64);
    input    psi_i("psi_i",    {n, y}, p_float64);
    input    color_weights("color_weights",    {wnum, tri}, p_int32);
    input    spin_weights("spin_weights",    {wnum, tri}, p_int32);

    computation Blocal_r_init("Blocal_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, expr((double) 0));
    computation Blocal_i_init("Blocal_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x}, expr((double) 0));

    std::pair<expr, expr> prop_0(prop_r(t, 0, iCprime, iSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y), prop_i(t, 0, iCprime, iSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y));
    std::pair<expr, expr> prop_2(prop_r(t, 2, kCprime, kSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y), prop_i(t, 2, kCprime, kSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y));
    std::pair<expr, expr> prop_0p(prop_r(t, 0, kCprime, kSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y), prop_i(t, 0, kCprime, kSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y));
    std::pair<expr, expr> prop_2p(prop_r(t, 2, iCprime, iSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y), prop_i(t, 2, iCprime, iSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y));
    std::pair<expr, expr> m1(mul_r(prop_0, prop_2) - mul_r(prop_0p, prop_2p), mul_i(prop_0, prop_2) - mul_i(prop_0p, prop_2p));
    std::pair<expr, expr> psi(psi_r(n, y), psi_i(n, y));
    std::pair<expr, expr> m2(mul_r(psi, m1), mul_i(psi, m1));
    expr prop_r_1 = prop_r(t, 1, jCprime, jSprime, color_weights(wnum, 1), spin_weights(wnum, 1), x, y);
    expr prop_i_1 = prop_i(t, 1, jCprime, jSprime, color_weights(wnum, 1), spin_weights(wnum, 1), x, y);
    std::pair<expr, expr> prop_1(prop_r_1, prop_i_1);

    computation Blocal_r_update("Blocal_r_update", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, y, wnum}, p_float64);
    Blocal_r_update.set_expression(Blocal_r_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x) + weights(wnum) * mul_r(m2, prop_1));

    computation Blocal_i_update("Blocal_i_update", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, y, wnum}, p_float64);
    Blocal_i_update.set_expression(Blocal_i_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x) + weights(wnum) * mul_i(m2, prop_1));

    computation Q_r_init("Q_r_init", {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y}, expr((double) 0));
    computation Q_i_init("Q_i_init", {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y}, expr((double) 0));

    computation Bsingle_r_init("Bsingle_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));
    computation Bsingle_i_init("Bsingle_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));

    computation Q_r_update("Q_r_update", {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y, wnum},
			Q_r_init(t, n, iCprime, iSprime, kCprime, kSprime, color_weights(wnum, 1), spin_weights(wnum, 1), x, y) + weights(wnum) * mul_r(psi, m1));
    Q_r_update.add_predicate((jCprime == color_weights(wnum, 1)) && (jSprime == spin_weights(wnum, 1)));

    computation Q_i_update("Q_i_update", {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y, wnum},
			Q_i_init(t, n, iCprime, iSprime, kCprime, kSprime, color_weights(wnum, 1), spin_weights(wnum, 1), x, y) + weights(wnum) * mul_i(psi, m1));
    Q_i_update.add_predicate((jCprime == color_weights(wnum, 1)) && (jSprime == spin_weights(wnum, 1)));

    std::pair<expr, expr> Q_update(Q_r_update(t, n, iCprime, iSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum), Q_i_update(t, n, iCprime, iSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum));
    std::pair<expr, expr> prop_1p(prop_r(t, 1, jCprime, jSprime, lCprime, lSprime, x2, y), prop_i(t, 1, jCprime, jSprime, lCprime, lSprime, x2, y));

    computation Bsingle_r_update("Bsingle_r_update", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bsingle_r_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) + mul_r(Q_update, prop_1p));

    computation Bsingle_i_update("Bsingle_i_update", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bsingle_i_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) + mul_i(Q_update, prop_1p));

    computation Bdouble_r_init("Bdouble_r_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));
    computation Bdouble_i_init("Bdouble_i_init", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2}, expr((double) 0));

    computation O_r_init("O_r_init", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y}, expr((double) 0));
    computation O_i_init("O_i_init", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y}, expr((double) 0));

    computation P_r_init("P_r_init", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y}, expr((double) 0));
    computation P_i_init("P_i_init", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y}, expr((double) 0));

    std::pair<expr, expr> m3(mul_r(psi, prop_1), mul_i(psi, prop_1));
    computation O_r_update("O_r_update", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y, wnum},
			O_r_init(t, n, jCprime, jSprime, kCprime, kSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y) + weights(wnum) * mul_r(m3, prop_2));
    O_r_update.add_predicate((iCprime == color_weights(wnum, 0)) && (iSprime == spin_weights(wnum, 0)));

    computation O_i_update("O_i_update", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y, wnum},
			O_i_init(t, n, jCprime, jSprime, kCprime, kSprime, color_weights(wnum, 0), spin_weights(wnum, 0), x, y) + weights(wnum) * mul_i(m3, prop_2));
    O_i_update.add_predicate((iCprime == color_weights(wnum, 0)) && (iSprime == spin_weights(wnum, 0)));

    std::pair<expr, expr> m4(mul_r(psi, prop_0p), mul_i(psi, prop_0p));
    computation P_r_update("P_r_update", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y, wnum},
			P_r_init(t, n, jCprime, jSprime, kCprime, kSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y) + weights(wnum) * mul_r(m4, prop_1));
    P_r_update.add_predicate((iCprime == color_weights(wnum, 2)) && (iSprime == spin_weights(wnum, 2)));

    computation P_i_update("P_i_update", {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y, wnum},
			P_i_init(t, n, jCprime, jSprime, kCprime, kSprime, color_weights(wnum, 2), spin_weights(wnum, 2), x, y) + weights(wnum) * mul_i(m4, prop_1));
    P_i_update.add_predicate((iCprime == color_weights(wnum, 2)) && (iSprime == spin_weights(wnum, 2)));

    std::pair<expr, expr> O_update(O_r_update(t, n, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum), O_i_update(t, n, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum));
    std::pair<expr, expr> P_update(P_r_update(t, n, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum), P_i_update(t, n, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, y, wnum));
    std::pair<expr, expr> prop_0pp(prop_r(t, 0, iCprime, iSprime, lCprime, lSprime, x2, y), prop_i(t, 0, iCprime, iSprime, lCprime, lSprime, x2, y));

    computation Bdouble_r_update0("Bdouble_r_update0", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bdouble_r_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) + mul_r(prop_0pp, O_update));

    computation Bdouble_i_update0("Bdouble_i_update0", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bdouble_i_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) + mul_i(prop_0pp, O_update));

    std::pair<expr, expr> prop_2pp(prop_r(t, 2, iCprime, iSprime, lCprime, lSprime, x2, y), prop_i(t, 2, iCprime, iSprime, lCprime, lSprime, x2, y));
    computation Bdouble_r_update1("Bdouble_r_update1", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bdouble_r_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) - mul_r(P_update, prop_2pp));

    computation Bdouble_i_update1("Bdouble_i_update1", {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, lCprime, lSprime, x, x2, y},
	    Bdouble_i_init(t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2) - mul_i(P_update, prop_2pp));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    block init_blk({&Blocal_r_init, &Blocal_i_init, &Q_r_init, &Q_i_init,
		&Bsingle_r_init, &Bsingle_i_init, &Bdouble_r_init, &Bdouble_i_init,
		&O_r_init, &O_i_init, &P_r_init, &P_i_init});

    block Blocal_blk({&Blocal_r_update, &Blocal_i_update, &Q_r_update, &Q_i_update,
		 &O_r_update, &O_i_update, &P_r_update, &P_i_update});

    block Bsingle_blk({&Bsingle_r_update, &Bsingle_i_update, &Bdouble_r_update0,
			&Bdouble_i_update0, &Bdouble_r_update1, &Bdouble_i_update1});

#if FUSE
    Bsingle_blk.interchange(iCprime, jCprime);
    Bsingle_blk.interchange(iSprime, jSprime);
    Bsingle_blk.interchange(iCprime, kCprime);
    Bsingle_blk.interchange(iSprime, kSprime);
#endif

    Blocal_r_init.then(Blocal_i_init, x)
		 .then(Q_r_init, computation::root)
		 .then(Q_i_init, y)
		 .then(Bsingle_r_init, x2)
		 .then(Bsingle_i_init, x2)
		 .then(Bdouble_r_init, x2)
		 .then(Bdouble_i_init, x2)
		 .then(O_r_init, y)
		 .then(O_i_init, y)
		 .then(P_r_init, y)
		 .then(P_i_init, y)
		 .then(Blocal_r_update, computation::root)
		 .then(Blocal_i_update, wnum)
		 .then(Q_r_update, jSprime)
		 .then(Q_i_update, wnum)
		 .then(O_r_update, wnum)
		 .then(O_i_update, wnum)
		 .then(P_r_update, wnum)
		 .then(P_i_update, wnum)
		 .then(Bsingle_r_update, n)
		 .then(Bsingle_i_update, y)
		 .then(Bdouble_r_update0, y)
		 .then(Bdouble_i_update0, y)
		 .then(Bdouble_r_update1, y)
		 .then(Bdouble_i_update1, y);


    if (PARALLEL)
    {
        Blocal_r_init.tag_parallel_level(t);
        Blocal_r_update.tag_parallel_level(t);
        Bsingle_r_update.tag_parallel_level(t);
    }

    Blocal_r_init.vectorize(x, Vsnk);
    Q_r_init.vectorize(y, Vsrc);

    Blocal_r_update.vectorize(x, Vsnk);
    Q_r_update.vectorize(y, Vsrc);

    Bsingle_r_update.vectorize(x2, Vsnk);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_Blocal_r("buf_Blocal_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_Blocal_i("buf_Blocal_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk}, p_float64, a_output);
    buffer buf_Q_r("buf_Q_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Q_i("buf_Q_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_O_r("buf_O_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_O_i("buf_O_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_P_r("buf_P_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_P_i("buf_P_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bsingle_r("buf_Bsingle_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bsingle_i("buf_Bsingle_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bdouble_r("buf_Bdouble_r", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);
    buffer buf_Bdouble_i("buf_Bdouble_i", {Lt, Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Vsnk}, p_float64, a_output);

    Blocal_r.store_in(&buf_Blocal_r);
    Blocal_i.store_in(&buf_Blocal_i);
    Blocal_r_init.store_in(&buf_Blocal_r);
    Blocal_i_init.store_in(&buf_Blocal_i);
    Blocal_r_update.store_in(&buf_Blocal_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x});
    Blocal_i_update.store_in(&buf_Blocal_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x});

    Q_r_init.store_in(&buf_Q_r);
    Q_i_init.store_in(&buf_Q_i);
    O_r_init.store_in(&buf_O_r);
    O_i_init.store_in(&buf_O_i);
    P_r_init.store_in(&buf_P_r);
    P_i_init.store_in(&buf_P_i);

    Q_r_update.store_in(&buf_Q_r, {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y});
    Q_i_update.store_in(&buf_Q_i, {t, n, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, x, y});
    O_r_update.store_in(&buf_O_r, {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y});
    O_i_update.store_in(&buf_O_i, {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y});
    P_r_update.store_in(&buf_P_r, {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y});
    P_i_update.store_in(&buf_P_i, {t, n, jCprime, jSprime, kCprime, kSprime, iCprime, iSprime, x, y});

    Bsingle_r_init.store_in(&buf_Bsingle_r);
    Bsingle_i_init.store_in(&buf_Bsingle_i);

    Bsingle_r_update.store_in(&buf_Bsingle_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    Bsingle_i_update.store_in(&buf_Bsingle_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});

    Bdouble_r_init.store_in(&buf_Bdouble_r);
    Bdouble_i_init.store_in(&buf_Bdouble_i);
    Bdouble_r_update0.store_in(&buf_Bdouble_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    Bdouble_i_update0.store_in(&buf_Bdouble_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    Bdouble_r_update1.store_in(&buf_Bdouble_r, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});
    Bdouble_i_update1.store_in(&buf_Bdouble_i, {t, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, x2});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_Blocal_r, &buf_Blocal_i, prop_r.get_buffer(), prop_i.get_buffer(), weights.get_buffer(), psi_r.get_buffer(), psi_i.get_buffer(), color_weights.get_buffer(), spin_weights.get_buffer(), Bsingle_r_update.get_buffer(), Bsingle_i_update.get_buffer(), Bdouble_r_init.get_buffer(), Bdouble_i_init.get_buffer(), &buf_O_r, &buf_O_i, &buf_P_r, &buf_P_i, &buf_Q_r, &buf_Q_i}, "generated_baryon.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
