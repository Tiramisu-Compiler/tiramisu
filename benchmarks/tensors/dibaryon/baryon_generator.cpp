#include <tiramisu/tiramisu.h>

#include <string.h>

#include "baryon_wrapper.h"

using namespace tiramisu;

/* Implementation log:
    - Sum: 135mn.
    - Started from 22:45 to 
 */

// TODO:
// I need to make the initialization of array to
// random so that I catch errors.

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
	x("x", 0, Vsnk),
	t("t", 0, Lt),
	wnum("wnum", 0, Nw),
	y("y", 0, Vsrc),
	tri("tri", 0, Nq);

    input Blocal_r("Blocal_r", {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t}, p_float64);
    input Blocal_i("Blocal_i", {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t}, p_float64);
    input   prop_r("prop_r",   {tri, iCprime, iSprime, jCprime, jSprime, x, t, y}, p_float64);
    input   prop_i("prop_i",   {tri, iCprime, iSprime, jCprime, jSprime, x, t, y}, p_float64);
    input  weights("weights",  {wnum}, p_float64);
    input    psi_r("psi_r",    {n, y}, p_float64);
    input    psi_i("psi_i",    {n, y}, p_float64);
    input    color_weights("color_weights",    {wnum, tri}, p_int32);
    input    spin_weights("spin_weights",    {wnum, tri}, p_int32);

    computation Blocal_r_init("Blocal_r_init", {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t}, expr((double) 0));
    computation Blocal_i_init("Blocal_i_init", {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t}, expr((double) 0));

    computation iC("iC", {wnum}, color_weights(wnum, 0));
    computation iS("iS", {wnum}, spin_weights(wnum, 0));
    computation jC("iC", {wnum}, color_weights(wnum, 1));
    computation jS("iS", {wnum}, spin_weights(wnum, 1));
    computation kC("iC", {wnum}, color_weights(wnum, 2));
    computation kS("iS", {wnum}, spin_weights(wnum, 2));

    computation Blocal_r_update("Blocal_r_update", {wnum, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t, y}, p_float64);
    Blocal_r_update.set_expression(Blocal_r_init(n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t) + weights(wnum) * psi_r(n, y) * (prop_r(0, iCprime, iSprime, iC(wnum), iS(wnum), x, t, y) * prop_r(2, kCprime, kSprime, kC(wnum), kS(wnum), x, t, y) - prop_r(0, kCprime, kSprime, iC(wnum), iS(wnum), x, t, y) * prop_r(2, iCprime, iSprime, kC(wnum), kS(wnum), x, t, y)) * prop_r(1, jCprime, jSprime, jC(wnum), jS(wnum), x, t, y));

    computation Blocal_i_update("Blocal_i_update", {wnum, n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t, y}, p_float64);
    Blocal_i_update.set_expression(Blocal_i_init(n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t) + weights(wnum) * psi_i(n, y) * (prop_i(0, iCprime, iSprime, iC(wnum), iS(wnum), x, t, y) * prop_i(2, kCprime, kSprime, kC(wnum), kS(wnum), x, t, y) - prop_i(0, kCprime, kSprime, iC(wnum), iS(wnum), x, t, y) * prop_i(2, iCprime, iSprime, kC(wnum), kS(wnum), x, t, y)) * prop_i(1, jCprime, jSprime, jC(wnum), jS(wnum), x, t, y));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    Blocal_r_init.then(Blocal_i_init, t)
	         .then(iC, computation::root)
		 .then(iS, wnum)
		 .then(jC, wnum)
		 .then(jS, wnum)
		 .then(kC, wnum)
		 .then(kS, wnum)
		 .then(Blocal_r_update, wnum)
		 .then(Blocal_i_update, y);

    //Blocal_r_update.tag_parallel_level(n);
    Blocal_r_update.vectorize(t, Lt);
    //Blocal_r_update.unroll(y, Vsrc);
    //Blocal_r_update.unroll(x, Vsnk);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_Blocal_r("buf_Blocal_r", {Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Lt}, p_float64, a_output);
    buffer buf_Blocal_i("buf_Blocal_i", {Nsrc, Nc, Ns, Nc, Ns, Nc, Ns, Vsnk, Lt}, p_float64, a_output);

    Blocal_r.store_in(&buf_Blocal_r);
    Blocal_i.store_in(&buf_Blocal_i);
    Blocal_r_init.store_in(&buf_Blocal_r);
    Blocal_i_init.store_in(&buf_Blocal_i);
    Blocal_r_update.store_in(&buf_Blocal_r, {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t});
    Blocal_i_update.store_in(&buf_Blocal_i, {n, iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, x, t});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_Blocal_r, &buf_Blocal_i, prop_r.get_buffer(), prop_i.get_buffer(), weights.get_buffer(), psi_r.get_buffer(), psi_i.get_buffer(), color_weights.get_buffer(), spin_weights.get_buffer()}, "generated_baryon.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
