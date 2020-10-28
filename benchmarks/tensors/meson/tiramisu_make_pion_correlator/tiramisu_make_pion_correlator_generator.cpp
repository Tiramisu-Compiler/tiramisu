#include <tiramisu/tiramisu.h>
#include <string.h>
#include "tiramisu_make_pion_correlator_wrapper.h"
#include "../../utils/complex_util.h"
#include "../../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 1
#define PARALLEL 1

void generate_function(std::string name)
{
    tiramisu::init(name);

   var r("r", 0, B0Nrows),
	rp("rp", 0, B0Nrows),
	q("q", 0, Mq),
	wnum("wnum", 0, Mw),
	wnumBlock("wnumBlock", 0, Mw),
        t("t", 0, Lt),
	x("x", 0, Vsnk),
	x_out("x_out", 0, Vsnk/sites_per_rank),
	x_in("x_in", 0, sites_per_rank),
        y("y", 0, Vsrc),
	m("m", 0, NsrcHex),
	n("n", 0, NsnkHex),
        tri("tri", 0, Mq),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, NsFull),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, NsFull);

   input C_r("C_r",      {t, x_out, rp, m, r, n}, p_float64);
   input C_i("C_i",      {t, x_out, rp, m, r, n}, p_float64);
   input prop_r("prop_r",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input prop_i("prop_i",   {tri, t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float64);
   input src_psi_r("src_psi_r",    {y, m}, p_float64);
   input src_psi_i("src_psi_i",    {y, m}, p_float64);
   input snk_psi_r("snk_psi_r", {x, n}, p_float64);
   input snk_psi_i("snk_psi_i", {x, n}, p_float64);
   input src_color_weights("src_color_weights", {r, wnum, q}, p_int32);
   input src_spin_weights("src_spin_weights", {r, wnum, q}, p_int32);
   input src_weights("src_weights", {r, wnum}, p_float64);
   input snk_color_weights("snk_color_weights", {r, wnum, q}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, wnum, q}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float64);

    complex_computation prop(&prop_r, &prop_i);

    complex_expr src_psi(src_psi_r(y, m), src_psi_i(y, m));

    /*
     * Computing pion block 
     */

    computation Blocal_r_init("Blocal_r_init", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, m}, expr((double) 0));
    computation Blocal_i_init("Blocal_i_init", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, m}, expr((double) 0));

    complex_computation Blocal_init(&Blocal_r_init, &Blocal_i_init);

    complex_expr prop_0 =  prop(0, t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr prop_1 = prop(1, t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);

    complex_expr props = ( prop_0 * prop_1 ) *  src_weights(0, wnumBlock);

    computation Blocal_r_props_init("Blocal_r_props_init", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y}, expr((double) 0));
    computation Blocal_i_props_init("Blocal_i_props_init", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y}, expr((double) 0));

    computation Blocal_r_props("Blocal_r_props", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y, wnumBlock}, Blocal_r_props_init(t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y) + props.get_real());
    computation Blocal_i_props("Blocal_i_props", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y, wnumBlock}, Blocal_i_props_init(t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y) + props.get_imag());

    complex_computation Blocal_props(&Blocal_r_props, &Blocal_i_props);

    complex_expr r1 = src_psi * Blocal_props(t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y, Nw-1);

    computation Blocal_r_update("Blocal_r_update", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y, m}, Blocal_r_init(t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, m) + r1.get_real());
    computation Blocal_i_update("Blocal_i_update", {t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, y, m}, Blocal_i_init(t, x_out, x_in, iCprime, iSprime, jCprime, jSprime, m) + r1.get_imag());

    /* Correlator */

    computation C_init_r("C_init_r", {t, x_out, rp, m, r, n}, expr((double) 0));
    computation C_init_i("C_init_i", {t, x_out, rp, m, r, n}, expr((double) 0));

    computation C_prop_init_r("C_prop_init_r", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    computation C_prop_init_i("C_prop_init_i", {t, x_out, x_in, rp, m, r}, expr((double) 0));
    
    complex_computation new_term_0("new_term_0", {t, x_out, x_in, rp, m, r, wnum}, Blocal_init(t, x_out, x_in, snk_color_weights(r, wnum, 0), snk_spin_weights(r, wnum, 0), snk_color_weights(r, wnum, 1), snk_spin_weights(r, wnum, 1), m) );

    complex_expr prefactor(cast(p_float64, snk_weights(r, wnum)), 0.0);

    complex_expr term_res = prefactor * new_term_0(t, x_out, x_in, rp, m, r, wnum);

    complex_expr snk_psi(snk_psi_r(x_out*sites_per_rank+x_in, n), snk_psi_i(x_out*sites_per_rank+x_in, n));

    computation C_prop_update_r("C_prop_update_r", {t, x_out, x_in, rp, m, r, wnum}, C_prop_init_r(t, x_out, x_in, rp, m, r) + term_res.get_real());
    computation C_prop_update_i("C_prop_update_i", {t, x_out, x_in, rp, m, r, wnum}, C_prop_init_i(t, x_out, x_in, rp, m, r) + term_res.get_imag());

    complex_computation C_prop_update(&C_prop_update_r, &C_prop_update_i);

    complex_expr term = C_prop_update(t, x_out, x_in, rp, m, r, Nw-1) * snk_psi;

    computation C_update_r("C_update_r", {t, x_out, x_in, rp, m, r, n}, C_init_r(t, x_out, rp, m, r, n) + term.get_real());
    computation C_update_i("C_update_i", {t, x_out, x_in, rp, m, r, n}, C_init_i(t, x_out, rp, m, r, n) + term.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    computation* handle = &(C_init_r
          .then(C_init_i, n)
    );

    // first the x only arrays
    handle = &(handle
        ->then(Blocal_r_init, t)
        .then(Blocal_i_init, jSprime)
        .then(Blocal_r_props_init, x_in)
        .then(Blocal_i_props_init, y)
        .then(Blocal_r_props, y)
        .then(Blocal_i_props, wnumBlock)
        .then(Blocal_r_update, y)
        .then(Blocal_i_update, m));

    handle = &(handle 
          ->then(C_prop_init_r, x_in) 
          .then(C_prop_init_i, r)
          .then( *(new_term_0.get_real()), r)
          .then( *(new_term_0.get_imag()), wnum)
          .then(C_prop_update_r, wnum) 
          .then(C_prop_update_i, wnum)
          .then(C_update_r, r) 
          .then(C_update_i, n));

#if VECTORIZED

#endif

#if PARALLEL

    C_init_r.tag_distribute_level(t);

    Blocal_r_init.tag_distribute_level(t);

    C_prop_init_r.tag_distribute_level(t);

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_Blocal_r("buf_Blocal_r",   {Nc, NsFull, Nc, NsFull, NsrcHex}, p_float64, a_temporary);
    buffer buf_Blocal_i("buf_Blocal_i",   {Nc, NsFull, Nc, NsFull, NsrcHex}, p_float64, a_temporary);
    Blocal_r_init.store_in(&buf_Blocal_r, {iCprime, iSprime, jCprime, jSprime, m});
    Blocal_i_init.store_in(&buf_Blocal_i, {iCprime, iSprime, jCprime, jSprime, m});
    Blocal_r_update.store_in(&buf_Blocal_r, {iCprime, iSprime, jCprime, jSprime, m});
    Blocal_i_update.store_in(&buf_Blocal_i, {iCprime, iSprime, jCprime, jSprime, m});
    buffer buf_Blocal_props_r("buf_Blocal_props_r",   {1}, p_float64, a_temporary);
    buffer buf_Blocal_props_i("buf_Blocal_props_i",   {1}, p_float64, a_temporary);
    Blocal_r_props_init.store_in(&buf_Blocal_props_r, {0});
    Blocal_i_props_init.store_in(&buf_Blocal_props_i, {0});
    Blocal_r_props.store_in(&buf_Blocal_props_r, {0});
    Blocal_i_props.store_in(&buf_Blocal_props_i, {0});

    /* Correlator */

    buffer buf_C_r("buf_C_r", {Lt, Vsnk/sites_per_rank, B0Nrows, NsrcHex, B0Nrows, NsnkHex}, p_float64, a_input);
    buffer buf_C_i("buf_C_i", {Lt, Vsnk/sites_per_rank, B0Nrows, NsrcHex, B0Nrows, NsnkHex}, p_float64, a_input);

    C_r.store_in(&buf_C_r);
    C_i.store_in(&buf_C_i);

    buffer* buf_new_term_r;
    buffer* buf_new_term_i;
    allocate_complex_buffers(buf_new_term_r, buf_new_term_i, {1}, "buf_new_term");

    new_term_0.get_real()->store_in(buf_new_term_r, {0});
    new_term_0.get_imag()->store_in(buf_new_term_i, {0});

    buffer buf_C_prop_r("buf_C_prop_r", {1}, p_float64, a_temporary);
    buffer buf_C_prop_i("buf_C_prop_i", {1}, p_float64, a_temporary);

    C_prop_init_r.store_in(&buf_C_prop_r, {0});
    C_prop_init_i.store_in(&buf_C_prop_i, {0});
    C_prop_update_r.store_in(&buf_C_prop_r, {0});
    C_prop_update_i.store_in(&buf_C_prop_i, {0});

    C_init_r.store_in(&buf_C_r, {t, x_out, rp, m, r, n});
    C_init_i.store_in(&buf_C_i, {t, x_out, rp, m, r, n});
    C_update_r.store_in(&buf_C_r, {t, x_out, rp, m, r, n});
    C_update_i.store_in(&buf_C_i, {t, x_out, rp, m, r, n});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
	     &buf_C_r, &buf_C_i,
        prop_r.get_buffer(), prop_i.get_buffer(),
        src_psi_r.get_buffer(), src_psi_i.get_buffer(), 
        snk_psi_r.get_buffer(), snk_psi_i.get_buffer(),
	     src_color_weights.get_buffer(),
	     src_spin_weights.get_buffer(),
	     src_weights.get_buffer(),
	     snk_color_weights.get_buffer(),
	     snk_spin_weights.get_buffer(),
	     snk_weights.get_buffer()
        }, 
        "generated_tiramisu_make_pion_correlator.o");
}

int main(int argc, char **argv)
{

    generate_function("tiramisu_make_pion_correlator");

    return 0;
}
