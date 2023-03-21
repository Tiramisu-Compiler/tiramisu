#include <tiramisu/tiramisu.h>
#include <string.h>
#include "gpu_single_tiramisu_make_fused_identical_baryon_blocks_correlator_wrapper.h"
#include "../../utils/complex_float_util.h"
#include "../../utils/util.h"

using namespace tiramisu;

#define VECTORIZED 0
#define PARALLEL 0
#define GPU_PARALLEL 1

#define TAG_PARALLEL_T 0

void generate_function(std::string name)
{
    tiramisu::init(name);

    var r("r", 0, B1Nrows),
        rp("rp", 0, B1Nrows),
        nperm("nperm", 0, B1Nperms),
        q("q", 0, Nq),
        wnum("wnum", 0, Nw),
        wnumBlock("wnumBlock", 0, Nw),
        t("t", 0, Lt),
        x("x", 0, Vsnk),
        x_out("x_out", 0, Vsnk/sites_per_rank),
        x_in("x_in", 0, sites_per_rank),
        y("y", 0, Vsrc), 
        m("m", 0, NsrcHex),
        n("n", 0, NsnkHex),
        iCprime("iCprime", 0, Nc),
        iSprime("iSprime", 0, Ns),
        jCprime("jCprime", 0, Nc),
        jSprime("jSprime", 0, Ns),
        kCprime("kCprime", 0, Nc),
        kSprime("kSprime", 0, Ns);

   input out_C_r("out_C_r",      {t, rp, m, r, n}, p_float32);
   input out_C_i("out_C_i",      {t, rp, m, r, n}, p_float32);
   input B1_prop_r("B1_prop_r",   {t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float32);
   input B1_prop_i("B1_prop_i",   {t, iCprime, iSprime, jCprime, jSprime, x, y}, p_float32);
   input src_psi_B1_r("src_psi_B1_r",    {y, m}, p_float32);
   input src_psi_B1_i("src_psi_B1_i",    {y, m}, p_float32);
   input snk_psi_r("snk_psi_r", {x, n}, p_float32); 
   input snk_psi_i("snk_psi_i", {x, n}, p_float32);
   input src_color_weights("src_color_weights", {rp, wnum, q}, p_int32);
   input src_spin_weights("src_spin_weights", {rp, wnum, q}, p_int32);
   input src_weights("src_weights", {rp, wnum}, p_float32);
   input snk_color_weights("snk_color_weights", {r, nperm, wnum, q}, p_int32);
   input snk_spin_weights("snk_spin_weights", {r, nperm, wnum, q}, p_int32);
   input snk_weights("snk_weights", {r, wnum}, p_float32);
   input src_spins("src_spins", {rp}, p_int32);
   input sigs("sigs", {nperm}, p_int32);

    complex_computation B1_prop(&B1_prop_r, &B1_prop_i);

    complex_expr src_psi_B1(src_psi_B1_r(y, m), src_psi_B1_i(y, m));

    /*
     * Computing B1_Blocal_r1
     */

    computation B1_Blocal_r1_r_init("B1_Blocal_r1_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((float) 0));
    computation B1_Blocal_r1_i_init("B1_Blocal_r1_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((float) 0));

    complex_computation B1_Blocal_r1_init(&B1_Blocal_r1_r_init, &B1_Blocal_r1_i_init);

    complex_expr B1_r1_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(0, wnumBlock, 0), src_spin_weights(0, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_1 =  B1_prop(t, jCprime, jSprime, src_color_weights(0, wnumBlock, 1), src_spin_weights(0, wnumBlock, 1), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r1_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(0, wnumBlock, 2), src_spin_weights(0, wnumBlock, 2), x_out*sites_per_rank+x_in, y);


    complex_expr B1_r1_diquark = ( B1_r1_prop_0 * B1_r1_prop_2 ) *  src_weights(0, wnumBlock); // prop_prod_02

    computation B1_Blocal_r1_r_props_init("B1_Blocal_r1_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((float) 0));
    computation B1_Blocal_r1_i_props_init("B1_Blocal_r1_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((float) 0));

    computation B1_Blocal_r1_r_diquark("B1_Blocal_r1_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_real());
    computation B1_Blocal_r1_i_diquark("B1_Blocal_r1_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r1_diquark.get_imag());

    complex_computation B1_Blocal_r1_diquark(&B1_Blocal_r1_r_diquark, &B1_Blocal_r1_i_diquark);

    complex_expr B1_r1_props = B1_r1_prop_1 * B1_Blocal_r1_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r1_r_props("B1_Blocal_r1_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_r1_props.get_real() + B1_Blocal_r1_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime));
    computation B1_Blocal_r1_i_props("B1_Blocal_r1_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_r1_props.get_imag() + B1_Blocal_r1_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime));

    complex_computation B1_Blocal_r1_props(&B1_Blocal_r1_r_props, &B1_Blocal_r1_i_props);

    complex_expr B1_r1 = src_psi_B1 * B1_Blocal_r1_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r1_r_update("B1_Blocal_r1_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_real());
    computation B1_Blocal_r1_i_update("B1_Blocal_r1_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r1_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r1.get_imag());

    /*
     * Computing B1_Blocal_r2
     */

    computation B1_Blocal_r2_r_init("B1_Blocal_r2_r_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((float) 0));
    computation B1_Blocal_r2_i_init("B1_Blocal_r2_i_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m}, expr((float) 0));

    complex_computation B1_Blocal_r2_init(&B1_Blocal_r2_r_init, &B1_Blocal_r2_i_init);

    complex_expr B1_r2_prop_0 =  B1_prop(t, iCprime, iSprime, src_color_weights(1, wnumBlock, 0), src_spin_weights(1, wnumBlock, 0), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_2 =  B1_prop(t, kCprime, kSprime, src_color_weights(1, wnumBlock, 2), src_spin_weights(1, wnumBlock, 2), x_out*sites_per_rank+x_in, y);
    complex_expr B1_r2_prop_1 = B1_prop(t, jCprime, jSprime, src_color_weights(1, wnumBlock, 1), src_spin_weights(1, wnumBlock, 1), x_out*sites_per_rank+x_in, y);



    complex_expr B1_r2_diquark = ( B1_r2_prop_0 * B1_r2_prop_2 ) *  src_weights(1, wnumBlock); // prop_prod_02 but for r2

    computation B1_Blocal_r2_r_props_init("B1_Blocal_r2_r_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((float) 0));
    computation B1_Blocal_r2_i_props_init("B1_Blocal_r2_i_props_init", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime}, expr((float) 0));

    computation B1_Blocal_r2_r_diquark("B1_Blocal_r2_r_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_real());
    computation B1_Blocal_r2_i_diquark("B1_Blocal_r2_i_diquark", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock}, B1_r2_diquark.get_imag());

    complex_computation B1_Blocal_r2_diquark(&B1_Blocal_r2_r_diquark, &B1_Blocal_r2_i_diquark);

    complex_expr B1_r2_props = B1_r2_prop_1 * B1_Blocal_r2_diquark(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock);

    computation B1_Blocal_r2_r_props("B1_Blocal_r2_r_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_r_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_real());
    computation B1_Blocal_r2_i_props("B1_Blocal_r2_i_props", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, wnumBlock, jCprime, jSprime}, B1_Blocal_r2_i_props_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime) + B1_r2_props.get_imag());



    complex_computation B1_Blocal_r2_props(&B1_Blocal_r2_r_props, &B1_Blocal_r2_i_props);

    complex_expr B1_r2 = src_psi_B1 * B1_Blocal_r2_props(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, Nw-1, jCprime, jSprime);

    computation B1_Blocal_r2_r_update("B1_Blocal_r2_r_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_r_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_real());
    computation B1_Blocal_r2_i_update("B1_Blocal_r2_i_update", {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, y, jCprime, jSprime, m}, B1_Blocal_r2_i_init(t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m) + B1_r2.get_imag());

    /* Correlator */

    computation C_init_r("C_init_r", {t, x_out, x_in, rp, m, r, n}, expr((float) 0));
    computation C_init_i("C_init_i", {t, x_out, x_in, rp, m, r, n}, expr((float) 0));

    computation C_init_r_cpu("C_init_r_cpu", {t, x_out, x_in, rp, m, r, n}, expr((float) 0));
    computation C_init_i_cpu("C_init_i_cpu", {t, x_out, x_in, rp, m, r, n}, expr((float) 0));

    computation out_C_init_r_cpu("out_C_init_r_cpu", {t, rp, m, r, n}, expr((float) 0));
    computation out_C_init_i_cpu("out_C_init_i_cpu", {t, rp, m, r, n}, expr((float) 0));

    computation C_prop_init_r("C_prop_init_r", {t, x_out, x_in, rp, m, r}, expr((float) 0));
    computation C_prop_init_i("C_prop_init_i", {t, x_out, x_in, rp, m, r}, expr((float) 0));
    
    int b=0;
    /* r1, b = 0 */
    complex_computation new_term_0_r1_b1("new_term_0_r1_b1", {t, x_out, x_in, rp, m, r, nperm, wnum}, B1_Blocal_r1_init(t, x_out, x_in, snk_color_weights(r, nperm, wnum, 0), snk_spin_weights(r, nperm, wnum, 0), snk_color_weights(r, nperm, wnum, 2), snk_spin_weights(r, nperm, wnum, 2), snk_color_weights(r, nperm, wnum, 1), snk_spin_weights(r, nperm, wnum, 1), m));
    new_term_0_r1_b1.add_predicate(src_spins(rp) == 1);
    /* r2, b = 0 */
    complex_computation new_term_0_r2_b1("new_term_0_r2_b1", {t, x_out, x_in, rp, m, r, nperm, wnum}, B1_Blocal_r2_init(t, x_out, x_in, snk_color_weights(r, nperm, wnum, 0), snk_spin_weights(r, nperm, wnum, 0), snk_color_weights(r, nperm, wnum, 2), snk_spin_weights(r, nperm, wnum, 2), snk_color_weights(r, nperm, wnum, 1), snk_spin_weights(r, nperm, wnum, 1), m));
    new_term_0_r2_b1.add_predicate(src_spins(rp) == 2);

    complex_expr prefactor(cast(p_float32, snk_weights(r, wnum))*cast(p_float32, sigs(nperm)), cast(p_float32, 0.0));

    complex_expr term_res_b1 = new_term_0_r1_b1(t, x_out, x_in, rp, m, r, nperm, wnum);

    complex_expr snk_psi(snk_psi_r(x_out*sites_per_rank+x_in, n), snk_psi_i(x_out*sites_per_rank+x_in, n));

    complex_expr term_res = prefactor * term_res_b1;

    computation C_prop_update_r("C_prop_update_r", {t, x_out, x_in, rp, m, r, nperm, wnum}, C_prop_init_r(t, x_out, x_in, rp, m, r) + term_res.get_real());
    computation C_prop_update_i("C_prop_update_i", {t, x_out, x_in, rp, m, r, nperm, wnum}, C_prop_init_i(t, x_out, x_in, rp, m, r) + term_res.get_imag());

    complex_computation C_prop_update(&C_prop_update_r, &C_prop_update_i);

    complex_expr term = C_prop_update(t, x_out, x_in, rp, m, r, B1Nperms-1, Nw-1) * snk_psi;

    computation C_update_r("C_update_r", {t, x_out, x_in, rp, m, r, n}, (expr)C_init_r(t, x_out, x_in, rp, m, r, n) + term.get_real());
    computation C_update_i("C_update_i", {t, x_out, x_in, rp, m, r, n}, (expr)C_init_i(t, x_out, x_in, rp, m, r, n) + term.get_imag());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    /* Correlator */

    // declaring buffers
    buffer out_buf_C_r_cpu("out_buf_C_r_cpu", {Lt, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);
    buffer out_buf_C_i_cpu("out_buf_C_i_cpu", {Lt, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);

    buffer buf_C_r("buf_C_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);
    buffer buf_C_i("buf_C_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);
    buffer buf_C_r_cpu("buf_C_r_cpu", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);
    buffer buf_C_i_cpu("buf_C_i_cpu", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, NsnkHex}, p_float32, a_temporary);

    out_C_r.store_in(&out_buf_C_r_cpu);
    out_C_i.store_in(&out_buf_C_i_cpu);
    buf_C_r.tag_gpu_global();
    buf_C_i.tag_gpu_global();

    C_init_r.store_in(&buf_C_r,   {t, x_out, x_in, rp, m, r, n});
    C_init_i.store_in(&buf_C_i,   {t, x_out, x_in, rp, m, r, n});
    C_init_r_cpu.store_in(&buf_C_r_cpu,   {t, x_out, x_in, rp, m, r, n});
    C_init_i_cpu.store_in(&buf_C_i_cpu,   {t, x_out, x_in, rp, m, r, n});
    out_C_init_r_cpu.store_in(&out_buf_C_r_cpu,   {t, rp, m, r, n});
    out_C_init_i_cpu.store_in(&out_buf_C_i_cpu,   {t, rp, m, r, n});
    C_update_r.store_in(&buf_C_r, {t, x_out, x_in, rp, m, r, n});
    C_update_i.store_in(&buf_C_i, {t, x_out, x_in, rp, m, r, n});

    computation reduce_buf_C_r_cpu("reduce_buf_C_r_cpu", {t, x_out, x_in, rp, m, r, n}, p_float32);
    reduce_buf_C_r_cpu.set_expression( reduce_buf_C_r_cpu( t, x_out, x_in, rp, m, r, n) + C_init_r_cpu( t, x_out, x_in, rp, m, r, n ) );
    computation reduce_buf_C_i_cpu("reduce_buf_C_i_cpu", {t, x_out, x_in, rp, m, r, n}, p_float32);
    reduce_buf_C_i_cpu.set_expression( reduce_buf_C_i_cpu( t, x_out, x_in, rp, m, r, n) + C_init_i_cpu( t, x_out, x_in, rp, m, r, n ) );
    reduce_buf_C_r_cpu.store_in( &out_buf_C_r_cpu, { t, rp, m, r, n } );
    reduce_buf_C_i_cpu.store_in( &out_buf_C_i_cpu, { t, rp, m, r, n } ); 

    buffer buf_B1_prop_r("buf_B1_prop_r",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Vsrc}, p_float32, a_temporary);
    buffer buf_B1_prop_i("buf_B1_prop_i",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Vsrc}, p_float32, a_temporary);
    buffer buf_src_psi_B1_r("buf_src_psi_B1_r",    {Vsrc, NsrcHex}, p_float32, a_temporary);
    buffer buf_src_psi_B1_i("buf_src_psi_B1_i",    {Vsrc, NsrcHex}, p_float32, a_temporary);
    buffer buf_snk_psi_r("buf_snk_psi_r", {Vsnk, NsnkHex}, p_float32, a_temporary);
    buffer buf_snk_psi_i("buf_snk_psi_i", {Vsnk, NsnkHex}, p_float32, a_temporary);
    buffer buf_src_color_weights("buf_src_color_weights", {B1Nrows, Nw, Nq}, p_int32, a_temporary);
    buffer buf_src_spin_weights("buf_src_spin_weights", {B1Nrows, Nw, Nq}, p_int32, a_temporary);
    buffer buf_src_weights("buf_src_weights", {B1Nrows, Nw}, p_float32, a_temporary);
    buffer buf_snk_color_weights("buf_snk_color_weights", {B1Nrows, B1Nperms, Nw, Nq}, p_int32, a_temporary);
    buffer buf_snk_spin_weights("buf_snk_spin_weights", {B1Nrows, B1Nperms, Nw, Nq}, p_int32, a_temporary);
    buffer buf_snk_weights("buf_snk_weights", {B1Nrows, Nw}, p_float32, a_temporary);
    // strange: needed to change buf_src_spins name from "buf_src_spins" to "src_spins" to work
    buffer buf_src_spins("src_spins", {B1Nrows}, p_int32, a_temporary);
    buffer buf_sigs("buf_sigs", {B1Nperms}, p_int32, a_temporary);

    B1_prop_r.store_in(&buf_B1_prop_r);
    B1_prop_i.store_in(&buf_B1_prop_i);
    src_psi_B1_r.store_in(&buf_src_psi_B1_r);
    src_psi_B1_i.store_in(&buf_src_psi_B1_i);
    snk_psi_r.store_in(&buf_snk_psi_r);
    snk_psi_i.store_in(&buf_snk_psi_i);
    src_color_weights.store_in(&buf_src_color_weights, {rp, wnum, q});
    src_spin_weights.store_in(&buf_src_spin_weights, {rp, wnum, q});
    src_weights.store_in(&buf_src_weights);
    snk_color_weights.store_in(&buf_snk_color_weights, {r, nperm, wnum, q});
    snk_spin_weights.store_in(&buf_snk_spin_weights, {r, nperm, wnum, q});
    snk_weights.store_in(&buf_snk_weights);
    src_spins.store_in(&buf_src_spins);
    sigs.store_in(&buf_sigs);


   buf_B1_prop_r.tag_gpu_global();
   buf_B1_prop_i.tag_gpu_global();
   buf_src_psi_B1_r.tag_gpu_global();
   buf_src_psi_B1_i.tag_gpu_global();
   buf_snk_psi_r.tag_gpu_global();
   buf_snk_psi_i.tag_gpu_global();
   buf_src_color_weights.tag_gpu_global();
   buf_src_spin_weights.tag_gpu_global();
   buf_src_weights.tag_gpu_global();
   buf_snk_color_weights.tag_gpu_global();
   buf_snk_spin_weights.tag_gpu_global();
   buf_snk_weights.tag_gpu_global();
   buf_src_spins.tag_gpu_global();
   buf_sigs.tag_gpu_global();

    buffer buf_new_term_r_b1( "buf_new_term_r_b1", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, B1Nperms, Nw}, p_float32, a_temporary );
    buffer buf_new_term_i_b1( "buf_new_term_i_b1", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows, B1Nperms, Nw}, p_float32, a_temporary );

    buffer buf_C_prop_r("buf_C_prop_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows}, p_float32, a_temporary);
    buffer buf_C_prop_i("buf_C_prop_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, B1Nrows, NsrcHex, B1Nrows}, p_float32, a_temporary);
   buf_C_prop_r.tag_gpu_global();
   buf_C_prop_i.tag_gpu_global();

    buffer B1_prop_r_cpu("B1_prop_r_cpu",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Vsrc}, p_float32, a_temporary);
    buffer B1_prop_i_cpu("B1_prop_i_cpu",   {Lt, Nc, Ns, Nc, Ns, Vsnk, Vsrc}, p_float32, a_temporary);
    buffer src_psi_B1_r_cpu("src_psi_B1_r_cpu",    {Vsrc, NsrcHex}, p_float32, a_temporary);
    buffer src_psi_B1_i_cpu("src_psi_B1_i_cpu",    {Vsrc, NsrcHex}, p_float32, a_temporary);
    buffer snk_psi_r_cpu("snk_psi_r_cpu", {Vsnk, NsnkHex}, p_float32, a_temporary);
    buffer snk_psi_i_cpu("snk_psi_i_cpu", {Vsnk, NsnkHex}, p_float32, a_temporary);
    buffer src_color_weights_cpu("src_color_weights_cpu", {B1Nrows, Nw, Nq}, p_int32, a_temporary);
    buffer src_spin_weights_cpu("src_spin_weights_cpu", {B1Nrows, Nw, Nq}, p_int32, a_temporary);
    buffer src_weights_cpu("src_weights_cpu", {B1Nrows, Nw}, p_float32, a_temporary);
    buffer snk_color_weights_cpu("snk_color_weights_cpu", {B1Nrows, B1Nperms, Nw, Nq}, p_int32, a_temporary);
    buffer snk_spin_weights_cpu("snk_spin_weights_cpu", {B1Nrows, B1Nperms, Nw, Nq}, p_int32, a_temporary);
    buffer snk_weights_cpu("snk_weights_cpu", {B1Nrows, Nw}, p_float32, a_temporary);
    buffer src_spins_cpu("src_spins_cpu", {B1Nrows}, p_int32, a_temporary);
    buffer sigs_cpu("sigs_cpu", {B1Nperms}, p_int32, a_temporary);

    buffer buf_B1_Blocal_r1_r("buf_B1_Blocal_r1_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float32, a_temporary);
    buffer buf_B1_Blocal_r1_i("buf_B1_Blocal_r1_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float32, a_temporary);
    buffer buf_B1_Blocal_diquark_r1_r("buf_B1_Blocal_diquark_r1_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, 1}, p_float32, a_temporary);
    buffer buf_B1_Blocal_diquark_r1_i("buf_B1_Blocal_diquark_r1_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, 1}, p_float32, a_temporary);
    
    buffer buf_B1_Blocal_props_r1_r("buf_B1_Blocal_props_r1_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns}, p_float32, a_temporary);
    buffer buf_B1_Blocal_props_r1_i("buf_B1_Blocal_props_r1_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns}, p_float32, a_temporary);

    buffer buf_B1_Blocal_r2_r("buf_B1_Blocal_r2_r",   {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float32, a_temporary);
    buffer buf_B1_Blocal_r2_i("buf_B1_Blocal_r2_i",   {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns, Nc, Ns, Nc, Ns, NsrcHex}, p_float32, a_temporary);
    buffer buf_B1_Blocal_diquark_r2_r("buf_B1_Blocal_diquark_r2_r", {Lt, Vsnk/sites_per_rank, sites_per_rank, 1}, p_float32, a_temporary);
    buffer buf_B1_Blocal_diquark_r2_i("buf_B1_Blocal_diquark_r2_i", {Lt, Vsnk/sites_per_rank, sites_per_rank, 1}, p_float32, a_temporary);
    
    buffer buf_B1_Blocal_props_r2_r("buf_B1_Blocal_props_r2_r",   {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns}, p_float32, a_temporary);
    buffer buf_B1_Blocal_props_r2_i("buf_B1_Blocal_props_r2_i",   {Lt, Vsnk/sites_per_rank, sites_per_rank, Nc, Ns}, p_float32, a_temporary);

    buf_new_term_r_b1.tag_gpu_global();
    buf_new_term_i_b1.tag_gpu_global();

    buf_C_prop_r.tag_gpu_global();
    buf_C_prop_i.tag_gpu_global();


    buf_B1_Blocal_r1_r.tag_gpu_global();
    buf_B1_Blocal_r1_i.tag_gpu_global();

    buf_B1_Blocal_diquark_r1_r.tag_gpu_global();
    buf_B1_Blocal_diquark_r1_i.tag_gpu_global();

    buf_B1_Blocal_props_r1_r.tag_gpu_global();
    buf_B1_Blocal_props_r1_i.tag_gpu_global();

    buf_B1_Blocal_r2_r.tag_gpu_global();
    buf_B1_Blocal_r2_i.tag_gpu_global();

    buf_B1_Blocal_diquark_r2_r.tag_gpu_global();
    buf_B1_Blocal_diquark_r2_i.tag_gpu_global();

    buf_B1_Blocal_props_r2_r.tag_gpu_global();
    buf_B1_Blocal_props_r2_i.tag_gpu_global();

    // {t, x_out, x_in, rp, m, r, nperm, wnum}
    new_term_0_r1_b1.get_real()->store_in(&buf_new_term_r_b1, {t, x_out, x_in, rp, m, r, nperm, wnum});
    new_term_0_r1_b1.get_imag()->store_in(&buf_new_term_i_b1, {t, x_out, x_in, rp, m, r, nperm, wnum});

    new_term_0_r2_b1.get_real()->store_in(&buf_new_term_r_b1, {t, x_out, x_in, rp, m, r, nperm, wnum});
    new_term_0_r2_b1.get_imag()->store_in(&buf_new_term_i_b1, {t, x_out, x_in, rp, m, r, nperm, wnum});

    C_prop_init_r.store_in(&buf_C_prop_r, {t, x_out, x_in, rp, m, r});
    C_prop_init_i.store_in(&buf_C_prop_i, {t, x_out, x_in, rp, m, r});
    C_prop_update_r.store_in(&buf_C_prop_r, {t, x_out, x_in, rp, m, r});
    C_prop_update_i.store_in(&buf_C_prop_i, {t, x_out, x_in, rp, m, r});

    B1_Blocal_r1_r_init.store_in(&buf_B1_Blocal_r1_r, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_init.store_in(&buf_B1_Blocal_r1_i, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_r_update.store_in(&buf_B1_Blocal_r1_r, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r1_i_update.store_in(&buf_B1_Blocal_r1_i, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});

    B1_Blocal_r1_r_diquark.store_in(&buf_B1_Blocal_diquark_r1_r, {t, x_out, x_in, 0});
    B1_Blocal_r1_i_diquark.store_in(&buf_B1_Blocal_diquark_r1_i, {t, x_out, x_in, 0});

    B1_Blocal_r1_r_props_init.store_in(&buf_B1_Blocal_props_r1_r, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r1_i_props_init.store_in(&buf_B1_Blocal_props_r1_i, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r1_r_props.store_in(&buf_B1_Blocal_props_r1_r, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r1_i_props.store_in(&buf_B1_Blocal_props_r1_i, {t, x_out, x_in, jCprime, jSprime});

    B1_Blocal_r2_r_init.store_in(&buf_B1_Blocal_r2_r, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_init.store_in(&buf_B1_Blocal_r2_i, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_r_update.store_in(&buf_B1_Blocal_r2_r, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});
    B1_Blocal_r2_i_update.store_in(&buf_B1_Blocal_r2_i, {t, x_out, x_in, iCprime, iSprime, kCprime, kSprime, jCprime, jSprime, m});

    B1_Blocal_r2_r_diquark.store_in(&buf_B1_Blocal_diquark_r2_r, {t, x_out, x_in, 0});
    B1_Blocal_r2_i_diquark.store_in(&buf_B1_Blocal_diquark_r2_i, {t, x_out, x_in, 0});

    B1_Blocal_r2_r_props_init.store_in(&buf_B1_Blocal_props_r2_r, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r2_i_props_init.store_in(&buf_B1_Blocal_props_r2_i, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r2_r_props.store_in(&buf_B1_Blocal_props_r2_r, {t, x_out, x_in, jCprime, jSprime});
    B1_Blocal_r2_i_props.store_in(&buf_B1_Blocal_props_r2_i, {t, x_out, x_in, jCprime, jSprime});


    computation copy_buf_C_r_host_to_device({}, memcpy(buf_C_r_cpu, buf_C_r));
    computation copy_buf_C_i_host_to_device({}, memcpy(buf_C_i_cpu, buf_C_i));
    computation copy_B1_prop_r_host_to_device({}, memcpy(B1_prop_r_cpu, buf_B1_prop_r));
    computation copy_B1_prop_i_host_to_device({}, memcpy(B1_prop_i_cpu, buf_B1_prop_i));
    computation copy_src_psi_B1_r_host_to_device({}, memcpy(src_psi_B1_r_cpu, buf_src_psi_B1_r));
    computation copy_src_psi_B1_i_host_to_device({}, memcpy(src_psi_B1_i_cpu, buf_src_psi_B1_i));
    computation copy_snk_psi_r_host_to_device({}, memcpy(snk_psi_r_cpu, buf_snk_psi_r));
    computation copy_snk_psi_i_host_to_device({}, memcpy(snk_psi_i_cpu, buf_snk_psi_i));
    computation copy_src_color_weights_host_to_device({}, memcpy(src_color_weights_cpu, buf_src_color_weights));
    computation copy_src_spin_weights_host_to_device({}, memcpy(src_spin_weights_cpu, buf_src_spin_weights));
    computation copy_src_weights_host_to_device({}, memcpy(src_weights_cpu, buf_src_weights));
    computation copy_src_spins_host_to_device({}, memcpy(src_spins_cpu, buf_src_spins));
    computation copy_snk_color_weights_host_to_device({}, memcpy(snk_color_weights_cpu, buf_snk_color_weights));
    computation copy_snk_spin_weights_host_to_device({}, memcpy(snk_spin_weights_cpu, buf_snk_spin_weights));
    computation copy_snk_weights_host_to_device({}, memcpy(snk_weights_cpu, buf_snk_weights));
    computation copy_sigs_host_to_device({}, memcpy(sigs_cpu, buf_sigs));

    computation copy_buf_C_r_device_to_host({}, memcpy(buf_C_r, buf_C_r_cpu));
    computation copy_buf_C_i_device_to_host({}, memcpy(buf_C_i, buf_C_i_cpu));

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

#if GPU_PARALLEL

// kernel_0
    C_init_r.tag_gpu_level(x_out, x_in);
    C_init_i.tag_gpu_level(x_out, x_in);

// kernel_1
    B1_Blocal_r1_r_init.tag_gpu_level(x_out, x_in);
    B1_Blocal_r1_i_init.tag_gpu_level(x_out, x_in);

// kernel_2
    B1_Blocal_r1_r_props_init.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);
    B1_Blocal_r1_i_props_init.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);

    B1_Blocal_r1_r_diquark.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);
    B1_Blocal_r1_i_diquark.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);

    B1_Blocal_r1_r_props.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);
    B1_Blocal_r1_i_props.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);

    B1_Blocal_r1_r_update.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);
    B1_Blocal_r1_i_update.tag_gpu_level(x_out, x_in);//, iCprime, iSprime, kCprime, kSprime);

// kernel_3

    B1_Blocal_r2_r_init.tag_gpu_level(x_out, x_in);
    B1_Blocal_r2_i_init.tag_gpu_level(x_out, x_in);

    B1_Blocal_r2_r_props_init.tag_gpu_level(x_out, x_in);
    B1_Blocal_r2_i_props_init.tag_gpu_level(x_out, x_in);

    B1_Blocal_r2_r_diquark.tag_gpu_level(x_out, x_in);
    B1_Blocal_r2_i_diquark.tag_gpu_level(x_out, x_in);

    B1_Blocal_r2_r_props.tag_gpu_level(x_out, x_in);
    B1_Blocal_r2_i_props.tag_gpu_level(x_out, x_in);

    B1_Blocal_r2_r_update.tag_gpu_level(x_out, x_in);
    B1_Blocal_r2_i_update.tag_gpu_level(x_out, x_in);

// kernel_4
    C_prop_init_r.tag_gpu_level(x_out, x_in);
    C_prop_init_i.tag_gpu_level(x_out, x_in);

    new_term_0_r1_b1.get_real()->tag_gpu_level(x_out, x_in);
    new_term_0_r1_b1.get_imag()->tag_gpu_level(x_out, x_in);

    new_term_0_r2_b1.get_real()->tag_gpu_level(x_out, x_in);
    new_term_0_r2_b1.get_imag()->tag_gpu_level(x_out, x_in);

    C_prop_update_r.tag_gpu_level(x_out, x_in);
    C_prop_update_r.tag_gpu_level(x_out, x_in);
    C_prop_update_i.tag_gpu_level(x_out, x_in);

    C_update_r.tag_gpu_level(x_out, x_in);
    C_update_i.tag_gpu_level(x_out, x_in);

#endif
    computation *handle = &copy_buf_C_r_host_to_device.then(copy_buf_C_i_host_to_device, computation::root);


    handle = &(handle->
            then(copy_B1_prop_r_host_to_device, computation::root)
            .then(copy_B1_prop_i_host_to_device, computation::root)
            .then(copy_src_psi_B1_r_host_to_device, computation::root)
            .then(copy_src_psi_B1_i_host_to_device, computation::root)
            .then(copy_snk_psi_r_host_to_device, computation::root)
            .then(copy_snk_psi_i_host_to_device, computation::root)
            .then(copy_src_color_weights_host_to_device, computation::root)
            .then(copy_src_spin_weights_host_to_device, computation::root)
            .then(copy_src_weights_host_to_device, computation::root)
            .then(copy_src_spins_host_to_device, computation::root)
            .then(copy_snk_color_weights_host_to_device, computation::root)
            .then(copy_snk_spin_weights_host_to_device, computation::root)
            .then(copy_snk_weights_host_to_device, computation::root)
            .then(copy_sigs_host_to_device, computation::root));

    handle = &(handle->then(C_init_r, computation::root).then(C_init_i, n));
    handle = &(handle->then(C_init_r_cpu, computation::root).then(C_init_i_cpu, n));
    handle = &(handle->then(out_C_init_r_cpu, computation::root).then(out_C_init_i_cpu, n));

    // // first the x only arrays
    handle = &(handle
        ->then(B1_Blocal_r1_r_init, computation::root)
        .then(B1_Blocal_r1_i_init, jSprime)
        //
        .then(B1_Blocal_r1_r_props_init, computation::root)
        .then(B1_Blocal_r1_i_props_init, jSprime)
        .then(B1_Blocal_r1_r_diquark, y)
        .then(B1_Blocal_r1_i_diquark, wnumBlock)
        .then(B1_Blocal_r1_r_props, wnumBlock)
        .then(B1_Blocal_r1_i_props, jSprime)
        .then(B1_Blocal_r1_r_update, y)
        .then(B1_Blocal_r1_i_update, m)
        //
        .then(B1_Blocal_r2_r_init, computation::root)
        .then(B1_Blocal_r2_i_init, jSprime)
        //
        .then(B1_Blocal_r2_r_props_init, computation::root)
        .then(B1_Blocal_r2_i_props_init, jSprime)
        .then(B1_Blocal_r2_r_diquark, y)
        .then(B1_Blocal_r2_i_diquark, wnumBlock)
        .then(B1_Blocal_r2_r_props, wnumBlock)
        .then(B1_Blocal_r2_i_props, jSprime)
        .then(B1_Blocal_r2_r_update, y)
        .then(B1_Blocal_r2_i_update, m)
        );


    handle = &(handle 
          ->then(C_prop_init_r, computation::root) 
          .then(C_prop_init_i, r)
          .then( *(new_term_0_r1_b1.get_real()), r)
          .then( *(new_term_0_r1_b1.get_imag()), wnum)
          .then( *(new_term_0_r2_b1.get_real()), wnum)
          .then( *(new_term_0_r2_b1.get_imag()), wnum)
          .then(C_prop_update_r, wnum) 
          .then(C_prop_update_i, wnum)
          .then(C_update_r, r) 
          .then(C_update_i, n));

    handle = &(handle->
            then(copy_buf_C_r_device_to_host, computation::root)
            .then(copy_buf_C_i_device_to_host, computation::root))
            ;

    handle = &handle->then( reduce_buf_C_r_cpu, x_out ).then( reduce_buf_C_i_cpu, n );

#if VECTORIZED

#endif

#if PARALLEL

    C_init_r.tag_distribute_level(t);

    B1_Blocal_r1_r_init.tag_distribute_level(t);
    B1_Blocal_r2_r_init.tag_distribute_level(t);

    C_prop_init_r.tag_distribute_level(t);

#endif
    // -------------------------------------------------------
    // Code Generation
    // -----------------------------------kernel_0<<--------------------
    tiramisu::codegen({
        &out_buf_C_r_cpu, &out_buf_C_i_cpu,
        &buf_C_r_cpu, &buf_C_i_cpu,
        &B1_prop_r_cpu, &B1_prop_i_cpu,
        &src_psi_B1_r_cpu, &src_psi_B1_i_cpu,
        &snk_psi_r_cpu, &snk_psi_i_cpu,
        &src_color_weights_cpu,
        &src_spin_weights_cpu,
        &src_weights_cpu,
        &src_spins_cpu, 
        &snk_color_weights_cpu,
        &snk_spin_weights_cpu,
        &snk_weights_cpu,
        &sigs_cpu,
        }, 
        "generated_gpu_single_tiramisu_make_fused_identical_baryon_blocks_correlator.o", true);
}

int main(int argc, char **argv)
{

    generate_function("gpu_single_tiramisu_make_fused_identical_baryon_blocks_correlator");

    return 0;
}
