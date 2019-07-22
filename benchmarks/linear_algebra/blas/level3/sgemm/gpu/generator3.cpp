#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // cache_shared

    tiramisu::init("matmul");

    int t_block = 16;  // Thread block
    int k_block = 4;  // K reduction split
    int r_block_i = 16; // Register blocks
    int r_block_j = 8;

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare loop iterators
    var i("i", 0, M), j("j", 0, N), k("k", 0, K);
    var i0("i0", 0, M / r_block_i), i1("i1", 0, r_block_i);
    var j0("j0", 0, N / r_block_j), j1("j1", 0, r_block_j);
    var i00("i00"), i01("i01");
    var j00("j00"), j01("j01");
    var k0("k0"), k1("k1");

    // Declare input wrappers
    input c_A({i, k}, DATA_PTYPE);
    input c_B({k, j}, DATA_PTYPE);
    c_A.get_buffer()->tag_gpu_global();
    c_B.get_buffer()->tag_gpu_global();
    input c_Consts({i}, DATA_PTYPE);
    constant c_alpha("alpha", c_Consts(0));
    constant c_beta("beta", c_Consts(1));

    // Declare computations
    // Accumulator
    computation c_acc_init({i, j}, (float) 0);
    c_acc_init.store_in({i % r_block_i, j % r_block_j}, {r_block_i, r_block_j});
    c_acc_init.get_buffer()->tag_gpu_local();
    computation c_acc({i, j, k}, DATA_PTYPE);
    c_acc.store_in(c_acc_init.get_buffer(), {i % r_block_i, j % r_block_j});
    c_acc.set_expression(c_acc(i, j, 0) + c_A(i, k) * c_B(k, j));
    // Output
    computation c_C({i, j}, DATA_PTYPE);
    c_C.get_buffer()->tag_gpu_global();
    c_C.set_expression(c_acc(i, j, 0) * c_alpha + c_C(i, j) * c_beta);
    computation c_acc_dec({i0, j0}, allocate(*c_acc_init.get_buffer()));

    // Declare cpu buffers.
    buffer b_A("b_A", {M, K}, DATA_PTYPE, a_input);
    buffer b_B("b_B", {K, N}, DATA_PTYPE, a_input);
    buffer b_C("b_C", {M, N}, DATA_PTYPE, a_output);

    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, *c_A.get_buffer()));
    computation copy_B_to_device({}, memcpy(b_B, *c_B.get_buffer()));
    computation copy_C_to_device({}, memcpy(b_C, *c_C.get_buffer()));
    computation copy_C_to_host({}, memcpy(*c_C.get_buffer(), b_C));

    // External call to timer function to find GPU time
    computation time_start({var("dummy", 0, 1)}, expr(o_call, "get_time", {int32_t(0)}, p_float32));
    computation time_end({var("dummy", 0, 1)}, expr(o_call, "get_time", {int32_t(0)}, p_float32));
    // Synchronize before timing
    computation final_sync({var("dummy", 0, 1)}, cuda_stream_synchronize());

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands

    block({&c_acc_init, &c_acc, &c_C}).tile(i, j, r_block_i, r_block_j, i0, j0, i1, j1);
    c_acc.interchange(j1, k);
    c_acc.interchange(i1, k);
    c_acc.split(k, k_block, k0, k1);

    block kernel_block({&c_acc_dec, &c_acc_init, &c_acc, &c_C});
    kernel_block.gpu_tile(i0, j0, t_block, t_block, i00, j00, i01, j01);

    copy_A_to_device.then(copy_B_to_device)
                    .then(copy_C_to_device)
                    .then(time_start)
                    .then(c_acc_dec)
                    .then(c_acc_init, j01)
                    .then(c_acc, j01)
                    .then(c_C, j01)
                    .then(final_sync)
                    .then(time_end)
                    .then(copy_C_to_host);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Use shared memory for global memory access
    c_acc.cache_shared(c_A, k0, {t_block * r_block_i, k_block}, {i00 * t_block * r_block_i, k0 * k_block}, true);
    c_acc.cache_shared(c_B, k0, {k_block, t_block * r_block_j}, {k0 * k_block, j00 * t_block * r_block_j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({c_Consts.get_buffer(), &b_A, &b_B, &b_C, time_start.get_buffer(), time_end.get_buffer()}, "fct.o", true);

    return 0;
}
