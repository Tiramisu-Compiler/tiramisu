#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare loop iterators
    var i("i", 0, M), j("j", 0, N), k("k", 0, K), i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0", 0, (K - 1) / BLOCK + 1), k1("k1");

    // Declare cpu buffers.
    buffer b_A("b_A", {M, K}, DATA_PTYPE, a_input);
    buffer b_B("b_B", {K, N}, DATA_PTYPE, a_input);
    buffer b_C("b_C", {M, N}, DATA_PTYPE, a_output);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {M, K}, DATA_PTYPE, a_temporary);
    buffer b_B_gpu("b_B_gpu", {K, N}, DATA_PTYPE, a_temporary);
    buffer b_C_gpu("b_C_gpu", {M, N}, DATA_PTYPE, a_temporary);
    buffer b_A_gpu_tile("b_A_gpu_tile", {BLOCK, BLOCK}, DATA_PTYPE, a_temporary);
    buffer b_B_gpu_tile("b_B_gpu_tile", {BLOCK, BLOCK}, DATA_PTYPE, a_temporary);
    buffer b_acc("b_acc", {1}, DATA_PTYPE, a_temporary);
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();
    b_A_gpu_tile.tag_gpu_shared();
    b_B_gpu_tile.tag_gpu_shared();
    b_acc.tag_gpu_register();

    // Declare wrappers for input buffers
    input c_A("c_A", {i, j, k0}, DATA_PTYPE);
    input c_B("c_B", {i, j, k0}, DATA_PTYPE);
    // Declare wrappers for shared arrays
    input c_A_tile("c_A_tile", {i, j}, DATA_PTYPE);
    computation c_A_tile_dec({i, j}, allocate(b_A_gpu_tile));
    computation c_A_tile_init("c_A_tile_init", {i, j, k0}, c_A(i, j, k0));
    input c_B_tile("c_B_tile", {i, j}, DATA_PTYPE);
    computation c_B_tile_dec({i, j}, allocate(b_B_gpu_tile));
    computation c_B_tile_init("c_B_tile_init", {i, j, k0}, c_B(i, j, k0));
    // Declare synchronizer computations
    computation sync1({i, j, k0}, tiramisu::sync());
    computation sync2({i, j, k0}, tiramisu::sync());
    // Declare wrapper computation for accumulator
    computation c_acc_dec({i, j}, allocate(b_acc));
    computation c_acc_init({i, j}, (DATA_TYPE) 0, DATA_PTYPE);
    computation c_acc({i, j, k}, DATA_PTYPE);
    c_acc.set_expression(c_acc(i, j, k - 1) + c_A_tile(i, k) * c_B_tile(k, j));
    // Declare wrapper for the output buffer
    computation c_C({i, j}, DATA_PTYPE);
    c_C.set_expression(c_acc(i, j, 0) * alpha + c_C(i, j) * beta);
    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, b_A_gpu));
    computation copy_B_to_device({}, memcpy(b_B, b_B_gpu));
    computation copy_C_to_device({}, memcpy(b_C, b_C_gpu));
    computation copy_C_to_host({}, memcpy(b_C_gpu, b_C));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands

    c_acc.split(k, BLOCK, k0, k1);
    c_C.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    sync1.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    sync2.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);

    copy_B_to_device.after(copy_A_to_device, computation::root);
    copy_C_to_device.after(copy_B_to_device, computation::root);
    c_A_tile_dec.after(copy_C_to_device, computation::root);
    c_B_tile_dec.after(c_A_tile_dec, j1);
    c_acc_dec.after(c_B_tile_dec, j1);
    c_acc_init.after(c_acc_dec, j1);
    c_A_tile_init.after(c_acc_init, j1);
    c_B_tile_init.after(c_A_tile_init, k0);
    sync1.after(c_B_tile_init, k0);
    c_acc.after(sync1, k0);
    sync2.after(c_acc, k0);
    c_C.after(sync2, j1);
    copy_C_to_host.after(c_C, computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    c_A.store_in(&b_A_gpu, {i, k0 * BLOCK + j % BLOCK});
    c_B.store_in(&b_B_gpu, {k0 * BLOCK + i % BLOCK, j});
    c_A_tile.store_in(&b_A_gpu_tile, {i % BLOCK, j % BLOCK});
    c_A_tile_init.store_in(&b_A_gpu_tile, {i % BLOCK, j % BLOCK});
    // Note the transpose
    c_B_tile.store_in(&b_B_gpu_tile, {j % BLOCK, i % BLOCK});
    c_B_tile_init.store_in(&b_B_gpu_tile, {j % BLOCK, i % BLOCK});

    c_acc_init.store_in(&b_acc, {});
    c_acc.store_in(&b_acc, {});

    c_C.store_in(&b_C_gpu);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({&b_A, &b_B, &b_C}, "fct.o", true);
    
    return 0;
}
