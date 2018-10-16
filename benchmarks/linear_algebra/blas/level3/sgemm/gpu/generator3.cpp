#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Double tiling with Register and Shared memory
    // Fused A_reg and non-square tiling

    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare loop iterators
    var i("i", 0, M), j("j", 0, N), k("k", 0, K);
    var i0("i0", 0, M / R_BLOCK_I), i1("i1", 0, R_BLOCK_I), j0("j0", 0, N / R_BLOCK_J), j1("j1", 0, R_BLOCK_J), k0("k0", 0, K / BLOCK), k1("k1", 0, BLOCK);
    var k0_skiplast("k0", 0, K / BLOCK - 1);
    var i00("i00"), i01("i01"), j00("j00"), j01("j01");

    // Declare cpu buffers.
    buffer b_A("b_A", {M, K}, DATA_PTYPE, a_input);
    buffer b_B("b_B", {K, N}, DATA_PTYPE, a_input);
    buffer b_C("b_C", {M, N}, DATA_PTYPE, a_output);
    buffer b_Consts("b_Consts", {2}, DATA_PTYPE, a_input);
    // Declare gpu buffers.
    buffer b_A_glb("b_A_glb", {M, K}, DATA_PTYPE, a_temporary);
    buffer b_B_glb("b_B_glb", {K, N}, DATA_PTYPE, a_temporary);
    buffer b_C_glb("b_C_glb", {M, N}, DATA_PTYPE, a_temporary);
    // "+ 1" to reduce shared memory bank conflicts
    buffer b_A_shr("b_A_shr", {2, BLOCK, BLOCK * R_BLOCK_I + 1}, DATA_PTYPE, a_temporary);
    buffer b_B_shr("b_B_shr", {2, BLOCK, BLOCK * R_BLOCK_J}, DATA_PTYPE, a_temporary);
    buffer b_A_reg("b_A_reg", {1}, DATA_PTYPE, a_temporary);
    buffer b_B_reg("b_B_reg", {R_BLOCK_J}, DATA_PTYPE, a_temporary);
    buffer b_acc("b_acc", {R_BLOCK_I, R_BLOCK_J}, DATA_PTYPE, a_temporary);
    b_A_glb.tag_gpu_global();
    b_B_glb.tag_gpu_global();
    b_C_glb.tag_gpu_global();
    b_A_shr.tag_gpu_shared();
    b_B_shr.tag_gpu_shared();
    b_A_reg.tag_gpu_register();
    b_B_reg.tag_gpu_local();
    b_acc.tag_gpu_local();


    // Declare input wrappers
    input c_A_glb({i0, j0, k0, i1}, DATA_PTYPE);
    input c_A_shr({i0, j0, k0, k1, i1}, DATA_PTYPE);
    input c_A({i, k}, DATA_PTYPE);
    input c_B_glb({i0, j0, k0, j1}, DATA_PTYPE);
    input c_B_shr({i0, j0, k0, k1, j1}, DATA_PTYPE);
    input c_B({k, j}, DATA_PTYPE);
    input c_Consts({i}, DATA_PTYPE);
    constant c_alpha("alpha", c_Consts(0));
    constant c_beta("beta", c_Consts(1));
    // Declare computations
    computation c_A_glb_to_shr_pre({i0, j0, i1}, c_A_glb(i0, j0, 0, i1));
    computation c_A_glb_to_shr({i0, j0, k0_skiplast, i1}, c_A_glb(i0, j0, k0_skiplast + 1, i1));
    computation c_A_shr_to_reg({i0, j0, k0, k1, i1}, c_A_shr(i0, j0, k0, k1, i1));
    computation c_B_glb_to_shr_pre({i0, j0, j1}, c_B_glb(i0, j0, 0, j1));
    computation c_B_glb_to_shr({i0, j0, k0_skiplast, j1}, c_B_glb(i0, j0, k0_skiplast + 1, j1));
    computation c_B_shr_to_reg({i0, j0, k0, k1, j1}, c_B_shr(i0, j0, k0, k1, j1));
    computation c_acc_init({i, j}, (float) 0);
    computation c_acc({i, j, k}, DATA_PTYPE);
    c_acc.set_expression(c_acc(i, j, 0) + c_A(i, k) * c_B(k, j));
    computation c_C({i, j}, DATA_PTYPE);
    c_C.set_expression(c_acc(i, j, 0) * c_alpha + c_C(i, j) * c_beta);
    // Declare declarations
    computation c_A_shr_dec({i0, j0}, allocate(b_A_shr));
    computation c_A_reg_dec({i0, j0}, allocate(b_A_reg));
    computation c_B_shr_dec({i0, j0}, allocate(b_B_shr));
    computation c_B_reg_dec({i0, j0}, allocate(b_B_reg));
    computation c_acc_dec({i0, j0}, allocate(b_acc));
    // Declare synchronizer computations
    computation c_sync1({i0, j0}, tiramisu::sync());
    computation c_sync2({i0, j0, k0}, tiramisu::sync());
    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, b_A_glb));
    computation copy_B_to_device({}, memcpy(b_B, b_B_glb));
    computation copy_C_to_device({}, memcpy(b_C, b_C_glb));
    computation copy_C_to_host({}, memcpy(b_C_glb, b_C));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands

    c_A_shr_dec.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_A_reg_dec.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_acc_dec.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_A_glb_to_shr_pre.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_A_glb_to_shr.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_A_shr_to_reg.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_B_shr_dec.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_B_reg_dec.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_B_glb_to_shr_pre.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_B_glb_to_shr.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_B_shr_to_reg.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_acc_init.tile(i, j, R_BLOCK_I, R_BLOCK_J, i0, j0, i1, j1);
    c_acc_init.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_acc.tile(i, j, R_BLOCK_I, R_BLOCK_J, i0, j0, i1, j1);
    c_acc.interchange(j1, k);
    c_acc.interchange(i1, k);
    c_acc.split(k, BLOCK, k0, k1);
    c_acc.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_C.tile(i, j, R_BLOCK_I, R_BLOCK_J, i0, j0, i1, j1);
    c_C.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_sync1.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);
    c_sync2.gpu_tile(i0, j0, BLOCK, BLOCK, i00, j00, i01, j01);

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(c_A_shr_dec, computation::root)
                    .then(c_B_shr_dec, j01)
                    .then(c_A_reg_dec, j01)
                    .then(c_B_reg_dec, j01)
                    .then(c_acc_dec, j01)
                    .then(c_acc_init, j01)
                    .then(c_A_glb_to_shr_pre, j01)
                    .then(c_B_glb_to_shr_pre, j01)
                    .then(c_sync1, j01)
                    .then(c_A_glb_to_shr, j01)
                    .then(c_B_glb_to_shr, k0)
                    .then(c_B_shr_to_reg, k0)
                    .then(c_A_shr_to_reg, k1)
                    .then(c_acc, i1)
                    .then(c_sync2, k0)
                    .then(c_C, j01)
                    .then(copy_C_to_host, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    c_A_glb.store_in(&b_A_glb, {(i0 - i0 % BLOCK + j0 % BLOCK) * R_BLOCK_I + i1, k0 * BLOCK + i0 % BLOCK});
    // Note the transpose:
    c_A_glb_to_shr_pre.store_in(&b_A_shr, {0, i0 % BLOCK, j0 % BLOCK * R_BLOCK_I + i1});
    c_A_glb_to_shr.store_in(&b_A_shr, {(k0_skiplast + 1) % 2, i0 % BLOCK, j0 % BLOCK * R_BLOCK_I + i1});
    c_A_shr.store_in(&b_A_shr, {k0 % 2, k1, i0 % BLOCK * R_BLOCK_I + i1});
    c_A_shr_to_reg.store_in(&b_A_reg, {0});
    // Note that we use a transposed mapping to assure memory coalescing
    // This requires R_BLOCK_J to be equal to BLOCK
    c_B_glb.store_in(&b_B_glb, {k0 * BLOCK + j1, j0 * R_BLOCK_J + i0 % BLOCK});
    c_B_glb_to_shr_pre.store_in(&b_B_shr, {0, j1, j0 % BLOCK * R_BLOCK_J + i0 % BLOCK});
    c_B_glb_to_shr.store_in(&b_B_shr, {(k0_skiplast + 1) % 2, j1, j0 % BLOCK * R_BLOCK_J + i0 % BLOCK});
    c_B_shr.store_in(&b_B_shr, {k0 % 2, k1, j0 % BLOCK * R_BLOCK_J + j1});
    c_B_shr_to_reg.store_in(&b_B_reg, {j1});
    c_A.store_in(&b_A_reg, {i % R_BLOCK_I});
    c_B.store_in(&b_B_reg, {j % R_BLOCK_J});
    c_acc_init.store_in(&b_acc, {i % R_BLOCK_I, j % R_BLOCK_J});
    c_acc.store_in(&b_acc, {i % R_BLOCK_I, j % R_BLOCK_J});
    c_C.store_in(&b_C_glb);
    c_Consts.store_in(&b_Consts, {i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({&b_Consts, &b_A, &b_B, &b_C}, "fct.o", true);

    return 0;
}
