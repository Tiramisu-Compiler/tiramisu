#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

#define N 128
#define M 128
#define K 128

int main(int argc, char **argv)
{
    // Testing cache_shared operation on GEMM with inner tiling and accumulator
    // Include buffer padding

    tiramisu::init("test_171");

    int block = 16;
    int k_block = 32;
    int r_block = 4;

    var i("i", 0, N), j("j", 0, M), k("k", 0, K);
    var i0("i0", 0, N / r_block), i1("i1", 0, r_block);
    var j0("j0", 0, M / r_block), j1("j1", 0, r_block);
    var i00("i00"), i01("i01");
    var j00("j00"), j01("j01");
    var k0("k0"), k1("k1");

    buffer b_A("b_A", {N, K}, p_float32, a_input);
    buffer b_B("b_B", {K, M}, p_float32, a_input);
    buffer b_C("b_C", {N, M}, p_float32, a_output);

    // Level 1
    buffer b_acc("b_acc", {r_block, r_block}, p_float32, a_temporary);
    b_acc.tag_gpu_local();
    computation c_acc_dec({i0, j0}, allocate(b_acc));
    computation c_acc_init({i, j}, (float) 0);

    input c_A({i, k}, p_float32);
    input c_B({k, j}, p_float32);
    c_A.get_buffer()->tag_gpu_global();
    c_B.get_buffer()->tag_gpu_global();
    computation c_acc({i, j, k}, p_float32);
    c_acc.set_expression(c_acc(i, j, 0) + c_A(i, k) * c_B(k, j));
    computation c_C({i, j}, p_float32);
    c_C.get_buffer()->tag_gpu_global();
    c_C.set_expression(c_C(i, j) + c_acc(i, j, 0));

    computation copy_A_to_device({}, memcpy(b_A, *c_A.get_buffer()));
    computation copy_B_to_device({}, memcpy(b_B, *c_B.get_buffer()));
    computation copy_C_to_device({}, memcpy(b_C, *c_C.get_buffer()));
    computation copy_C_to_host({}, memcpy(*c_C.get_buffer(), b_C));

    // Level 2
    c_acc_dec.gpu_tile(i0, j0, block, block, i00, j00, i01, j01);
    c_acc_init.tile(i, j, r_block, r_block, i0, j0, i1, j1);
    c_acc_init.gpu_tile(i0, j0, block, block, i00, j00, i01, j01);
    c_acc.tile(i, j, r_block, r_block, i0, j0, i1, j1);
    c_acc.interchange(j1, k);
    c_acc.interchange(i1, k);
    c_acc.split(k, k_block, k0, k1);
    c_acc.gpu_tile(i0, j0, block, block, i00, j00, i01, j01);
    c_C.tile(i, j, r_block, r_block, i0, j0, i1, j1);
    c_C.gpu_tile(i0, j0, block, block, i00, j00, i01, j01);

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(c_acc_dec, computation::root)
                    .then(c_acc_init, j01)
                    .then(c_acc, j01)
                    .then(c_C, j01)
                    .then(copy_C_to_host, computation::root);

    // Level 3
    c_acc_init.store_in(&b_acc, {i % 4, j % 4});
    c_acc.store_in(&b_acc, {i % 4, j % 4});
    // Buffer padding enabled:
//    c_acc.cache_shared(c_A, k0, {block * r_block, k_block}, {i00 * block * r_block, k0 * k_block}, true);
//    c_acc.cache_shared(c_B, k0, {k_block, block * r_block}, {k0 * k_block, j00 * block * r_block}, true);

    tiramisu::codegen({&b_A, &b_B, &b_C}, "build/generated_fct_test_171.o", true);

    return 0;
}
