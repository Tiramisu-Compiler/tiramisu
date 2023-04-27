#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Testing cache_shared operation on GEMM with inner tiling

    tiramisu::init("test_169");

    var i("i", 0, 128), j("j", 0, 128), k("k", 0, 128);
    var i0("i0"), i1("i1");
    var j0("j0"), j1("j1");
    var i00("i00"), i01("i01");
    var j00("j00"), j01("j01");
    var k0("k0"), k1("k1");

    buffer b_A("b_A", {128, 128}, p_float32, a_input);
    buffer b_B("b_B", {128, 128}, p_float32, a_input);
    buffer b_C("b_C", {128, 128}, p_float32, a_output);

    input c_A({i, k}, p_float32);
    input c_B({k, j}, p_float32);
    c_A.get_buffer()->tag_gpu_global();
    c_B.get_buffer()->tag_gpu_global();
    computation c_C({i, j, k}, p_float32);
    c_C.store_in({i, j}, {128, 128});
    c_C.get_buffer()->tag_gpu_global();
    c_C.set_expression(c_C(i, j, 0) + c_A(i, k) * c_B(k, j));

    computation copy_A_to_device({}, memcpy(b_A, *c_A.get_buffer()));
    computation copy_B_to_device({}, memcpy(b_B, *c_B.get_buffer()));
    computation copy_C_to_device({}, memcpy(b_C, *c_C.get_buffer()));
    computation copy_C_to_host({}, memcpy(*c_C.get_buffer(), b_C));

    int block = 16;
    int k_block = 32;
    int r_block = 4;
    c_C.tile(i, j, r_block, r_block, i0, j0, i1, j1);
    c_C.interchange(j1, k);
    c_C.interchange(i1, k);
    c_C.split(k, k_block, k0, k1);
    c_C.gpu_tile(i0, j0, block, block, i00, j00, i01, j01);

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(c_C, computation::root)
                    .then(copy_C_to_host, computation::root);

//    c_C.cache_shared(c_A, k0, {block * r_block, k_block}, {i00 * block * r_block, k0 * k_block});
//    c_C.cache_shared(c_B, k0, {k_block, block * r_block}, {k0 * k_block, j00 * block * r_block});

    tiramisu::codegen({&b_A, &b_B, &b_C}, "build/generated_fct_test_169.o", true);

    return 0;
}
