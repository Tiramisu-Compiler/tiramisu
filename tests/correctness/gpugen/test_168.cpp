#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Testing cache_shared operation on simple transpose
    tiramisu::init("test_168");

    var i("i", 0, 128), j("j", 0, 256);
    var i0("i0"), i1("i1");
    var j0("j0"), j1("j1");

    buffer b_A("b_A", {128, 256}, p_float32, a_input);
    buffer b_B("b_B", {256, 128}, p_float32, a_input);

    input c_A({i, j}, p_float32);
    c_A.get_buffer()->tag_gpu_global();
    computation c_B({j, i}, c_A(i, j));
    c_B.get_buffer()->tag_gpu_global();

    computation copy_A_to_device({}, memcpy(b_A, *c_A.get_buffer()));
    computation copy_B_to_host({}, memcpy(*c_B.get_buffer(), b_B));

    int block = 16;
    c_B.gpu_tile(j, i, block, block, j0, i0, j1, i1);

    copy_A_to_device.then(c_B, computation::root)
                    .then(copy_B_to_host, computation::root);

//    c_B.cache_shared(c_A, i1, {16, 16}, {i0 * block, j0 * block});

    tiramisu::codegen({&b_A, &b_B}, "generated_fct_test_168.o", true);

    return 0;
}
