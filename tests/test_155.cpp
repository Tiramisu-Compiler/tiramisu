#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // A simple gemm to check store_in and implicit buffers.
    tiramisu::init("test_155");

    // Configuration
    input sizes("sizes", {"s_iter"}, {3}, p_int32);
    constant M("M", sizes(0));
    constant N("N", sizes(1));
    constant K("K", sizes(2));
    var i("i", 0, M),
        j("j", 0, N),
        k("k", 0, K);

    // Layer I
    input A({i, j}, p_int32);
    input B({j, k}, p_int32);
    computation C({i, j, k}, p_int32);
    C.set_expression(C(i, 0, k) + A(i, j) * B(j, k));
    computation C_init({i, k}, int32_t(0));

    // Layer II
    C.interchange(j, k);
    C_init.then(C, k);

    // Layer III
    C.store_in({i, k}, {M, K});
    C_init.store_in(C.get_buffer());

    tiramisu::codegen({sizes.get_buffer(), A.get_buffer(), B.get_buffer(), C.get_buffer()}, "build/generated_fct_test_155.o");

    return 0;
}
