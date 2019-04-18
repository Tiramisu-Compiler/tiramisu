#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_162");

    input sizes("sizes", {"s_i"}, {3}, p_int32);
    input params("params", {"p_i"}, {2}, p_float32);

    constant M("M", sizes(0));
    constant N("N", sizes(1));
    constant K("K", sizes(2));
    constant alpha("alpha", params(0));
    constant beta("beta", params(1));

    input A("A", {"A_i", "A_j"}, {M, K}, p_float32);
    input B("B", {"B_i", "B_j"}, {K, N}, p_float32);
    input C("C", {"C_i", "C_j"}, {M, N}, p_float32);

    buffer b_A_glb("b_A_glb", {M, K}, p_float32, a_temporary);
    buffer b_B_glb("b_B_glb", {K, N}, p_float32, a_temporary);
    buffer b_C_glb("b_C_glb", {M, N}, p_float32, a_temporary);
    b_A_glb.tag_gpu_global();
    b_B_glb.tag_gpu_global();
    b_C_glb.tag_gpu_global();

    computation copy_A_to_device({}, memcpy(*A.get_buffer(), b_A_glb));
    computation copy_B_to_device({}, memcpy(*B.get_buffer(), b_B_glb));
    computation copy_C_to_device({}, memcpy(*C.get_buffer(), b_C_glb));
    computation copy_C_to_host({}, memcpy(b_C_glb, *C.get_buffer()));

    computation gemm({var("dummy", 0, 1)},
        cublas_gemm(b_A_glb, b_B_glb, b_C_glb, M, N, K, alpha, beta));

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(gemm, computation::root)
                    .then(copy_C_to_host, computation::root);

    tiramisu::codegen({sizes.get_buffer(), params.get_buffer(),
                       A.get_buffer(), B.get_buffer(), C.get_buffer()},
                      "build/generated_fct_test_162.o", true);

    return 0;
}
