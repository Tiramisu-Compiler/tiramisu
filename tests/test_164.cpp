#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_164");

    input sizes("sizes", {"s_i"}, {12}, p_int32);
    input params("params", {"p_i"}, {2}, p_float32);
    input transposes("transposes", {"t_i"}, {2}, p_boolean);

    constant M("M", sizes(0));
    constant N("N", sizes(1));
    constant K("K", sizes(2));
    constant rowsA("rowsA", sizes(3));
    constant colsA("colsA", sizes(4));
    constant rowsB("rowsB", sizes(5));
    constant colsB("colsB", sizes(6));
    constant rowsC("rowsC", sizes(7));
    constant colsC("colsC", sizes(8));
    constant offsetA("offsetA", sizes(9));
    constant offsetB("offsetB", sizes(10));
    constant offsetC("offsetC", sizes(11));
    constant alpha("alpha", params(0));
    constant beta("beta", params(1));
    constant transposeA("transposeA", transposes(0));
    constant transposeB("transposeB", transposes(1));

    input A("A", {"A_i", "A_j"}, {rowsA, colsA}, p_float32);
    input B("B", {"B_i", "B_j"}, {rowsB, colsB}, p_float32);
    input C("C", {"C_i", "C_j"}, {rowsC, colsC}, p_float32);

    buffer b_A_glb("b_A_glb", {rowsA, colsA}, p_float32, a_temporary);
    buffer b_B_glb("b_B_glb", {rowsB, colsB}, p_float32, a_temporary);
    buffer b_C_glb("b_C_glb", {rowsC, colsC}, p_float32, a_temporary);
    b_A_glb.tag_gpu_global();
    b_B_glb.tag_gpu_global();
    b_C_glb.tag_gpu_global();

    computation copy_A_to_device({}, memcpy(*A.get_buffer(), b_A_glb));
    computation copy_B_to_device({}, memcpy(*B.get_buffer(), b_B_glb));
    computation copy_C_to_device({}, memcpy(*C.get_buffer(), b_C_glb));
    computation copy_C_to_host({}, memcpy(b_C_glb, *C.get_buffer()));

    computation gemm({var("dummy", 0, 1)},
        cublas_sgemm(b_A_glb, b_B_glb, b_C_glb,
                     M, N, K,
                     alpha, beta,
                     colsA, colsB, colsC,
                     offsetA, offsetB, offsetC,
                     transposeA, transposeB));

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(gemm, computation::root)
                    .then(copy_C_to_host, computation::root);

    tiramisu::codegen({sizes.get_buffer(), params.get_buffer(), transposes.get_buffer(),
                       A.get_buffer(), B.get_buffer(), C.get_buffer()},
                      "build/generated_fct_test_164.o", true);

    return 0;
}
