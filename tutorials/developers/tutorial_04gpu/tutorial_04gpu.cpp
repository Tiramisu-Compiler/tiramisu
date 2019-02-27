/* 
    This tutorial shows how to write a simple matrix multiplication for gpu.
     
     To run this tutorial
     
     cd build/
     make run_developers_tutorial_04gpu

*/

#include <tiramisu/tiramisu.h>

#define SIZE0 100

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant N("N", SIZE0);

    // Declare loop iterators
    var i("i", 0, N), j("j", 0, N), k("k", 0, N);

    // Declare cpu buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary);
    buffer b_B_gpu("b_B_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary);
    buffer b_C_gpu("b_C_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary);
    // Tag the GPU buffers to be stored in global memory.
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();

    // Declare inputs.
    input A("A", {"i", "j"}, {N, N}, p_uint8);
    input B("B", {"i", "j"}, {N, N}, p_uint8);

    // Declare a computation to initialize the reduction.
    computation C_init("C_init", {i, j}, expr((uint8_t) 0));

    // Declare the reduction operation.
    computation C("C", {i,j,k}, p_uint8);
    // Note that the previous computation has an empty expression,
    // because we can only use C in an expression after its declaration.
    C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));

    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, b_A_gpu));
    computation copy_B_to_device({}, memcpy(b_B, b_B_gpu));
    computation copy_C_to_host({}, memcpy(b_C_gpu, b_C));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // A simple tiling.
    C_init.gpu_tile(i, j, 16, 16);
    C.gpu_tile(i, j, 16, 16);

    // Scheduling commands
    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(C_init, computation::root)
                    .then(C, computation::root)
                    .then(copy_C_to_host, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Map the computations to a buffer.
    A.store_in(&b_A_gpu);
    B.store_in(&b_B_gpu);

    // Store C_init[i,j] in b_C[i,j]
    C_init.store_in(&b_C_gpu);
    // Store C[i,j,k] in b_C[i,j]
    C.store_in(&b_C_gpu, {i,j});
    // Note that both of the computations C_init and C store their
    // results in the buffer b_C.

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. The last argument triggers cuda compilation.
    tiramisu::codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_04gpu.o", true);

    return 0;
}
