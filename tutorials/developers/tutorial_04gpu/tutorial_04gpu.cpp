/* 
    This tutorial shows how to write a simple matrix multiplication in gpu (C = A * B)

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = 0;
            for k = 0 .. N
                C[i,j] = C[i,j] + A[i,k] * B[k,j];
     
     To run this tutorial
     
     cd build/
     make run_developers_tutorial_04gpu

*/

#include <tiramisu/tiramisu.h>

#define SIZE0 1000

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant p0("N", expr((int32_t) SIZE0));

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

    // Declare inputs (b_A and b_B)
    input c_A({i,j}, p_uint8);
    input c_B({i,j}, p_uint8);

    // Declare a computation to initialize the reduction c[i,j]
    computation C_init({i,j}, expr((uint8_t) 0));
    
    // Declare the reduction operation.
    computation c_C({i,j,k}, p_uint8);
    // Note that the previous computation has an empty expression (because we can only use c_C in an expression after its declaration)
    c_C.set_expression(c_C(i, j, k - 1) + c_A(i, k) * c_B(k, j));

    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, b_A_gpu));
    computation copy_B_to_device({}, memcpy(b_B, b_B_gpu));
    computation copy_C_to_host({}, memcpy(b_C_gpu, b_C));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands
    copy_B_to_device.after(copy_A_to_device, computation::root);
    C_init.after(copy_B_to_device, computation::root);
    c_C.after(C_init, computation::root);
    copy_C_to_host.after(c_C, computation::root);

    // A simple tiling.
    C_init.gpu_tile(i, j, 16, 16);
    c_C.gpu_tile(i, j, 16, 16);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Map the computations to a buffer.
    c_A.store_in(&b_A_gpu);
    c_B.store_in(&b_B_gpu);

    // Store C_init[i,j,k] in b_C[i,j]
    C_init.store_in(&b_C_gpu, {i,j});
    // Store c_C[i,j,k] in b_C[i,j]
    c_C.store_in(&b_C_gpu, {i,j});
    // Note that both of the computations C_init and c_C store their
    // results in the buffer b_C.

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_04gpu.o", true);

    return 0;
}
