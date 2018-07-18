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
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    /*
     * Declare a function matmul.
     */
    function matmul("matmul");

    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &matmul);

    // Declare loop iterators
    var i("i"), j("j"), k("k");

    // Declare cpu buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output, &matmul);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    buffer b_B_gpu("b_B_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    buffer b_C_gpu("b_C_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();

    // Declare computations that represents the input buffers (b_A and b_B)
    computation c_A("[N]->{c_A[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);
    computation c_B("[N]->{c_B[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);

    // Declare a computation to initialize the reduction c[i,j]
    computation C_init("[N]->{C_init[i,j,-1]: 0<=i<N and 0<=j<N}", expr((uint8_t) 0), true, p_uint8, &matmul);
    
    // Declare the reduction operation.
    computation c_C("[N]->{c_C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", expr(), true, p_uint8, &matmul);
    // Note that the previous computation has an empty expression (because we can only use c_C in an expression after its declaration)
    c_C.set_expression(c_C(i, j, k - 1) + c_A(i, k) * c_B(k, j));

    // Declare host-gpu transfer computations.
    computation copy_A_to_device("{copy_A_to_device[0]}", memcpy(b_A, b_A_gpu), true, p_none, &matmul);
    computation copy_B_to_device("{copy_B_to_device[0]}", memcpy(b_B, b_B_gpu), true, p_none, &matmul);
    computation copy_C_to_device("{copy_C_to_device[0]}", memcpy(b_C, b_C_gpu), true, p_none, &matmul);
    computation copy_A_to_host("{copy_A_to_host[0]}", memcpy(b_A_gpu, b_A), true, p_none, &matmul);
    computation copy_B_to_host("{copy_B_to_host[0]}", memcpy(b_B_gpu, b_B), true, p_none, &matmul);
    computation copy_C_to_host("{copy_C_to_host[0]}", memcpy(b_C_gpu, b_C), true, p_none, &matmul);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands
    // TODO: optimizations
    copy_B_to_device.after(copy_A_to_device, computation::root);
    copy_C_to_device.after(copy_B_to_device, computation::root);
    C_init.after(copy_C_to_device, computation::root);
    c_C.after(C_init, computation::root);
    copy_A_to_host.after(c_C, computation::root);
    copy_B_to_host.after(copy_A_to_host, computation::root);
    copy_C_to_host.after(copy_B_to_host, computation::root);

    // TODO: Optimizations
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
    matmul.codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_04gpu.o", true);
    
    // Dump the generated Halide statement (just for debugging).
    matmul.dump_halide_stmt();

    return 0;
}
