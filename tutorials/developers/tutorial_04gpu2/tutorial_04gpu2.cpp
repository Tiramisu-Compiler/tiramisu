/*
    This tutorial shows how to write a GPU multiplication with simplified
    GPU interface.

    To run this tutorial

    cd build/
    make run_developers_tutorial_04gpu2

*/

#include <tiramisu/tiramisu.h>

#define SIZE0 128

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
    var i0("i0"), i1("i1");
    var j0("j0"), j1("j1");
    var k0("k0"), k1("k1");

    // Declare inputs.
    input A("A", {i, k}, p_uint8);
    input B("B", {k, j}, p_uint8);
    A.get_buffer()->tag_gpu_global();
    B.get_buffer()->tag_gpu_global();

    // Declare a computation to initialize the reduction.
    computation C_init("C_init", {i, j}, expr((uint8_t) 0));
    C_init.get_buffer()->tag_gpu_global();

    // Declare the reduction operation.
    computation C("C", {i, j, k}, p_uint8);
    // Note that the previous computation has an empty expression,
    // because we can only use C in an expression after its declaration.
    C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));

    // Declare cpu buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);
    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, *A.get_buffer()));
    computation copy_B_to_device({}, memcpy(b_B, *B.get_buffer()));
    computation copy_C_to_host({}, memcpy(*C_init.get_buffer(), b_C));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // A simple tiling.
    C.split(k, 16, k0, k1);
    C_init.gpu_tile(i, j, 16, 16, i0, j0, i1, j1);
    C.gpu_tile(i, j, 16, 16, i0, j0, i1, j1);

    // Scheduling commands
    copy_A_to_device.then(copy_B_to_device)
                    .then(C_init)
                    .then(C, j1)  // Fuse
                    .then(copy_C_to_host);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Tiramisu automatically allocates buffers for A, B, and C_init

    // Store C in the same buffer as C_init[i,j]
    C.store_in(C_init.get_buffer(), {i, j});

    // Use shared memory as an intermediate layer for global access
    C.cache_shared(A, k0, {16, 16}, {i0 * 16, k0 * 16});
    C.cache_shared(B, k0, {16, 16}, {k0 * 16, j0 * 16});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. The last argument triggers cuda compilation.
    tiramisu::codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_04gpu2.o", true);

    return 0;
}
