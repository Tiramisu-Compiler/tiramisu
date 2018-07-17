/* 
    This tutorial shows how to write a simple matrix multiplication (C = A * B)

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = 0;
            for k = 0 .. N
                C[i,j] = C[i,j] + A[i,k] * B[k,j];
     
     To run this tutorial
     
     cd build/
     make run_developers_tutorial_04

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

    // Declare computations that represents the input buffers (b_A and b_B)
    computation c_A("[N]->{c_A[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);
    computation c_B("[N]->{c_B[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);

    // Declare loop iterators
    var i("i"), j("j"), k("k"), i0("i0"), j0("j0"), i1("i1"), j1("j1");

    // Declare a computation to initialize the reduction c[i,j]
    computation C_init("[N]->{C_init[i,j,-1]: 0<=i<N and 0<=j<N}", expr((uint8_t) 0), true, p_uint8, &matmul);
    
    // Declare the reduction operation.
    computation c_C("[N]->{c_C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", expr(), true, p_uint8, &matmul);
    // Note that the previous computation has an empty expression (because we can only use c_C in an expression after its declaration)
    c_C.set_expression(c_C(i, j, k - 1) + c_A(i, k) * c_B(k, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Tile both computations: C_init and c_C
    // This tiles the loop levels i and j and produces the loop levels by a 32x32 tile.
    // i0, j0, i1 and j1 where i0 is the outermost loop level and j1 is the innermost.
    C_init.tile(i, j, 32, 32, i0, j0, i1, j1);
    c_C.tile(i, j, 32, 32, i0, j0, i1, j1);

    // Parallelize the outermost loop level i0
    c_C.paralleliz(i0);

    // Indicate that c_C is after C_init at the loop level j (this means,
    // they share the two outermost loops i and j and starting from j c_C
    // is ordered after C_init).
    c_C.after(C_init, j);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Declare the buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output, &matmul);

    // Map the computations to a buffer.
    c_A.store_in(&b_A);
    c_B.store_in(&b_B);

    // Store C_init[i,j,k] in b_C[i,j]
    C_init.store_in(&b_C, {i,j});
    // Store c_C[i,j,k] in b_C[i,j]
    c_C.store_in(&b_C, {i,j});
    // Note that both of the computations C_init and c_C store their
    // results in the buffer b_C.

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    matmul.codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_04A.o");
    
    // Dump the generated Halide statement (just for debugging).
    matmul.dump_halide_stmt();

    return 0;
}
