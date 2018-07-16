/* Matrix multiplication.

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = 0;
            for k = 0 .. N
                C[i,j] = C[i,j] + A[i,k] * B[k,j];
}
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

    // Declare computations that represents the input buffer (b_A and b_B)
    computation c_A("[N]->{c_A[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);
    computation c_B("[N]->{c_B[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);

    // Indices
    var i("i"), j("j"), k("k"), i0("i0"), j0("j0"), i1("i1"), j1("j1");

    // Declare a computation to initialize the reduction c[i,j]
    computation C_init("[N]->{c_C[i,j,-1]: 0<=i<N and 0<=j<N}", expr((uint8_t) 0), true, p_uint8, &matmul);
    computation c_C("[N]->{c_C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", expr(), true, p_uint8, &matmul);
    expr e1 = c_C(i, j, k - 1) + c_A(i, k) * c_B(k, j);
    c_C.set_expression(e1);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Set the schedule of each computation.
    // The identity schedule means that the program order is not modified
    // (i.e. no optimization is applied).
    C_init.tile(i, j, 32, 32, i0, j0, i1, j1);
    c_C.after(C_init, j);
    c_C.tile(i, j, 32, 32, i0, j0, i1, j1);
    c_C.tag_parallel_level(i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output, &matmul);

    // Map the computations to a buffer.
    c_A.set_access("{c_A[i,j]->b_A[i,j]}");
    c_B.set_access("{c_B[i,j]->b_B[i,j]}");
    C_init.set_access("{c_C[i,j,k]->b_C[i,j]}");
    c_C.set_access("{c_C[i,j,k]->b_C[i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to blurxy
    matmul.codegen({&b_A, &b_B, &b_C}, "build/generated_fct_developers_tutorial_03.o");

    return 0;
}
