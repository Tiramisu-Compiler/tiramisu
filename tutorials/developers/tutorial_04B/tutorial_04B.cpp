/* 
    This tutorial shows how to write two successive matrix multiplications:
	C = A * B
	E = A * D
    We want to perform these two matrix multiplications in the same i, j loop
    (i.e., fuse them).

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = 0;
            E[i,j] = 0;
            for k = 0 .. N
                C[i,j] = C[i,j] + A[i,k] * B[k,j];
            for k = 0 .. N
                E[i,j] = E[i,j] + A[i,k] * D[k,j];
    
     To run this tutorial
     
     cd build/
     make run_developers_tutorial_04B

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

    constant p0("N", expr((int32_t) SIZE0));

    var i("i", 0, p0), j("j", 0, p0), k("k", 0, p0);

    // Declare computations that represents the input buffers.
    input A("A", {"i", "j"}, {SIZE0, SIZE0}, p_uint8);
    input B("B", {"i", "j"}, {SIZE0, SIZE0}, p_uint8);
    input D("D", {"i", "j"}, {SIZE0, SIZE0}, p_uint8);

    // Declare a computation to initialize the reductions.
    computation C_init("C_init", {i,j}, expr((uint8_t) 0));
    computation E_init("E_init", {i,j}, expr((uint8_t) 0));
    
    // Declare the first reduction.  Do not provide any expression during declaration.
    computation C("C", {i,j,k}, p_uint8);
    // Note that the previous computation has an empty expression (because we can only use C in an expression after its declaration)
    C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));

    // Declare the second reduction.  Do not provide any expression during declaration.
    computation E("E", {i,j,k}, p_uint8);
    E.set_expression(E(i, j, k - 1) + A(i, k) * D(k, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Declare loop iterators
    var i0("i0"), j0("j0"), i1("i1"), j1("j1");

    // Tile all the computations: C_init, C, E_init, E
    // This tiles the loop levels i and j and produces the loop levels by a 32x32 tile.
    // i0, j0, i1 and j1 where i0 is the outermost loop level and j1 is the innermost.
    C_init.tile(i, j, 32, 32, i0, j0, i1, j1);
    C.tile(i, j, 32, 32, i0, j0, i1, j1);
    E_init.tile(i, j, 32, 32, i0, j0, i1, j1);
    E.tile(i, j, 32, 32, i0, j0, i1, j1);

    // Parallelize the outermost loop level i0. By parallelizing this loop all
    // the other computations are parallelized too because they all share the
    // same outer loop i0.
    C.parallelize(i0);

    // Specify the order between C, E, C_init and E_init.
    E_init.after(C_init, j1);
    C.after(E_init, j1);
    E.after(C, j1);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Declare the buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);
    buffer b_D("b_D", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_E("b_E", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);


    // Map the computations to a buffer.
    A.store_in(&b_A);
    B.store_in(&b_B);
    D.store_in(&b_D);

    // Store C_init[i,j,k] in b_C[i,j] and E_init[i,j,k] in b_E[i,j]
    C_init.store_in(&b_C, {i,j});
    E_init.store_in(&b_E, {i,j});

    // Store C[i,j,k] in b_C[i,j] and E[i,j,k] in b_E[i,j]
    C.store_in(&b_C, {i,j});
    E.store_in(&b_E, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_A, &b_B, &b_C, &b_D, &b_E}, "build/generated_fct_developers_tutorial_04B.o");

    /** Generated code

  let N = 1000
  parallel (c1, 0, ((int32(floor_f32(float32(((N + -1)/32)))) + 1) - 0)) {
    for (c3, 0, ((int32(floor_f32(float32(((N + -1)/32)))) + 1) - 0)) {
      for (c5, 0, (min((N - (c1*32)), 32) - 0)) {
        for (c7, 0, (min((N - (c3*32)), 32) - 0)) {
          b_C[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] = (uint8)0
          b_E[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] = (uint8)0
          for (c9, 0, (N - 0)) {
            b_C[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] = (b_C[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] + (b_A[((0 + int32((int64(c9)*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))]*b_B[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(c9)*(int64)1000)))]))
          }
          for (c9, 0, (N - 0)) {
            b_E[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] = (b_E[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))] + (b_A[((0 + int32((int64(c9)*(int64)1))) + int32((int64(((32*c1) + c5))*(int64)1000)))]*b_D[((0 + int32((int64(((32*c3) + c7))*(int64)1))) + int32((int64(c9)*(int64)1000)))]))
          }
        }
      }
    }
   }
      */

    return 0;
}
