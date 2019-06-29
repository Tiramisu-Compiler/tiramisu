
#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

/*
  this is an implementation of the BLAS SYR2 (Symetric rank-two update) function in Tiramisu 
  the performed operation is :
   A = A + alpha*x*y^T + alpha*y*x^T

   A : NxN Hermitian input matrix
   x,y : size N input vectors
   ^T : transpose operator
   alpha : input scalar 
   
   The C version of this function is as follow : 
   
    for(int i=0; i<N:i++){
          for(int j=0; j<N; j++){
	         A[i,j] += alpha*x[i]*y[j] + alpha*y[i]*x[j]; 
          }
    }
   
*/



using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("syr2");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    
    var i("i", 0, N), j("j", 0, N);
    constant DIM("N", expr(N));

    input A("A", {"i", "j"}, {DIM, DIM}, p_float64);
    input x("x", {"i"}, {DIM}, p_float64);
    input y("y", {"j"}, {DIM}, p_float64);
    input alpha("alpha", {}, p_float64);

    computation SYR2("C", {i,j}, p_float64);
    SYR2.set_expression( A(i, j) + alpha(0)*x(i)*y(j) + alpha(0)*x(j)*y(i));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var i0("i0"), j0("j0"), i1("i1"), j1("j1");
    SYR2.tile(i, j, 32, 32, i0, j0, i1, j1);
    SYR2.parallelize(i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_A("b_A", {expr(DIM), expr(DIM)}, p_float64, a_output);
    buffer b_x("b_x", {expr(DIM)}, p_float64, a_input);   
    buffer b_y("b_y", {expr(DIM)}, p_float64, a_input);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input);

    A.store_in(&b_A);
    x.store_in(&b_x);
    y.store_in(&b_y);
    alpha.store_in(&b_alpha);
    SYR2.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_A, &b_x,&b_y,&b_alpha}, "generated_syr2.o");

    return 0;
}
