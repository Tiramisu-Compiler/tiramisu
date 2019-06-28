

#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

/*
  IMPLEMENTATION OF THE SGER FUNCTION IN TIRAMISU 
  
   A = A + a*x*y**T 

   where : 
   A : is a NxM matrix
   x : is a size N vector 
   y : is a size M vector ( y**T : transpose of y) 
   a : scalar 
   
   THE C CODE OF THIS FUNCTION IS AS FOLLOW : 
   
    for(int i=0; i<N:i++){

        for(int j=0; j<M; j++){

	        A[i,j] += a*x[i]*y[j]; 
        }
    }
   

*/



using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("sger");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant NN("N", expr(N));
    constant MM("M", expr(M));

    var i("i", 0, N), j("j", 0, M);

    input A("A", {"i", "j"}, {NN, MM}, p_float64);
    input x("x", {"i"}, {NN},p_float64);
    input y("y", {"j"}, {MM}, p_float64);
    input alpha("alpha", {}, p_float64);
    computation C("C", {i,j}, p_float64);
    C.set_expression( A(i, j) + x(i)*y(j)*alpha(0));



	// -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

	C.parallelize(i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_A("b_A", {expr(NN), expr(MM)}, p_float64, a_output);
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);   
    buffer b_y("b_y", {expr(MM)}, p_float64, a_input);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input);

    A.store_in(&b_A);
    x.store_in(&b_x);
    y.store_in(&b_y);
	alpha.store_in(&b_alpha);

   
	C.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_A, &b_x,&b_y,&b_alpha}, "generated_sger.o");

    return 0;
}
