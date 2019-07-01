#include <tiramisu/tiramisu.h>

using namespace tiramisu;


/**
*   Implementation of SYMM Benchmark in Tiramisu : 
*
*        result = alpha*A*B + beta*C 
*     or result = alpha*B*A + beta*C with A is symetric 
*
*     A : a symmetric N by N matrix 
*     B, C : a N by M matrix
*     alpha, beta : scalars
* 
* 
*    The C code is as follow: 
*	for (i = 0; i < N; i++){
* 	   for (j = 0; j < M; j++){
*		temp = 0;
*		for (k = 0; k < j - 1; k++){
*	   		C[k][j] += alpha * A[k][i] * B[i][j];
*	    		temp += B[k][j] * A[k][i];
*	  	}
*	   Result[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * temp;
*	   }
* 	}
* 
**/ 

int main(int argc, char **argv)
{
	tiramisu::init("symm");

	
	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------

	// Constants
	constant N("N", expr((int32_t) N));
	constant M("M", expr((int32_t) M));

	// Iteration variables
	var i("i", 0, N), j("j", 0, M), k("k", 0, j);

	// Scalars  
	computation alpha("{alpha[0]}", expr());
	computation beta("{beta[0]}", expr());
	constant a("a", alpha(0));
	constant b("b", beta(0));

	// Matrix 
	input B("B", {"k", "j"}, {N, M}, p_float64);
	input A("A", {"i", "k"}, {N, N}, p_float64);
    	
	input C_0("C_0", {"i", "j"}, {N, M}, p_float64);
	
	// Computations 
	// Initialisation à 0 
	computation init("init", {i,j}, expr((p_float64) 0));
	
	//  A * B
	computation mat_mul_a_b("mat_mul_a_b", {i,j,k}, p_float64);

	// a * temp 
	computation mult_alpha("mult_alpha",  {i,j,k}, p_float64);

	// Reduction on C 
	computation C_1("C_1", {i,j,k}, C_0(k-1,j) + mult_alpha(i,j,k));


	//  B * A 
	computation mat_mul_b_a("mat_mul_b_a", {i,j,k}, p_float64);


	computation mat_mul_a_b_e("mat_mul_a_b_e", {i,j}, p_float64);

	
	computation mult_alpha_k("mult_alpha_k", {i,j}, p_float64);

	computation mult_beta("mult_beta", {i,j}, p_float64);

	computation add_all("add_all", {i,j}, p_float64);

	mat_mul_a_b.set_expression(expr(mat_mul_a_b(i, j, k-1) + A(k, i) * B(i, j)));
	mult_alpha.set_expression(expr(alpha(0) * mat_mul_a_b(i, j , j-1)));
	mat_mul_b_a.set_expression(expr(mat_mul_b_a(i, j, k-1) + B(k, j) * A(k, i)));
	mat_mul_a_b_e.set_expression(expr(A(i, i) * B(i, j)));
	mult_alpha_k.set_expression(expr(alpha(0) * mat_mul_a_b_e(i, j)));
	mult_beta.set_expression(expr(beta(0) *  C_1(i,j, j-1)));
	add_all.set_expression(expr(add_all(i,j-1) + mat_mul_a_b_e(i, j) + mult_alpha(i,j, j-1) + mult_beta(i,j)));


	// -------------------------------------------------------
	// Layer II
	// -------------------------------------------------------
	add_all.after(mult_beta, j);
	mult_beta.after(mult_alpha_k, j);
	mult_alpha_k.after(mat_mul_a_b_e, j);
	mat_mul_a_b_e.after(mat_mul_b_a, j);
	mat_mul_b_a.after(mult_alpha, k);
	mult_alpha.after(mat_mul_a_b,k);
	mat_mul_a_b.after(init, j);

	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------
	
	buffer buf_A("buf_A", {N,N}, p_float64, a_input);
	buffer buf_B("buf_B", {N,M}, p_float64, a_input);
	buffer buf_C("buf_C", {N,M}, p_float64, a_input);
	buffer buf_alpha("buf_alpha", {1}, p_float64, a_input);
	buffer buf_beta("buf_beta", {1}, p_float64, a_input);


	buffer buf_result("buf_result", {N,M}, p_float64, a_output);

	//Store inputs
	A.store_in(&buf_A);
	B.store_in(&buf_B);
	C_0.store_in(&buf_C);
	alpha.store_in(&buf_alpha[0]);
	beta.store_in(&buf_beta[0]);

	init.store_in(&buf_result, {i,j});

	add_all.store_in(&buf_result, {i,j});
	C_1.store_in(&buf_result, {i,j});
	mult_beta.store_in(&buf_result, {i,j});
	mat_mul_a_b_e.store_in(&buf_result, {i,j});
	mat_mul_b_a.store_in(&buf_result, {i,j});
	mult_alpha.store_in(&buf_result, {i,j});
	mat_mul_a_b.store_in(&buf_result, {i,j});

	
	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	tiramisu::codegen({&buf_A, &buf_B, &buf_C, &buf_alpha, &buf_beta, &buf_result}, "generated_symm.o");

	
	return 0;

}
