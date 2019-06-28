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
	tiramisu::init();

	function symm("symm");
	
	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------

	computation SIZES("{SIZES[e]: 0<=e<2}", expr(), false, p_float64, &symm);

	// Constants
	constant N("N", SIZES(0), p_int32, true, NULL, 0, &symm);
	constant M("M", SIZES(1), p_int32, true, NULL, 0, &symm);

	// Iteration variables
	var i("i"), j("j"), k("k");

	// Scalars  
	tiramisu::computation alpha("{alpha[0]}", expr(), false, p_float64, &symm);
	tiramisu::computation beta("{beta[0]}", expr(), false, p_float64, &symm);
	tiramisu::constant a("a", alpha(0),p_float64, true, NULL, 0, &symm);
	tiramisu::constant b("b", beta(0), p_float64,true, NULL, 0,   &symm);

	// Matrix 
	computation B("[N,M]->{B[k,j]: 0<=k<N and 0<=j<M}", expr(), false, p_float64, &symm);
	computation A("[N]->{A[i,k]: 0<=i<N and 0<=k<N}", expr(), false, p_float64, &symm);
    	
	computation C_0("[N,M]->{C_0[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_float64, &symm);
	
	// Computations 
	// Initialisation à 0 
	computation init("[N, M]->{init[i,j]: 0<=i<N and 0<=j<=M}", expr(cast(p_float64, 0)), true, p_float64, &symm);
	
	//  A * B
	computation mat_mul_a_b("[N, M]->{mat_mul_a_b[i,j,k]: 0<=i<N and 0<=j<=M and 0<=k<j-1}", expr(), true, p_float64, &symm);

	// a * temp 
	computation mult_alpha("[N, M]->{mult_alpha[i,j,k]: 0<=i<N and 0<=j<=M and 0<=k<j-1}", expr(), true, p_float64, &symm);

	// Reduction on C 
	computation C_1("[N,M]->{C_1[i,j,k]: 0<=k<N and 0<=j<M}", C_0(k-1,j) + mult_alpha(i,j,k) , false, p_float64, &symm);


	//  B * A 
	computation mat_mul_b_a("[N, M]->{mat_mul_b_a[i,j,k]: 0<=i<N and 0<=j<=M and 0<=k<j-1}", expr(), true, p_float64, &symm);


	computation mat_mul_a_b_e("[N, M]->{mat_mul_a_b_e[i,j]: 0<=i<N and 0<=j<=M }", expr(), true, p_float64, &symm);

	
	computation mult_alpha_k("[N, M]->{mult_alpha_k[i,j]: 0<=i<N and 0<=j<=M}", expr(), true, p_float64, &symm);

	computation mult_beta("[N, M]->{mult_beta[i,j]: 0<=i<N and 0<=j<M}", expr(), true, p_float64, &symm);
	

	computation add_all("[N, M]->{add_all[i,j]: 0<=i<N and 0<=j<=M }", expr(), true, p_float64, &symm);

	

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
	mult_alpha_k.after(mat_mul_a_b_e, j ); 
	mat_mul_a_b_e.after(mat_mul_b_a, j); 
	mat_mul_b_a.after(mult_alpha, k); 
	mult_alpha.after(mat_mul_a_b,k);
	mat_mul_a_b.after(init, j);

	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------

	buffer buf_SIZES("buf_SIZES", {2}, tiramisu::p_int32, a_input, &symm);
	
	buffer buf_A("buf_A", {N,N}, p_float64, a_input, &symm);
	buffer buf_B("buf_B", {N,M}, p_float64, a_input, &symm);
	buffer buf_C("buf_C", {N,M}, p_float64, a_input, &symm);
	buffer buf_alpha("buf_alpha", {1}, p_float64, a_input, &symm);
	buffer buf_beta("buf_beta", {1}, p_float64, a_input, &symm);


	buffer buf_result("buf_result", {N,M}, p_float64, a_output, &symm);

	//Store inputs
	SIZES.set_access("{SIZES[e]->buf_SIZES[e]: 0<=e<2}");
	
	A.set_access("{A[i,k]->buf_A[i,k]}");
	A.set_access("{A[i,i]->buf_A[i,i]}");
	B.set_access("{B[k,j]->buf_B[k,j]}");
	B.set_access("{B[i,j]->buf_B[i,j]}");
	C_0.set_access("{C_0[i,j]->buf_C[i,j]}");
	alpha.set_access("{alpha[0]->buf_alpha[0]}");
	beta.set_access("{beta[0]->buf_beta[0]}");

	add_all.set_access("{add_all[i,j]->buf_result[i,j]}");
	C_1.set_access("{C_1[i,j,k]->buf_result[i,j]}");
	mult_beta.set_access("{mult_beta[i,j]->buf_result[i,j]}");
	mult_alpha_k.set_access("{mult_alpha_k[i,j]->buf_result[i,j]}");
	mat_mul_a_b_e.set_access("{mat_mul_a_b_e[i,j]->buf_result[i,j]}");
	mat_mul_b_a.set_access("{mat_mul_b_a[i,j,k]->buf_result[i,j]}");
	mult_alpha.set_access("{mult_alpha[i,j,k]->buf_result[i,j]}");
	mat_mul_a_b.set_access("{mat_mul_a_b[i,j, k]->buf_result[i,j]}");
	init.set_access("{init[i,j]->buf_result[i,j]}");

	
	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	symm.set_arguments({&buf_SIZES, &buf_A, &buf_B, &buf_C, &buf_alpha, &buf_beta, &buf_result});
	symm.gen_time_space_domain();
	symm.gen_isl_ast();
	symm.gen_halide_stmt();
	symm.gen_halide_obj("generated_symm.o");
	
	return 0;

}
