#include <tiramisu/tiramisu.h>

#include "benchmarks.h"

#define UNROLL_FACTOR 32

using namespace tiramisu;

/**
*  Benchmark for BLAS SYR2K
*     out = alpha * A * B' + alpha * B* A' + beta * C
*
*     A : a N by K matrix
*     B : a N by K matrix
*     C : a symmetric N by N matrix
*     alpha, beta : scalars
*     A' is the transpose of A
*     B' is the transpose of B
*/
/**
	We will make a tiramisu implementation of this code :
	for(int i = 0; i<N; i++){
		for(int j = 0; j<=i; j++){
			int tmp = 0;
			for(k=0; k<K; k++)
				tmp += A[i][k] * B[j][k] + B[i][k] * A[j][k];
			RESULT[i][j] = alpha * tmp + beta* C[i][j];
		}
	}
*/

int main(int argc, char **argv)
{
	tiramisu::init();
	
	function syr2k("syr2k");
	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------
	computation SIZES("{SIZES[e]: 0<=e<2}", expr(), false, p_float64, &syr2k);
	//Constants
	constant NN("NN", SIZES(0), p_int32, true, NULL, 0, &syr2k);
	constant KK("KK", SIZES(1), p_int32, true, NULL, 0, &syr2k);
	
	//Iteration variables
	var i("i"), j("j"), k("k");
	
	//Inputs
	computation A("[NN,KK]->{A[i,k]: 0<=i<NN and 0<=k<KK}", expr(), false,  p_float64, &syr2k);
	computation B("[NN,KK]->{B[i,k]: 0<=i<NN and 0<=k<KK}", expr(), false,  p_float64, &syr2k);
	computation C("[NN]->{C[i,j]: 0<=i<NN and 0<=j<NN}", expr(), false, p_float64, &syr2k);
	computation alpha("{alpha[0]}", expr(), false, p_float64, &syr2k);
	computation beta("{beta[0]}", expr(), false, p_float64, &syr2k);
	
	//Computations
	computation result_init("[NN]->{result_init[i,j]: 0<=i<NN and 0<=j<=i}", expr(cast(p_float64, 0)), true, p_float64, &syr2k);
	computation mat_mul1("[NN,KK]->{mat_mul1[i,j,k]: 0<=i<NN and 0<=j<=i and 0<=k<KK}", expr(), true, p_float64, &syr2k);
	computation mat_mul2("[NN,KK]->{mat_mul2[i,j,k]: 0<=i<NN and 0<=j<=i and 0<=k<KK}", expr(), true, p_float64, &syr2k);
	computation mult_alpha("[NN]->{mult_alpha[i,j]: 0<=i<NN and 0<=j<=i}", expr(), true, p_float64, &syr2k);
	computation add_beta_C("[NN]->{add_beta_C[i,j]: 0<=i<NN and 0<=j<=i}", expr(), true, p_float64, &syr2k);
	
	computation copy_symmetric_part("[NN]->{copy_symmetric_part[i,j]: 0<=i<NN and i<j<NN}", expr(), true, p_float64, &syr2k);
	
	mat_mul1.set_expression(expr(mat_mul1(i, j, k-1) + A(i, k) * B(j, k)));
	mat_mul2.set_expression(expr(mat_mul2(i, j, k-1) + B(i, k) * A(j, k)));
	mult_alpha.set_expression(expr(alpha(0) * mat_mul2(i, j, NN-1)));
	add_beta_C.set_expression(expr(mult_alpha(i, j) + beta(0) * C(i, j)));
	
	copy_symmetric_part.set_expression(expr(add_beta_C(j, i)));
	
	// -------------------------------------------------------
	// Layer II
	// -------------------------------------------------------

	copy_symmetric_part.after(add_beta_C, computation::root_dimension);
	add_beta_C.after(mult_alpha, i);
	mult_alpha.after(mat_mul2,i);
	mat_mul2.after(mat_mul1, i);
	mat_mul1.after(result_init, i);

	
#if TIRAMISU_LARGE
	mat_mul1.unroll(k, UNROLL_FACTOR);
	mat_mul2.unroll(k, UNROLL_FACTOR);
#endif

	//Parallelization
	mat_mul1.parallelize(i);
	mat_mul2.parallelize(i);
	copy_symmetric_part.parallelize(i);

	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------
	//Input Buffers
	buffer buf_SIZES("buf_SIZES", {2}, tiramisu::p_int32, a_input, &syr2k);
	
	buffer buf_A("buf_A", {NN,KK}, p_float64, a_input, &syr2k);
	buffer buf_B("buf_B", {NN,KK}, p_float64, a_input, &syr2k);
	buffer buf_C("buf_C", {NN,NN}, p_float64, a_input, &syr2k);
	buffer buf_alpha("buf_alpha", {1}, p_float64, a_input, &syr2k);
	buffer buf_beta("buf_beta", {1}, p_float64, a_input, &syr2k);
	
	//Output Buffers
	buffer buf_result("buf_result", {NN, NN}, p_float64, a_output, &syr2k);
	
	//Store inputs
	SIZES.set_access("{SIZES[e]->buf_SIZES[e]: 0<=e<2}");
	
	A.set_access("{A[i,k]->buf_A[i,k]}");
	B.set_access("{B[i,k]->buf_B[i,k]}");
	C.set_access("{C[i,j]->buf_C[i,j]}");
	alpha.set_access("{alpha[0]->buf_alpha[0]}");
	beta.set_access("{beta[0]->buf_beta[0]}");
	
	//Store computations
	result_init.set_access("{result_init[i,j]->buf_result[i,j]}");
	mat_mul1.set_access("{mat_mul1[i,j,k]->buf_result[i,j]}");
	mat_mul2.set_access("{mat_mul2[i,j,k]->buf_result[i,j]}");
	mult_alpha.set_access("{mult_alpha[i,j]->buf_result[i,j]}");
	add_beta_C.set_access("{add_beta_C[i,j]->buf_result[i,j]}");
	add_beta_C.set_access("{add_beta_C[j,i]->buf_result[j,i]}");
	copy_symmetric_part.set_access("{copy_symmetric_part[i,j]->buf_result[i,j]}");
	
	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	syr2k.set_arguments({&buf_SIZES, &buf_A, &buf_B, &buf_C, &buf_alpha, &buf_beta, &buf_result});
	syr2k.gen_time_space_domain();
	syr2k.gen_isl_ast();
	syr2k.gen_halide_stmt();
	syr2k.gen_halide_obj("generated_syr2k.o");
	
	return 0;
}
