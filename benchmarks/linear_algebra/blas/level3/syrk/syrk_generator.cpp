#include <tiramisu/tiramisu.h>

#include "benchmarks.h"

#define UNROLL_FACTOR 32

using namespace tiramisu;

/**
*  Benchmark for BLAS SYRK
*     out = alpha * A * A' + beta * C
*
*     A : a N by K matrix
*     C : a symmetric N by N matrix
*     alpha, beta : scalars
*     A' is the transpose of A
*/
/**
	We will make a tiramisu implementation of this code :
	for(int i = 0; i<N; i++){
		for(int j = 0; j<=i; j++){
			int tmp = 0;
			for(k=0; k<K; k++)
				tmp += A[i][k] * A[j][k];
			RESULT[i][j] = alpha * tmp + beta* C[i][j];
		}
	}
*/

int main(int argc, char **argv)
{
	tiramisu::init();
	
	function syrk("syrk");
	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------
	computation SIZES("{SIZES[e]: 0<=e<2}", expr(), false, p_float64, &syrk);
	//Constants
	constant NN("NN", SIZES(0), p_int32, true, NULL, 0, &syrk);
	constant KK("KK", SIZES(1), p_int32, true, NULL, 0, &syrk);
	
	//Iteration variables
	var i("i"), j("j"), k("k");
	
	//Inputs
	computation A("[NN,KK]->{A[i,k]: 0<=i<NN and 0<=k<KK}", expr(), false,  p_float64, &syrk);
	computation C("[NN]->{C[i,j]: 0<=i<NN and 0<=j<NN}", expr(), false, p_float64, &syrk);
	computation alpha("{alpha[0]}", expr(), false, p_float64, &syrk);
	computation beta("{beta[0]}", expr(), false, p_float64, &syrk);
	
	//Computations
	computation result_init("[NN]->{result_init[i,j]: 0<=i<NN and 0<=j<=i}", expr(cast(p_float64, 0)), true, p_float64, &syrk);
	computation mat_mul("[NN,KK]->{mat_mul[i,j,k]: 0<=i<NN and 0<=j<=i and 0<=k<KK}", expr(), true, p_float64, &syrk);
	computation mult_alpha("[NN]->{mult_alpha[i,j]: 0<=i<NN and 0<=j<=i}", expr(), true, p_float64, &syrk);
	computation add_beta_C("[NN]->{add_beta_C[i,j]: 0<=i<NN and 0<=j<=i}", expr(), true, p_float64, &syrk);
	
	computation copy_symmetric_part("[NN]->{copy_symmetric_part[i,j]: 0<=i<NN and i<j<NN}", expr(), true, p_float64, &syrk);
	
	mat_mul.set_expression(expr(mat_mul(i, j, k-1) + A(i, k) * A(j, k)));
	mult_alpha.set_expression(expr(alpha(0) * mat_mul(i, j, NN-1)));
	add_beta_C.set_expression(expr(mult_alpha(i, j) + beta(0) * C(i, j)));
	
	copy_symmetric_part.set_expression(expr(add_beta_C(j, i)));
	
	// -------------------------------------------------------
	// Layer II
	// -------------------------------------------------------
	copy_symmetric_part.after(add_beta_C, computation::root_dimension);
	add_beta_C.after(mult_alpha, i);
	mult_alpha.after(mat_mul, i);
	mat_mul.after(result_init, i);
	
#if TIRAMISU_LARGE
	mat_mul.unroll(k, UNROLL_FACTOR);
#endif

	//Parallelization
	mat_mul.parallelize(i);
	copy_symmetric_part.parallelize(i);

	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------
	//Input Buffers
	buffer buf_SIZES("buf_SIZES", {2}, tiramisu::p_int32, a_input, &syrk);
	
	buffer buf_A("buf_A", {NN,KK}, p_float64, a_input, &syrk);
	buffer buf_C("buf_C", {NN,NN}, p_float64, a_input, &syrk);
	buffer buf_alpha("buf_alpha", {1}, p_float64, a_input, &syrk);
	buffer buf_beta("buf_beta", {1}, p_float64, a_input, &syrk);
	
	//Output Buffers
	buffer buf_result("buf_result", {NN, NN}, p_float64, a_output, &syrk);
	
	//Store inputs
	SIZES.set_access("{SIZES[e]->buf_SIZES[e]: 0<=e<2}");
	
	A.set_access("{A[i,k]->buf_A[i,k]}");
	C.set_access("{C[i,j]->buf_C[i,j]}");
	alpha.set_access("{alpha[0]->buf_alpha[0]}");
	beta.set_access("{beta[0]->buf_beta[0]}");
	
	//Store computations
	result_init.set_access("{result_init[i,j]->buf_result[i,j]}");
	mat_mul.set_access("{mat_mul[i,j,k]->buf_result[i,j]}");
	mult_alpha.set_access("{mult_alpha[i,j]->buf_result[i,j]}");
	add_beta_C.set_access("{add_beta_C[i,j]->buf_result[i,j]}");
	add_beta_C.set_access("{add_beta_C[j,i]->buf_result[j,i]}");
	copy_symmetric_part.set_access("{copy_symmetric_part[i,j]->buf_result[i,j]}");
	
	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	syrk.set_arguments({&buf_SIZES, &buf_A, &buf_C, &buf_alpha, &buf_beta, &buf_result});
	syrk.gen_time_space_domain();
	syrk.gen_isl_ast();
	syrk.gen_halide_stmt();
	syrk.gen_halide_obj("generated_syrk.o");
	
	return 0;
}
