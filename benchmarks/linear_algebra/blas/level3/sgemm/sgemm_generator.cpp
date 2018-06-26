#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "benchmarks.h"
#include "sgemm_wrapper.h"

using namespace tiramisu;

/**
 * Benchmark sgemm
 *     C = aAB + bC
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::var i("i"), j("j"), k("k");
    tiramisu::computation SIZES("{SIZES[i]}", tiramisu::expr(), false, p_int32, &function0);
    tiramisu::computation alpha("{alpha[i]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::computation beta("{beta[i]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::computation A("{A[i,j]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::computation B("{B[i,j]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::computation C("{C[i,j]}", tiramisu::expr(), false, p_float32, &function0);
    tiramisu::constant N_cst("N", SIZES(0), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M_cst("M", SIZES(1), p_int32, true, NULL, 0, &function0);
    tiramisu::constant K_cst("K", SIZES(2), p_int32, true, NULL, 0, &function0);
    tiramisu::constant a("a", alpha(0), p_float32, true, NULL, 0, &function0);
    tiramisu::constant b("b", beta(0), p_float32, true, NULL, 0, &function0);

#define PACK_ARRAY 1
#define AUTO_SCHEDULE 0
#define INNER_SPLIT 1

/**
  Best parameters:
  - Lanka:
	#define B0 64
	#define B1 (SIZE_IS_MULTIPLE_OF_TILE?64:32)
	#define B2 32

	#define L3_B0 2
	#define L3_B1 32
	#define L3_B2 32

	#define U1 64
    - MacBookPro
	#define B0 64
	#define B1 (SIZE_IS_MULTIPLE_OF_TILE?32:32)
	#define B2 32

	#define L3_B0 2
	#define L3_B1 4
	#define L3_B2 4

	#define U1 32

  */


#if AUTO_SCHEDULE
	#include "SCHEDULE.h"
	#define L3_TILING 1
#else
	#define B0 64
	#define B1 (SIZE_IS_MULTIPLE_OF_TILE?32:32)
	#define B2 32

	#define L3_B0 2
	#define L3_B1 4
	#define L3_B2 4

	#define U1 32

	#define L3_TILING (SIZE_IS_MULTIPLE_OF_TILE?1:0)
#endif

    std::string B0s = std::to_string(B0);
    std::string B1s = std::to_string(B1);
    std::string B2s = std::to_string(B2);

    tiramisu::computation reduced_AB_0   ("[N, M, K]->{reduced_AB_0   [i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and                        0<=j<"+B1s+"*floor(M/"+B1s+")}", (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p0("[N, M, K]->{reduced_AB_0_p0[i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p1("[N, M, K]->{reduced_AB_0_p1[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and                        0<=j<"+B1s+"*floor(M/"+B1s+")}",              (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p2("[N, M, K]->{reduced_AB_0_p2[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);

#if PACK_ARRAY
    tiramisu::computation packed_B   ("[N, M, K]->{packed_B   [j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(j,k), true, p_float32, &function0);
    tiramisu::computation packed_B_p0("[N, M, K]->{packed_B_p0[j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(j,k), true, p_float32, &function0);
    tiramisu::computation packed_B_p1("[N, M, K]->{packed_B_p1[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(j,k), true, p_float32, &function0);
    tiramisu::computation packed_B_p2("[N, M, K]->{packed_B_p2[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(j,k), true, p_float32, &function0);
#endif

    tiramisu::computation reduced_AB_1   ("[N, M, K]->{reduced_AB_1   [i,j,k]: 0<=i<N and              0<=j<"+B1s+"*floor(M/"+B1s+") and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p0("[N, M, K]->{reduced_AB_1_p0[i,j,k]: 0<=i<N and	       0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p1("[N, M, K]->{reduced_AB_1_p1[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p2("[N, M, K]->{reduced_AB_1_p2[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);

    tiramisu::computation result   ("[N, M, K]->{result   [i,j]: 0<=i<"+B0s+"*floor(N/"+B0s+") and                        0<=j<"+B1s+"*floor(M/"+B1s+")}", tiramisu::var(p_float32, "a") * reduced_AB_1(i,j,0) + tiramisu::var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    tiramisu::computation result_p0("[N, M, K]->{result_p0[i,j]: 0<=i<"+B0s+"*floor(N/"+B0s+") and "+B1s+"*floor(M/"+B1s+")<=j<M}",              tiramisu::var(p_float32, "a") * reduced_AB_1(i,j,0) + tiramisu::var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    tiramisu::computation result_p1("[N, M, K]->{result_p1[i,j]: "+B0s+"*floor(N/"+B0s+")<=i<N and                        0<=j<"+B1s+"*floor(M/"+B1s+")}",              tiramisu::var(p_float32, "a") * reduced_AB_1(i,j,0) + tiramisu::var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    tiramisu::computation result_p2("[N, M, K]->{result_p2[i,j]: "+B0s+"*floor(N/"+B0s+")<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M}",              tiramisu::var(p_float32, "a") * reduced_AB_1(i,j,0) + tiramisu::var(p_float32, "b") * C(i,j) , true, p_float32, &function0);



#if SIZE_IS_MULTIPLE_OF_TILE
     function0.add_context_constraints("[N, M, K]->{:N>"+B0s+" and M>"+B1s+" and K>"+B2s+" and N%"+B0s+"=0 and M%"+B1s+"=0 and K%"+B2s+"=0}");
#else
     function0.add_context_constraints("[N, M, K]->{:N>"+B0s+" and M>"+B1s+" and K>"+B2s+" and N%"+B0s+">0 and M%"+B1s+">0 and K%"+B2s+">0}");
#endif


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------



    function0.align_schedules();

    // ----------------------------------------------------------------------------------------------------------------
    // Parallelization
    // ----------------------------------------------------------------------------------------------------------------
    result.tag_parallel_level(0);

#if PACK_ARRAY
    packed_B.tag_parallel_level(0);
    packed_B_p1.tag_parallel_level(0);
    packed_B_p0.tag_parallel_level(0);
#endif



    // ----------------------------------------------------------------------------------------------------------------
    // L2 tiling
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.tile(0,1, B0,B1);
    reduced_AB_0_p0.tile(0,1, B0,B1);
    reduced_AB_0_p1.tile(0,1, B0,B1);
    reduced_AB_0_p2.tile(0,1, B0,B1);

#if PACK_ARRAY
    packed_B.tile(0,1, B1,B2);
    packed_B_p0.tile(0,1, B1,B2);
    packed_B_p1.tile(0,1, B1,B2);
    packed_B_p2.tile(0,1, B1,B2);
#endif

    reduced_AB_1.tile(0,1,2, B0,B1,B2);
    reduced_AB_1_p0.tile(0,1,2, B0,B1,B2);
    reduced_AB_1_p1.tile(0,1,2, B0,B1,B2);
    reduced_AB_1_p2.tile(0,1,2, B0,B1,B2);

    result.tile(0,1, B0,B1);
    result_p0.tile(0,1, B0,B1);
    result_p1.tile(0,1, B0,B1);
    result_p2.tile(0,1, B0,B1);



    // ----------------------------------------------------------------------------------------------------------------
    // L3 tiling (only if SIZE_IS_MULTIPLE_OF_TILE)
    // ----------------------------------------------------------------------------------------------------------------
    int lev0 = 0, lev1 = 0, lev2 = 0;
#if L3_TILING
    lev0 = 1;
    lev1 = 1;
    lev2 = 1;
    
    reduced_AB_0.tile(0,1, L3_B0, L3_B1);
    reduced_AB_0_p0.tile(0,1, L3_B0, L3_B1);
    reduced_AB_0_p1.tile(0,1, L3_B0, L3_B1);
    reduced_AB_0_p2.tile(0,1, L3_B0, L3_B1);

    #if PACK_ARRAY
    packed_B.tile(0,1, L3_B1,L3_B2);
    packed_B_p0.tile(0,1, L3_B1,L3_B2);
    packed_B_p1.tile(0,1, L3_B1,L3_B2);
    packed_B_p2.tile(0,1, L3_B1,L3_B2);
    #endif

    reduced_AB_1.tile(0,1,2, L3_B0, L3_B1, L3_B2);
    reduced_AB_1_p0.tile(0,1,2, L3_B0, L3_B1, L3_B2);
    reduced_AB_1_p1.tile(0,1,2, L3_B0, L3_B1, L3_B2);
    reduced_AB_1_p2.tile(0,1,2, L3_B0, L3_B1, L3_B2);

    result.tile(0,1, L3_B0, L3_B1);
    result_p0.tile(0,1, L3_B0, L3_B1);
    result_p1.tile(0,1, L3_B0, L3_B1);
    result_p2.tile(0,1, L3_B0, L3_B1);
#endif




    // ----------------------------------------------------------------------------------------------------------------
    // Ordering
    // ----------------------------------------------------------------------------------------------------------------
#if PACK_ARRAY
    packed_B_p1.after_low_level(packed_B, 1);
    packed_B_p2.after_low_level(packed_B_p1, 1);
    reduced_AB_0.after_low_level(packed_B_p2, -1);
    reduced_AB_1.after_low_level(packed_B_p2, -1);
    result.after_low_level(packed_B_p2, -1);
#endif

    reduced_AB_0_p0.after_low_level(reduced_AB_0, 1);
    reduced_AB_0_p1.after_low_level(reduced_AB_0_p0, 1);
    reduced_AB_0_p2.after_low_level(reduced_AB_0_p1, 1);

    reduced_AB_1.after_low_level(reduced_AB_0_p2, 0);
    reduced_AB_1_p0.after_low_level(reduced_AB_1, 3);
    reduced_AB_1_p1.after_low_level(reduced_AB_1_p0, 1);
    reduced_AB_1_p2.after_low_level(reduced_AB_1_p1, 1);

    result.after_low_level(reduced_AB_1_p2, 0);
    result_p0.after_low_level(result, 1);
    result_p1.after_low_level(result_p0, 1);
    result_p2.after_low_level(result_p1, 1);

 
    // ----------------------------------------------------------------------------------------------------------------
    // Split to prepare for unrolling
    // ----------------------------------------------------------------------------------------------------------------
    int split_AB_0 = 0;
    int split_AB_1 = 0;
    int split_result = 0;
#if INNER_SPLIT
if (U1 < B1)
{
    #if PACK_ARRAY
    packed_B_p1.split(lev0+lev1+4, U1);
    #endif

    int original_depth_AB_0 = reduced_AB_0.compute_maximal_AST_depth();
    reduced_AB_0.split(lev0+lev1+2, U1);
    int new_depth_AB_0 = reduced_AB_0.compute_maximal_AST_depth();
    if (new_depth_AB_0 > original_depth_AB_0)
	split_AB_0 = 1;

    int original_depth_AB_1 = reduced_AB_1.compute_maximal_AST_depth();
    reduced_AB_1.split(lev0+lev1+lev2+5, U1);
    int new_depth_AB_1 = reduced_AB_1.compute_maximal_AST_depth();
    if (new_depth_AB_1 > original_depth_AB_1)
	split_AB_1 = 1;

    reduced_AB_1_p1.split(lev0+lev1+lev2+4, U1);

    int original_depth_result = result.compute_maximal_AST_depth();
    result.split(lev0+lev1+2, U1);
    int new_depth_result = result.compute_maximal_AST_depth();
    if (new_depth_result > original_depth_result)
	split_result = 1;
}
#endif



    // ----------------------------------------------------------------------------------------------------------------
    // Unrolling
    // ----------------------------------------------------------------------------------------------------------------
#if PACK_ARRAY
    packed_B_p1.tag_unroll_level(lev0+lev1+4);
#endif
    reduced_AB_0.tag_unroll_level(lev0+lev1+split_AB_0+2);
    reduced_AB_1.tag_unroll_level(lev0+lev1+lev2+split_AB_1+5);
    reduced_AB_1_p1.tag_unroll_level(lev0+lev1+lev2+split_AB_1+4);
    result.tag_unroll_level(lev0+lev1+split_result+2);



    // ----------------------------------------------------------------------------------------------------------------
    // Vectorization
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.tag_vector_level(lev0+lev1+split_AB_0+3, B1);
    if (SIZE_IS_MULTIPLE_OF_TILE)
	reduced_AB_0_p0.tag_vector_level(lev0+lev1+split_AB_0+1, B1);
    reduced_AB_0_p1.tag_vector_level(lev0+lev1+split_AB_0+3, B1);
    reduced_AB_1.tag_vector_level(lev0+lev1+lev2+4, B1);
    reduced_AB_1_p0.tag_vector_level(lev0+lev1+lev2+split_AB_1+3, B1);
    result.tag_vector_level(lev0+lev1+split_result+3, B1);
    result_p1.tag_vector_level(lev0+lev1+split_result+3, B1);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    tiramisu::buffer buf_SIZES("buf_SIZES", {3}, tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer buf_alpha("buf_alpha", {1}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_beta("buf_beta", {1}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_A("buf_A", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "K")}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_B("buf_B", {tiramisu::var(p_int32, "K"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_input, &function0);
    std::vector<tiramisu::expr> sz = {(tiramisu::var(p_int32, "M")/B1), tiramisu::var(p_int32, "K"), (tiramisu::var(p_int32, "M")%B1)};
    tiramisu::buffer buf_Bp("buf_Bp", sz, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_temps("buf_temp", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_C("buf_C", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_output, &function0);

    SIZES.set_access("{SIZES[i]->buf_SIZES[i]}");
    alpha.set_access("{alpha[i]->buf_alpha[i]}");
    beta.set_access("{beta[i]->buf_beta[i]}");
    A.set_access("{A[i,j]->buf_A[i,j]}");
    B.set_access("{B[i,j]->buf_B[i,j]}");
#if PACK_ARRAY
    packed_B.set_access("{packed_B[j,k]->buf_Bp[j/"+B1s+",k,j%"+B1s+"]}");
    packed_B_p0.set_access("{packed_B_p0[j,k]->buf_Bp[j/"+B1s+",k,j%"+B1s+"]}");
    packed_B_p1.set_access("{packed_B_p1[j,k]->buf_Bp[j/"+B1s+",k,j%"+B1s+"]}");
    packed_B_p2.set_access("{packed_B_p2[j,k]->buf_Bp[j/"+B1s+",k,j%"+B1s+"]}");
#endif
    C.set_access("{C[i,j]->buf_C[i,j]}");
    reduced_AB_0.set_access   ("{reduced_AB_0   [i,j,k]->buf_temp[i,j]}");
    reduced_AB_0_p0.set_access("{reduced_AB_0_p0[i,j,k]->buf_temp[i,j]}");
    reduced_AB_0_p1.set_access("{reduced_AB_0_p1[i,j,k]->buf_temp[i,j]}");
    reduced_AB_0_p2.set_access("{reduced_AB_0_p2[i,j,k]->buf_temp[i,j]}");
    reduced_AB_1.set_access   ("{reduced_AB_1   [i,j,k]->buf_temp[i,j]}");
    reduced_AB_1_p0.set_access("{reduced_AB_1_p0[i,j,k]->buf_temp[i,j]}");
    reduced_AB_1_p1.set_access("{reduced_AB_1_p1[i,j,k]->buf_temp[i,j]}");
    reduced_AB_1_p2.set_access("{reduced_AB_1_p2[i,j,k]->buf_temp[i,j]}");
    result.set_access   ("{result[i,j]->buf_C[i,j]}");
    result_p0.set_access("{result_p0[i,j]->buf_C[i,j]}");
    result_p1.set_access("{result_p1[i,j]->buf_C[i,j]}");
    result_p2.set_access("{result_p2[i,j]->buf_C[i,j]}");



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    function0.set_arguments({&buf_SIZES, &buf_alpha, &buf_beta, &buf_A, &buf_B, &buf_C});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_sgemm.o");
}

int main(int argc, char **argv)
{
    generate_function("sgemm_tiramisu");

    return 0;
}
