#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

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


#define L3_B0 2
#define L3_B1 4
#define L3_B2 4
#define B0 32
#if SIZE_IS_MULTIPLE_OF_TILE
    #define B1 64
#else
    #define B1 32
#endif
#define B2 32

#if SIZE_IS_MULTIPLE_OF_TILE
    #define THREE_D_L3_TILING 1
#else
    #define THREE_D_L3_TILING 0
    #define TWO_D_L3_TILING 0
#endif

//#include "TUNED_SIZES.h"

    std::string B0s = std::to_string(B0);
    std::string B1s = std::to_string(B1);
    std::string B2s = std::to_string(B2);

    tiramisu::computation reduced_AB_0   ("[N, M, K]->{reduced_AB_0   [i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and                        0<=j<"+B1s+"*floor(M/"+B1s+")}", (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p0("[N, M, K]->{reduced_AB_0_p0[i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p1("[N, M, K]->{reduced_AB_0_p1[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and                        0<=j<"+B1s+"*floor(M/"+B1s+")}",              (float) 0, true, p_float32, &function0);
    tiramisu::computation reduced_AB_0_p2("[N, M, K]->{reduced_AB_0_p2[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);

    tiramisu::computation packed_B   ("[N, M, K]->{packed_B   [j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(k,j), true, p_float32, &function0);
    tiramisu::computation packed_B_p0("[N, M, K]->{packed_B_p0[j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(k,j), true, p_float32, &function0);
    tiramisu::computation packed_B_p1("[N, M, K]->{packed_B_p1[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(k,j), true, p_float32, &function0);
    tiramisu::computation packed_B_p2("[N, M, K]->{packed_B_p2[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(k,j), true, p_float32, &function0);

    tiramisu::computation reduced_AB_1   ("[N, M, K]->{reduced_AB_1   [i,j,k]: 0<=i<N and              0<=j<"+B1s+"*floor(M/"+B1s+") and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*packed_B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p0("[N, M, K]->{reduced_AB_1_p0[i,j,k]: 0<=i<N and	       0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*packed_B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p1("[N, M, K]->{reduced_AB_1_p1[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*packed_B(k,j), true, p_float32, &function0);
    tiramisu::computation reduced_AB_1_p2("[N, M, K]->{reduced_AB_1_p2[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*packed_B(k,j), true, p_float32, &function0);

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
    packed_B_p1.tag_parallel_level(0);
    packed_B_p0.tag_parallel_level(0);



    // ----------------------------------------------------------------------------------------------------------------
    // L2 tiling
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_0   [0, 0, i, 0, j, 0, 0, 0]->reduced_AB_0   [0, 0, i0, 0, j0, 0, i1, 0, j1, 0, 0,  0,  0, 0]:"
        "i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    reduced_AB_0_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p0[0, 0, i, 0, j, 0, 0, 0]->reduced_AB_0_p0[0, 0, i0, 0, j0, 0, i1, 0, j1, 0, 0,  0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    reduced_AB_0_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p1[0, 0, i, 0, j, 0, 0, 0]->reduced_AB_0_p1[0, 0, i0, 0, j0, 0, i1, 0, j1, 0, 0,  0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    reduced_AB_0_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p2[0, 0, i, 0, j, 0, 0, 0]->reduced_AB_0_p2[0, 0, i0, 0, j0, 0, i1, 0, j1, 0, 0,  0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");

    packed_B.apply_transformation_on_schedule   ("[N,M,K]->{packed_B   [0, 0, j, 0, k, 0, 0, 0]->packed_B   [0, 0, 0, 0, j0, 0, k0, 0, 0, 0, j1, 0, k1, 0]:"
	"j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    packed_B_p0.apply_transformation_on_schedule   ("[N,M,K]->{packed_B_p0[0, 0, j, 0, k, 0, 0, 0]->packed_B_p0[0, 0, 0, 0, j0, 0, k0, 0, 0, 0, j1, 0, k1, 0]:"
	"j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    packed_B_p1.apply_transformation_on_schedule   ("[N,M,K]->{packed_B_p1[0, 0, j, 0, k, 0, 0, 0]->packed_B_p1[0, 0, 0, 0, j0, 0, k0, 0, 0, 0, j1, 0, k1, 0]:"
	"j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    packed_B_p2.apply_transformation_on_schedule   ("[N,M,K]->{packed_B_p2[0, 0, j, 0, k, 0, 0, 0]->packed_B_p2[0, 0, 0, 0, j0, 0, k0, 0, 0, 0, j1, 0, k1, 0]:"
	"j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");

    reduced_AB_1.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_1   [0, 0, i, 0, j, 0, k, 0]->reduced_AB_1   [0, 0, i0, 0, j0, 0, k0, 0, i1, 0, j1, 0, k1, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    reduced_AB_1_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p0[0, 0, i, 0, j, 0, k, 0]->reduced_AB_1_p0[0, 0, i0, 0, j0, 0, k0, 0, i1, 0, j1, 0, k1, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    reduced_AB_1_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p1[0, 0, i, 0, j, 0, k, 0]->reduced_AB_1_p1[0, 0, i0, 0, j0, 0, k0, 0, i1, 0, j1, 0, k1, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");
    reduced_AB_1_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p2[0, 0, i, 0, j, 0, k, 0]->reduced_AB_1_p2[0, 0, i0, 0, j0, 0, k0, 0, i1, 0, j1, 0, k1, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+" and k0=floor(k/"+B2s+") and k1=k%"+B2s+"}");

    result.apply_transformation_on_schedule      ("[N,M,K]->{result      [0, 0, i, 0, j, 0, 0, 0]->result      [0, 0, i0, 0, j0, 0, i1, 0, j1, 0,  0, 0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    result_p0.apply_transformation_on_schedule   ("[N,M,K]->{result_p0   [0, 0, i, 0, j, 0, 0, 0]->result_p0   [0, 0, i0, 0, j0, 0, i1, 0, j1, 0,  0, 0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    result_p1.apply_transformation_on_schedule   ("[N,M,K]->{result_p1   [0, 0, i, 0, j, 0, 0, 0]->result_p1   [0, 0, i0, 0, j0, 0, i1, 0, j1, 0,  0, 0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");
    result_p2.apply_transformation_on_schedule   ("[N,M,K]->{result_p2   [0, 0, i, 0, j, 0, 0, 0]->result_p2   [0, 0, i0, 0, j0, 0, i1, 0, j1, 0,  0, 0,  0, 0]:"
	"i0=floor(i/"+B0s+") and i1=i%"+B0s+" and j0=floor(j/"+B1s+") and j1=j%"+B1s+"}");



    // ----------------------------------------------------------------------------------------------------------------
    // L3 tiling (only if SIZE_IS_MULTIPLE_OF_TILE)
    // ----------------------------------------------------------------------------------------------------------------
    int lev0 = 0, lev1 = 0, lev2 = 0;
#if THREE_D_L3_TILING
    lev0 = 1;
    lev1 = 1;
    lev2 = 1;
    
    reduced_AB_0.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_0   [0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0   [0, 0, i00, 0, j00, 0, i01, 0, j01, 0,  i1, 0,  j1, 0, 0,  0,  0, 0,  0, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    reduced_AB_0_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p0[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p0[0, 0, i00, 0, j00, 0, i01, 0, j01, 0,  i1, 0,  j1, 0, 0,  0,  0, 0,  0, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    reduced_AB_0_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p1[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p1[0, 0, i00, 0, j00, 0, i01, 0, j01, 0,  i1, 0,  j1, 0, 0,  0,  0, 0,  0, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    reduced_AB_0_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p2[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p2[0, 0, i00, 0, j00, 0, i01, 0, j01, 0,  i1, 0,  j1, 0, 0,  0,  0, 0,  0, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");

    packed_B.apply_transformation_on_schedule   ("[N,M,K]->{packed_B[0, 0, 0,  0,  j0,  0,   k0,  0,  0, 0,    j1, 0,    k1, 0]->"
							   "packed_B[0, 0, 0,  0, j00,  0,  k00,  0,  0, 0,   j01, 0,   k01, 0, 0, 0, j1, 0, k1, 0]:"
							   "j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    packed_B_p0.apply_transformation_on_schedule("[N,M,K]->{packed_B_p0[0, 0, 0,  0,  j0,  0,   k0,  0,  0, 0,    j1, 0,    k1, 0]->"
						           "packed_B_p0[0, 0, 0,  0, j00,  0,  k00,  0,  0, 0,   j01, 0,   k01, 0, 0, 0, j1, 0, k1, 0]:"
							   "j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    packed_B_p1.apply_transformation_on_schedule("[N,M,K]->{packed_B_p1[0, 0, 0,  0,  j0,  0,   k0,  0,  0, 0,    j1, 0,    k1, 0]->"
							   "packed_B_p1[0, 0, 0,  0, j00,  0,  k00,  0,  0, 0,   j01, 0,   k01, 0, 0, 0, j1, 0, k1, 0]:"
							   "j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    packed_B_p2.apply_transformation_on_schedule("[N,M,K]->{packed_B_p2[0, 0, 0,  0,  j0,  0,   k0,  0,  0, 0,    j1, 0,    k1, 0]->"
							   "packed_B_p2[0, 0, 0,  0, j00,  0,  k00,  0,  0, 0,   j01, 0,   k01, 0, 0, 0, j1, 0, k1, 0]:"
							   "j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");

    reduced_AB_1.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_1   [0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1[0, 0, i00, 0, j00, 0, k00, 0, i01, 0, j01, 0, k01, 0, i1, 0, j1, 0, k1, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    reduced_AB_1_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p0[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p0[0, 0, i00, 0, j00, 0, k00, 0, i01, 0, j01, 0, k01, 0, i1, 0, j1, 0, k1, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    reduced_AB_1_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p1[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p1[0, 0, i00, 0, j00, 0, k00, 0, i01, 0, j01, 0, k01, 0, i1, 0, j1, 0, k1, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");
    reduced_AB_1_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p2[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p2[0, 0, i00, 0, j00, 0, k00, 0, i01, 0, j01, 0, k01, 0, i1, 0, j1, 0, k1, 0]:"
							       "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4 and k00=floor(k0/4) and k01=k0%4}");

    result.apply_transformation_on_schedule      ("[N,M,K]->{      result[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "      result[0, 0, i00, 0, j00, 0, i01, 0, j01, 0, i1,  0,  j1, 0,  0, 0,  0, 0,  0, 0]:"
							    "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    result_p0.apply_transformation_on_schedule   ("[N,M,K]->{   result_p0[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "   result_p0[0, 0, i00, 0, j00, 0, i01, 0, j01, 0, i1,  0,  j1, 0,  0, 0,  0, 0,  0, 0]:"
							    "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    result_p1.apply_transformation_on_schedule   ("[N,M,K]->{   result_p1[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "   result_p1[0, 0, i00, 0, j00, 0, i01, 0, j01, 0, i1,  0,  j1, 0,  0, 0,  0, 0,  0, 0]:"
							    "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");
    result_p2.apply_transformation_on_schedule   ("[N,M,K]->{   result_p2[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "   result_p2[0, 0, i00, 0, j00, 0, i01, 0, j01, 0, i1,  0,  j1, 0,  0, 0,  0, 0,  0, 0]:"
							    "i00=floor(i0/2) and i01=i0%2 and j00=floor(j0/4) and j01=j0%4}");

    // ----------------------------------------------------------------------------------------------------------------
    // No L3 tiling (if SIZE_IS_MULTIPLE_OF_TILE == false)
    // ----------------------------------------------------------------------------------------------------------------
#else  // No L3 Tiling
    lev0 = 0;
    lev1 = 0;
    lev2 = 0;

    reduced_AB_0.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_0   [0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0   [0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p0[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p0[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p1[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p1[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p2[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0]->"
							       "reduced_AB_0_p2[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0]}");

    packed_B.apply_transformation_on_schedule   ("[N,M,K]->{packed_B   [0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0]->"
							   "packed_B   [0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    packed_B_p0.apply_transformation_on_schedule("[N,M,K]->{packed_B_p0[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0]->"
							   "packed_B_p0[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    packed_B_p1.apply_transformation_on_schedule("[N,M,K]->{packed_B_p1[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0]->"
							   "packed_B_p1[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    packed_B_p2.apply_transformation_on_schedule("[N,M,K]->{packed_B_p2[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0]->"
							   "packed_B_p2[0, 0,  0,  0,  j0, 0,  k0, 0,  0,  0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");

    reduced_AB_1.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_1   [0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1   [0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_1_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p0[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p0[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_1_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p1[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p1[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_1_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p2[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0]->"
							       "reduced_AB_1_p2[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, 0, 0, 0, 0, 0, 0]}");

    result.apply_transformation_on_schedule      ("[N,M,K]->{            result[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "            result[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p0.apply_transformation_on_schedule   ("[N,M,K]->{         result_p0[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "         result_p0[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p1.apply_transformation_on_schedule   ("[N,M,K]->{         result_p1[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "         result_p1[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p2.apply_transformation_on_schedule   ("[N,M,K]->{         result_p2[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0]->"
							    "         result_p2[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  0,  0,   0, 0, 0, 0, 0, 0, 0, 0]}");
#endif




    // ----------------------------------------------------------------------------------------------------------------
    // Ordering
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_0   [0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							       "reduced_AB_0   [0, 1, i0,  0, j0,  0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p0[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							       "reduced_AB_0_p0[0, 1, i0,  0, j0,  1,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p1[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							       "reduced_AB_0_p1[0, 1, i0,  0, j0,  2,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    reduced_AB_0_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_0_p2[0, 0, i0,  0, j0,  0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							       "reduced_AB_0_p2[0, 1, i0,  0, j0,  3,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");

    packed_B.apply_transformation_on_schedule   ("[N,M,K]->{packed_B           [0, 0,  0,  0, j00, 0, k00, 0,  0,  0, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]->"
							   "packed_B           [0, 0,  0,  2, j00, 4, k00, 0,  0,  2, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]}");
    reduced_AB_1.apply_transformation_on_schedule   ("[N,M,K]->{reduced_AB_1   [0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, x0, 0, y0, 0, z0 , 0]->"
							       "reduced_AB_1   [0, 1, i0,  2,  j0, 4,  k0, 0,  i1, 3,  j1, 0,  k1, 0, x0, 1, y0, 0, z0, 0]}");
    packed_B_p0.apply_transformation_on_schedule("[N,M,K]->{packed_B_p0        [0, 0,  0,  0, j00, 0, k00, 0,  0,  0, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]->"
							   "packed_B_p0        [0, 0,  0,  2, j00, 4, k00, 0,  0,  4, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]}");
    reduced_AB_1_p0.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p0[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, x0, 0, y0, 0, z0, 0]->"
							       "reduced_AB_1_p0[0, 1, i0,  2,  j0, 4,  k0, 0,  i1, 5,  j1, 0,  k1, 0, x0, 1, y0, 0, z0, 0]}");
    packed_B_p1.apply_transformation_on_schedule("[N,M,K]->{packed_B_p1        [0, 0,  0,  0, j00, 0, k00, 0,  0,  0, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]->"
							   "packed_B_p1        [0, 0,  0,  2, j00, 4, k00, 0,  0,  6, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]}");
    reduced_AB_1_p1.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p1[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, x0, 0, y0, 0, z0, 0]->"
							       "reduced_AB_1_p1[0, 1, i0,  2,  j0, 6,  k0, 0,  i1, 7,  j1, 0,  k1, 0, x0, 1, y0, 0, z0, 0]}");
    packed_B_p2.apply_transformation_on_schedule("[N,M,K]->{packed_B_p2        [0, 0,  0,  0, j00, 0, k00, 0,  0,  0, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]->"
							   "packed_B_p2        [0, 0,  0,  2, j00, 6, k00, 0,  0,  8, j01, 0, k01, 0, 0,  0, j1, 0, k1, 0]}");
    reduced_AB_1_p2.apply_transformation_on_schedule("[N,M,K]->{reduced_AB_1_p2[0, 0, i0,  0,  j0, 0,  k0, 0,  i1, 0,  j1, 0,  k1, 0, x0, 0, y0, 0, z0, 0]->"
							       "reduced_AB_1_p2[0, 1, i0,  2,  j0, 7,  k0, 0,  i1, 9,  j1, 0,  k1, 0, x0, 1, y0, 0, z0, 0]}");

    result.apply_transformation_on_schedule      ("[N,M,K]->{            result[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							    "            result[0, 1, i0,  3,  j0, 0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p0.apply_transformation_on_schedule   ("[N,M,K]->{         result_p0[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							    "         result_p0[0, 1, i0,  3,  j0, 1,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p1.apply_transformation_on_schedule   ("[N,M,K]->{         result_p1[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							    "         result_p1[0, 1, i0,  3,  j0, 2,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");
    result_p2.apply_transformation_on_schedule   ("[N,M,K]->{         result_p2[0, 0, i0,  0,  j0, 0,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]->"
							    "         result_p2[0, 1, i0,  3,  j0, 3,  i1, 0,  j1, 0,  x0, 0,  y0, 0, 0, 0, 0, 0, 0, 0]}");



    // ----------------------------------------------------------------------------------------------------------------
    // Vectorization
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.tag_vector_level(lev0+lev1+3, B1);
    if (SIZE_IS_MULTIPLE_OF_TILE)
	reduced_AB_0_p0.tag_vector_level(lev0+lev1+1, B1);
    reduced_AB_0_p1.tag_vector_level(lev0+lev1+3, B1);
    reduced_AB_1.tag_vector_level(lev0+lev1+lev2+4, B1);
    reduced_AB_1_p0.tag_vector_level(lev0+lev1+lev2+3, B1);
    result.tag_vector_level(lev0+lev1+3, B1);
    result_p1.tag_vector_level(lev0+lev1+3, B1);



    // ----------------------------------------------------------------------------------------------------------------
    // Unrolling
    // ----------------------------------------------------------------------------------------------------------------
    reduced_AB_0.tag_unroll_level(lev0+lev1+2);
    packed_B_p1.tag_unroll_level(lev0+lev1+4);
    reduced_AB_1.tag_unroll_level(lev0+lev1+lev2+5);
    reduced_AB_1_p1.tag_unroll_level(lev0+lev1+lev2+4);
    result.tag_unroll_level(lev0+lev1+2);



    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    tiramisu::buffer buf_SIZES("buf_SIZES", {3}, tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer buf_alpha("buf_alpha", {1}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_beta("buf_beta", {1}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_A("buf_A", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "K")}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_B("buf_B", {tiramisu::var(p_int32, "K"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_input, &function0);
    tiramisu::buffer buf_Bp("buf_Bp", {tiramisu::var(p_int32, "M"), tiramisu::var(p_int32, "K")}, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_temps("buf_temp", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_temporary, &function0);
    tiramisu::buffer buf_C("buf_C", {tiramisu::var(p_int32, "N"), tiramisu::var(p_int32, "M")}, tiramisu::p_float32, a_output, &function0);

    SIZES.set_access("{SIZES[i]->buf_SIZES[i]}");
    alpha.set_access("{alpha[i]->buf_alpha[i]}");
    beta.set_access("{beta[i]->buf_beta[i]}");
    A.set_access("{A[i,j]->buf_A[i,j]}");
    B.set_access("{B[i,j]->buf_B[i,j]}");
    packed_B.set_access("{packed_B[j,k]->buf_Bp[j%"+B1s+",k%"+B2s+"]}");
    packed_B_p0.set_access("{packed_B_p0[j,k]->buf_Bp[j%"+B1s+",k%"+B2s+"]}");
    packed_B_p1.set_access("{packed_B_p1[j,k]->buf_Bp[j%"+B1s+",k%"+B2s+"]}");
    packed_B_p2.set_access("{packed_B_p2[j,k]->buf_Bp[j%"+B1s+",k%"+B2s+"]}");
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
