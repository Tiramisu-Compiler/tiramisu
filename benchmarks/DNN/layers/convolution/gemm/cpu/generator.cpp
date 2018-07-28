#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

using namespace tiramisu;

void generate_function(std::string name)
{
    global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function function0(name);

    // IM2COL

    // parameters[0]: N_B - batch size
    // parameters[1]: W - width
    // parameters[2]: H - height
    // parameters[3]: F_In - input filters
    // parameters[4]: F_Out - input filters
    // parameters[5]: K_W
    // parameters[6]: K_H
    computation parameters("{parameters[i]: 0<=i<7}", expr(), false, p_int32, &function0);
    constant N_B("N_B", parameters(0), p_int32, true, NULL, 0, &function0);
    constant W("W", parameters(1), p_int32, true, NULL, 0, &function0);
    constant H("H", parameters(2), p_int32, true, NULL, 0, &function0);
    constant F_In("F_In", parameters(3), p_int32, true, NULL, 0, &function0);
    constant F_Out("F_Out", parameters(4), p_int32, true, NULL, 0, &function0);
    constant K_W("K_W", parameters(5), p_int32, true, NULL, 0, &function0);
    constant K_H("K_H", parameters(6), p_int32, true, NULL, 0, &function0);

    var n_b("n_b"), x("x"), y("y"), f_in("f_in"), k_x("k_x"), k_y("k_y");

    computation input_padded("[N_B, W, H, F_In, K_W, K_H]->{input_padded[n_b, x, y, f_in]: 0<=n_b<N_B and 0<=x<W+K_W-1 and 0<=y<H+K_H-1 and 0<=f_in<F_In}", expr(), false, p_float32, &function0);
    computation input_col("[N_B, W, H, F_In, K_W, K_H]->{input_col[n_b, x, y, f_in, k_x, k_y]: 0<=n_b<N_B and  0<=f_in<F_In and 0<=k_y<K_H and 0<=k_x<K_W and 0<=y<H and 0<=x<W}", input_padded(n_b, x + k_x, y + k_y, f_in), true, p_float32, &function0);

    function0.add_context_constraints("[N_B, W, H, F_In, F_Out, K_W, K_H]->{:N_B>0 and W>0 and H>0 and F_In>0 and F_Out>0 and K_W>0 and K_H>0}");

    // SGEMM

    var i("i"), j("j"), k("k");
    computation A("{A[i,j]}", expr(), false, p_float32, &function0);
    computation B("{B[i,j]}", expr(), false, p_float32, &function0);
    computation C("{C[i,j]}", expr(), false, p_float32, &function0);
    constant N("N", expr(N_B) * W * H, p_int32, true, NULL, 0, &function0);
    constant M("M", F_Out, p_int32, true, NULL, 0, &function0);
    constant K("K", expr(F_In) * K_W * K_H, p_int32, true, NULL, 0, &function0);
    constant a("a", expr((float) 1), p_float32, true, NULL, 0, &function0);
    constant b("b", expr((float) 0), p_float32, true, NULL, 0, &function0);

#define PACK_ARRAY 1
#define AUTO_SCHEDULE 0
#define INNER_SPLIT 1
#define SIZE_IS_MULTIPLE_OF_TILE 0

#if AUTO_SCHEDULE
	#include "SCHEDULE.h"
	#define L3_TILING 1
#else
	#define B0 64
	#define B1 (SIZE_IS_MULTIPLE_OF_TILE?64:32)
	#define B2 32

	#define L3_B0 2
	#define L3_B1 32
	#define L3_B2 32

	#define U1 64

	#define L3_TILING (SIZE_IS_MULTIPLE_OF_TILE?1:0)
#endif

    std::string B0s = std::to_string(B0);
    std::string B1s = std::to_string(B1);
    std::string B2s = std::to_string(B2);

    computation reduced_AB_0   ("[N, M, K]->{reduced_AB_0   [i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and                        0<=j<"+B1s+"*floor(M/"+B1s+")}", (float) 0, true, p_float32, &function0);
    computation reduced_AB_0_p0("[N, M, K]->{reduced_AB_0_p0[i,j,0]: 0<=i<"+B0s+"*floor(N/"+B0s+") and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);
    computation reduced_AB_0_p1("[N, M, K]->{reduced_AB_0_p1[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and                        0<=j<"+B1s+"*floor(M/"+B1s+")}",              (float) 0, true, p_float32, &function0);
    computation reduced_AB_0_p2("[N, M, K]->{reduced_AB_0_p2[i,j,0]: "+B0s+"*floor(N/"+B0s+")<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M}",              (float) 0, true, p_float32, &function0);

#if PACK_ARRAY
    computation packed_B   ("[N, M, K]->{packed_B   [j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(j,k), true, p_float32, &function0);
    computation packed_B_p0("[N, M, K]->{packed_B_p0[j,k]: 0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(j,k), true, p_float32, &function0);
    computation packed_B_p1("[N, M, K]->{packed_B_p1[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and 0<=k<"+B2s+"*floor(K/"+B2s+")}", B(j,k), true, p_float32, &function0);
    computation packed_B_p2("[N, M, K]->{packed_B_p2[j,k]: "+B1s+"*floor(M/"+B1s+")<=j<M and "+B2s+"*floor(K/"+B2s+")<=k<K}", B(j,k), true, p_float32, &function0);
#endif

    computation reduced_AB_1   ("[N, M, K]->{reduced_AB_1   [i,j,k]: 0<=i<N and              0<=j<"+B1s+"*floor(M/"+B1s+") and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    computation reduced_AB_1_p0("[N, M, K]->{reduced_AB_1_p0[i,j,k]: 0<=i<N and	       0<=j<"+B1s+"*floor(M/"+B1s+") and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    computation reduced_AB_1_p1("[N, M, K]->{reduced_AB_1_p1[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and                        0<=k<"+B2s+"*floor(K/"+B2s+")}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);
    computation reduced_AB_1_p2("[N, M, K]->{reduced_AB_1_p2[i,j,k]: 0<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M              and "+B2s+"*floor(K/"+B2s+")<=k<K}", reduced_AB_0(i,j,0) + A(i,k)*B(k,j), true, p_float32, &function0);

    computation result   ("[N, M, K]->{result   [i,j]: 0<=i<"+B0s+"*floor(N/"+B0s+") and                        0<=j<"+B1s+"*floor(M/"+B1s+")}", var(p_float32, "a") * reduced_AB_1(i,j,0) + var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    computation result_p0("[N, M, K]->{result_p0[i,j]: 0<=i<"+B0s+"*floor(N/"+B0s+") and "+B1s+"*floor(M/"+B1s+")<=j<M}",              var(p_float32, "a") * reduced_AB_1(i,j,0) + var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    computation result_p1("[N, M, K]->{result_p1[i,j]: "+B0s+"*floor(N/"+B0s+")<=i<N and                        0<=j<"+B1s+"*floor(M/"+B1s+")}",              var(p_float32, "a") * reduced_AB_1(i,j,0) + var(p_float32, "b") * C(i,j) , true, p_float32, &function0);
    computation result_p2("[N, M, K]->{result_p2[i,j]: "+B0s+"*floor(N/"+B0s+")<=i<N and "+B1s+"*floor(M/"+B1s+")<=j<M}",              var(p_float32, "a") * reduced_AB_1(i,j,0) + var(p_float32, "b") * C(i,j) , true, p_float32, &function0);



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
    buffer parameters_buf("parameters_buf", {7}, p_int32, a_input, &function0);
    buffer input_padded_buf("input_padded_buf", {N_B, expr(W) + K_W - 1, expr(H) + K_H - 1, F_In}, p_float32, a_input, &function0);
    buffer input_col_buf("input_col_buf", {N_B, W, H, F_In, K_W, K_H}, p_float32, a_output, &function0);

    parameters.set_access("{parameters[i]->parameters_buf[i]}");
    input_padded.set_access("{input_padded[n, x, y, c]->input_padded_buf[n, x, y, c]}");
    input_col.set_access("{input_col[n, x, y, c, k_x, k_y]->input_col_buf[n, x, y, c, k_x, k_y]}");


    buffer buf_A("buf_A", {N, K}, p_float32, a_input, &function0);
    buffer buf_B("buf_B", {K, M}, p_float32, a_input, &function0);
    buffer buf_Bp("buf_Bp", {(expr(M)/B1), K, expr(M)%B1}, p_float32, a_temporary, &function0);
    buffer buf_temps("buf_temp", {N, M}, p_float32, a_temporary, &function0);
    buffer buf_C("buf_C", {N, M}, p_float32, a_output, &function0);

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
    function0.set_arguments({&parameters_buf, &input_padded_buf, &input_col_buf, &buf_A, &buf_B, &buf_C});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("fct.o");
}

int main(int argc, char **argv)
{
    generate_function("gemm_conv");
    return 0;
}
