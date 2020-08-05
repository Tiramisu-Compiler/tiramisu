#define __TIRAMISU_GENERATOR__
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
#include "configure.h"
using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    std::string FOUT_BLs = std::to_string(FOUT_BL);
    std::string FOUT_NB_BLs = std::to_string(FOUT_NB_BL);
    std::string X_BL1s = std::to_string(X_BL1);
    std::string Y_BL1s = std::to_string(Y_BL1);
    std::string X_NB_BL1s = std::to_string(X_NB_BL1);
    std::string Y_NB_BL1s = std::to_string(Y_NB_BL1);

    std::string X_BL2s = std::to_string(X_BL2);
    std::string Y_BL2s = std::to_string(Y_BL2);
    std::string X_NB_BL2s = std::to_string(X_NB_BL2);
    std::string Y_NB_BL2s = std::to_string(Y_NB_BL2);


    function sparse_conv_relu_conv_relu("sparse_conv_relu_conv_relu_512_512_14_tiramisu");

    // Inputs
    computation SIZES("{SIZES[e]: 0<=e<1}", expr(), false, p_int32, &sparse_conv_relu_conv_relu);
    computation c_input("[BATCHSIZE,IND_RANGE1]->{c_input[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE1}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);


    // First convolution weights
    computation c_filter_values1("[FNNZ1]->{c_filter_values1[k1]: 0<=k1<FNNZ1}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);
    computation c_filter_idx1("[FNNZ1]->{c_filter_idx1[k1]: 0<=k1<FNNZ1}", expr(), false, p_int32, &sparse_conv_relu_conv_relu);
    computation c_filter_finptr1("[filter_finptr_size]->{c_filter_finptr1[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &sparse_conv_relu_conv_relu);

    computation c_bias1("[F_Out]->{c_bias1[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);

    // Second convolution weights
    computation c_input2_view("[BATCHSIZE,IND_RANGE2]->{c_input2_view[b,ind2]: 0<=b<BATCHSIZE and 0<=ind2<IND_RANGE2}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);

    computation c_filter_values2("[FNNZ2]->{c_filter_values2[k2]: 0<=k2<FNNZ2}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);
    computation c_filter_idx2("[FNNZ2]->{c_filter_idx2[k2]: 0<=k2<FNNZ2}", expr(), false, p_int32, &sparse_conv_relu_conv_relu);
    computation c_filter_finptr2("[filter_finptr_size]->{c_filter_finptr2[fin2]: 0<=fin2<filter_finptr_size}", expr(), false, p_int32, &sparse_conv_relu_conv_relu);

    computation c_bias2("[F_Out]->{c_bias2[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &sparse_conv_relu_conv_relu);


    constant IND_RANGE1("IND_RANGE1",(N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant IND_RANGE2("IND_RANGE2",(N + 2) * (N + 2) * FOut, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant FNNZ1("FNNZ1", SIZES(0), p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant FNNZ2("FNNZ2", SIZES(1), p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);

    constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant F_In("F_In", FIn, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant height("height", N + 2, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant width("width", N + 2, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);

    constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant KK("KK", K, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);

    constant output_height("output_height", N, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);
    constant output_width("output_width", N, p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);

    constant filter_finptr_size("filter_finptr_size", expr(FOut + 1), p_int32, true, NULL, 0, &sparse_conv_relu_conv_relu);

    var k1("k1"), l("l"), fout("fout"), b("b"), fin("fin");
    var output_x("output_x"), output_y("output_y");
    var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");

    var k2("k2");
    var fout_b("fout_b"), ffout("ffout");

    #if PAD_OUTPUT
    computation result_init_zero_for_padding("[BATCHSIZE,F_Out,height,width]->{result_init_zero_for_padding[b,fout,y,x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y<height and 0<=x<width}", cast(p_float32, 0), true, p_float32, &sparse_conv_relu_conv_relu);
    #endif

    //First convolution + ReLU
    computation conv1_init_zero("[BATCHSIZE,height,width]->{conv1_init_zero[b,fout_b,ffout,output_y,output_x]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=output_y<height and 0<=output_x<width}",  cast(p_float32, 0), true, p_float32, &sparse_conv_relu_conv_relu);

    computation conv1_init("[BATCHSIZE]->{conv1_init[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL1s+" and 0<=xx<"+X_BL1s+" and 0<=y_b<"+Y_NB_BL1s+" and 0<=yy<"+Y_BL1s+"}",  c_bias1(fout_b * FOUT_BL + ffout), true, p_float32, &sparse_conv_relu_conv_relu);
    computation conv1_relu_store("[BATCHSIZE]->{conv1_relu_store[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL1s+" and 0<=xx<"+X_BL1s+" and 0<=y_b<"+Y_NB_BL1s+" and 0<=yy<"+Y_BL1s+"}",  expr(), true, p_float32, &sparse_conv_relu_conv_relu);

    computation convolve1("[BATCHSIZE,k1_range0,k1_range1]->{convolve1[b,fout_b,y_b,x_b,ffout,k1,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL1s+" and 0<=xx<"+X_BL1s+" and 0<=y_b<"+Y_NB_BL1s+" and 0<=yy<"+Y_BL1s+" and k1_range0<=k1<k1_range1}", expr(), true, p_float32, &sparse_conv_relu_conv_relu);
    conv1_relu_store.set_expression(expr(o_max, 0.f, convolve1(b, fout_b, y_b, x_b, ffout, 0, yy, xx)));
    // Loop invariants
    constant filter_k1("filter_k1", c_filter_idx1(k1), p_int32, false, &convolve1, 5, &sparse_conv_relu_conv_relu);

    constant k1_range0("k1_range0", c_filter_finptr1(fout_b * FOUT_BL + ffout), p_int32, false, &convolve1, 4, &sparse_conv_relu_conv_relu);
    constant k1_range1("k1_range1", c_filter_finptr1(fout_b * FOUT_BL + ffout + 1), p_int32, false, &convolve1, 4, &sparse_conv_relu_conv_relu);

    convolve1.set_expression(convolve1(b, fout_b, y_b, x_b, ffout, k1, yy, xx) + c_input(b, (y_b * Y_BL1 + yy) * (N + 2) + x_b * X_BL1 + xx) * c_filter_values1(k1));

    // Second convolution + ReLU

    computation conv2_init("[BATCHSIZE]->{conv2_init[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+"}",  c_bias2(fout_b * FOUT_BL + ffout), true, p_float32, &sparse_conv_relu_conv_relu);
    computation conv2_relu_store("[BATCHSIZE]->{conv2_relu_store[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+"}",  expr(), true, p_float32, &sparse_conv_relu_conv_relu);

    computation convolve2("[BATCHSIZE,k2_range0,k2_range1]->{convolve2[b,fout_b,y_b,x_b,ffout,k2,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+" and k2_range0<=k2<k2_range1}", expr(), true, p_float32, &sparse_conv_relu_conv_relu);
    conv2_relu_store.set_expression(expr(o_max, 0.f, convolve2(b, fout_b, y_b, x_b, ffout, 0, yy, xx)));

    // Loop invariants
    constant filter_k2("filter_k2", c_filter_idx2(k2), p_int32, false, &convolve2, 5, &sparse_conv_relu_conv_relu);

    constant k2_range0("k2_range0", c_filter_finptr2(fout_b * FOUT_BL + ffout), p_int32, false, &convolve2, 4, &sparse_conv_relu_conv_relu);
    constant k2_range1("k2_range1", c_filter_finptr2(fout_b * FOUT_BL + ffout + 1), p_int32, false, &convolve2, 4, &sparse_conv_relu_conv_relu);

    convolve2.set_expression(convolve2(b, fout_b, y_b, x_b, ffout, k2, yy, xx) + c_input2_view(b, (y_b * Y_BL2 + yy) * (N + 2) + x_b * X_BL2 + xx) * c_filter_values2(k2));

    sparse_conv_relu_conv_relu.set_context_set("[k1_range0,k1_range1,k2_range0,k2_range1]->{: k1_range0>0 and k1_range1>0 and k1_range1>k1_range0 and k2_range0>0 and k2_range1>0 and k2_range1>k2_range0}");

    // -----------------------------------------------------------------
    // Layer II
    // -----------------------------------------------------------------

    // Temporary buffer for the first convolution
    buffer b_workspace1("b_workspace1", {Y_BL1, X_BL1}, p_float32, a_temporary, &sparse_conv_relu_conv_relu);
    computation *alloc1 = b_workspace1.allocate_at(conv1_init, ffout);

    // Temporary buffer for the second convolution
    buffer b_workspace2("b_workspace2", {Y_BL2, X_BL2}, p_float32, a_temporary, &sparse_conv_relu_conv_relu);
    computation *alloc2 = b_workspace2.allocate_at(conv2_init, ffout);

    #if PAD_OUTPUT
      result_init_zero_for_padding.after_low_level(conv1_init_zero, computation::root_dimension);
      k1_range0.after_low_level(result_init_zero_for_padding, computation::root_dimension);
    #else
      k1_range0.after_low_level(conv1_init_zero, computation::root_dimension);
    #endif
    k1_range1.after_low_level(k1_range0, 4);
    alloc1->after_low_level(k1_range1, 4);
    conv1_init.after_low_level(*alloc1, 4);
    filter_k1.after_low_level(conv1_init, 4);
    convolve1.after_low_level(filter_k1, 5);
    conv1_relu_store.after_low_level(convolve1, 4);

    k2_range0.after_low_level(conv1_relu_store, 0);
    k2_range1.after_low_level(k2_range0, 4);
    alloc2->after_low_level(k2_range1, 4);
    conv2_init.after_low_level(*alloc2, 4);
    filter_k2.after_low_level(conv2_init,4);
    convolve2.after_low_level(filter_k2, 5);
    conv2_relu_store.after_low_level(convolve2, 4);

    convolve1.tag_parallel_level(0);
    convolve1.tag_parallel_level(1);
    if(X_BL1<=8)
      convolve1.tag_vector_level(xx, X_BL1);

    convolve2.tag_parallel_level(0);
    convolve2.tag_parallel_level(1);

    if(X_BL2<=8)
      convolve2.tag_vector_level(xx, X_BL2);

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    // Input
    buffer b_SIZES("b_SIZES", {expr(2)}, p_int32, a_input, &sparse_conv_relu_conv_relu);

    buffer b_input("b_input", {BATCHSIZE, expr(F_In)*expr(height)*expr(width)}, p_float32, a_input, &sparse_conv_relu_conv_relu);

    // Conv1 Weights in CSR format
    buffer b_filter_values1("b_filter_values1", {expr(FNNZ1)}, p_float32, a_input, &sparse_conv_relu_conv_relu);
    buffer b_filter_idx1("b_filter_idx1", {FNNZ1}, p_int32, a_input, &sparse_conv_relu_conv_relu);
    buffer b_filter_finptr1("b_filter_finptr1", {expr(filter_finptr_size)}, p_int32, a_input, &sparse_conv_relu_conv_relu);

    buffer b_bias1("b_bias1", {F_Out}, p_float32, a_input, &sparse_conv_relu_conv_relu);

    // Conv2 Weights in CSR format
    buffer b_filter_values2("b_filter_values2", {expr(FNNZ2)}, p_float32, a_input, &sparse_conv_relu_conv_relu);
    buffer b_filter_idx2("b_filter_idx2", {FNNZ2}, p_int32, a_input, &sparse_conv_relu_conv_relu);
    buffer b_filter_finptr2("b_filter_finptr2", {expr(filter_finptr_size)}, p_int32, a_input, &sparse_conv_relu_conv_relu);

    buffer b_bias2("b_bias2", {F_Out}, p_float32, a_input, &sparse_conv_relu_conv_relu);

    // Output
    buffer b_conv1("b_conv1", {BATCHSIZE, F_Out, height, width}, p_float32, a_input, &sparse_conv_relu_conv_relu);
    buffer b_result("b_result", {BATCH_SIZE, F_Out, N + 2 * PAD_OUTPUT, N + 2 * PAD_OUTPUT}, p_float32, a_output, &sparse_conv_relu_conv_relu);

    // Mapping computations
    // First convolution inputs
    SIZES.set_access("{SIZES[e]->b_SIZES[e]}");
    c_input.set_access("[filter_k1]->{c_input[b,ind]->b_input[b,ind + filter_k1]}");

    c_filter_values1.set_access("{c_filter_values1[k1]->b_filter_values1[k1]}");
    c_filter_idx1.set_access("{c_filter_idx1[k1]->b_filter_idx1[k1]}");
    c_filter_finptr1.set_access("{c_filter_finptr1[fin]->b_filter_finptr1[fin]}");

    c_bias1.set_access("{c_bias1[fout]->b_bias1[fout]}");

    // Second convolution inputs
    c_input2_view.set_access("[filter_k2]->{c_input2_view[b,ind2]->b_conv1[b,0,0,ind2 + filter_k2]}");

    c_filter_values2.set_access("{c_filter_values2[k2]->b_filter_values2[k2]}");
    c_filter_idx2.set_access("{c_filter_idx2[k2]->b_filter_idx2[k2]}");
    c_filter_finptr2.set_access("{c_filter_finptr2[fin2]->b_filter_finptr2[fin2]}");

    c_bias2.set_access("{c_bias2[fout]->b_bias2[fout]}");

    // First convolution computations mapping

    conv1_init_zero.set_access("{conv1_init_zero[b,fout_b,ffout,output_y,output_x]->b_conv1[b, fout_b * "+FOUT_BLs+" + ffout, output_y, output_x]}");

    conv1_init.set_access("{conv1_init[b,fout_b,y_b,x_b,ffout,yy,xx]->b_workspace1[yy,xx]}");

    conv1_relu_store.set_access("{conv1_relu_store[b,fout_b,y_b,x_b,ffout,yy,xx]->b_conv1[b, fout_b * "+FOUT_BLs+" + ffout, y_b *"+Y_BL1s+" + yy + 1, x_b *"+X_BL1s+" + xx + 1]}");

    convolve1.set_access("{convolve1[b,fout_b,y_b,x_b,ffout,k1,yy,xx]->b_workspace1[yy,xx]}");

    // Second convolution computations mapping

    conv2_init.set_access("{conv2_init[b,fout_b,y_b,x_b,ffout,yy,xx]->b_workspace2[yy,xx]}");

    #if PAD_OUTPUT
      result_init_zero_for_padding.set_access("{result_init_zero_for_padding[b,fout,y,x]->b_result[b,fout,y,x]}");
      conv2_relu_store.set_access("{conv2_relu_store[b,fout_b,y_b,x_b,ffout,yy,xx]->b_result[b,fout_b * "+FOUT_BLs+" + ffout,(y_b*"+Y_BL2s+"+yy) + 1,(x_b * "+X_BL2s+" + xx) + 1]}");
    #else
      conv2_relu_store.set_access("{conv2_relu_store[b,fout_b,y_b,x_b,ffout,yy,xx]->b_result[b,fout_b * "+FOUT_BLs+" + ffout,(y_b*"+Y_BL2s+"+yy),(x_b * "+X_BL2s+" + xx)]}");
    #endif
    convolve2.set_access("{convolve2[b,fout_b,y_b,x_b,ffout,k2,yy,xx]->b_workspace2[yy,xx]}");

    // ------------------------------------------------------------------
    // Generate code &b_idx, &b_finptr,
    // ------------------------------------------------------------------
    sparse_conv_relu_conv_relu.set_arguments({&b_SIZES, &b_input, &b_filter_values1, &b_filter_idx1, &b_filter_finptr1, &b_bias1, &b_conv1 , &b_filter_values2, &b_filter_idx2, &b_filter_finptr2, &b_bias2, &b_result});
    sparse_conv_relu_conv_relu.gen_time_space_domain();
    sparse_conv_relu_conv_relu.gen_isl_ast();
    sparse_conv_relu_conv_relu.gen_halide_stmt();
    sparse_conv_relu_conv_relu.gen_halide_obj("generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o");

    return 0;
}
