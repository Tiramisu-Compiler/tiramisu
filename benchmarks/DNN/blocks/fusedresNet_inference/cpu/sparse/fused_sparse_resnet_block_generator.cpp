#include <tiramisu/core.h>

#include <string.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
  // Set default tiramisu options.
  global::set_default_tiramisu_options();

  function fused_sparse_resnet_block("fused_sparse_resnet_block");
  std::string FOUT_BLs = std::to_string(FOUT_BL);
  std::string FOUT_NB_BLs = std::to_string(FOUT_NB_BL);
  std::string X_BLs = std::to_string(X_BL);
  std::string Y_BLs = std::to_string(Y_BL);
  std::string X_NB_BLs = std::to_string(X_NB_BL);
  std::string Y_NB_BLs = std::to_string(Y_NB_BL);

  std::string FOUT_BL2s = std::to_string(FOUT_BL2);
  std::string FOUT_NB_BL2s = std::to_string(FOUT_NB_BL2);
  std::string X_BL2s = std::to_string(X_BL2);
  std::string Y_BL2s = std::to_string(Y_BL2);
  std::string X_NB_BL2s = std::to_string(X_NB_BL2);
  std::string Y_NB_BL2s = std::to_string(Y_NB_BL2);

  // Iteration variables
  var k("k"), fout("fout"), b("b"), fin("fin");
  var output_x("output_x"), output_y("output_y");
  var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");
  var fout_b("fout_b"), ffout("ffout");
  var x("x"), y("y");
  // Inputs
  // First convolution and bn inputs
  computation SIZES("{SIZES[e]: 0<=e<2}", expr(), false, p_int32, &fused_sparse_resnet_block);
  computation c_input("[BATCHSIZE,IND_RANGE]->{c_input[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_filter_values("[FNNZ]->{c_filter_values[k]: 0<=k<FNNZ}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_filter_idx("[FNNZ]->{c_filter_idx[k]: 0<=k<FNNZ}", expr(), false, p_int32, &fused_sparse_resnet_block);
  computation c_filter_finptr("[filter_finptr_size]->{c_filter_finptr[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &fused_sparse_resnet_block);

  computation c_bias("[F_Out]->{c_bias[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_bn_scale("[F_Out]->{c_bn_scale[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_bn_shift("[F_Out]->{c_bn_shift[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_bn_mean("[F_Out]->{c_bn_mean[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_bn_variance("[F_Out]->{c_bn_variance[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  // Second convolution and bn inputs
  computation c_input2_view("[BATCHSIZE,IND_RANGE2]->{c_input2_view[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE2}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_filter_values2("[FNNZ2]->{c_filter_values2[k]: 0<=k<FNNZ2}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_filter_idx2("[FNNZ2]->{c_filter_idx2[k]: 0<=k<FNNZ2}", expr(), false, p_int32, &fused_sparse_resnet_block);
  computation c_filter_finptr2("[filter_finptr_size]->{c_filter_finptr2[fin2]: 0<=fin2<filter_finptr_size}", expr(), false, p_int32, &fused_sparse_resnet_block);

  computation c_bias2("[F_Out]->{c_bias2[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_bn2_scale("[F_Out]->{c_bn2_scale[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_bn2_shift("[F_Out]->{c_bn2_shift[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  computation c_bn2_mean("[F_Out]->{c_bn2_mean[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);
  computation c_bn2_variance("[F_Out]->{c_bn2_variance[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &fused_sparse_resnet_block);

  constant IND_RANGE("IND_RANGE",(N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant FNNZ("FNNZ", SIZES(0), p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant IND_RANGE2("IND_RANGE2",(N + 2) * (N + 2) * FOut, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant FNNZ2("FNNZ2", SIZES(1), p_int32, true, NULL, 0, &fused_sparse_resnet_block);

  constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant F_In("F_In", FIn, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant height("height", N + 2, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant width("width", N + 2, p_int32, true, NULL, 0, &fused_sparse_resnet_block);

  constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant KK("KK", K, p_int32, true, NULL, 0, &fused_sparse_resnet_block);

  constant output_height("output_height", N, p_int32, true, NULL, 0, &fused_sparse_resnet_block);
  constant output_width("output_width", N, p_int32, true, NULL, 0, &fused_sparse_resnet_block);

  constant filter_finptr_size("filter_finptr_size", expr(FOut + 1), p_int32, true, NULL, 0, &fused_sparse_resnet_block);

  // First Convolution
  computation conv_output_init_zero("[BATCHSIZE,F_Out,height,width]->{conv_output_init_zero[b,fout,y,x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y<height and 0<=x<width}", cast(p_float32, 0), true, p_float32, &fused_sparse_resnet_block);

  computation init_conv("[BATCHSIZE]->{init_conv[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BLs+" and 0<=xx<"+X_BLs+" and 0<=y_b<"+Y_NB_BLs+" and 0<=yy<"+Y_BLs+"}",  c_bias(fout_b * FOUT_BL + ffout), true, p_float32, &fused_sparse_resnet_block);

  computation convolve("[BATCHSIZE,k_range0,k_range1]->{convolve[b,fout_b,y_b,x_b,ffout,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BLs+" and 0<=xx<"+X_BLs+" and 0<=y_b<"+Y_NB_BLs+" and 0<=yy<"+Y_BLs+" and k_range0<=k<k_range1}", expr(), true, p_float32, &fused_sparse_resnet_block);

  computation store_conv_bn_relu("[BATCHSIZE]->{store_conv_bn_relu[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BLs+" and 0<=ffout<"+FOUT_BLs+" and 0<=x_b<"+X_NB_BLs+" and 0<=xx<"+X_BLs+" and 0<=y_b<"+Y_NB_BLs+" and 0<=yy<"+Y_BLs+"}",  expr(), true, p_float32, &fused_sparse_resnet_block);

  // Loop invariants
  constant filter_k("filter_k", c_filter_idx(k), p_int32, false, &convolve, 5, &fused_sparse_resnet_block);

  constant k_range0("k_range0", c_filter_finptr(fout_b * FOUT_BL + ffout), p_int32, false, &convolve, 4, &fused_sparse_resnet_block);
  constant k_range1("k_range1", c_filter_finptr(fout_b * FOUT_BL + ffout + 1), p_int32, false, &convolve, 4, &fused_sparse_resnet_block);

  convolve.set_expression(convolve(b, fout_b, y_b, x_b, ffout, k, yy, xx) + c_input(b, (y_b * Y_BL + yy) * (N + 2) + x_b * X_BL + xx) * c_filter_values(k));

  store_conv_bn_relu.set_expression(expr(o_max, 0.f, c_bn_scale(fout_b * FOUT_BL + ffout) * ((convolve(b, fout_b, y_b, x_b, ffout, 0, yy, xx) - c_bn_mean(fout_b * FOUT_BL + ffout))/ expr(o_sqrt, c_bn_variance(fout_b * FOUT_BL + ffout) + cast(p_float32, EPSILON))) + c_bn_shift(fout_b * FOUT_BL + ffout)));

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
  // Second Convolution

  computation init_conv2("[BATCHSIZE]->{init_conv2[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BL2s+" and 0<=ffout<"+FOUT_BL2s+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+"}", c_bias2(fout_b * FOUT_BL + ffout), true, p_float32, &fused_sparse_resnet_block);
  computation store_conv_bn2("[BATCHSIZE]->{store_conv_bn2[b,fout_b,y_b,x_b,ffout,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BL2s+" and 0<=ffout<"+FOUT_BL2s+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+"}", expr(), true, p_float32, &fused_sparse_resnet_block);

  computation convolve2("[BATCHSIZE,k2_range0,k2_range1]->{convolve2[b,fout_b,y_b,x_b,ffout,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout_b<"+FOUT_NB_BL2s+" and 0<=ffout<"+FOUT_BL2s+" and 0<=x_b<"+X_NB_BL2s+" and 0<=xx<"+X_BL2s+" and 0<=y_b<"+Y_NB_BL2s+" and 0<=yy<"+Y_BL2s+" and k2_range0<=k<k2_range1}", expr(), true, p_float32, &fused_sparse_resnet_block);

  computation conv2_result_view("[BATCHSIZE,F_Out,output_height,output_width]->{conv2_result_view[b,fout,y,x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y<output_height and 0<=x<output_width}", expr(), false, p_float32, &fused_sparse_resnet_block);
  // Loop invariants
  constant filter_k2("filter_k2", c_filter_idx2(k), p_int32, false, &convolve2, 5, &fused_sparse_resnet_block);

  constant k2_range0("k2_range0", c_filter_finptr2(fout_b * FOUT_BL2 + ffout), p_int32, false, &convolve2, 4, &fused_sparse_resnet_block);
  constant k2_range1("k2_range1", c_filter_finptr2(fout_b * FOUT_BL2 + ffout + 1), p_int32, false, &convolve2, 4, &fused_sparse_resnet_block);

  convolve2.set_expression(convolve2(b, fout_b, y_b, x_b, ffout, k, yy, xx) + c_input2_view(b, (y_b * Y_BL2 + yy) * (N + 2) + x_b * X_BL2 + xx) * c_filter_values2(k));

  // First Batchnorm + ReLU

  store_conv_bn2.set_expression((c_bn2_scale(fout_b * FOUT_BL + ffout) * ((convolve2(b, fout_b, y_b, x_b, ffout, 0, yy, xx) - c_bn2_mean(fout_b * FOUT_BL + ffout)) / expr(o_sqrt, c_bn2_variance(fout_b * FOUT_BL + ffout) + cast(p_float32, EPSILON)))) + c_bn2_shift(fout_b * FOUT_BL + ffout));

  fused_sparse_resnet_block.set_context_set("[k_range0,k_range1,k2_range0,k2_range1]->{: k_range0>0 and k_range1>0 and k_range1>k_range0 and k2_range0>0 and k2_range1>0 and k2_range1>k2_range0}");

  // -----------------------------------------------------------------
  // Layer II
  // -----------------------------------------------------------------
  // We need to allocate the temporary register at level ffout to get private array per thread
  buffer b_workspace("b_workspace", {Y_BL, X_BL}, p_float32, a_temporary, &fused_sparse_resnet_block);
  computation *alloc = b_workspace.allocate_at(init_conv, ffout);

  k_range0.after_low_level(conv_output_init_zero, computation::root_dimension);
  k_range1.after_low_level(k_range0, 4);
  alloc->after_low_level(k_range1, 4);
  init_conv.after_low_level(*alloc, 4);
  filter_k.after_low_level(init_conv, 4);
  convolve.after_low_level(filter_k, 5);
  store_conv_bn_relu.after_low_level(convolve, 4);

  buffer b_workspace2("b_workspace2", {Y_BL2, X_BL2}, p_float32, a_temporary, &fused_sparse_resnet_block);
  computation *alloc2 = b_workspace2.allocate_at(init_conv2, ffout);

  k2_range0.after_low_level(store_conv_bn_relu, 0);
  k2_range1.after_low_level(k2_range0, 4);
  alloc2->after_low_level(k2_range1, 4);
  init_conv2.after_low_level(*alloc2, 4);
  filter_k2.after_low_level(init_conv2, 4);
  convolve2.after_low_level(filter_k2, 5);
  store_conv_bn2.after_low_level(convolve2, 4);

  // Parallelization
  convolve.parallelize(b);

  convolve2.parallelize(b);

  // ---------------------------------------------------------------------------------
  // Layer III
  // ---------------------------------------------------------------------------------
  // Input
  buffer b_SIZES("b_SIZES", {expr(2)}, p_int32, a_input, &fused_sparse_resnet_block);

  buffer b_input("b_input", {BATCHSIZE, expr(F_In)*expr(height)*expr(width)}, p_float32, a_input, &fused_sparse_resnet_block);

  // First convolution Weights in CSR format
  buffer b_filter_values("b_filter_values", {expr(FNNZ)}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_filter_idx("b_filter_idx", {FNNZ}, p_int32, a_input, &fused_sparse_resnet_block);
  buffer b_filter_finptr("b_filter_finptr", {expr(filter_finptr_size)}, p_int32, a_input, &fused_sparse_resnet_block);

  buffer b_bias("b_bias", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  buffer b_bn_scale("b_bn_scale", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_bn_shift("b_bn_shift", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  buffer b_bn_mean("b_bn_mean", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_bn_variance("b_bn_variance", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  buffer b_conv1_output("b_conv1_output", {BATCH_SIZE, FOut, N + 2, N + 2}, p_float32, a_input, &fused_sparse_resnet_block);

  // Second convolution Weights in CSR format
  buffer b_filter_values2("b_filter_values2", {FNNZ2}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_filter_idx2("b_filter_idx2", {FNNZ2}, p_int32, a_input, &fused_sparse_resnet_block);
  buffer b_filter_finptr2("b_filter_finptr2", {expr(filter_finptr_size)}, p_int32, a_input, &fused_sparse_resnet_block);

  buffer b_bias2("b_bias2", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  buffer b_bn2_scale("b_bn2_scale", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_bn2_shift("b_bn2_shift", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  buffer b_bn2_mean("b_bn2_mean", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);
  buffer b_bn2_variance("b_bn2_variance", {F_Out}, p_float32, a_input, &fused_sparse_resnet_block);

  // Output
  buffer b_result("b_result", {BATCHSIZE, F_Out, N, N}, p_float32, a_output, &fused_sparse_resnet_block);

  // Mapping computations
  SIZES.set_access("{SIZES[e]->b_SIZES[e]}");
  c_input.set_access("[filter_k]->{c_input[b,ind]->b_input[b,ind + filter_k]}");

  c_filter_values.set_access("{c_filter_values[k]->b_filter_values[k]}");
  c_filter_idx.set_access("{c_filter_idx[k]->b_filter_idx[k]}");
  c_filter_finptr.set_access("{c_filter_finptr[fin]->b_filter_finptr[fin]}");

  c_bias.set_access("{c_bias[fout]->b_bias[fout]}");

  // BN
  c_bn_scale.set_access("{c_bn_scale[fout]->b_bn_scale[fout]}");
  c_bn_shift.set_access("{c_bn_shift[fout]->b_bn_shift[fout]}");

  c_bn_mean.set_access("{c_bn_mean[fout]->b_bn_mean[fout]}");
  c_bn_variance.set_access("{c_bn_variance[fout]->b_bn_variance[fout]}");

  conv_output_init_zero.set_access("{conv_output_init_zero[b,fout,y,x]->b_conv1_output[b,fout,y,x]}");

  init_conv.set_access("{init_conv[b,fout_b,y_b,x_b,ffout,yy,xx]->b_workspace[yy,xx]}");

  store_conv_bn_relu.set_access("{store_conv_bn_relu[b,fout_b,y_b,x_b,ffout,yy,xx]->b_conv1_output[b,fout_b * "+FOUT_BLs+"+ffout,y_b * "+Y_BLs+"+yy + 1,x_b * "+X_BLs+" + xx + 1]}");

  convolve.set_access("{convolve[b,fout_b,y_b,x_b,ffout,k,yy,xx]->b_workspace[yy,xx]}");

  //View for next convolution input
  c_input2_view.set_access("[filter_k2]->{c_input2_view[b,ind]->b_conv1_output[b,0,0,ind + filter_k2]}");

  // Second convolution
  c_filter_values2.set_access("{c_filter_values2[k]->b_filter_values2[k]}");
  c_filter_idx2.set_access("{c_filter_idx2[k]->b_filter_idx2[k]}");
  c_filter_finptr2.set_access("{c_filter_finptr2[fin]->b_filter_finptr2[fin]}");

  c_bias2.set_access("{c_bias2[fout]->b_bias2[fout]}");

  //BN2
  c_bn2_scale.set_access("{c_bn2_scale[fout]->b_bn2_scale[fout]}");
  c_bn2_shift.set_access("{c_bn2_shift[fout]->b_bn2_shift[fout]}");

  c_bn2_mean.set_access("{c_bn2_mean[fout]->b_bn2_mean[fout]}");
  c_bn2_variance.set_access("{c_bn2_variance[fout]->b_bn2_variance[fout]}");

  init_conv2.set_access("{init_conv2[b,fout_b,y_b,x_b,ffout,yy,xx]->b_workspace2[yy,xx]}");

  store_conv_bn2.set_access("{store_conv_bn2[b,fout_b,y_b,x_b,ffout,yy,xx]->b_result[b,fout_b * "+FOUT_BL2s+"+ffout,y_b * "+Y_BL2s+"+yy,x_b * "+X_BL2s+" + xx]}");

  convolve2.set_access("{convolve2[b,fout_b,y_b,x_b,ffout,k,yy,xx]->b_workspace2[yy,xx]}");

  // ------------------------------------------------------------------
  // Generate code &b_idx, &b_finptr,
  // ------------------------------------------------------------------
  fused_sparse_resnet_block.set_arguments({&b_SIZES,
                        &b_input,
                              &b_filter_values,
                              &b_filter_idx,
                              &b_filter_finptr,
                              &b_bias,
                              &b_bn_scale,
                              &b_bn_shift,
                              &b_bn_mean,
                              &b_bn_variance,
                        &b_conv1_output,
                              &b_filter_values2,
                              &b_filter_idx2,
                              &b_filter_finptr2,
                              &b_bias2,
                              &b_bn2_scale,
                              &b_bn2_shift,
                              &b_bn2_mean,
                              &b_bn2_variance,
                        &b_result});
  fused_sparse_resnet_block.gen_time_space_domain();
  fused_sparse_resnet_block.gen_isl_ast();
  fused_sparse_resnet_block.gen_halide_stmt();
  fused_sparse_resnet_block.gen_halide_obj("generated_fused_sparse_resnet_block.o");

  return 0;
}
