#include <tiramisu/core.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
  // Set default tiramisu options.
  global::set_default_tiramisu_options();

  function spconv_relu_maxpool("spconv_relu_maxpool");

  // Inputs
  computation SIZES("{SIZES[e]: 0<=e<1}", expr(), false, p_int32, &spconv_relu_maxpool);
  computation c_input("[BATCHSIZE,IND_RANGE]->{c_input[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE}", expr(), false, p_float32, &spconv_relu_maxpool);

  // Weights in CSR format
  computation c_filter_values("[FNNZ]->{c_filter_values[k]: 0<=k<FNNZ}", expr(), false, p_float32, &spconv_relu_maxpool);
  computation c_filter_idx("[FNNZ]->{c_filter_idx[k]: 0<=k<FNNZ}", expr(), false, p_int32, &spconv_relu_maxpool);
  computation c_filter_finptr("[filter_finptr_size]->{c_filter_finptr[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &spconv_relu_maxpool);

  computation c_bias("[F_Out]->{c_bias[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &spconv_relu_maxpool);

  constant filter_finptr_size("filter_finptr_size", expr(FOut + 1), p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant IND_RANGE("IND_RANGE",(N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant FNNZ("FNNZ", SIZES(0), p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant F_In("F_In", FIn, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant height("height", N + 2, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant width("width", N + 2, p_int32, true, NULL, 0, &spconv_relu_maxpool);

  constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant KK("KK", K, p_int32, true, NULL, 0, &spconv_relu_maxpool);

  // Convolution output dimensions
  constant conv_output_height("conv_output_height", N, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant conv_output_width("conv_output_width", N, p_int32, true, NULL, 0, &spconv_relu_maxpool);

  // Maxpool output dimensions
  constant maxpool_height("maxpool_height", N / 2 + 2 * PAD_OUTPUT, p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant maxpool_width("maxpool_width", N / 2 + 2 * PAD_OUTPUT, p_int32, true, NULL, 0, &spconv_relu_maxpool);

  // X and Y blocking constants
  constant X_BLOCKING("X_BLOCKING", expr(X_BL), p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant X_NB_BLOCKS("X_NB_BLOCKS", expr(X_NB_BL), p_int32, true, NULL, 0, &spconv_relu_maxpool);

  constant Y_BLOCKING("Y_BLOCKING", expr(Y_BL), p_int32, true, NULL, 0, &spconv_relu_maxpool);
  constant Y_NB_BLOCKS("Y_NB_BLOCKS", expr(Y_NB_BL), p_int32, true, NULL, 0, &spconv_relu_maxpool);

  var k("k"), fout("fout"), b("b"), fin("fin");
  var output_x("output_x"), output_y("output_y");
  var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");
  //Initialize the output
  computation init_conv("[BATCHSIZE,F_Out,Y_NB_BLOCKS,Y_BLOCKING, X_BLOCKING,X_NB_BLOCKS]->{init_conv[b,fout,y_b,x_b,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING}",  c_bias(fout), true, p_float32, &spconv_relu_maxpool);
  computation init_output("[BATCHSIZE,F_Out,maxpool_height,maxpool_width]->{init_output[b,fout,output_y, output_x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=output_y<maxpool_height and 0<=output_x<maxpool_width}",  cast(p_float32, 0), true, p_float32, &spconv_relu_maxpool);

  computation maxpool("[BATCHSIZE,F_Out,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS,Y_NB_BLOCKS]->{maxpool[b,fout,y_b,x_b,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING}",  expr(), true, p_float32, &spconv_relu_maxpool);

  computation convolve("[BATCHSIZE,F_Out,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS,Y_NB_BLOCKS,k_range0,k_range1]->{convolve[b,fout,y_b,x_b,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and k_range0<=k<k_range1}", expr(), true, p_float32, &spconv_relu_maxpool);
  maxpool.set_expression(expr(o_max, maxpool(b, fout, y_b, x_b, yy, xx), convolve(b, fout, y_b, x_b, 0, yy, xx)));

  // Loop invariants
  constant filter_k("filter_k", c_filter_idx(k), p_int32, false, &convolve, 4, &spconv_relu_maxpool);

  constant k_range0("k_range0", c_filter_finptr(fout), p_int32, false, &convolve, 2, &spconv_relu_maxpool);
  constant k_range1("k_range1", c_filter_finptr(fout + 1), p_int32, false, &convolve, 2, &spconv_relu_maxpool);

  convolve.set_expression(init_conv(b, fout, y_b, x_b, yy, xx) + c_input(b, (y_b * Y_BL + yy) * (N + 2) + x_b * X_BL + xx) * c_filter_values(k));

  spconv_relu_maxpool.set_context_set("[k_range0,k_range1]->{: k_range0>0 and k_range1>0 and k_range1>k_range0}");

  // -----------------------------------------------------------------
  // Layer II
  // -----------------------------------------------------------------
  // Buffer for storing accumulations temporarly
  buffer b_workspace("b_workspace", {Y_BL, X_BL}, p_float32, a_temporary, &spconv_relu_maxpool);
  computation *alloc = b_workspace.allocate_at(init_conv, x_b);

  k_range0.after_low_level(init_output, 1);
  k_range1.after_low_level(k_range0, 2);
  alloc->after_low_level(k_range1, 3);
  init_conv.after_low_level(*alloc, 3);
  filter_k.after_low_level(init_conv, 3);
  convolve.after_low_level(filter_k, 4);
  maxpool.after_low_level(convolve, 3);
  #if !NO_BATCH
    convolve.parallelize(b);
  #endif
  convolve.parallelize(fout);

  convolve.tag_vector_level(xx, X_BL);
  // ---------------------------------------------------------------------------------
  // Layer III
  // ---------------------------------------------------------------------------------
  // Input
  buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &spconv_relu_maxpool);

  buffer b_input("b_input", {BATCHSIZE, expr(F_In)*expr(height)*expr(width)}, p_float32, a_input, &spconv_relu_maxpool);

  buffer b_filter_values("b_filter_values", {expr(FNNZ)}, p_float32, a_input, &spconv_relu_maxpool);
  buffer b_filter_idx("b_filter_idx", {FNNZ}, p_int32, a_input, &spconv_relu_maxpool);
  buffer b_filter_finptr("b_filter_finptr", {expr(filter_finptr_size)}, p_int32, a_input, &spconv_relu_maxpool);

  buffer b_bias("b_bias", {F_Out}, p_float32, a_input, &spconv_relu_maxpool);

  // Output
  buffer b_result("b_result", {BATCHSIZE, F_Out, maxpool_height, maxpool_width}, p_float32, a_output, &spconv_relu_maxpool);

  // Mapping computations
  SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
  c_input.set_access("[filter_k]->{c_input[b,ind]->b_input[b,ind + filter_k]}");

  c_filter_values.set_access("{c_filter_values[k]->b_filter_values[k]}");
  c_filter_idx.set_access("{c_filter_idx[k]->b_filter_idx[k]}");
  c_filter_finptr.set_access("{c_filter_finptr[fin]->b_filter_finptr[fin]}");

  c_bias.set_access("{c_bias[fout]->b_bias[fout]}");

  std::stringstream init_conv_access, init_output_access, maxpool_access, convolve_access;

  init_conv_access <<"{init_conv[b,fout,y_b,x_b,yy,xx]->b_workspace[yy,xx]}";
  init_conv.set_access(init_conv_access.str());

  init_output_access <<"{init_output[b,fout,output_y,output_x]->b_result[b,fout,output_y,output_x]}";
  init_output.set_access(init_output_access.str());

  maxpool_access << "{maxpool[b,fout,y_b,x_b,yy,xx]->b_result[b,fout,(y_b*"<<Y_BL<<"+yy)/2 + "<<PAD_OUTPUT<<",(x_b * "<<X_BL<<" + xx)/2 + "<<PAD_OUTPUT<<"]}";
  maxpool.set_access(maxpool_access.str());

  convolve_access << "{convolve[b,fout,y_b,x_b,k,yy,xx]->b_workspace[yy,xx]}";
  convolve.set_access(convolve_access.str());

  // ------------------------------------------------------------------
  // Generate code &b_idx, &b_finptr,
  // ------------------------------------------------------------------
  spconv_relu_maxpool.set_arguments({&b_SIZES, &b_input, &b_filter_values, &b_filter_idx, &b_filter_finptr, &b_bias, &b_result});
  spconv_relu_maxpool.gen_time_space_domain();
  spconv_relu_maxpool.gen_isl_ast();
  spconv_relu_maxpool.gen_halide_stmt();
  spconv_relu_maxpool.gen_halide_obj("generated_spconv_relu_maxpool.o");

  return 0;
}
