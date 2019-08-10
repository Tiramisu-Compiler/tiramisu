#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

//#include <bits/stdc++.h>

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

  function spconv("spconv");

  // Inputs
  computation SIZES("{SIZES[e]: 0<=e<1}", expr(), false, p_int32, &spconv);
  computation c_input("[BATCHSIZE,IND_RANGE]->{c_input[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE}", expr(), false, p_float32, &spconv);

  computation c_filter_values("[FNNZ]->{c_filter_values[k]: 0<=k<FNNZ}", expr(), false, p_float32, &spconv);
  computation c_filter_idx("[FNNZ]->{c_filter_idx[k]: 0<=k<FNNZ}", expr(), false, p_int32, &spconv);
  computation c_filter_finptr("[filter_finptr_size]->{c_filter_finptr[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &spconv);

  computation c_bias("[F_Out]->{c_bias[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &spconv);

  constant IND_RANGE("IND_RANGE", (N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &spconv);
  constant FNNZ("FNNZ", SIZES(0), p_int32, true, NULL, 0, &spconv);
  constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &spconv);
  constant F_In("F_In", FIn, p_int32, true, NULL, 0, &spconv);
  constant height("height", N + 2, p_int32, true, NULL, 0, &spconv);
  constant width("width", N + 2, p_int32, true, NULL, 0, &spconv);

  constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &spconv);
  constant KK("KK", K, p_int32, true, NULL, 0, &spconv);

  constant output_height("output_height", N, p_int32, true, NULL, 0, &spconv);
  constant output_width("output_width", N, p_int32, true, NULL, 0, &spconv);

  constant filter_finptr_size("filter_finptr_size", FOut + 1, p_int32, true, NULL, 0, &spconv);

  constant X_BLOCKING("X_BLOCKING", X_BL, p_int32, true, NULL, 0, &spconv);
  constant X_NB_BLOCKS("X_NB_BLOCKS", X_NB_BL, p_int32, true, NULL, 0, &spconv);

  constant Y_BLOCKING("Y_BLOCKING", Y_BL, p_int32, true, NULL, 0, &spconv);
  constant Y_NB_BLOCKS("Y_NB_BLOCKS", Y_NB_BL, p_int32, true, NULL, 0, &spconv);

  var k("k"), l("l"), fout("fout"), b("b"), fin("fin");
  var output_x("output_x"), output_y("output_y");
  var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");

  //Initialize the output
  computation init_zero("[BATCHSIZE,F_Out,output_height,output_width]->{init_zero[b,fout,output_y,output_x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=output_y<output_height and 0<=output_x<output_width}",  cast(p_float32, 0), true, p_float32, &spconv);
  computation add_bias("[BATCHSIZE,F_Out,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS, Y_NB_BLOCKS]->{add_bias[b,fout,y_b,x_b,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING}",  expr(), true, p_float32, &spconv);
  add_bias.set_expression(add_bias(b, fout, y_b, x_b, yy, xx) + c_bias(fout));

  computation convolve("[BATCHSIZE,F_Out,X_BLOCKING,Y_BLOCKING,X_NB_BLOCKS,Y_NB_BLOCKS,k_range0,k_range1]->{convolve[b,fout,y_b,x_b,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and k_range0<=k<k_range1}", expr(), true, p_float32, &spconv);

  // Loop invariants
  constant filter_k("filter_k", c_filter_idx(k), p_int32, false, &convolve, 4, &spconv);

  constant k_range0("k_range0", c_filter_finptr(fout), p_int32, false, &convolve, 2, &spconv);
  constant k_range1("k_range1", c_filter_finptr(fout + 1), p_int32, false, &convolve, 2, &spconv);

  convolve.set_expression(convolve(b, fout, y_b, x_b, k, yy, xx) + c_input(b, (y_b * Y_BL + yy) * (N + 2) + x_b * X_BL + xx) * c_filter_values(k));

  spconv.set_context_set("[k_range0,k_range1]->{: k_range0>0 and k_range1>0 and k_range1>k_range0}");

  // -----------------------------------------------------------------
  // Layer II
  // -----------------------------------------------------------------
  k_range0.after_low_level(init_zero, 1);
  k_range1.after_low_level(k_range0, 2);
  filter_k.after_low_level(k_range1, 3);
  convolve.after_low_level(filter_k, 4);
  add_bias.after_low_level(convolve, 3);

  convolve.tag_parallel_level(0);
  convolve.tag_parallel_level(1);

  // ---------------------------------------------------------------------------------
  // Layer III
  // ---------------------------------------------------------------------------------
  // Input
  buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &spconv);

  buffer b_input("b_input", {BATCHSIZE, expr(F_In) * expr(height) * expr(width)}, p_float32, a_input, &spconv);

  // Weights in CSR format
  buffer b_filter_values("b_filter_values", {FNNZ}, p_float32, a_input, &spconv);
  buffer b_filter_idx("b_filter_idx", {FNNZ}, p_int32, a_input, &spconv);
  buffer b_filter_finptr("b_filter_finptr", {filter_finptr_size}, p_int32, a_input, &spconv);

  buffer b_bias("b_bias", {F_Out}, p_float32, a_input, &spconv);

  // Output
  buffer b_result("b_result", {BATCHSIZE, F_Out, output_height, output_width}, p_float32, a_output, &spconv);

  // Mapping computations
  SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
  c_input.set_access("[filter_k]->{c_input[b,ind]->b_input[b,ind + filter_k]}");

  c_filter_values.set_access("{c_filter_values[k]->b_filter_values[k]}");
  c_filter_idx.set_access("{c_filter_idx[k]->b_filter_idx[k]}");
  c_filter_finptr.set_access("{c_filter_finptr[fin]->b_filter_finptr[fin]}");

  c_bias.set_access("{c_bias[fout]->b_bias[fout]}");

  std::stringstream init_zero_access;
  std::stringstream add_bias_access;
  std::stringstream convolve_access;
  init_zero_access <<"{init_zero[b,fout,output_y,output_x]->b_result[b,fout,output_y,output_x]}";
  init_zero.set_access(init_zero_access.str());

  add_bias_access << "{add_bias[b,fout,y_b,x_b,yy,xx]->b_result[b, fout, y_b *"<< Y_BL <<" + yy, x_b *"<< X_BL << " + xx]}";
  add_bias.set_access(add_bias_access.str());

  convolve_access << "{convolve[b,fout,y_b,x_b,k,yy,xx]->b_result[b, fout, y_b *"<< Y_BL <<" + yy, x_b *"<< X_BL <<" + xx]}";
  convolve.set_access(convolve_access.str());

  // ------------------------------------------------------------------
  // Generate code &b_idx, &b_finptr,
  // ------------------------------------------------------------------
  spconv.set_arguments({&b_SIZES, &b_input, &b_filter_values, &b_filter_idx, &b_filter_finptr, &b_bias, &b_result});
  spconv.gen_time_space_domain();
  spconv.gen_isl_ast();
  spconv.gen_halide_stmt();
  spconv.gen_halide_obj("generated_spconv.o");

  return 0;
}
