/* 
    This benchmark represents a ResNet Block which includes the next layers in order:
    - convolution layer
    - 2D batch normalization
    - relu
    - convolution 
    - 2D batch normalization
    Each convolution function is fused with the BN layer that follows it.
*/

#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
  init("fused_resnet_block");

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------
  // parameters
  // N: parameters[0]
  // BATCH_SIZE: parameters[1]

  var i("i", 0, 2);
  input parameters("parameters", {i}, p_int32);

  constant C_N("C_N", parameters(0));
  constant C_NX("C_NX", parameters(0));
  constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(1));
  constant C_NB_ELEMENTS("C_NB_ELEMENTS", parameters(0) * parameters(0) * parameters(1));
  constant C_N_PAD("C_N_PAD", parameters(0) + 2); // input padded size
  constant C_N_PADD("C_N_PADD", parameters(0) + 1);

  var x("x", 0, C_N), y("y", 0, C_N), z("z", 0, 3), n("n", 0, C_BATCH_SIZE); // input
  var x11("x11", 1, C_N), y11("y11", 1, C_N), n11("n11", 1, C_BATCH_SIZE);
  var x1("x1", 0, C_N_PAD), y1("y1", 0, C_N_PAD);                                 // inputpadd
  var x2("x2", 1, C_N_PADD), y2("y2", 1, C_N_PADD);                               // init_input
  var k_y("k_y", 0, 3), k_x("k_x", 0, 3), k_z("k_z", 0, 64), k_z1("k_z1", 0, 64); // kernel conv1
  var x_m("x_m", C_N - 1, C_N), y_m("y_m", C_N - 1, C_N);

  // Input computations
  input c_input("c_input", {n, z, y, x}, p_float64);
  input filter("filter", {k_z, z, k_y, k_x}, p_float64);
  input filter2("filter2", {k_z, k_z1, k_y, k_x}, p_float64);

  // Conv1 computation
  computation inputPadd("inputPadd", {n, z, y1, x1}, cast(p_float64, 0));
  computation init_input("init_input", {n, z, y2, x2}, c_input(n, z, y2 - 1, x2 - 1));
  computation init_conv1("init_conv1", {n, k_z, y, x}, cast(p_float64, 0));
  computation conv1("conv1", {n, k_z, y, x, z, k_y, k_x}, init_conv1(n, k_z, y, x) + filter(k_z, z, k_y, k_x) * inputPadd(n, z, y + k_y, x + k_x));

  // BN1 computation
  // Calculate the sum of the input values in parallel
  computation init_sum("init_sum", {n, k_z, y, x}, conv1(n, k_z, y, x, 0, 0, 0));
  computation x_sum("x_sum", {n, k_z, y, x11}, init_sum(n, k_z, y, x11) + init_sum(n, k_z, y, x11 - 1));
  computation y_sum("y_sum", {n, k_z, y11, x_m}, x_sum(n, k_z, y11, x_m) + x_sum(n, k_z, y11 - 1, x_m));
  computation sum("sum", {n11, k_z, y_m, x_m}, y_sum(n11, k_z, y_m, x_m) + y_sum(n11 - 1, k_z, y_m, x_m));

  // Calculate the sum of the input values squared in parallel
  computation init_sumSquared("init_sumSquared", {n, k_z, y, x}, conv1(n, k_z, y, x, 0, 0, 0) * conv1(n, k_z, y, x, 0, 0, 0));
  computation x_sumSquared("x_sumSquared", {n, k_z, y, x11}, init_sumSquared(n, k_z, y, x11) + init_sumSquared(n, k_z, y, x11 - 1));
  computation y_sumSquared("y_sumSquared", {n, k_z, y11, x_m}, x_sumSquared(n, k_z, y11, x_m) + x_sumSquared(n, k_z, y11 - 1, x_m));
  computation sumSquared("sumSquared", {n11, k_z, y_m, x_m}, y_sumSquared(n11, k_z, y_m, x_m) + y_sumSquared(n11 - 1, k_z, y_m, x_m));

  computation bn1("bn1", {n, k_z, y, x}, (conv1(n, k_z, y, x, 0, 0, 0) - sum(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) / expr(o_sqrt, sumSquared(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS) - (sum(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) * (sum(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS))));

  // Relu computation
  computation relu("relu", {n, k_z, y, x}, expr(o_max, cast(p_float64, 0), bn1(n, k_z, y, x)));

  // Conv2 computation
  computation reluPadd("reluPadd", {n, k_z, y1, x1}, cast(p_float64, 0));
  computation init_relu("init_relu", {n, k_z, y2, x2}, relu(n, k_z, y2 - 1, x2 - 1));
  computation init_conv2("init_conv2", {n, k_z, y, x}, cast(p_float64, 0));
  computation conv2("conv2", {n, k_z, y, x, k_z1, k_y, k_x}, init_conv2(n, k_z, y, x) + filter2(k_z, k_z1, k_y, k_x) * reluPadd(n, k_z1, y + k_y, x + k_x));

  // BN2 computation
  // Calculate the sum of the input values in parallel
  computation init_sum2("init_sum2", {n, k_z, y, x}, conv2(n, k_z, y, x, 0, 0, 0));
  computation x_sum2("x_sum2", {n, k_z, y, x11}, init_sum2(n, k_z, y, x11) + init_sum2(n, k_z, y, x11 - 1));
  computation y_sum2("y_sum2", {n, k_z, y11, x_m}, x_sum2(n, k_z, y11, x_m) + x_sum2(n, k_z, y11 - 1, x_m));
  computation sum2("sum2", {n11, k_z, y_m, x_m}, y_sum2(n11, k_z, y_m, x_m) + y_sum2(n11 - 1, k_z, y_m, x_m));

  // Calculate the sum of the input values squared in parallel
  computation init_sumSquared2("init_sumSquared2", {n, k_z, y, x}, conv2(n, k_z, y, x, 0, 0, 0) * conv2(n, k_z, y, x, 0, 0, 0));
  computation x_sumSquared2("x_sumSquared2", {n, k_z, y, x11}, init_sumSquared2(n, k_z, y, x11) + init_sumSquared2(n, k_z, y, x11 - 1));
  computation y_sumSquared2("y_sumSquared2", {n, k_z, y11, x_m}, x_sumSquared2(n, k_z, y11, x_m) + x_sumSquared2(n, k_z, y11 - 1, x_m));
  computation sumSquared2("sumSquared2", {n11, k_z, y_m, x_m}, y_sumSquared2(n11, k_z, y_m, x_m) + y_sumSquared2(n11 - 1, k_z, y_m, x_m));

  computation bn2("bn2", {n, k_z, y, x}, (conv2(n, k_z, y, x, 0, 0, 0) - sum2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) / expr(o_sqrt, sumSquared2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS) - (sum2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) * (sum2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS))));

  // ResNet block with each convolution fused with its following bn function
  if (FUSED_SCHEDULE)
  {
    init_input.after(inputPadd, x2);

    init_conv1.after(init_input, x);

    conv1.after(init_conv1, n);

    init_sum.tag_parallel_level(n);
    init_sum.after(conv1, n);

    x_sum.tag_parallel_level(n);
    x_sum.after(init_sumSquared, x11);

    y_sum.tag_parallel_level(n);
    y_sum.after(x_sumSquared, y11);

    sum.after(y_sumSquared, computation::root_dimension);

    init_sumSquared.tag_parallel_level(n);
    init_sumSquared.after(init_sum, x);

    x_sumSquared.tag_parallel_level(n);
    x_sumSquared.after(x_sum, x11);

    y_sumSquared.tag_parallel_level(n);
    y_sumSquared.after(y_sum, y11);

    sumSquared.after(sum, computation::root_dimension);

    bn1.after(sumSquared, computation::root_dimension);
    bn1.tag_unroll_level(x);

    relu.after(bn1, x);

    reluPadd.after(relu, x1);

    init_relu.tag_parallel_level(n);
    init_relu.after(reluPadd, x2);

    init_conv2.tag_parallel_level(n);
    init_conv2.after(init_relu, computation::root_dimension);

    conv2.tag_parallel_level(n);
    conv2.after(init_conv2, y);

    init_sum2.tag_parallel_level(n);
    init_sum2.after(conv2, x);

    x_sum2.tag_parallel_level(n);
    x_sum2.after(init_sumSquared2, x11);

    y_sum2.after(x_sumSquared2, y11);

    sum2.after(y_sumSquared2, computation::root_dimension);

    init_sumSquared2.after(init_sum2, computation::root_dimension);

    x_sumSquared2.after(x_sum2, x11);

    y_sumSquared2.after(y_sum2, y11);

    sumSquared2.after(sum2, computation::root_dimension);

    bn2.tag_parallel_level(n);
    bn2.after(sumSquared2, computation::root_dimension);
    bn2.tag_unroll_level(x);
  }
  
  // ResNet block wihtout fusing convolution and bn layers
  else
  {
    init_input.after(inputPadd, x2);

    init_conv1.after(init_input, x);

    conv1.after(init_conv1, n);

    init_sum.tag_parallel_level(n);
    init_sum.after(conv1, n);

    x_sum.tag_parallel_level(n);
    x_sum.after(init_sum, computation::root_dimension);

    y_sum.tag_parallel_level(n);
    y_sum.after(x_sum, computation::root_dimension);

    sum.after(y_sum, computation::root_dimension);

    init_sumSquared.after(sum, n);

    x_sumSquared.after(init_sumSquared, computation::root_dimension);

    y_sumSquared.after(x_sumSquared, computation::root_dimension);

    sumSquared.after(y_sumSquared, computation::root_dimension);

    bn1.after(sumSquared, computation::root_dimension);
    bn1.tag_unroll_level(x);

    relu.after(bn1, n);

    reluPadd.after(relu, n);

    init_relu.tag_parallel_level(n);
    init_relu.after(reluPadd, computation::root_dimension);

    init_conv2.tag_parallel_level(n);
    init_conv2.after(init_relu, computation::root_dimension);

    conv2.tag_parallel_level(n);
    conv2.after(init_conv2, computation::root_dimension);

    init_sum2.tag_parallel_level(n);
    init_sum2.after(conv2, n); 

    x_sum2.after(init_sum2, computation::root_dimension); 

    y_sum2.after(x_sum2, computation::root_dimension); 

    sum2.after(y_sum2, computation::root_dimension);

    init_sumSquared2.after(sum2, computation::root_dimension);

    x_sumSquared2.after(init_sumSquared2,computation::root_dimension); 

    y_sumSquared2.after(x_sumSquared2, computation::root_dimension);

    sumSquared2.after(y_sumSquared2, computation::root_dimension);

    bn2.tag_parallel_level(n);
    bn2.after(sumSquared2, computation::root_dimension);
    bn2.tag_unroll_level(x);
  }

  buffer parameters_buf("parameters_buf", {2}, p_int32, a_input);
  buffer input_buf("input_buf", {C_BATCH_SIZE, 3, C_N, C_N}, p_float64, a_input);
  buffer filter_buf("filter_buf", {64, 3, 3, 3}, p_float64, a_input);
  buffer filter2_buf("filter2_buf", {64, 64, 3, 3}, p_float64, a_input);
  buffer inputPadd_buf("inputPadd_buf", {C_BATCH_SIZE, 3, C_N_PAD, C_N_PAD}, p_float64, a_output);
  buffer conv1_buf("conv1_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer reluPadd_buf("reluPadd_buf", {C_BATCH_SIZE, 64, C_N_PAD, C_N_PAD}, p_float64, a_output);
  buffer conv2_buf("conv2_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer sum_buff("sum_buff", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer bn1_buf("bn1_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer sumSquared_buff("sumSquared_buff", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer bn2_buf("bn2_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);

  c_input.store_in(&input_buf);
  parameters.store_in(&parameters_buf);
  filter.store_in(&filter_buf);
  filter2.store_in(&filter2_buf);

  inputPadd.store_in(&inputPadd_buf);
  init_input.store_in(&inputPadd_buf);
  init_conv1.store_in(&conv1_buf);
  conv1.store_in(&conv1_buf, {n, k_z, y, x});

  init_sum.store_in(&sum_buff);
  sum.store_in(&sum_buff);
  x_sum.store_in(&sum_buff);
  y_sum.store_in(&sum_buff);
  init_sumSquared.store_in(&sumSquared_buff);
  x_sumSquared.store_in(&sumSquared_buff);
  y_sumSquared.store_in(&sumSquared_buff);
  sumSquared.store_in(&sumSquared_buff);
  bn1.store_in(&bn1_buf);
  relu.store_in(&bn1_buf);

  reluPadd.store_in(&reluPadd_buf);
  init_relu.store_in(&reluPadd_buf);
  init_conv2.store_in(&conv2_buf);
  conv2.store_in(&conv2_buf, {n, k_z, y, x});

  init_sum2.store_in(&sum_buff);
  sum2.store_in(&sum_buff);
  x_sum2.store_in(&sum_buff);
  y_sum2.store_in(&sum_buff);
  init_sumSquared2.store_in(&sumSquared_buff);
  x_sumSquared2.store_in(&sumSquared_buff);
  y_sumSquared2.store_in(&sumSquared_buff);
  sumSquared2.store_in(&sumSquared_buff);
  bn2.store_in(&bn2_buf);

  tiramisu::codegen({&parameters_buf, &filter_buf, &filter2_buf, &input_buf, &inputPadd_buf, &conv1_buf, &sum_buff, &sumSquared_buff, &bn1_buf, &reluPadd_buf, &conv2_buf, &bn2_buf}, "fused_resnet_block_generator_tiramisu.o");
  return 0;
}
