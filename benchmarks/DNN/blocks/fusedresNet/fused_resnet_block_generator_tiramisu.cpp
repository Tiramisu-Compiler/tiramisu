#include <tiramisu/tiramisu.h>

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
  computation init_mean("init_mean", {n, k_z, y, x}, conv1(n, k_z, y, x, 0, 0, 0));
  computation x_mean("x_mean", {n, k_z, y, x11}, init_mean(n, k_z, y, x11) + init_mean(n, k_z, y, x11 - 1));
  computation y_mean("y_mean", {n, k_z, y11, x_m}, x_mean(n, k_z, y11, x_m) + x_mean(n, k_z, y11 - 1, x_m));
  computation mean("mean", {n11, k_z, y_m, x_m}, y_mean(n11, k_z, y_m, x_m) + y_mean(n11 - 1, k_z, y_m, x_m));

  computation init_variance("init_variance", {n, k_z, y, x}, conv1(n, k_z, y, x, 0, 0, 0) * conv1(n, k_z, y, x, 0, 0, 0));

  computation x_variance("x_variance", {n, k_z, y, x11}, init_variance(n, k_z, y, x11) + init_variance(n, k_z, y, x11 - 1));
  computation y_variance("y_variance", {n, k_z, y11, x_m}, x_variance(n, k_z, y11, x_m) + x_variance(n, k_z, y11 - 1, x_m));
  computation variance("variance", {n11, k_z, y_m, x_m}, y_variance(n11, k_z, y_m, x_m) + y_variance(n11 - 1, k_z, y_m, x_m));

  computation bn1("bn1", {n, k_z, y, x}, (conv1(n, k_z, y, x, 0, 0, 0) - mean(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) / expr(o_sqrt, variance(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS) - (mean(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) * (mean(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS))));

  // Relu computation
  computation relu("relu", {n, k_z, y, x}, expr(o_max, cast(p_float64, 0), bn1(n, k_z, y, x)));

  // Conv2 computation
  computation reluPadd("reluPadd", {n, k_z, y1, x1}, cast(p_float64, 0));
  computation init_relu("init_relu", {n, k_z, y2, x2}, relu(n, k_z, y2 - 1, x2 - 1));
  computation init_conv2("init_conv2", {n, k_z, y, x}, cast(p_float64, 0));
  computation conv2("conv2", {n, k_z, y, x, k_z1, k_y, k_x}, init_conv2(n, k_z, y, x) + filter2(k_z, k_z1, k_y, k_x) * reluPadd(n, k_z1, y + k_y, x + k_x));

  // BN2 computation
  computation init_mean2("init_mean2", {n, k_z, y, x}, conv2(n, k_z, y, x, 0, 0, 0));
  computation x_mean2("x_mean2", {n, k_z, y, x11}, init_mean2(n, k_z, y, x11) + init_mean2(n, k_z, y, x11 - 1));
  computation y_mean2("y_mean2", {n, k_z, y11, x_m}, x_mean2(n, k_z, y11, x_m) + x_mean2(n, k_z, y11 - 1, x_m));
  computation mean2("mean2", {n11, k_z, y_m, x_m}, y_mean2(n11, k_z, y_m, x_m) + y_mean2(n11 - 1, k_z, y_m, x_m));

  computation init_variance2("init_variance2", {n, k_z, y, x}, conv2(n, k_z, y, x, 0, 0, 0) * conv2(n, k_z, y, x, 0, 0, 0));

  computation x_variance2("x_variance2", {n, k_z, y, x11}, init_variance2(n, k_z, y, x11) + init_variance2(n, k_z, y, x11 - 1));
  computation y_variance2("y_variance2", {n, k_z, y11, x_m}, x_variance2(n, k_z, y11, x_m) + x_variance2(n, k_z, y11 - 1, x_m));
  computation variance2("variance2", {n11, k_z, y_m, x_m}, y_variance2(n11, k_z, y_m, x_m) + y_variance2(n11 - 1, k_z, y_m, x_m));

  computation bn2("bn2", {n, k_z, y, x}, (conv2(n, k_z, y, x, 0, 0, 0) - mean2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) / expr(o_sqrt, variance2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS) - (mean2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS)) * (mean2(C_BATCH_SIZE - 1, k_z, C_N - 1, C_NX - 1) / cast(p_float64, C_NB_ELEMENTS))));
  init_input.after(inputPadd, 3);

  init_conv1.after(init_input, 3);

  conv1.after(init_conv1, 0);

  init_mean.tag_parallel_level(0);
  init_mean.after(conv1, 0);

  x_mean.tag_parallel_level(0);
  x_mean.after(init_variance, 3);

  y_mean.tag_parallel_level(0);
  y_mean.after(x_variance, 2);

  mean.after(y_variance, computation::root_dimension);

  init_variance.tag_parallel_level(0);
  init_variance.after(init_mean, 3);

  x_variance.tag_parallel_level(0);
  x_variance.after(x_mean, 3);

  y_variance.tag_parallel_level(0);
  y_variance.after(y_mean, 2);

  variance.after(mean, computation::root_dimension);

  //bn1.tag_parallel_level(0);
  bn1.after(variance, computation::root_dimension);
  bn1.tag_unroll_level(3);
  relu.after(bn1, 3);

  // reluPadd.tag_parallel_level(0);
  reluPadd.after(relu, 3);

  init_relu.tag_parallel_level(0);
  init_relu.after(reluPadd, 3);

  init_conv2.tag_parallel_level(0);
  init_conv2.after(init_relu, computation::root_dimension);

  conv2.tag_parallel_level(0);
  conv2.after(init_conv2, 2);

  init_mean2.tag_parallel_level(0);
  init_mean2.after(conv2, 3);

  x_mean2.tag_parallel_level(0);
  x_mean2.after(init_variance2, 3);

  //y_mean2.tag_parallel_level(0);
  y_mean2.after(x_variance2, 2);

  mean2.after(y_variance2, computation::root_dimension);

  //init_variance2.tag_parallel_level(0);
  init_variance2.after(init_mean2, computation::root_dimension);

  //x_variance2.tag_parallel_level(0);
  x_variance2.after(x_mean2, 3);

  //y_variance2.tag_parallel_level(0);
  y_variance2.after(y_mean2, 2);

  variance2.after(mean2, computation::root_dimension);

  bn2.tag_parallel_level(0);
  bn2.after(variance2, computation::root_dimension);
  bn2.tag_unroll_level(3);

  buffer parameters_buf("parameters_buf", {2}, p_int32, a_input);
  buffer input_buf("input_buf", {C_BATCH_SIZE, 3, C_N, C_N}, p_float64, a_input);
  buffer filter_buf("filter_buf", {64, 3, 3, 3}, p_float64, a_input);
  buffer filter2_buf("filter2_buf", {64, 64, 3, 3}, p_float64, a_input);
  buffer inputPadd_buf("inputPadd_buf", {C_BATCH_SIZE, 3, C_N_PAD, C_N_PAD}, p_float64, a_output);
  buffer conv1_buf("conv1_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer reluPadd_buf("reluPadd_buf", {C_BATCH_SIZE, 64, C_N_PAD, C_N_PAD}, p_float64, a_output);
  buffer conv2_buf("conv2_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer mean_buff("mean_buff", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer bn1_buf("bn1_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer variance_buff("variance_buff", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);
  buffer bn2_buf("bn2_buf", {C_BATCH_SIZE, 64, C_N, C_N}, p_float64, a_output);

  c_input.store_in(&input_buf);
  parameters.store_in(&parameters_buf);
  filter.store_in(&filter_buf);
  filter2.store_in(&filter2_buf);

  inputPadd.store_in(&inputPadd_buf);
  init_input.store_in(&inputPadd_buf);
  init_conv1.store_in(&conv1_buf);
  conv1.store_in(&conv1_buf, {n, k_z, y, x});

  init_mean.store_in(&mean_buff);
  mean.store_in(&mean_buff);
  x_mean.store_in(&mean_buff);
  y_mean.store_in(&mean_buff);
  init_variance.store_in(&variance_buff);
  x_variance.store_in(&variance_buff);
  y_variance.store_in(&variance_buff);
  variance.store_in(&variance_buff);
  bn1.store_in(&bn1_buf);
  relu.store_in(&bn1_buf);

  reluPadd.store_in(&reluPadd_buf);
  init_relu.store_in(&reluPadd_buf);
  init_conv2.store_in(&conv2_buf);
  conv2.store_in(&conv2_buf, {n, k_z, y, x});

  init_mean2.store_in(&mean_buff);
  mean2.store_in(&mean_buff);
  x_mean2.store_in(&mean_buff);
  y_mean2.store_in(&mean_buff);
  init_variance2.store_in(&variance_buff);
  x_variance2.store_in(&variance_buff);
  y_variance2.store_in(&variance_buff);
  variance2.store_in(&variance_buff);
  bn2.store_in(&bn2_buf);

  tiramisu::codegen({&parameters_buf, &filter_buf, &filter2_buf, &input_buf, &inputPadd_buf, &conv1_buf, &mean_buff, &variance_buff, &bn1_buf, &reluPadd_buf, &conv2_buf, &bn2_buf}, "generated_fused_resnet_block.o");
  return 0;
}