#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Reference LSTM implementation for correctness check
    tiramisu::init("lstm_ref");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    // Inner dimensions
    var i("i", 0, FEATURE_SIZE), j("j", 0, FEATURE_SIZE), k("k", 0, BATCH_SIZE);
    // Outer dimensions
    var l("l", 0, NUM_LAYERS), m("m", 0, SEQ_LENGTH);

    input R_i("R_i", {l, i, j}, p_float32);
    input R_z("R_z", {l, i, j}, p_float32);
    input R_o("R_o", {l, i, j}, p_float32);
    input R_f("R_f", {l, i, j}, p_float32);
    input W_i("W_i", {l, i, j}, p_float32);
    input W_z("W_z", {l, i, j}, p_float32);
    input W_o("W_o", {l, i, j}, p_float32);
    input W_f("W_f", {l, i, j}, p_float32);
    input b_i("b_i", {l, i}, p_float32);
    input b_z("b_z", {l, i}, p_float32);
    input b_o("b_o", {l, i}, p_float32);
    input b_f("b_f", {l, i}, p_float32);
    input x({m, k, i}, p_float32);

    // h(m, l) is the output of the block (m, l)
    // which takes h(m - 1, l) and h(m, l - 1) as inputs
    // initial hidden states are h(-1, l) and c(-1, l)
    // input x is copied to h(m, -1)
    computation h({m, l, k, i}, p_float32);
    computation c({m, l, k, i}, p_float32);
    computation h_init({l, k, i}, expr(float(0)));
    computation c_init({l, k, i}, expr(float(0)));
    computation h_copy_x({m, k, i}, x(m, k, i));
    computation sum_i_init({m, l, k, i}, b_i(l, i));
    computation sum_z_init({m, l, k, i}, b_z(l, i));
    computation sum_o_init({m, l, k, i}, b_o(l, i));
    computation sum_f_init({m, l, k, i}, b_f(l, i));
    computation sum_i({m, l, k, i, j}, sum_i_init(m, l, k, i) + R_i(l, i, j) * h(m - 1, l, k, j) + W_i(l, i, j) * h(m, l - 1, k, j));
    computation sum_z({m, l, k, i, j}, sum_z_init(m, l, k, i) + R_z(l, i, j) * h(m - 1, l, k, j) + W_z(l, i, j) * h(m, l - 1, k, j));
    computation sum_o({m, l, k, i, j}, sum_o_init(m, l, k, i) + R_o(l, i, j) * h(m - 1, l, k, j) + W_o(l, i, j) * h(m, l - 1, k, j));
    computation sum_f({m, l, k, i, j}, sum_f_init(m, l, k, i) + R_f(l, i, j) * h(m - 1, l, k, j) + W_f(l, i, j) * h(m, l - 1, k, j));
    #define sigmoid(x) expr(float(1)) / (1 + expr(o_expo, -(x)))
    computation sig_i({m, l, k, i}, sigmoid(sum_i(m, l, k, i, 0)));
    computation tnh_z({m, l, k, i}, expr(o_tanh, sum_z(m, l, k, i, 0)));
    computation sig_o({m, l, k, i}, sigmoid(sum_o(m, l, k, i, 0)));
    computation sig_f({m, l, k, i}, sigmoid(sum_f(m, l, k, i, 0)));
    computation mul_iz({m, l, k, i}, sig_i(m, l, k, i) * tnh_z(m, l, k, i));
    computation mul_fc({m, l, k, i}, sig_f(m, l, k, i) * c(m - 1, l, k, i));
    c.set_expression(mul_iz(m, l, k, i) + mul_fc(m, l, k, i));
    computation tnh_c({m, l, k, i}, expr(o_tanh, c(m, l, k, i)));
    h.set_expression(tnh_c(m, l, k, i) * sig_o(m, l, k, i));
    // Output is the last layer
    computation y({m, k, i}, h(m, NUM_LAYERS - 1, k, i));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands
    h_init.then(c_init, computation::root)
          .then(h_copy_x, computation::root)
          .then(sum_i_init, computation::root)
          .then(sum_z_init, l)
          .then(sum_o_init, l)
          .then(sum_f_init, l)
          .then(sum_i, l)
          .then(sum_z, l)
          .then(sum_o, l)
          .then(sum_f, l)
          .then(sig_i, l)
          .then(tnh_z, l)
          .then(sig_o, l)
          .then(sig_f, l)
          .then(mul_iz, l)
          .then(mul_fc, l)
          .then(c, l)
          .then(tnh_c, l)
          .then(h, l)
          .then(y, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_params("buf_params", {4}, p_int32, a_input);
    buffer buf_Weights("buf_Weights", {NUM_LAYERS, 2, 4 * FEATURE_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_biases("buf_biases", {NUM_LAYERS, 4 * FEATURE_SIZE}, p_float32, a_input);
    buffer buf_x("buf_x", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_y("buf_y", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_output);
    buffer buf_tmp_i("buf_tmp_i", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_h("buf_h", {SEQ_LENGTH + 1, NUM_LAYERS + 1, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_c("buf_c", {SEQ_LENGTH + 1, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);

    // store_in workaround
    R_i.store_in(&buf_Weights, {l, 0, i + 0 * FEATURE_SIZE, j});
    R_z.store_in(&buf_Weights, {l, 0, i + 1 * FEATURE_SIZE, j});
    R_o.store_in(&buf_Weights, {l, 0, i + 2 * FEATURE_SIZE, j});
    R_f.store_in(&buf_Weights, {l, 0, i + 3 * FEATURE_SIZE, j});
    W_i.store_in(&buf_Weights, {l, 1, i + 0 * FEATURE_SIZE, j});
    W_z.store_in(&buf_Weights, {l, 1, i + 1 * FEATURE_SIZE, j});
    W_o.store_in(&buf_Weights, {l, 1, i + 2 * FEATURE_SIZE, j});
    W_f.store_in(&buf_Weights, {l, 1, i + 3 * FEATURE_SIZE, j});
    b_i.store_in(&buf_biases, {l, i + 0 * FEATURE_SIZE});
    b_z.store_in(&buf_biases, {l, i + 1 * FEATURE_SIZE});
    b_o.store_in(&buf_biases, {l, i + 2 * FEATURE_SIZE});
    b_f.store_in(&buf_biases, {l, i + 3 * FEATURE_SIZE});
    x.store_in(&buf_x);
    y.store_in(&buf_y);
    sum_i_init.store_in(&buf_tmp_i, {k, i});
    sum_z_init.store_in(&buf_tmp_z, {k, i});
    sum_o_init.store_in(&buf_tmp_o, {k, i});
    sum_f_init.store_in(&buf_tmp_f, {k, i});
    sum_i.store_in(&buf_tmp_i, {k, i});
    sum_z.store_in(&buf_tmp_z, {k, i});
    sum_o.store_in(&buf_tmp_o, {k, i});
    sum_f.store_in(&buf_tmp_f, {k, i});
    sig_i.store_in(&buf_tmp_i, {k, i});
    tnh_z.store_in(&buf_tmp_z, {k, i});
    sig_o.store_in(&buf_tmp_o, {k, i});
    sig_f.store_in(&buf_tmp_f, {k, i});
    mul_iz.store_in(&buf_tmp_i, {k, i});
    mul_fc.store_in(&buf_tmp_f, {k, i});
    tnh_c.store_in(&buf_tmp_i, {k, i});
    h.store_in(&buf_h, {m + 1, l + 1, k, i});
    c.store_in(&buf_c, {m + 1, l, k, i});
    h_init.store_in(&buf_h, {0, l + 1, k, i});
    c_init.store_in(&buf_c, {0, l, k, i});
    h_copy_x.store_in(&buf_h, {m + 1, 0, k, i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files.
    tiramisu::codegen({
            &buf_Weights,
            &buf_biases,
            &buf_x,
            &buf_y,
        }, "lstm_ref.o");

    return 0;
}
