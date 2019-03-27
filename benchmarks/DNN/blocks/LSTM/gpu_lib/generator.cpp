#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Single LSTM block without minibatching
    tiramisu::init("lstm");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Inner dimensions
    var i("i", 0, FEATURE_SIZE), j("j", 0, FEATURE_SIZE), k("k", 0, BATCH_SIZE);
    var i_merged("i_merged", 0, 4 * FEATURE_SIZE);
    // Outer dimensions
    var l("l", 0, NUM_LAYERS), m("m", 0, SEQ_LENGTH);

    input R("R", {l, i_merged, j}, p_float32);
    input W("W", {l, i_merged, j}, p_float32);
    input b("b", {l, i_merged}, p_float32);
    input x({m, k, i}, p_float32);

    buffer buf_tmp("buf_tmp", {BATCH_SIZE, 4 * FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_Weights("buf_Weights", {NUM_LAYERS, 2, 4 * FEATURE_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_h("buf_h", {SEQ_LENGTH + 1, NUM_LAYERS + 1, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);

    // h(m, l) is the output of the block (m, l)
    // which takes h(m - 1, l) and h(m, l - 1) as inputs
    // initial hidden states are h(-1, l) and c(-1, l)
    // input x is copied to h(m, -1)
    computation h({m, l, k, i}, p_float32);
    computation c({m, l, k, i}, p_float32);
    computation h_init({l, k, i}, expr(float(0)));
    computation c_init({l, k, i}, expr(float(0)));
    computation h_copy_x({m, k, i}, x(m, k, i));
    computation sum_init({m, l, k, i_merged}, b(l, i_merged));
    computation sum1({m, l, k, i_merged, j}, sum_init(m, l, k, i_merged) + R(l, i_merged, j) * h(m - 1, l, k, j));
    computation sum2({m, l, k, i_merged, j}, sum_init(m, l, k, i_merged) + W(l, i_merged, j) * h(m, l - 1, k, j));
    #define sigmoid(x) expr(float(1)) / (1 + expr(o_expo, -(x)))
    computation sig_i({m, l, k, i}, sigmoid(sum_init(m, l, k, i)));
    computation tnh_z({m, l, k, i}, expr(o_tanh, sum_init(m, l, k, i + FEATURE_SIZE)));
    computation sig_o({m, l, k, i}, sigmoid(sum_init(m, l, k, i + 2 * FEATURE_SIZE)));
    computation sig_f({m, l, k, i}, sigmoid(sum_init(m, l, k, i + 3 * FEATURE_SIZE)));
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

    // TODO: Add tiling and better schedule

    // Scheduling commands
    h_init.then(c_init, computation::root)
          .then(h_copy_x, computation::root)
          .then(sum_init, computation::root)
          .then(sum1, l)
          .then(sum2, l)
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

    buffer buf_biases("buf_biases", {NUM_LAYERS, 4 * FEATURE_SIZE}, p_float32, a_input);
    buffer buf_x("buf_x", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_y("buf_y", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_output);
    // TODO: Does not support parallel
    buffer buf_tmp_i("buf_tmp_i", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    // TODO: As in cuDNN LSTM example, we store every output at separate places in a huge tensor.
    // This can be made more compact.
    buffer buf_c("buf_c", {SEQ_LENGTH + 1, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);

    // Weights and biases are packed
    R.store_in(&buf_Weights, {l, 0, i_merged, j});
    W.store_in(&buf_Weights, {l, 1, i_merged, j});
    b.store_in(&buf_biases, {l, i_merged});
    x.store_in(&buf_x);
    y.store_in(&buf_y);
    sum_init.store_in(&buf_tmp, {k, i_merged});
    sum1.store_in(&buf_tmp, {k, i_merged});
    sum2.store_in(&buf_tmp, {k, i_merged});
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
        }, "lstm.o");

    return 0;
}
