#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Reference LSTM implementation for correctness check
    tiramisu::init("lstm_ref");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var p_i("p_i", 0, 4);
    input params({p_i}, p_int32);
    constant feature_size("feature_size", params(0));
    constant batch_size("batch_size", params(1));
    constant num_layers("num_layers", params(2));
    constant seq_length("seq_length", params(3));

    // Inner dimensions
    var i("i", 0, feature_size), j("j", 0, feature_size), k("k", 0, batch_size);
    // Outer dimensions
    var l("l", 0, num_layers), m("m", 0, seq_length);

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
    computation y({m, k, i}, h(m, num_layers - 1, k, i));

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
    buffer buf_Weights("buf_Weights", {num_layers, 2, 4 * feature_size, feature_size}, p_float32, a_input);
    buffer buf_biases("buf_biases", {num_layers, 4 * feature_size}, p_float32, a_input);
    buffer buf_x("buf_x", {seq_length, batch_size, feature_size}, p_float32, a_input);
    buffer buf_y("buf_y", {seq_length, batch_size, feature_size}, p_float32, a_output);
    buffer buf_tmp_i("buf_tmp_i", {batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_h("buf_h", {seq_length + 1, num_layers + 1, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_c("buf_c", {seq_length + 1, num_layers, batch_size, feature_size}, p_float32, a_temporary);

    params.store_in(&buf_params);
    // store_in workaround
    R_i.set_access("[feature_size] -> {R_i[l, i, j] -> buf_Weights[l, 0, i + 0 * feature_size, j]}");
    R_z.set_access("[feature_size] -> {R_z[l, i, j] -> buf_Weights[l, 0, i + 1 * feature_size, j]}");
    R_o.set_access("[feature_size] -> {R_o[l, i, j] -> buf_Weights[l, 0, i + 2 * feature_size, j]}");
    R_f.set_access("[feature_size] -> {R_f[l, i, j] -> buf_Weights[l, 0, i + 3 * feature_size, j]}");
    W_i.set_access("[feature_size] -> {W_i[l, i, j] -> buf_Weights[l, 1, i + 0 * feature_size, j]}");
    W_z.set_access("[feature_size] -> {W_z[l, i, j] -> buf_Weights[l, 1, i + 1 * feature_size, j]}");
    W_o.set_access("[feature_size] -> {W_o[l, i, j] -> buf_Weights[l, 1, i + 2 * feature_size, j]}");
    W_f.set_access("[feature_size] -> {W_f[l, i, j] -> buf_Weights[l, 1, i + 3 * feature_size, j]}");
    b_i.set_access("[feature_size] -> {b_i[l, i] -> buf_biases[l, i + 0 * feature_size]}");
    b_z.set_access("[feature_size] -> {b_z[l, i] -> buf_biases[l, i + 1 * feature_size]}");
    b_o.set_access("[feature_size] -> {b_o[l, i] -> buf_biases[l, i + 2 * feature_size]}");
    b_f.set_access("[feature_size] -> {b_f[l, i] -> buf_biases[l, i + 3 * feature_size]}");
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
            &buf_params,
            &buf_Weights,
            &buf_biases,
            &buf_x,
            &buf_y,
        }, "lstm_ref.o");

    return 0;
}
