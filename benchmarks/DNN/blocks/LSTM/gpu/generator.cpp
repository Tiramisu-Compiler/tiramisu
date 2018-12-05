#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Single LSTM block without minibatching
    tiramisu::init("lstm");

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
    var i0("i0"), i1("i1"), k0("k0"), k1("k1");

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

    buffer buf_params("buf_params", {4}, p_int32, a_input);
    buffer buf_Weights("buf_Weights", {num_layers, 8, feature_size, feature_size}, p_float32, a_input);
    buffer buf_biases("buf_biases", {num_layers, 4, feature_size}, p_float32, a_input);
    buffer buf_x("buf_x", {seq_length, batch_size, feature_size}, p_float32, a_input);
    buffer buf_y("buf_y", {seq_length, batch_size, feature_size}, p_float32, a_output);
    buffer buf_Weights_gpu("buf_Weights_gpu", {num_layers, 8, feature_size, feature_size}, p_float32, a_temporary);
    buffer buf_biases_gpu("buf_biases_gpu", {num_layers, 4, feature_size}, p_float32, a_temporary);
    buffer buf_x_gpu("buf_x_gpu", {seq_length, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_y_gpu("buf_y_gpu", {seq_length, batch_size, feature_size}, p_float32, a_temporary);
    // TODO: Does not support parallel
    buffer buf_tmp_i("buf_tmp_i", {num_layers, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {num_layers, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {num_layers, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {num_layers, batch_size, feature_size}, p_float32, a_temporary);
    // TODO: As in cuDNN LSTM example, we store every output at separate places in a huge tensor.
    // This can be made more compact.
    buffer buf_h("buf_h", {seq_length + 1, num_layers + 1, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_c("buf_c", {seq_length + 1, num_layers, batch_size, feature_size}, p_float32, a_temporary);

    buf_Weights_gpu.tag_gpu_global();
    buf_biases_gpu.tag_gpu_global();
    buf_x_gpu.tag_gpu_global();
    buf_y_gpu.tag_gpu_global();
    buf_tmp_i.tag_gpu_global();
    buf_tmp_z.tag_gpu_global();
    buf_tmp_o.tag_gpu_global();
    buf_tmp_f.tag_gpu_global();
    buf_h.tag_gpu_global();
    buf_c.tag_gpu_global();

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
    // Copies
    computation copy_Weights_to_device({}, memcpy(buf_Weights, buf_Weights_gpu));
    computation copy_biases_to_device({}, memcpy(buf_biases, buf_biases_gpu));
    computation copy_x_to_device({}, memcpy(buf_x, buf_x_gpu));
    computation copy_y_to_host({}, memcpy(buf_y_gpu, buf_y));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Fuse initialization
    h_init.interchange(l, k);
    h_init.interchange(l, i);
    c_init.interchange(l, k);
    c_init.interchange(l, i);
    h_copy_x.interchange(m, k);
    h_copy_x.interchange(m, i);
    y.interchange(m, k);
    y.interchange(m, i);

    int block = 16;
    h_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    c_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    h_copy_x.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_i_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_z_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_o_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_f_init.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_i.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_z.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_o.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sum_f.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sig_i.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    tnh_z.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sig_o.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    sig_f.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    mul_iz.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    mul_fc.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    c.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    tnh_c.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    h.gpu_tile(k, i, block, block, k0, i0, k1, i1);
    y.gpu_tile(k, i, block, block, k0, i0, k1, i1);

    // Scheduling commands
    copy_Weights_to_device.then(copy_biases_to_device, computation::root)
        .then(copy_x_to_device, computation::root)
        .then(h_init, i1)
        .then(c_init, i1)
        .then(h_copy_x, i1)
        .then(sum_i_init, computation::root)
        .then(sum_i, i1)
        .then(sum_z_init, computation::root)
        .then(sum_z, i1)
        .then(sum_o_init, computation::root)
        .then(sum_o, i1)
        .then(sum_f_init, computation::root)
        .then(sum_f, i1)
        .then(sig_i, computation::root)
        .then(tnh_z, i1)
        .then(sig_o, i1)
        .then(sig_f, i1)
        .then(mul_iz, i1)
        .then(mul_fc, i1)
        .then(c, i1)
        .then(tnh_c, i1)
        .then(h, i1)
        .then(y, computation::root)
        .then(copy_y_to_host, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    params.store_in(&buf_params);
    // Weights and biases are packed
    R_i.store_in(&buf_Weights_gpu, {l, 0, i, j});
    R_z.store_in(&buf_Weights_gpu, {l, 1, i, j});
    R_o.store_in(&buf_Weights_gpu, {l, 2, i, j});
    R_f.store_in(&buf_Weights_gpu, {l, 3, i, j});
    W_i.store_in(&buf_Weights_gpu, {l, 4, i, j});
    W_z.store_in(&buf_Weights_gpu, {l, 5, i, j});
    W_o.store_in(&buf_Weights_gpu, {l, 6, i, j});
    W_f.store_in(&buf_Weights_gpu, {l, 7, i, j});
    b_i.store_in(&buf_biases_gpu, {l, 0, i});
    b_z.store_in(&buf_biases_gpu, {l, 1, i});
    b_o.store_in(&buf_biases_gpu, {l, 2, i});
    b_f.store_in(&buf_biases_gpu, {l, 3, i});
    x.store_in(&buf_x_gpu);
    y.store_in(&buf_y_gpu);
    sum_i_init.store_in(&buf_tmp_i, {l, k, i});
    sum_z_init.store_in(&buf_tmp_z, {l, k, i});
    sum_o_init.store_in(&buf_tmp_o, {l, k, i});
    sum_f_init.store_in(&buf_tmp_f, {l, k, i});
    sum_i.store_in(&buf_tmp_i, {l, k, i});
    sum_z.store_in(&buf_tmp_z, {l, k, i});
    sum_o.store_in(&buf_tmp_o, {l, k, i});
    sum_f.store_in(&buf_tmp_f, {l, k, i});
    sig_i.store_in(&buf_tmp_i, {l, k, i});
    tnh_z.store_in(&buf_tmp_z, {l, k, i});
    sig_o.store_in(&buf_tmp_o, {l, k, i});
    sig_f.store_in(&buf_tmp_f, {l, k, i});
    mul_iz.store_in(&buf_tmp_i, {l, k, i});
    mul_fc.store_in(&buf_tmp_f, {l, k, i});
    tnh_c.store_in(&buf_tmp_i, {l, k, i});
    h.store_in(&buf_h, {m + 1, l + 1, k, i});
    c.store_in(&buf_c, {m + 1, l, k, i});
    h_init.store_in(&buf_h, {0, l, k, i});
    c_init.store_in(&buf_c, {0, l, k, i});
    h_copy_x.store_in(&buf_h, {m, 0, k, i});

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
        }, "lstm.o", true);

    return 0;
}
