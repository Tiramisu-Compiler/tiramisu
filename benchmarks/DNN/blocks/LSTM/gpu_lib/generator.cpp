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
    var i0("i0"), i1("i1"), k0("k0"), k1("k1");
    // Outer dimensions
    var l("l", 0, NUM_LAYERS), m("m", 0, SEQ_LENGTH);

    input R("R", {l, i_merged, j}, p_float32);
    input W("W", {l, i_merged, j}, p_float32);
    input b("b", {l, i_merged}, p_float32);
    input x({m, k, i}, p_float32);

    buffer buf_Weights_cpu("buf_Weights_cpu", {NUM_LAYERS, 2, 4 * FEATURE_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_Weights("buf_Weights", {NUM_LAYERS, 2, 4 * FEATURE_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_h("buf_h", {NUM_LAYERS + 1, SEQ_LENGTH + 1, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp("buf_tmp", {SEQ_LENGTH, BATCH_SIZE, 4 * FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_biases_cpu("buf_biases_cpu", {NUM_LAYERS, 4 * FEATURE_SIZE}, p_float32, a_input);
    buffer buf_x_cpu("buf_x_cpu", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_input);
    buffer buf_y_cpu("buf_y_cpu", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_output);
    buffer buf_biases("buf_biases", {NUM_LAYERS, 4 * FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_x("buf_x", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_y("buf_y", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_i("buf_tmp_i", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buffer buf_c("buf_c", {SEQ_LENGTH + 1, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE}, p_float32, a_temporary);
    buf_Weights.tag_gpu_global();
    buf_h.tag_gpu_global();
    buf_tmp.tag_gpu_global();
    buf_biases.tag_gpu_global();
    buf_x.tag_gpu_global();
    buf_y.tag_gpu_global();
    buf_tmp_i.tag_gpu_global();
    buf_tmp_z.tag_gpu_global();
    buf_tmp_o.tag_gpu_global();
    buf_tmp_f.tag_gpu_global();
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
    computation sum_init({m, l, k, i_merged}, b(l, i_merged));
    computation sum1({m, l},
        cublas_sgemm(buf_h, buf_Weights, buf_tmp,
                     BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
                     1, 1,  // alpha, beta
                     0, 0, 0,  // ldABC
                     ((l + 1) * (SEQ_LENGTH + 1) + m) * BATCH_SIZE * FEATURE_SIZE,  //offsetA
                     (l * 2) * 4 * FEATURE_SIZE * FEATURE_SIZE,  //offsetB
                     m * BATCH_SIZE * 4 * FEATURE_SIZE,  // offsetC
                     false, true));
    computation sum2({m, l},
        cublas_sgemm(buf_h, buf_Weights, buf_tmp,
                     BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
                     1, 1,  // alpha, beta
                     0, 0, 0,  // ldABC
                     (l * (SEQ_LENGTH + 1) + m + 1) * BATCH_SIZE * FEATURE_SIZE,  //offsetA
                     (l * 2 + 1) * 4 * FEATURE_SIZE * FEATURE_SIZE,  //offsetB
                     m * BATCH_SIZE * 4 * FEATURE_SIZE,  // offsetC
                     false, true));
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
    // Copies
    computation copy_Weights_to_device({}, memcpy(buf_Weights_cpu, buf_Weights));
    computation copy_biases_to_device({}, memcpy(buf_biases_cpu, buf_biases));
    computation copy_x_to_device({}, memcpy(buf_x_cpu, buf_x));
    computation copy_y_to_host({}, memcpy(buf_y, buf_y_cpu));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    block({&h_init, &c_init, &h_copy_x, &sig_i, &tnh_z, &sig_o, &sig_f, &mul_iz,
           &mul_fc, &c, &tnh_c, &h, &y}).gpu_tile(k, i, 16, 16, k0, i0, k1, i1);
    sum_init.gpu_tile(k, i_merged, 16, 16, k0, i0, k1, i1);

    // Scheduling commands
    copy_Weights_to_device
            .then(copy_biases_to_device, computation::root)
            .then(copy_x_to_device, computation::root)
            .then(h_init, computation::root)
            .then(c_init, computation::root)
            .then(h_copy_x, computation::root)
            .then(sum_init, computation::root)
            .then(sum1, l)
            .then(sum2, l)
            .then(sig_i, i1)
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

    // Weights and biases are packed
    R.store_in(&buf_Weights, {l, 0, i_merged, j});
    W.store_in(&buf_Weights, {l, 1, i_merged, j});
    b.store_in(&buf_biases, {l, i_merged});
    x.store_in(&buf_x);
    y.store_in(&buf_y);
    sum_init.store_in(&buf_tmp, {m, k, i_merged});
    sig_i.store_in(&buf_tmp_i, {k, i});
    tnh_z.store_in(&buf_tmp_z, {k, i});
    sig_o.store_in(&buf_tmp_o, {k, i});
    sig_f.store_in(&buf_tmp_f, {k, i});
    mul_iz.store_in(&buf_tmp_i, {k, i});
    mul_fc.store_in(&buf_tmp_f, {k, i});
    tnh_c.store_in(&buf_tmp_i, {k, i});
    h.store_in(&buf_h, {l + 1, m + 1, k, i});
    c.store_in(&buf_c, {m + 1, l, k, i});
    h_init.store_in(&buf_h, {l + 1, 0, k, i});
    c_init.store_in(&buf_c, {0, l, k, i});
    h_copy_x.store_in(&buf_h, {0, m + 1, k, i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files.
    tiramisu::codegen({
            &buf_Weights_cpu,
            &buf_biases_cpu,
            &buf_x_cpu,
            &buf_y_cpu,
        }, "lstm.o", true);

    return 0;
}
