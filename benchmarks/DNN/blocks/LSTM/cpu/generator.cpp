#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Single LSTM block without minibatching
    tiramisu::init("lstm");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var p_i("p_i", 0, 1);
    input params({p_i}, p_int32);
    constant feature_size("feature_size", params(0));

    var i("i", 0, feature_size), j("j", 0, feature_size);

    input R_i("R_i", {i, j}, p_float32);
    input R_z("R_z", {i, j}, p_float32);
    input R_o("R_o", {i, j}, p_float32);
    input R_f("R_f", {i, j}, p_float32);
    input W_i("W_i", {i, j}, p_float32);
    input W_z("W_z", {i, j}, p_float32);
    input W_o("W_o", {i, j}, p_float32);
    input W_f("W_f", {i, j}, p_float32);
    input b_i("b_i", {i}, p_float32);
    input b_z("b_z", {i}, p_float32);
    input b_o("b_o", {i}, p_float32);
    input b_f("b_f", {i}, p_float32);
    input h_prev({j}, p_float32);
    input c_prev({j}, p_float32);
    input x({j}, p_float32);

    computation sum_i_init({i}, b_i(i));
    computation sum_z_init({i}, b_z(i));
    computation sum_o_init({i}, b_o(i));
    computation sum_f_init({i}, b_f(i));
    computation sum_i({i, j}, sum_i_init(i) + R_i(i, j) * h_prev(j) + W_i(i, j) * x(j));
    computation sum_z({i, j}, sum_z_init(i) + R_z(i, j) * h_prev(j) + W_z(i, j) * x(j));
    computation sum_o({i, j}, sum_o_init(i) + R_o(i, j) * h_prev(j) + W_o(i, j) * x(j));
    computation sum_f({i, j}, sum_f_init(i) + R_f(i, j) * h_prev(j) + W_f(i, j) * x(j));
    computation sig_i({i}, expr(float(1)) / (1 + expr(o_expo, -sum_i(i, -1))));
    computation tnh_z({i}, expr(o_tanh, sum_z(i, -1)));
    computation sig_o({i}, expr(float(1)) / (1 + expr(o_expo, -sum_o(i, -1))));
    computation sig_f({i}, expr(float(1)) / (1 + expr(o_expo, -sum_f(i, -1))));
    computation mul_iz({i}, sig_i(i) * tnh_z(i));
    computation mul_fc({i}, sig_f(i) * c_prev(i));
    computation c({i}, mul_iz(i) + mul_fc(i));
    computation tnh_c({i}, expr(o_tanh, c(i)));
    computation h({i}, tnh_c(i) * sig_o(i));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands
    sum_i_init.then(sum_z_init, computation::root)
              .then(sum_o_init, computation::root)
              .then(sum_f_init, computation::root)
              .then(sum_i, computation::root)
              .then(sum_z, computation::root)
              .then(sum_o, computation::root)
              .then(sum_f, computation::root)
              .then(sig_i, computation::root)
              .then(tnh_z, computation::root)
              .then(sig_o, computation::root)
              .then(sig_f, computation::root)
              .then(mul_iz, computation::root)
              .then(mul_fc, computation::root)
              .then(c, computation::root)
              .then(tnh_c, computation::root)
              .then(h, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer buf_params("buf_params", {1}, p_int32, a_input);
    buffer buf_Weights("buf_Weights", {feature_size * 8, feature_size}, p_float32, a_input);
    buffer buf_biases("buf_biases", {feature_size * 4}, p_float32, a_input);
    buffer buf_h_prev("buf_h_prev", {feature_size}, p_float32, a_input);
    buffer buf_c_prev("buf_c_prev", {feature_size}, p_float32, a_input);
    buffer buf_x("buf_x", {feature_size}, p_float32, a_input);
    buffer buf_tmp_i("buf_tmp_i", {feature_size}, p_float32, a_temporary);
    buffer buf_tmp_z("buf_tmp_z", {feature_size}, p_float32, a_temporary);
    buffer buf_tmp_o("buf_tmp_o", {feature_size}, p_float32, a_temporary);
    buffer buf_tmp_f("buf_tmp_f", {feature_size}, p_float32, a_temporary);
    buffer buf_c("buf_c", {feature_size}, p_float32, a_output);
    buffer buf_h("buf_h", {feature_size}, p_float32, a_output);

    params.store_in(&buf_params);
    R_i.set_access("[feature_size]->{R_i[i, j]->buf_Weights[i + feature_size * 0, j]}");
    R_z.set_access("[feature_size]->{R_z[i, j]->buf_Weights[i + feature_size * 1, j]}");
    R_o.set_access("[feature_size]->{R_o[i, j]->buf_Weights[i + feature_size * 2, j]}");
    R_f.set_access("[feature_size]->{R_f[i, j]->buf_Weights[i + feature_size * 3, j]}");
    W_i.set_access("[feature_size]->{W_i[i, j]->buf_Weights[i + feature_size * 4, j]}");
    W_z.set_access("[feature_size]->{W_z[i, j]->buf_Weights[i + feature_size * 5, j]}");
    W_o.set_access("[feature_size]->{W_o[i, j]->buf_Weights[i + feature_size * 6, j]}");
    W_f.set_access("[feature_size]->{W_f[i, j]->buf_Weights[i + feature_size * 7, j]}");
    b_i.set_access("[feature_size]->{b_i[i]->buf_biases[i + feature_size * 0]}");
    b_z.set_access("[feature_size]->{b_z[i]->buf_biases[i + feature_size * 1]}");
    b_o.set_access("[feature_size]->{b_o[i]->buf_biases[i + feature_size * 2]}");
    b_f.set_access("[feature_size]->{b_f[i]->buf_biases[i + feature_size * 3]}");
    h_prev.store_in(&buf_h_prev);
    c_prev.store_in(&buf_c_prev);
    x.store_in(&buf_x);
    sum_i_init.store_in(&buf_tmp_i);
    sum_z_init.store_in(&buf_tmp_z);
    sum_o_init.store_in(&buf_tmp_o);
    sum_f_init.store_in(&buf_tmp_f);
    sum_i.store_in(&buf_tmp_i, {i});
    sum_z.store_in(&buf_tmp_z, {i});
    sum_o.store_in(&buf_tmp_o, {i});
    sum_f.store_in(&buf_tmp_f, {i});
    sig_i.store_in(&buf_tmp_i);
    tnh_z.store_in(&buf_tmp_z);
    sig_o.store_in(&buf_tmp_o);
    sig_f.store_in(&buf_tmp_f);
    mul_iz.store_in(&buf_tmp_i);
    mul_fc.store_in(&buf_tmp_f);
    c.store_in(&buf_c);
    tnh_c.store_in(&buf_tmp_i);
    h.store_in(&buf_h);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({
            &buf_params,
            &buf_Weights,
            &buf_biases,
            &buf_h_prev,
            &buf_c_prev,
            &buf_x,
            &buf_h,
            &buf_c
        }, "lstm.o");

    return 0;
}
