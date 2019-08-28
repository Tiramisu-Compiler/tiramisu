#define __TIRAMISU_GENERATOR__
#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

#define sigmoid(x) (expr(DATA_TYPE(1)) / (1 + expr(o_expo, -(x))))
#define tanh(x) ((expr(o_expo, 2*(x)) - 1) / (expr(o_expo, 2*(x)) + 1))

int main(int argc, char **argv)
{
    tiramisu::init("lstm");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Inner dimensions
    var i("i", 0, FEATURE_SIZE), j("j", 0, FEATURE_SIZE), k("k", 0, BATCH_SIZE);
    var i_merged("i_merged", 0, 4 * FEATURE_SIZE);
    var w_i("w_i", 0, 2);

    // Outer dimensions
    var l("l", 0, NUM_LAYERS), s("s", 0, SEQ_LENGTH);
    var s0("s0", 0, SEQ_LENGTH / GEMM_BATCH);

    input weights("weights", {l, w_i, j, i_merged}, DATA_TYPE_P);
    input x("x", {s, k, i}, DATA_TYPE_P);
    input biases("biases", {l, i_merged}, DATA_TYPE_P);

    input tmp("tmp", {s, k, i_merged}, DATA_TYPE_P);

    // h(l, s) is the output of the block (l, s)
    // which takes h(l, s - 1) and h(l - 1, s) as inputs
    // Initial hidden states are h(l, -1) and c(l, -1)
    // Input x is copied to h(-1, s)
    computation h("h", {l, s, k, i}, DATA_TYPE_P);
    computation c("c", {l, s, k, i}, DATA_TYPE_P);

    // Pad buffers to make room for edges
    h.store_in({s + 1, k, i}, {SEQ_LENGTH + 1, BATCH_SIZE, FEATURE_SIZE});
    c.store_in({k, i}, {BATCH_SIZE, FEATURE_SIZE});

    // Initial sets and stores
    computation h_init("h_init", {l, k, i}, cast(DATA_TYPE_P, 0));
    computation c_init("c_init", {l, k, i}, cast(DATA_TYPE_P, 0));
    computation h_copy_x("h_copy_x", {s, k, i}, x(s, k, i));

    // Multiplication from input is batched
    computation sum1("sum1", {l, s0},
        cblas_gemm(
            *h.get_buffer(), *weights.get_buffer(), *tmp.get_buffer(),
            GEMM_BATCH * BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
            1, 0, // alpha, beta
            0, 0, 0, // ldABC
            (s0 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE, //offsetA
            (l * 2) * 4 * FEATURE_SIZE * FEATURE_SIZE, //offsetB
            s0 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE, // offsetC
            false, false
        )
    );

    computation sum2("sum2", {l, s},
        cblas_gemm(
            *h.get_buffer(), *weights.get_buffer(), *tmp.get_buffer(),
            BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
            1, 1, // alpha, beta
            0, 0, 0, // ldABC
            s * BATCH_SIZE * FEATURE_SIZE, //offsetA
            (l * 2 + 1) * 4 * FEATURE_SIZE * FEATURE_SIZE, //offsetB
            s * BATCH_SIZE * 4 * FEATURE_SIZE, // offsetC
            false, false
        )
    );

    // Nonlinear operations as well as biases
    computation sig_i({l, s, k, i}, sigmoid(tmp(s, k, i + 0 * FEATURE_SIZE) + biases(l, i + 0 * FEATURE_SIZE)));
    computation sig_f({l, s, k, i}, sigmoid(tmp(s, k, i + 1 * FEATURE_SIZE) + biases(l, i + 1 * FEATURE_SIZE)));
    computation tnh_z({l, s, k, i}, tanh(tmp(s, k, i + 2 * FEATURE_SIZE) + biases(l, i + 2 * FEATURE_SIZE)));
    computation sig_o({l, s, k, i}, sigmoid(tmp(s, k, i + 3 * FEATURE_SIZE) + biases(l, i + 3 * FEATURE_SIZE)));

    c.set_expression(sig_i(l, s, k, i) * tnh_z(l, s, k, i) + sig_f(l, s, k, i) * c(l, s - 1, k, i));
    h.set_expression(tanh(c(l, s, k, i)) * sig_o(l, s, k, i));

    // Output is the last layer
    computation copy_output({s, k, i}, h(NUM_LAYERS - 1, s, k, i));

    computation dummy_c({}, weights(0, 0, 0, 0));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    dummy_c.then(h_copy_x, computation::root)
           .then(h_init, computation::root)
           .then(c_init, i)
           .then(sum1, l)
           .then(sum2, l)
           .then(sig_i, s)
           .then(sig_f, i)
           .then(tnh_z, i)
           .then(sig_o, i)
           .then(c, i)
           .then(h, i)
           .then(copy_output, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    h_init.store_in(h.get_buffer(), {0, k, i});
    h_copy_x.store_in(h.get_buffer(), {s + 1, k, i});
    c_init.store_in(c.get_buffer(), {k, i});

    buffer sig_buf("sig_buf", {4, VEC_LEN}, DATA_TYPE_P, a_temporary);

    sig_i.store_in(&sig_buf, {0, i%VEC_LEN});
    sig_f.store_in(&sig_buf, {1, i%VEC_LEN});
    tnh_z.store_in(&sig_buf, {2, i%VEC_LEN});
    sig_o.store_in(&sig_buf, {3, i%VEC_LEN});

    buffer dummy_buf("dummy_buf", {1}, DATA_TYPE_P, a_temporary);
    dummy_c.store_in(&dummy_buf, {});

    buffer gemm_ret("gemm_ret", {1}, DATA_TYPE_P, a_temporary);
    sum1.store_in(&gemm_ret, {});
    sum2.store_in(&gemm_ret, {});

    buffer buf_output("buf_output", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, DATA_TYPE_P, a_output);
    copy_output.store_in(&buf_output);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        weights.get_buffer(),
        biases.get_buffer(),
        x.get_buffer(),
        h.get_buffer(),
        c.get_buffer(),
        &buf_output
    }, "lstm.o");

    return 0;
}
