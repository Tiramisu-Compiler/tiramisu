#include <tiramisu/tiramisu.h>

#include "configure.h"

#define GEMM_BATCH 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("lstm");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Inner dimensions
    var i("i", 0, FEATURE_SIZE), j("j", 0, FEATURE_SIZE), k("k", 0, BATCH_SIZE);
    var i_merged("i_merged", 0, 4 * FEATURE_SIZE);
    var i0("i0"), i1("i1"), k0("k0"), k1("k1");
    var w_i("w_i", 0, 2);

    // Outer dimensions
    var l("l", 0, NUM_LAYERS), s("s", 0, SEQ_LENGTH);
    var s0("s0", 0, SEQ_LENGTH / GEMM_BATCH), s1("s1", 0, GEMM_BATCH);
    
    // After skewing
    var l_s("l_s"), s_s("s_s");

    input weights("weights", {l, w_i, j, i_merged}, DATA_TYPE_P);
    input x("x", {s, k, i}, DATA_TYPE_P);
    input tmp("tmp", {s, k, i_merged}, DATA_TYPE_P);
    input biases("biases", {l, i_merged}, DATA_TYPE_P);

    // h(l, s) is the output of the block (l, s)
    // which takes h(l, s - 1) and h(l - 1, s) as inputs
    // Initial hidden states are h(l, -1) and c(l, -1)
    // Input x is copied to h(-1, s)
    computation h({l, s, k, i}, DATA_TYPE_P);
    computation c({l, s, k, i}, DATA_TYPE_P);
    
    // Pad buffers to make room for edges
    h.store_in({l + 1, s + 1, k, i}, {NUM_LAYERS + 1, SEQ_LENGTH + 1, BATCH_SIZE, FEATURE_SIZE});
    c.store_in({l, s + 1, k, i}, {NUM_LAYERS, SEQ_LENGTH + 1, BATCH_SIZE, FEATURE_SIZE});
        
    // Initial sets and stores
    computation h_init({l, k, i}, expr(DATA_TYPE(0)));
    computation c_init({l, k, i}, expr(DATA_TYPE(0)));
    computation h_copy_x({s, k, i}, x(s, k, i));

    // Multiplication from input is batched
    computation sum1({l, s0},
    cblas_gemm(*h.get_buffer(), *weights.get_buffer(), *tmp.get_buffer(),
                    GEMM_BATCH * BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
                    1, 0,  // alpha, beta
                    0, 0, 0,  // ldABC
                    (l * (SEQ_LENGTH + 1) + s0 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE,  //offsetA
                    (l * 2) * 4 * FEATURE_SIZE * FEATURE_SIZE,  //offsetB
                    s0 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE,  // offsetC
                    false, false));

    computation sum2({l, s},
    cblas_gemm(*h.get_buffer(), *weights.get_buffer(), *tmp.get_buffer(),
                    BATCH_SIZE, 4 * FEATURE_SIZE, FEATURE_SIZE,
                    1, 1,  // alpha, beta
                    0, 0, 0,  // ldABC
                    ((l + 1) * (SEQ_LENGTH + 1) + s) * BATCH_SIZE * FEATURE_SIZE,  //offsetA
                    (l * 2 + 1) * 4 * FEATURE_SIZE * FEATURE_SIZE,  //offsetB
                    s * BATCH_SIZE * 4 * FEATURE_SIZE,  // offsetC
                    false, false));

    // Nonlinear operations as well as biases
    #define sigmoid(x) expr(DATA_TYPE(1)) / (1 + expr(o_expo, -(x)))
    #define tanh(x) (expr(o_expo, x) - expr(o_expo, -(x))) / (expr(o_expo, x) + expr(o_expo, -(x)))

    computation sig_i({l, s, k, i},      sigmoid(tmp(s, k, i + 0 * FEATURE_SIZE) + biases(l, i + 0 * FEATURE_SIZE)));
    computation sig_f({l, s, k, i},      sigmoid(tmp(s, k, i + 1 * FEATURE_SIZE) + biases(l, i + 1 * FEATURE_SIZE)));
    computation tnh_z({l, s, k, i},      tanh(tmp(s, k, i + 2 * FEATURE_SIZE) + biases(l, i + 2 * FEATURE_SIZE)));
    computation sig_o({l, s, k, i},      sigmoid(tmp(s, k, i + 3 * FEATURE_SIZE) + biases(l, i + 3 * FEATURE_SIZE)));

    c.set_expression(sig_i(l, s, k, i) * tnh_z(l, s, k, i) + sig_f(l, s, k, i) * c(l, s - 1, k, i));
    h.set_expression(tanh(c(l, s, k, i)) * sig_o(l, s, k, i));
   
    // Output is the last layer
    computation y({s, k, i}, h(NUM_LAYERS - 1, s, k, i));

    computation dummy_c({}, weights(0, 0, 0, 0));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    h_init.interchange(l, k);
    h_init.interchange(l, i);
    c_init.interchange(l, k);
    c_init.interchange(l, i);
    h_copy_x.interchange(s, k);
    h_copy_x.interchange(s, i);
    y.interchange(s, k);
    y.interchange(s, i);

    block nonlinear_block({&sig_i, &tnh_z, &sig_o, &sig_f, &c, &h});

    block({&sum2, &nonlinear_block}).split(s, GEMM_BATCH, s0, s1);
    block({&h_init, &c_init, &h_copy_x, &nonlinear_block, &y}).tile(k, i, 16, 16, k0, i0, k1, i1);
    block lstm_block({&sum1, &sum2, &nonlinear_block});

    // Skew and interchange to get diagonal traversal
    lstm_block.skew(l, s0, 1, l_s, s_s);
    lstm_block.interchange(l_s, s_s);
    // Parallelize diagonal traversal
    // Due to a bug in tagging system we only need to parallelize a single computation
    sum1.parallelize(l_s);

    // Scheduling commands
    dummy_c.then(h_init, computation::root)
           .then(c_init, l)
          .then(h_copy_x, computation::root)          
          .then(sum1, computation::root)   
          .then(sum2, l_s)
          .then(sig_i,s1 )
          .then(sig_f,i1)
          .then(tnh_z, i1)
          .then(sig_o, i1)
          .then(c, i1)
          .then(h, i1)
          .then(y, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer buf_y_cpu("buf_y_cpu", {SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE}, DATA_TYPE_P, a_output);

    sig_i.store_in(tmp.get_buffer(), {s, k, i + 0 * FEATURE_SIZE});
    sig_f.store_in(tmp.get_buffer(), {s, k, i + 1 * FEATURE_SIZE});
    tnh_z.store_in(tmp.get_buffer(), {s, k, i + 2 * FEATURE_SIZE});
    sig_o.store_in(tmp.get_buffer(), {s, k, i + 3 * FEATURE_SIZE});
    h_init.store_in(h.get_buffer(), {l + 1, 0, k, i});
    c_init.store_in(c.get_buffer(), {l, 0, k, i});
    h_copy_x.store_in(h.get_buffer(), {0, s + 1, k, i});

    y.store_in(&buf_y_cpu);

    buffer dummy_buf("dummy_buf", {1}, DATA_TYPE_P, a_temporary);
    dummy_c.store_in(&dummy_buf, {});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        weights.get_buffer(),
        biases.get_buffer(),
        x.get_buffer(),
        &buf_y_cpu
    }, "lstm.o");

    return 0;
}
