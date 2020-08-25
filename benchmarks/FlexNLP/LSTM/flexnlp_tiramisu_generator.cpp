#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("flexnlp_test");

    // ----------------------------------------------------------------
    // Layer I
    // ----------------------------------------------------------------
    var i("i", 0, INPUT_SIZE), j("j", 0, FEATURE_SIZE), k("k", 0, FEATURE_SIZE);
    var s("s", 0, SEQ_LENGTH);
    var l("l", 0, NUM_LAYERS);
    var gate("gate", 0, 4);
    var dummy("dummy", 0, 1);
    constant zero("zero", 0);
    // Declare CPU Inputs
    input input_cpu_input("input_cpu_input", {s}, p_float32);
    input input_cpu_Wx("input_cpu_Wx", {l}, p_float32);
    input input_cpu_Wh("input_cpu_Wh", {l}, p_float32);
    input input_cpu_bx("input_cpu_bx", {l}, p_float32);
    input input_cpu_bh("input_cpu_bh", {l}, p_float32);
    input input_cpu_h_in("input_cpu_h_in", {l}, p_float32);


    input input_cpu_c("input_cpu_c", {l}, p_float32);
    input input_cpu_output("input_cpu_output", {s}, p_float32);

    input tmp_dummy_comp("tmp_dummy_comp", {dummy}, p_float32);


    // Declare CPU Output
    buffer c_buf("c_buf", {NUM_LAYERS}, p_float32, a_output);
    buffer output_buf("output_buf", {SEQ_LENGTH}, p_float32, a_output);

    // Makes sure the inputs are not discared because no access has been
    // done on them
    computation dummy_computation("dummy_computation", {}, input_cpu_input(0) + input_cpu_Wx(0) + input_cpu_Wh(0) + input_cpu_bx(0) + input_cpu_bh(0) + input_cpu_h_in(0) + input_cpu_c(0) + input_cpu_output(0));
    computation dummy_buf_c("dummy_buf_c", {}, input_cpu_c(0));
    computation dummy_buf_c_back("dummy_buf_c_back", {}, tmp_dummy_comp(0));

    computation dummy_buf_output("dummy_buf_output", {}, input_cpu_output(0));
    computation dummy_buf_output_back("dummy_buf_output_back", {}, tmp_dummy_comp(0));

    computation initialize_flexnlp("initialize_flexnlp", {}, flexnlp_init(1));

    // Runs the LSTM cell
    computation run_lstm_cell("run_lstm_cell", {l, s}, flexnlp_lstm_cell(*input_cpu_Wx.get_buffer(), *input_cpu_Wh.get_buffer(), *input_cpu_bx.get_buffer(), *input_cpu_bh.get_buffer(),
                                                                         *input_cpu_input.get_buffer(), *input_cpu_h_in.get_buffer(), output_buf,
                                                                         c_buf, expr(0)));

    // ----------------------------------------------------------------
    // Layer II:Apply schedules and specify computations order
    // ----------------------------------------------------------------
    dummy_computation.then(dummy_buf_c, computation::root)
                     .then(dummy_buf_c_back, computation::root)
                     .then(dummy_buf_output, computation::root)
                     .then(dummy_buf_output_back, computation::root)

                     .then(initialize_flexnlp, computation::root)
                     //.then(copy_input, computation::root)
                     .then(run_lstm_cell, computation::root);
                     //.then(copy_output, s)

    // ----------------------------------------------------------------
    // Layer III:Specify access to data
    // ----------------------------------------------------------------
    buffer tmp_buf("tmp_buf", {1}, p_float32, a_temporary);

    dummy_computation.store_in(&tmp_buf, {0});
    dummy_buf_c.store_in(tmp_dummy_comp.get_buffer(), {0});
    dummy_buf_output.store_in(tmp_dummy_comp.get_buffer(), {0});
    dummy_buf_c_back.store_in(&c_buf, {0});
    dummy_buf_output_back.store_in(&output_buf, {0});

    initialize_flexnlp.store_in(&tmp_buf, {0});
    run_lstm_cell.store_in(&tmp_buf, {0});

    input_cpu_c.store_in(&c_buf, {l});
    input_cpu_output.store_in(&output_buf, {s});
    // ----------------------------------------------------------------
    // Code Generation:Generate code for execution on a FlexNLP device
    // ----------------------------------------------------------------
    tiramisu::codegen({
        input_cpu_input.get_buffer(),
        input_cpu_Wx.get_buffer(),
        input_cpu_Wh.get_buffer(),
        input_cpu_bx.get_buffer(),
        input_cpu_bh.get_buffer(),
        input_cpu_h_in.get_buffer(),
        &c_buf,
        &output_buf
    },"generated_flexnlp_test.o", tiramisu::hardware_architecture_t::arch_flexnlp);

    return 0;
}
