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
    constant zero("zero", 0);
    // Declare CPU Inputs
    input input_cpu_input("input_cpu_input", {s}, p_float32);
    input input_cpu_Wx("input_cpu_Wx", {l}, p_float32);
    input input_cpu_Wh("input_cpu_Wh", {l}, p_float32);
    input input_cpu_bx("input_cpu_bx", {l}, p_float32);
    input input_cpu_bh("input_cpu_bh", {l}, p_float32);
    input input_cpu_h_in("input_cpu_h_in", {l}, p_float32);

    // Declare CPU Output
    buffer c_buf("c_buf", {NUM_LAYERS}, p_float32, a_output);
    buffer output_buf("output_buf", {SEQ_LENGTH}, p_float32, a_output);

    computation initialize_flexnlp("initialize_flexnlp", {}, flexnlp_init(1));

    // Runs the LSTM cell
    computation run_lstm_cell("run_lstm_cell", {l, s}, flexnlp_lstm_cell(*input_cpu_Wx.get_buffer(), *input_cpu_Wh.get_buffer(), *input_cpu_bx.get_buffer(), *input_cpu_bh.get_buffer(),
                                                                         *input_cpu_input.get_buffer(), *input_cpu_h_in.get_buffer(), output_buf,
                                                                         c_buf, expr(0)));

    // ----------------------------------------------------------------
    // Layer II:Apply schedules and specify computations order
    // ----------------------------------------------------------------

    initialize_flexnlp.then(run_lstm_cell, computation::root);

    // ----------------------------------------------------------------
    // Layer III:Specify access to data
    // ----------------------------------------------------------------
    buffer tmp_buf("tmp_buf", {1}, p_float32, a_temporary);

    initialize_flexnlp.store_in(&tmp_buf, {0});
    run_lstm_cell.store_in(&tmp_buf, {0});

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
