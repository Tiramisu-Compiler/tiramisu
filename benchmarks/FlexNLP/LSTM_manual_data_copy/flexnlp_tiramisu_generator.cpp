#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("flexnlp_lstm");

    // ----------------------------------------------------------------
    // Layer I
    // ----------------------------------------------------------------
    var s("s", 0, SEQ_LENGTH);
    var b("b", 0, BATCH_SIZE);
    var i("i", 0, INPUT_SIZE);

    var gate("gate", 0, 4);
    var j("j", 0, HIDDEN_SIZE);
    var k("k", 0, INPUT_SIZE + HIDDEN_SIZE);
    var o("o", 0, HIDDEN_SIZE);

    var l("l", 0, NUM_LAYERS);

    // Declare CPU Inputs
    input input_cpu_input("input_cpu_input", {s, b, i}, p_int8);
    input input_cpu_W("input_cpu_W", {l, gate, o, k}, p_int8);
    input input_cpu_output("input_cpu_output", {s, b, j}, p_int8);
    input input_cpu_h_out("input_cpu_h_out", {l, b, j}, p_int8);

    // Declare CPU Output
    computation initialize_flexnlp("initialize_flexnlp", {}, flexnlp_init(1));

    // We declare a computation to copy the Input
    computation copy_input("copy_input", {l},
          flexnlp_load_input(*input_cpu_input.get_buffer(), // Input
                               0, // Offset in Host buffer
                               SEQ_LENGTH * BATCH_SIZE * INPUT_SIZE) // Number of elements
    );

    // We declare a computation to copy the weights
    computation copy_weights("copy_weights", {l},
          flexnlp_load_weights(*input_cpu_W.get_buffer(), // Weights
                               l * (4 * OUTPUT_SIZE * (INPUT_SIZE + HIDDEN_SIZE)), // Offset in host buffer (input_cpu_W)
                               4 * OUTPUT_SIZE * (INPUT_SIZE + HIDDEN_SIZE)) // Number of elements
    );
    // Runs the LSTM cell
    computation run_lstm("run_lstm", {l},
          flexnlp_lstm_cell_manual(*input_cpu_input.get_buffer(), *input_cpu_W.get_buffer(), // Weights
                                   *input_cpu_output.get_buffer(), *input_cpu_h_out.get_buffer(),
                                   l));

    computation copy_output("copy_output", {l},
          flexnlp_store_output(*input_cpu_output.get_buffer(),
                               0,
                               BATCH_SIZE * OUTPUT_SIZE)
    );

    // ----------------------------------------------------------------
    // Layer II:Apply schedules and specify computations order
    // ----------------------------------------------------------------
    initialize_flexnlp.then(copy_input, computation::root)
                      .then(copy_weights, l)
                      .then(run_lstm, l)
                      .then(copy_output, l);

    // ----------------------------------------------------------------
    // Layer III : Specify access to data
    // ----------------------------------------------------------------
    buffer tmp_buf("tmp_buf", {1}, p_float32, a_temporary);

    initialize_flexnlp.store_in(&tmp_buf, {0});
    run_lstm.store_in(&tmp_buf, {0});

    // ----------------------------------------------------------------
    // Code Generation:Generate code for execution on a FlexNLP device
    // ----------------------------------------------------------------
    tiramisu::codegen({
        input_cpu_input.get_buffer(),
        input_cpu_W.get_buffer(),
        input_cpu_output.get_buffer(),
        input_cpu_h_out.get_buffer()
    },"generated_flexnlp_test.o", tiramisu::hardware_architecture_t::arch_flexnlp);

    return 0;
}
