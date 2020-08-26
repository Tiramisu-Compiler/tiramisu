#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

// Set to true to perform autoscheduling
bool perform_autoscheduling = false;

// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python;

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/data/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)
{
    tiramisu::init("conv");
    
    var n("n", 0, 8), fout("fout", 0, 2), y("y", 0, 1024), x("x", 0, 1024), fin("fin", 0, 3) , 
        k_y("k_y", 0, 3), k_x("k_x", 0, 3) , y_pad("y_pad", 0, 1026) , x_pad("x_pad", 0, 1026);
    
    // Declare computations
    input bias("bias", {fout}, p_int32);
    input src("src", {n, fin, y_pad, x_pad}, p_int32);
    input weights("weights", {n, fout, y, x}, p_int32);

    computation conv_init("conv_init", {n, fout, y, x}, bias(fout));
    computation conv("conv", {n, fout, y, x, fin, k_y, k_x}, p_int32);
    conv.set_expression(conv(n, fout, y, x, fin, k_y, k_x) + src(n, fin, y + k_y, x + k_x) * weights(fout, fin, k_y, k_x));
    
    conv_init.then(conv, x);
    
    // Declare buffers
    buffer buf_bias("buf_bias", {2}, p_int32, a_input);
    buffer buf_src("buf_src", {8, 3, 1026, 1026}, p_int32, a_input);
    buffer buf_weights("buf_weights", {2, 3, 3, 3}, p_int32, a_input);
    
    buffer buf_output("buf_output", {8, 2, 1024, 1024}, p_int32, a_output);

    bias.store_in(&buf_bias);
    src.store_in(&buf_src);
    weights.store_in(&buf_weights);
    
    conv_init.store_in(&buf_output);
    conv.store_in(&buf_output, {n, fout, y, x});
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
        tiramisu::codegen({
            &buf_output, 
            &buf_bias, 
            &buf_src, 
            &buf_weights
        }, "function.o");
       
        return 0;
    }

    // Some parameters for the search methods
    const int beam_size = 2;
    const int max_depth = 4;

    const int nb_samples = 5;
    const int topk = 1;
    
    // An object used by search methods to generate schedules
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    // An evaluation function that measures execution time by compiling and executing the program
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf_output, &buf_bias, &buf_src, &buf_weights}, 
                                      "function.o", "./wrapper");
    
    // An evaluation function that uses an ML model to estimate speedup
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    
    // Two search methods : Beam Search and MCTS
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    // Create the autoscheduler and start search
    auto_scheduler::auto_scheduler as(bs, model_eval);
    as.set_exec_evaluator(exec_eval);
    as.find_schedule();
    as.apply_best_schedule();

    delete scheds_gen;
    delete exec_eval;
    delete bs;
    delete mcts;
    
    return 0;
}
