#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

using namespace tiramisu;

// Set to true to perform autoscheduling


// Path to python (please give absolute path)
const std::string py_cmd_path = "/usr/bin/python3";

// Path to a script that executes the ML model (please give absolute path)
const std::string py_interface_path = "/home/nassim/Desktop/tiramisu_raw/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)
{
    tiramisu::init("conv");
    
    var t("t", 0, 200), y("y", 0, 1024), x("x", 0, 1024);

    //var  yy("yy", 1, 1021), xx("xx", 1, 1023);
    var  yy("yy", 10, 778), xx("xx", 1, 1023);

    var t2("t2"),t1("t1"),y1("y1"),x1("x1"),y2("y2"),x2("x2") ,x0("x0");
    
    // Declare computations

    input src("src", {x, y}, p_int32);


    computation conv("conv", {t,xx,yy}, p_int32);
    //conv.set_expression(  2*src(xx,yy));
    //conv.set_expression( src(xx-1,yy-1) - src(xx-1,yy)+src(xx-1,yy+1)-src(xx,yy-1)        + src(xx,yy+1)+src(xx+1,yy-1)-src(xx+1,yy)+src(xx+1,yy+1) );
    conv.set_expression( src(xx-1,yy) -src(xx,yy)        +src(xx+1,yy) );
    // Declare buffers
    
    buffer buf_output("buf_output", {1024, 1024}, p_int32, a_output);

    src.store_in(&buf_output);
    
    conv.store_in(&buf_output, {xx, yy});

    prepare_schedules_for_legality_checks();
    performe_full_dependency_analysis();

    bool perform_autoscheduling = false;

    perform_autoscheduling=true;
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
        //conv.skew(t,x,2,1,t1,x0);
        //conv.tile(x0,y,64,32,x1,y1,x2,y2);
        //conv.parallelize(x1);

        tiramisu::codegen({
            &buf_output
        }, "function.o");
       
        return 0;
    }

    // Some parameters for the search methods
    const int beam_size = 4;
    const int max_depth = 6;

    const int nb_samples = 5;
    const int topk = 1;
    
    // An object used by search methods to generate schedules
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    // An evaluation function that measures execution time by compiling and executing the program
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf_output}, 
                                      "function.o", "./wrapper");
    
    // An evaluation function that uses an ML model to estimate speedup
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    
    // Two search methods : Beam Search and MCTS
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    // Create the autoscheduler and start search
    auto_scheduler::auto_scheduler as(bs, exec_eval);
    as.set_exec_evaluator(exec_eval);
    as.find_schedule();
    as.apply_best_schedule();

    delete scheds_gen;
    delete exec_eval;
    delete bs;
    delete mcts;
    
    return 0;
}
