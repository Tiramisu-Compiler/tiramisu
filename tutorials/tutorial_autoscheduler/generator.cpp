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
    
    var t("t", 0, 100), y("y", 0, 1024), x("x", 0, 1024),z("z", 0, 128);;

    //var  yy("yy", 1, 223), xx("xx", 1, 223);
    var  yy("yy", 1, 257), xx("xx", 1, 129), zz("zz", 1, 128);

    var t2("t2"),t1("t1"),y1("y1"),x1("x1"),y2("y2"),x2("x2") ,x0("x0");
    
    // Declare computations

    input A("A", {x,y}, p_int32);
    input B("B", {x,y}, p_int32);



    computation B_out("B_out", {t,xx,yy}, A(xx, yy) + A(xx, yy-1) + A(xx, 1+yy) + A(1+xx, yy) + A(xx-1, yy));

    computation A_out("A_out", {t,xx,yy}, B(xx, yy) + B(xx, yy-1) + B(xx, 1+yy) + B(1+xx, yy) + B(xx-1, yy));

    

    buffer b_A("buffA", {1024,1024}, p_int32, a_output);    
    buffer b_B("buffB", {1024,1024}, p_int32, a_output); 
    A.store_in(&b_A);
    B.store_in(&b_B);

    //Store computations
    A_out.store_in(&b_A, {xx,yy});
    B_out.store_in(&b_B, {xx,yy});  


    B_out.then(A_out, t);
//    B_out.interchange(1,2);
    // the code above is the initial unfused code since we used "B_out.then(A_out, t)" 
    // we want to dependency analysis to be performed on the original code correctly

    prepare_schedules_for_legality_checks(true);
    performe_full_dependency_analysis();
    // result stored in the function class


    // this is a major change in the program, as we fuse to the yy loop level
    //B_out.then(A_out,yy);

    // we must prepare schedules for our solvers like this, since we want applied changes on sched_graph to be reflected in schedules
    //prepare_schedules_for_legality_checks(true);

    bool perform_autoscheduling = false;

    // this is the fusion solver
    /*auto shiftings = global::get_implicit_function()->correcting_loop_fusion_with_shifting({&B_out},A_out,{t,xx,yy});

    assert(shiftings.size() > 0);// asserts that a fusion is possible (shiftings.size() == 0 means impossible fusion)

    // shift A_out loops which was the target computation to shift in the solver
    for(auto const& tup:shiftings)
    {
        A_out.shift(
            std::get<0>(tup),
            std::get<1>(tup)
            );
    }*/

    perform_autoscheduling= true;
    
    // Generate a program with no schedule
    if (!perform_autoscheduling)
    {
//        A_out.interchange(1,2);
//        global::get_implicit_function()->reset_schedules();
//        B_out.then(A_out, t);
//        A_out.interchange(1,2);
//
//        B_out.parallelize(xx);

        B_out.then(A_out, 4);
        A_out.shift(1,1);
//        A_out.interchange(0,1);
//        B_out.interchange(0,1);
//        A_out.tile(1,2,20,10);
//        A_out.tile(1,2,20,10);
//        A_out.tile(1,2,32,10);
        A_out.skew(0,1,4,1);
        B_out.skew(0,1,4,1);
//        A_out.tag_parallel_level(1);
//        B_out.tag_parallel_level(1);
        A_out.tile(0,1,64,32);
        B_out.tile(0,1,64,32);



        tiramisu::codegen({
            &b_A,&b_B
        }, "function.o");
       
        return 0;
    }

    // Some parameters for the search methods
    const int beam_size = 5;
    const int max_depth = 6;

    const int nb_samples = 5;
    const int topk = 1;
    
    // An object used by search methods to generate schedules
    auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
    
    // An evaluation function that measures execution time by compiling and executing the program
    auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A,&b_B}, 
                                      "function.o", "./wrapper");
    
    // An evaluation function that uses an ML model to estimate speedup
    auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    
    // Two search methods : Beam Search and MCTS
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
//    auto_scheduler::mcts *mcts = new auto_scheduler::mcts(nb_samples, topk, max_depth, model_eval, exec_eval, scheds_gen);
    
    // Create the autoscheduler and start search
    auto_scheduler::auto_scheduler as(bs, exec_eval);
    as.set_exec_evaluator(exec_eval);
//    as.find_schedule();
    as.sample_search_space("test.json");
//    as.apply_best_schedule();

    delete scheds_gen;
    delete exec_eval;
    delete bs;
//    delete mcts;
    
    return 0;
}
