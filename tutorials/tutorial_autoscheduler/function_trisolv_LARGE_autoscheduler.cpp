#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_trisolv_LARGE_wrapper.h"



const std::string py_cmd_path = "/data/scratch/mmerouani/anaconda/envs/base-tig/bin/python";
const std::string py_interface_path = "/data/scratch/mmerouani/tiramisu3/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_trisolv_LARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 2000);

    //Iteration variables    
    var i("i", 0, 2000), j("j");
    

    //inputs
    input L("L", {i, i}, p_float64);
    input b("b", {i}, p_float64);


    //Computations

    computation x_init("x_init", {i}, b(i));
    computation x_sub("[NN]->{x_sub[i,j]: 0<=i<NN and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    x_sub.set_expression(x_sub(i,j) - L(i,j) * x_init(j));
    computation x_out("x_out", {i}, x_sub(i,0) / L(i,i));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    x_init.then(x_sub,i)
          .then(x_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_L("b_L", {2000,2000}, p_float64, a_input);    
    buffer b_b("b_b", {2000}, p_float64, a_input);    
    buffer b_x("b_x", {2000}, p_float64, a_output);    

    //Store inputs
    L.store_in(&b_L);    
    b.store_in(&b_b);    

    //Store computations
    x_init.store_in(&b_x);
    x_sub.store_in(&b_x, {i});
    x_out.store_in(&b_x);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_L, &b_b, &b_x}, "function_trisolv_LARGE.o", "./function_trisolv_LARGE_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_trisolv_LARGE_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
