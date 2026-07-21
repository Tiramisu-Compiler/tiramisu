#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_trisolv_XLARGE_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_trisolv_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 4000), j("j");
    

    //inputs
    input L("L", {i, i}, p_float64);
    input b("b", {i}, p_float64);
    input x("x", {i}, p_float64);


    //Computations

    computation x_init("{x_init[i]: 0<=i<4000 }", expr(), true, p_float64, global::get_implicit_function());
    x_init.set_expression(b(i));
    computation x_sub("{x_sub[i,j]: 0<=i<4000 and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    x_sub.set_expression(x(i) - L(i,j) * x(j));
    computation x_out("{x_out[i]: 0<=i<4000 }", expr(), true, p_float64, global::get_implicit_function());
    x_out.set_expression(x(i) / L(i,i));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    x_init.then(x_sub,i)
            .then(x_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_L("b_L", {4000,4000}, p_float64, a_input);    
    buffer b_b("b_b", {4000}, p_float64, a_input);    
    buffer b_x("b_x", {4000}, p_float64, a_output);    

    //Store inputs
    L.store_in(&b_L);
    b.store_in(&b_b);
    x.store_in(&b_x);   

    //Store computations
    x_init.store_in(&b_x);
    x_sub.store_in(&b_x, {i});
    x_out.store_in(&b_x);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_L, &b_b, &b_x}, "function_trisolv_XLARGE.o", "./function_trisolv_XLARGE_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_trisolv_XLARGE_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_trisolv_XLARGE_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
