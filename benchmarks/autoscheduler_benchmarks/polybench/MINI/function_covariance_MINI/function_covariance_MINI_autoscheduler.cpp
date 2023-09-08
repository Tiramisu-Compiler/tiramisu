#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_covariance_MINI_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_covariance_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 28), j("j", 0, 28), k("k", 0, 32), l("l", 0, 32);
    

     //inputs
    input data("data", {l, j}, p_float64);
    input mean("mean", {j}, p_float64);
    input cov("cov", {i,j}, p_float64);


    //Computations
    
    computation mean_init("mean_init", {j}, 0.0);
    computation mean_sum("mean_sum", {j,l}, mean(j) + data(l,j));

    computation mean_div("mean_div", {j}, mean(j) /expr(cast(p_float64, 32)));

    computation data_sub("data_sub", {l,j}, data(l,j)-mean(j));

    computation cov_init("{cov_init[i,j]: 0<=i<28 and i<=j<28}", expr(0.0), true, p_float64, global::get_implicit_function());
    
    computation cov_prod("{cov_prod[i,j,k]: 0<=i<28 and i<=j<28 and 0<=k<32}", expr(), true, p_float64, global::get_implicit_function());
    cov_prod.set_expression(cov(i,j) + data(k,i)*data(k,j));

    computation cov_div("{cov_div[i,j]: 0<=i<28 and i<=j<28}", expr(0.0), true, p_float64, global::get_implicit_function());
    cov_div.set_expression(cov(i,j)/expr(cast(p_float64, 32-1)));

    computation cov_sym("{cov_sym[i,j]: 0<=i<28 and i<=j<28}", expr(0.0), true, p_float64, global::get_implicit_function());
    cov_sym.set_expression(cov(i,j));
    

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    mean_init.then(mean_sum, j)
             .then(mean_div,j)
             .then(data_sub,computation::root)
             .then(cov_init, computation::root)
             .then(cov_prod, j)
             .then(cov_div, j)
             .then(cov_sym, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_data("b_data", {32,28}, p_float64, a_input);
    buffer b_mean("b_mean", {28}, p_float64, a_temporary);
    buffer b_cov("b_cov", {28,28}, p_float64, a_output);   
    

    //Store inputs
    data.store_in(&b_data);
    mean.store_in(&b_mean);
    cov.store_in(&b_cov);
    

    //Store computations
    mean_init.store_in(&b_mean);
    mean_sum.store_in(&b_mean, {j});
    mean_div.store_in(&b_mean, {j});
    data_sub.store_in(&b_data);
    cov_init.store_in(&b_cov);
    cov_prod.store_in(&b_cov, {i,j});
    cov_div.store_in(&b_cov, {i,j});
    cov_sym.store_in(&b_cov, {j,i});


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_data, &b_cov}, "function_covariance_MINI.o", "./function_covariance_MINI_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_covariance_MINI_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_covariance_MINI_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}