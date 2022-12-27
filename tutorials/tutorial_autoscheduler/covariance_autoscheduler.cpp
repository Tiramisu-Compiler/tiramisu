#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "covariance_wrapper.h"


using namespace tiramisu;

const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";

int main(int argc, char **argv)

{
    tiramisu::init("covariance");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------     
    //Iteration variables    
    var i("i", 0, 28), j("j", 0, 28), k("k", 0, 32), l("l", 0, 32);
    

    //inputs
    input data("data", {l, j}, p_float64);


    //Computations
    
    computation mean_init("mean_init", {j}, 0.0);
    computation mean("mean", {l,j}, p_float64);
    mean.set_expression(mean(l,j) + data(l,j)/expr(cast(p_float64, 32)));
    
    computation cov_init("conv_init", {i,j}, 0.0);
    computation cov("cov", {i,j,k}, p_float64);
    cov.set_expression(cov(i,j,k) + (data(k,i)-mean(0,i))*(data(k,j)-mean(0,j))/expr(cast(p_float64, 32-1)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    mean_init.then(mean, computation::root)
             .then(cov_init, computation::root)
             .then(cov, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_data("b_data", {32,28}, p_float64, a_input);
    buffer b_mean("b_mean", {28}, p_float64, a_temporary);
    buffer b_cov("b_cov", {28,28}, p_float64, a_output);   
    

    //Store inputs
    data.store_in(&b_data);
    

    //Store computations
    mean_init.store_in(&b_mean);
    mean.store_in(&b_mean, {j});
    cov_init.store_in(&b_cov);
    cov.store_in(&b_cov, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_data, &b_cov}, "covariance.o", "./covariance_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, model_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./covariance_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete model_eval;
	delete bs;
	return 0;
}
