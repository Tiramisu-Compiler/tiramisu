#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_correlation_SMALL_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_correlation_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 80-1), j("j", 0, 80), j_up("j_up"), l("l", 0, 100), k("k");
    

    //inputs
    input data("data", {l, j}, p_float64);
    input mean("mean", {j}, p_float64);
    input stddev("stddev", {j}, p_float64);
    input corr("corr", {j, j}, p_float64);


    //Computations
    
    computation mean_init("mean_init", {j}, 0.0);
    computation mean_sum("mean_sum", {j,l}, mean(j) + data(l,j));
    computation mean_div("mean_div", {j}, mean(j) /expr(cast(p_float64, 100)));

    computation stddev_init("stddev_init", {j}, 0.0);
    computation stddev_prod("stddev_prod", {j,l}, stddev(j) + (data(l,j)-mean(j))*(data(l,j)-mean(j)));
    computation stddev_div("stddev_div", {j}, stddev(j)/expr(cast(p_float64, 100)));
    computation stddev_sqrt("stddev_sqrt", {j}, expr(o_sqrt,stddev(j)));
    computation stddev_tern("stddev_tern", {j}, (stddev(j)<=0.1)*1.0+(1-(stddev(j)<=0.1))*stddev(j));

    computation data_sub("data_sub", {l,j}, data(l,j)-mean(j));
    computation data_div("data_div",{l,j},data(l,j)/(expr(o_sqrt,expr(cast(p_float64, 100)))*stddev(j)));
    computation corr_init_diag("{corr_init_diag[i]: 0<=i<80-1}", 1.0, true, p_float64, global::get_implicit_function());

    computation corr_init_upper("{corr_init_upper[i,j_up]: 0<=i<80-1 and i+1<=j_up<80}", 0.0, true, p_float64, global::get_implicit_function());
    computation corr_prod("{corr_prod[i,j_up,k]: 0<=i<80-1 and i+1<=j_up<80 and 0<=k<100}", corr(i,j_up)+data(k,i)*data(k,j_up),true, p_float64,global::get_implicit_function());
    computation corr_transpose("{corr_transpose[i, j_up]: 0<=i<80-1 and i+1<=j_up<80}", corr(i,j_up),true, p_float64, global::get_implicit_function());

    computation corr_last("corr_last", {}, 1.0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    mean_init.then(mean_sum, j)
            .then(mean_div,j)
            .then(stddev_init, computation::root)
            .then(stddev_prod, j)
            .then(stddev_div, j)
            .then(stddev_sqrt, j)
            .then(stddev_tern, j)
            .then(data_sub, computation::root)
            .then(data_div, j)
            .then(corr_init_diag, computation::root)
            .then(corr_init_upper, i)
            .then(corr_prod, j_up)
            .then(corr_transpose, j_up)
            .then(corr_last, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_data("b_data", {100,80}, p_float64, a_input);
    buffer b_mean("b_mean", {80}, p_float64, a_temporary);
    buffer b_std("b_std", {80}, p_float64, a_temporary); 
    buffer b_corr("b_corr", {80,80}, p_float64, a_output);   
    

    //Store inputs
    data.store_in(&b_data);
    mean.store_in(&b_mean);
    stddev.store_in(&b_std);
    corr.store_in(&b_corr);
    

    //Store computations
    data.store_in(&b_data);
    mean.store_in(&b_mean);
    stddev.store_in(&b_std);
    corr.store_in(&b_corr);


    //Store computations
    mean_init.store_in(&b_mean);
    mean_sum.store_in(&b_mean, {j});
    mean_div.store_in(&b_mean, {j});

    stddev_init.store_in(&b_std);
    stddev_prod.store_in(&b_std, {j});
    stddev_div.store_in(&b_std, {j});
    stddev_sqrt.store_in(&b_std, {j});
    stddev_tern.store_in(&b_std, {j});

    data_sub.store_in(&b_data);
    data_div.store_in(&b_data);

    corr_init_diag.store_in(&b_corr, {i,i});
    corr_init_upper.store_in(&b_corr, {i,j_up});
    corr_prod.store_in(&b_corr, {i,j_up});
    corr_transpose.store_in(&b_corr, {j_up,i});

    corr_last.store_in(&b_corr,{80-1,80-1});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_data, &b_corr}, "function_correlation_SMALL.o", "./function_correlation_SMALL_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_correlation_SMALL_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_correlation_SMALL_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
