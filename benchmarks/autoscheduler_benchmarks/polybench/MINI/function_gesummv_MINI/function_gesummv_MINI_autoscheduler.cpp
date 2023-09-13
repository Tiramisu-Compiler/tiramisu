#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gesummv_MINI_wrapper.h"



const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_gesummv_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 30), j("j", 0, 30);
    

    //inputs
    input A("A", {i, j}, p_float64);
    input B("B", {i, j}, p_float64);
    input x("x", {i}, p_float64);
    input y("y", {i}, p_float64);
    input tmp("tmp", {i}, p_float64);

    //Computations
    computation tmp_init("tmp_init", {i}, 0.0);
    computation y_init("y_init", {i}, 0.0);
    computation tmp_comp("tmp_comp", {i,j}, tmp(i)+A(i,j)*x(j));
    computation y_comp1("y_comp1", {i,j}, y(i)+B(i,j)*x(j));
    computation y_comp2("y_comp2", {i}, tmp(i)*1.5 + y(i)*1.2);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    tmp_init.then(y_init, i)
            .then(tmp_comp,i)
            .then(y_comp1,{j})
            .then(y_comp2,i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_tmp("b_tmp", {30}, p_float64, a_temporary);
    buffer b_A("b_A", {30,30}, p_float64, a_input);
    buffer b_B("b_B", {30,30}, p_float64, a_input);
    buffer b_x("b_x", {30}, p_float64, a_input);
    buffer b_y("b_y", {30}, p_float64, a_output);     

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    x.store_in(&b_x);
    y.store_in(&b_y);
    tmp.store_in(&b_tmp);
    

    //Store computations
    tmp_init.store_in(&b_tmp);
    tmp_comp.store_in(&b_tmp,{i});
    y_init.store_in(&b_y);
    y_comp1.store_in(&b_y,{i});
    y_comp2.store_in(&b_y);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_B, &b_x, &b_y}, "function_gesummv_MINI.o", "./function_gesummv_MINI_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gesummv_MINI_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gesummv_MINI_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}

