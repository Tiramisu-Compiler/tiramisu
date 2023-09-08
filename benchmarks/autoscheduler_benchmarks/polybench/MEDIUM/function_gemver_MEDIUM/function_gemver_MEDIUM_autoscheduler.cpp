#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gemver_MEDIUM_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_gemver_MEDIUM");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 400), j("j", 0, 400);
    

    //inputs
    input A("A", {i, j}, p_float64);
    input u1("u1", {i}, p_float64);
    input u2("u2", {i}, p_float64);
    input v1("v1", {i}, p_float64);
    input v2("v2", {i}, p_float64);
    input y("y", {i}, p_float64);
    input z("z", {i}, p_float64);


    //Computations
    
    computation A_hat("A_hat", {i,j}, A(i, j) + u1(i)*v1(j) + u2(i)*v2(j));
    computation x_temp("x_temp", {i,j}, p_float64);
    x_temp.set_expression(x_temp(i,j) + A_hat(j, i)*y(j)*1.2);
    computation x("x", {i}, x_temp(i, 0) + z(i));
    computation w("w", {i,j}, p_float64);
    w.set_expression(w(i,j) + A_hat(i, j) * x(j)*1.5);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_hat.then(x_temp, computation::root)
         .then(x, computation::root)
         .then(w, computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {400,400}, p_float64, a_input);
    buffer b_u1("b_u1", {400}, p_float64, a_input);
    buffer b_u2("b_u2", {400}, p_float64, a_input);
    buffer b_v1("b_v1", {400}, p_float64, a_input);
    buffer b_v2("b_v2", {400}, p_float64, a_input);
    buffer b_z("b_z", {400}, p_float64, a_input);
    buffer b_y("b_y", {400}, p_float64, a_input);
    buffer b_A_hat("b_A_hat", {400,400}, p_float64, a_output);
    buffer b_x("b_x", {400}, p_float64, a_output);
    buffer b_w("b_w", {400}, p_float64, a_output);

    //Store inputs
    A.store_in(&b_A);
    u1.store_in(&b_u1);
    u2.store_in(&b_u2);
    v1.store_in(&b_v1);
    v2.store_in(&b_v2);
    y.store_in(&b_y);
    z.store_in(&b_z);
    
    //Store computations
    A_hat.store_in(&b_A_hat);
    x_temp.store_in(&b_x, {i});
    x.store_in(&b_x);
    w.store_in(&b_w, {i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_u1, &b_u2, &b_v1, &b_v2, &b_y, &b_z, &b_A_hat, &b_x, &b_w}, "function_gemver_MEDIUM.o", "./function_gemver_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gemver_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gemver_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
