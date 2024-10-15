#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_atax_MEDIUM_wrapper.h"



const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)
 {
        tiramisu::init("function_atax_MEDIUM");

        // -------------------------------------------------------
        // Layer I
        // -------------------------------------------------------

        // Iteration variables
        var i("i", 0, 390), j("j", 0, 410);

        // inputs
        input A("A", {i, j}, p_float64);
        input x("x", {j}, p_float64);

        // Computations
        computation Ax_init("Ax_init", {i}, 0.0);
        computation Ax("Ax", {i, j}, p_float64);
        Ax.set_expression(Ax(i, j) + A(i, j) * x(j));
        computation y_init("y_init", {j}, 0.0);
        computation y("y", {i, j}, p_float64);
        y.set_expression(y(i, j) + A(i, j) * Ax(i, 0));

        // -------------------------------------------------------
        // Layer II
        // -------------------------------------------------------
        y_init.then(Ax_init, computation::root)
         .then(Ax, i)
         .then(y, i);

        // -------------------------------------------------------
        // Layer III
        // -------------------------------------------------------
        // Input Buffers
        buffer b_A("b_A", {390, 410}, p_float64, a_input);
        buffer b_Ax("b_Ax", {390}, p_float64, a_temporary);
        buffer b_x("b_x", {410}, p_float64, a_input);
        buffer b_y("b_y", {410}, p_float64, a_output);

        // Store inputs
        A.store_in(&b_A);
        x.store_in(&b_x);

        // Store computations
        Ax_init.store_in(&b_Ax);
        Ax.store_in(&b_Ax, {i});
        y_init.store_in(&b_y);
        y.store_in(&b_y, {j});

        // -------------------------------------------------------
        // Code Generation
        // -------------------------------------------------------
        prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_x, &b_y}, "function_atax_MEDIUM.o", "./function_atax_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_atax_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_atax_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
