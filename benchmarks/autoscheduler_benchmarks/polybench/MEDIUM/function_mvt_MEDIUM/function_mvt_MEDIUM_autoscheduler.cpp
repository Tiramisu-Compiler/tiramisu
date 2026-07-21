#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_mvt_MEDIUM_wrapper.h"



const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)
 {
  tiramisu::init("function_mvt_MEDIUM");

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Iteration variables
  var i("i", 0, 400), j("j", 0, 400);

  // inputs
  input A("A", {i, j}, p_float64);
  input y1("y1", {j}, p_float64);
  input y2("y2", {j}, p_float64);
  input x1_inp("x1_inp", {i}, p_float64);
  input x2_inp("x2_inp", {i}, p_float64);

  // Computations
  computation x1("x1", {i, j}, p_float64);
  x1.set_expression(x1_inp(i) + A(i, j) * y1(j));
  computation x2("x2", {i, j}, p_float64);
  x2.set_expression(x2_inp(i) + A(j, i) * y2(j));


  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------
  x1.then(x2, computation::root);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------
  // Input Buffers
  buffer b_A("b_A", {400, 400}, p_float64, a_input);
  buffer b_y1("b_y1", {400}, p_float64, a_input);
  buffer b_y2("b_y2", {400}, p_float64, a_input);
  buffer b_x1("b_x1", {400}, p_float64, a_output);
  buffer b_x2("b_x2", {400}, p_float64, a_output);

  // Store inputs
  A.store_in(&b_A);
  y1.store_in(&b_y1);
  y2.store_in(&b_y2);
  x1_inp.store_in(&b_x1);
  x2_inp.store_in(&b_x2);


  // Store computations
  x1.store_in(&b_x1, {i});
  x2.store_in(&b_x2, {i});

  // -------------------------------------------------------
  // Code Generation
  // -------------------------------------------------------
  prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_y1, &b_y2, &b_x1, &b_x2}, "function_mvt_MEDIUM.o", "./function_mvt_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_mvt_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_mvt_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
