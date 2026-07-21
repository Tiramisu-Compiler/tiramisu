#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_bicg_MEDIUM_wrapper.h"



const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)
 {
  tiramisu::init("function_bicg_MEDIUM");

  // -------------------------------------------------------
  // Layer I
  // -------------------------------------------------------

  // Iteration variables
  var i("i", 0, 410), j("j", 0, 390);

  // inputs
  input A("A", {i, j}, p_float64);
  input p("p", {i}, p_float64);
  input r("r", {j}, p_float64);

  // Computations
  computation q_init("q_init", {i}, 0.0);
  computation q("q", {i, j}, p_float64);
  q.set_expression(q(i, j) + A(i, j) * p(j));
  computation s_init("s_init", {j}, 0.0);
  computation s("s", {i, j}, p_float64);
  s.set_expression(s(i, j) + A(i, j) * r(i));

  // -------------------------------------------------------
  // Layer II
  // -------------------------------------------------------
  s_init.then(q_init, computation::root)
        .then(s, i)
        .then(q, j);

  // -------------------------------------------------------
  // Layer III
  // -------------------------------------------------------
  // Input Buffers
  buffer b_A("b_A", {410, 390}, p_float64, a_input);
  buffer b_p("b_p", {390}, p_float64, a_input);
  buffer b_r("b_r", {410}, p_float64, a_input);
  buffer b_q("b_q", {410}, p_float64, a_output);
  buffer b_s("b_s", {390}, p_float64, a_output);

  // Store inputs
  A.store_in(&b_A);
  p.store_in(&b_p);
  r.store_in(&b_r);
  q.store_in(&b_q);
  s.store_in(&b_s);

  // Store computations
  q_init.store_in(&b_q);
  q.store_in(&b_q, {i});
  s_init.store_in(&b_s);
  s.store_in(&b_s, {j});

  // -------------------------------------------------------
  // Code Generation
  // -------------------------------------------------------
  prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_p, &b_r, &b_q, &b_s}, "function_bicg_MEDIUM.o", "./function_bicg_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_bicg_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_bicg_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
