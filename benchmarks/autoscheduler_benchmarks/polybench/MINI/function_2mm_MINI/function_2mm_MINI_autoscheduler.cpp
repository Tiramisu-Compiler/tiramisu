#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_2mm_MINI_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_2mm_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
     var i("i", 0, 16), j("j", 0, 24), k("k", 0, 22), l("l", 0, 18);

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, l}, p_float64);
    input C("C", {l, j}, p_float64);
    input D("D", {i, j}, p_float64);
    input tmp("tmp", {i,l}, p_float64);
    
    //Computations
    computation tmp_init("tmp_init",{i,l}, 0.0);
    computation tmp_prod("tmp_prod",{i,l,k}, tmp(i,l) + A(i,k)*B(k,l)*1.5);

    computation D_beta("D_beta", {i,j}, D(i,j)*1.2);
    computation D_prod("D_prod", {i,j,l}, D(i,j)+tmp(i,l)*C(l,j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    tmp_init.then(tmp_prod,l)
            .then(D_beta, computation::root)
            .then(D_prod, j);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {16,22}, p_float64, a_input);
    buffer b_B("b_B", {22,18}, p_float64, a_input);
    buffer b_C("b_C", {18,24}, p_float64, a_input);
    buffer b_D("b_D", {16,24}, p_float64, a_output);
    buffer b_tmp("b_tmp", {16,18}, p_float64, a_temporary);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    D.store_in(&b_D);
    tmp.store_in(&b_tmp);

    //Store computations
    tmp_init.store_in(&b_tmp);
    tmp_prod.store_in(&b_tmp, {i,l});
    D_beta.store_in(&b_D);
    D_prod.store_in(&b_D, {i,j});
   

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_B, &b_C, &b_D}, "function_2mm_MINI.o", "./function_2mm_MINI_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_2mm_MINI_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_2mm_MINI_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}