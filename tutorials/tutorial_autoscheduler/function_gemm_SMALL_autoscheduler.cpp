#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gemm_SMALL_wrapper.h"




const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_gemm_SMALL");

    //Iteration variables    
    var i("i", 0, 60), j("j", 0, 70), k("k", 0, 80);
    

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, j}, p_float64);
    input C("C", {i, j}, p_float64);


    //Computations
    
    computation C_init("C_init", {i,j}, C(i,j)*1.2);
    computation C_out("C_out", {i,j,k}, p_float64);
    C_out.set_expression(C_out(i,j,k)+A(i,k)*B(k,j)*1.5);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    C_init.then(C_out, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {60,80}, p_float64, a_input);
    buffer b_B("b_B", {80,70}, p_float64, a_input);
    buffer b_C("b_C", {60,70}, p_float64, a_output);
    // buffer b_C_out("b_C_out", {60,70}, p_float64, a_output); 
    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    

    //Store computations
    C_init.store_in(&b_C);
    C_out.store_in(&b_C, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_B, &b_C}, "function_gemm_SMALL.o", "./function_gemm_SMALL_wrapper");
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, model_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_gemm_SMALL_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete model_eval;
	delete bs;
	return 0;
}

