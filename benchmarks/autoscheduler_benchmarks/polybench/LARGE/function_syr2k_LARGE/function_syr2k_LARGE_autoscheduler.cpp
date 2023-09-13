#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_syr2k_LARGE_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_syr2k_LARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
//     var i("i", 0, 1200), j("j", 0, 1200), k("k", 0, 1000);
    var i("i", 0, 1200), j("j"), k("k", 0, 1000);
    

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {i, k}, p_float64);
    input C("C", {i, j}, p_float64);


    //Computations
    computation C_beta("{C_beta[i,j]: 0<=i<1200 and 0<=j<=i}", expr(), true, p_float64, global::get_implicit_function());
    C_beta.set_expression(C(i,j)*1.2);
    computation C_out("{C_out[i,k,j]: 0<=i<1200 and 0<=j<=i and 0<=k<1000}", expr(), true, p_float64, global::get_implicit_function());
    C_out.set_expression(C(i,j)+ A(j, k)*B(i, k)*1.5 + B(j, k)*A(i, k)*1.5);

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    C_beta.then(C_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {1200,1000}, p_float64, a_input);
    buffer b_B("b_B", {1200,1000}, p_float64, a_input);
    buffer b_C("b_C", {1200,1200}, p_float64, a_output);
    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    

    //Store computations
    C_beta.store_in(&b_C);
    C_out.store_in(&b_C, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_B, &b_C}, "function_syr2k_LARGE.o", "./function_syr2k_LARGE_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_syr2k_LARGE_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_syr2k_LARGE_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}

