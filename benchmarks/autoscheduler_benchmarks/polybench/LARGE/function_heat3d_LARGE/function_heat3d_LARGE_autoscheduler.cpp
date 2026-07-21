#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_heat3d_LARGE_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_heat3d_LARGE");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------     
    //Iteration variables    
    var i_f("i_f", 0, 120), j_f("j_f", 0, 120), k_f("k_f", 0, 120);
    var t("t", 1, 500+1), i("i", 1, 120-1), j("j", 1, 120-1), k("k", 1, 120-1);
    
    //inputs
    input A("A", {i_f, j_f, k_f}, p_float64);
    input B("B", {i_f, j_f, k_f}, p_float64);

    //Computations
    computation B_out("B_out", {t,i,j,k}, (A(i+1, j, k) - A(i, j, k)*2.0 + A(i-1, j, k))*0.125
                                        + (A(i, j+1, k) - A(i, j, k)*2.0 + A(i, j-1, k))*0.125
                                        + (A(i, j, k+1) - A(i, j, k)*2.0 + A(i, j, k-1))*0.125
                                        + A(i, j, k));

    computation A_out("A_out", {t,i,j,k}, (B(i+1, j, k) - B(i, j, k)*2.0 + B(i-1, j, k))*0.125
                                        + (B(i, j+1, k) - B(i, j, k)*2.0 + B(i, j-1, k))*0.125
                                        + (B(i, j, k+1) - B(i, j, k)*2.0 + B(i, j, k-1))*0.125
                                        + B(i, j, k));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    B_out.then(A_out, t);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {120,120,120}, p_float64, a_output);    
    buffer b_B("b_B", {120,120,120}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);

    //Store computations
    A_out.store_in(&b_A, {i,j,k});
    B_out.store_in(&b_B, {i,j,k});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A,&b_B}, "function_heat3d_LARGE.o", "./function_heat3d_LARGE_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_heat3d_LARGE_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_heat3d_LARGE_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}