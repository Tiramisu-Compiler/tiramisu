#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_jacobi2d_XLARGE_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_jacobi2d_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i_f("i_f", 0, 2800), j_f("j_f", 0, 2800);
    var t("t", 0, 1000), i("i", 1, 2800-1), j("j", 1, 2800-1);
    
    //inputs
    input A("A", {i_f, j_f}, p_float64);
    input B("B", {i_f, j_f}, p_float64);

    //Computations
    computation B_comp("B_comp", {t,i,j}, (A(i, j) + A(i, j-1) + A(i, 1+j) + A(1+i, j) + A(i-1, j))*0.2);
    computation A_comp("A_comp", {t,i,j}, (B(i, j) + B(i, j-1) + B(i, 1+j) + B(1+i, j) + B(i-1, j))*0.2);

    //Ordering
    B_comp.then(A_comp,t);

    //Input Buffers
    buffer b_A("b_A", {2800,2800}, p_float64, a_output);    
    buffer b_B("b_B", {2800,2800}, p_float64, a_output);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);


    //Store computations
    B_comp.store_in(&b_B, {i,j});
    A_comp.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A,&b_B}, "function_jacobi2d_XLARGE.o", "./function_jacobi2d_XLARGE_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_jacobi2d_XLARGE_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_jacobi2d_XLARGE_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}