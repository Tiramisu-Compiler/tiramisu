#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_lu_SMALL_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_lu_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i"), j("j"), k("k"), l("l"), m("m");
    

    //inputs
    input A("A", {i, i}, p_float64);


    //Computations
    computation A_sub("{A_sub[i,j,k]: 0<=i<120 and 0<=j<i and 0<=k<j}", expr(), true, p_float64, global::get_implicit_function());
    A_sub.set_expression(A_sub(i,j,k) - A(i,k)*A(k,j));
    computation A_div("{A_div[i,j]: 0<=i<120 and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    A_div.set_expression(A_sub(i,j,0)/A_sub(j,j,0));
    computation A_out("{A_out[i,l,m]: 0<=i<120 and i<=l<120 and 0<=m<i}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(A_out(i,l,m) - A_div(i,m)*A_div(m,l));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_sub.then(A_div,j)
         .then(A_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {120,120}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);    

    //Store computations
    A_sub.store_in(&b_A, {i,j});
    A_div.store_in(&b_A);
    A_out.store_in(&b_A, {i,l});


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A}, "function_lu_SMALL.o", "./function_lu_SMALL_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_lu_SMALL_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_lu_SMALL_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
