#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_cholesky_LARGE_wrapper.h"



const std::string py_cmd_path = "/data/scratch/mmerouani/anaconda/envs/base-tig/bin/python";
const std::string py_interface_path = "/data/scratch/mmerouani/tiramisu3/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_cholesky_LARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 2000);

    //Iteration variables    
    var i("i"), j("j"), k("k"), l("l"), m("m");
    

    //inputs
    input A("A", {i, i}, p_float64);


    //Computations
    computation A_sub("[NN]->{A_sub[i,j,k]: 0<=i<NN and 0<=j<i and 0<=k<j}", expr(), true, p_float64, global::get_implicit_function());
    A_sub.set_expression(A_sub(i,j,k) - A(i,k)*A(j,k));
    computation A_div("[NN]->{A_div[i,j]: 0<=i<NN and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    A_div.set_expression(A_sub(i,j,0)/A_sub(j,j,0));
    computation A_diag("[NN]->{A_diag[i,m]: 0<=i<NN and 0<=m<i}", expr(), true, p_float64, global::get_implicit_function());
    A_diag.set_expression(A_diag(i,m) - A_div(i,m)*A_div(i,m));
    computation A_out("[NN]->{A_out[i]: 0<=i<NN}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(expr(o_sqrt, A_diag(i,i)));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_sub.then(A_div,j)
         .then(A_diag, i)
         .then(A_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {2000,2000}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);    

    //Store computations
    A_sub.store_in(&b_A, {i,j});
    A_div.store_in(&b_A);
    A_diag.store_in(&b_A, {i,i});
    A_out.store_in(&b_A,{i,i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A}, "function_cholesky_LARGE.o", "./function_cholesky_LARGE_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_cholesky_LARGE_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
