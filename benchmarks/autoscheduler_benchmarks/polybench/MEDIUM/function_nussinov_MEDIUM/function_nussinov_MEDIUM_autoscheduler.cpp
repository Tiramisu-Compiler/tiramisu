#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_nussinov_MEDIUM_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_nussinov_MEDIUM");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 500), j("j", 0, 500), k("k");
    var i_reversed("i_reversed");

    //inputs
    input table("table", {i, j}, p_int32);
    input seq("seq", {i}, p_int32);


    //Computations
    computation table_1("{table_1[i,j]: -500+1<=i<1 and 1-i<=j<500 and 0<=j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_1.set_expression(expr(o_max, table(-i, j), table(-i, j-1)));
    computation table_2("{table_2[i,j]: -500+1<=i<1 and 1-i<=j<500 and 1-i<500}", expr(), true, p_int32, global::get_implicit_function());
    table_2.set_expression(expr(o_max, table(-i, j), table(1-i, j)));
    computation table_3("{table_3[i,j]: -500+1<=i<1 and 1-i<=j<500 and 0<=j-1 and 1-i<500 and -i<j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_3.set_expression(expr(o_max, table(-i, j), table(1-i, j-1)+cast(p_int32, ((seq(-i)+seq(j))==3))));
    computation table_4("{table_4[i,j]: -500+1<=i<1 and 1-i<=j<500 and 0<=j-1 and 1-i<500 and -i>=j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_4.set_expression(expr(o_max, table(-i, j), table(1-i, j-1)));
    computation table_5("{table_5[i,j,k]: -500+1<=i<1 and 1-i<=j<500 and 1-i<=k<j}", expr(), true, p_int32, global::get_implicit_function());
    table_5.set_expression(expr(o_max, table(-i, j), table(-i, k) + table(k+1, j)));
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    table_1.then(table_2, j)
           .then(table_3, j)
           .then(table_4, j)
           .then(table_5, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_table("b_table", {500,500}, p_int32, a_output);    
    buffer b_seq("b_seq", {500}, p_int32, a_input);    

    //Store inputs
    table.store_in(&b_table);  
    seq.store_in(&b_seq);  

    //Store computations
    table_1.store_in(&b_table, {-i, j});
    table_2.store_in(&b_table, {-i, j});
    table_3.store_in(&b_table, {-i, j});
    table_4.store_in(&b_table, {-i, j});
    table_5.store_in(&b_table, {-i, j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_table, &b_seq}, "function_nussinov_MEDIUM.o", "./function_nussinov_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_nussinov_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_nussinov_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
