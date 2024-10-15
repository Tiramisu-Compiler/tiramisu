#include <tiramisu/tiramisu.h>80
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gramschmidt_SMALL_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_gramschmidt_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 60), j("j", 0, 80), k("k", 0, 80), l("l"), m("m");
    
    //inputs
    input A("A", {i, k}, p_float64);
    input Q("Q", {i, k}, p_float64);
    input R("R", {j, j}, p_float64);
    input nrm("nrm", {}, p_float64);
    
    //Computations
    computation nrm_init("{nrm_init[k]: 0<=k<80}", expr(), true, p_float64, global::get_implicit_function());
    nrm_init.set_expression(0);

    computation nrm_comp("{nrm_comp[k,i]: 0<=k<80 and 0<=i<60}", expr(), true, p_float64, global::get_implicit_function());
    nrm_comp.set_expression(nrm(0) + A(i, k) * A(i, k));

    computation R_diag("{R_diag[k]: 0<=k<80}", expr(), true, p_float64, global::get_implicit_function());
    R_diag.set_expression(expr(o_sqrt, nrm(0)));

    computation Q_out("{Q_out[k,i]: 0<=k<80 and 0<=i<60}", expr(), true, p_float64, global::get_implicit_function());
    Q_out.set_expression(A(i,k) / R(k,k));

    computation R_up_init("{R_up_init[k,j]: 0<=k<80 and k+1<=j<80}", expr(), true, p_float64, global::get_implicit_function());
    R_up_init.set_expression(0.0);

    computation R_up("{R_up[k,j,i]: 0<=k<80 and k+1<=j<80 and 0<=i<60}", expr(), true, p_float64, global::get_implicit_function());
    R_up.set_expression(R(k,j) + Q(i,k) * A(i, j)); 

    computation A_out("{A_out[k,j,i]: 0<=k<80 and k+1<=j<80 and 0<=i<60}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(A(i,j) - Q(i,k) * R(k, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    nrm_init.then(nrm_comp, k)
            .then(R_diag, k)
            .then(Q_out, k)
            .then(R_up_init, k)
            .then(R_up, j)
            .then(A_out, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {60,80}, p_float64, a_output);
    buffer b_nrm("b_nrm", {1}, p_float64, a_temporary);
    buffer b_R("b_R", {80,80}, p_float64, a_output);
    buffer b_Q("b_Q", {60,80}, p_float64, a_output);  

    //Store inputs
    A.store_in(&b_A);    
    Q.store_in(&b_Q);    
    R.store_in(&b_R);    
    nrm.store_in(&b_nrm);
    
    //Store computations
    nrm_init.store_in(&b_nrm, {});
    nrm_comp.store_in(&b_nrm, {});
    R_diag.store_in(&b_R, {k,k});
    Q_out.store_in(&b_Q, {i,k});
    R_up_init.store_in(&b_R);
    R_up.store_in(&b_R, {k,j});
    A_out.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_Q, &b_R}, "function_gramschmidt_SMALL.o", "./function_gramschmidt_SMALL_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gramschmidt_SMALL_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_gramschmidt_SMALL_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
