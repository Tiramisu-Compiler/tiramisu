#include <tiramisu/tiramisu.h>240
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gramschmidt_MEDIUM_wrapper.h"



const std::string py_cmd_path = "/data/scratch/mmerouani/anaconda/envs/base-tig/bin/python";
const std::string py_interface_path = "/data/scratch/mmerouani/tiramisu3/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_gramschmidt_MEDIUM");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 240), MM("MM", 200);

    //Iteration variables    
    var i("i", 0, 200), j("j", 0, 240), k("k", 0, 240), l("l"), m("m");
    
    //inputs
    input A("A", {i, k}, p_float64);
    input Q("Q", {i, k}, p_float64);
    input R("R", {j, j}, p_float64);

    //Computations
    computation nrm_init("nrm_init", {k}, 0.000001);
    computation nrm("nrm", {k, i}, p_float64);
    nrm.set_expression(nrm(k,i) + A(i, k) * A(i, k));

    computation R_diag("[NN]->{R_diag[k]: 0<=k<NN}", expr(), true, p_float64, global::get_implicit_function());
    R_diag.set_expression(expr(o_sqrt, nrm(k,0)));

    computation Q_out("Q_out", {k,i}, A(i,k) / R(k,k));

    computation R_up_init("[NN]->{R_up_init[k,j]: 0<=k<NN and k+1<=j<NN}", expr(), true, p_float64, global::get_implicit_function());
    R_up_init.set_expression(0.0);

    computation R_up("[NN,MM]->{R_up[k,j,i]: 0<=k<NN and k+1<=j<NN and 0<=i<MM}", expr(), true, p_float64, global::get_implicit_function());
    R_up.set_expression(R(k,j) + Q(i,k) * A(i, j)); 

    computation A_out("[NN,MM]->{A_out[k,j,i]: 0<=k<NN and k+1<=j<NN and 0<=i<MM}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(A(i,j) - Q(i,k) * R(k, j)) ;

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    nrm_init.then(nrm, k)
            .then(R_diag, k)
            .then(Q_out, k)
            .then(R_up_init, k)
            .then(R_up, j)
            .then(A_out, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {200,240}, p_float64, a_output);
    buffer b_nrm("b_nrm", {240}, p_float64, a_temporary);
    buffer b_R("b_R", {240,240}, p_float64, a_output);
    buffer b_Q("b_Q", {200,240}, p_float64, a_output);  

    //Store inputs
    A.store_in(&b_A);    
    Q.store_in(&b_Q);    
    R.store_in(&b_R);    

    //Store computations
    nrm_init.store_in(&b_nrm);
    nrm.store_in(&b_nrm, {k});
    R_diag.store_in(&b_R, {k,k});
    Q_out.store_in(&b_Q, {i,k});
    R_up_init.store_in(&b_R);
    R_up.store_in(&b_R, {k,j});
    A_out.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_Q, &b_R}, "function_gramschmidt_MEDIUM.o", "./function_gramschmidt_MEDIUM_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_gramschmidt_MEDIUM_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
