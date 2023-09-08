#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_adi_MEDIUM_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    double DX, DY, DT;
    double B1, B2;
    double mul1, mul2;
    double a, b, c, d, e, f;

    DX = 1.0/(double)200;
    DY = 1.0/(double)200;
    DT = 1.0/(double)100;
    B1 = 2.0;
    B2 = 1.0;
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 /  2.0;
    b = 1.0+mul1;
    c = a;
    d = -mul2 / 2.0;
    e = 1.0+mul2;
    f = d;
    
    tiramisu::init("function_adi_MEDIUM");
    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 1, 200-1), j("j", 1, 200-1), t("t", 1, 100+1), j_reversed("j_reversed", -200+2, 0);
    var i_f("i_f", 0, 200), j_f("j_f", 0, 200);

    //inputs
    input u("u", {i, j}, p_float64);
    input v("v", {i, j}, p_float64);
    input p("p", {i, j}, p_float64);
    input q("q", {i, j}, p_float64);

    //Computations
    computation v_comp("v_comp", {t, i}, 1.0);
    computation p_comp("p_comp", {t, i}, 0.0);
    computation q_comp("q_comp", {t, i}, v(0,i));

    computation p_col("p_col", {t,i,j}, expr(-c) / (p(i, j-1)*a+b));
    computation q_col("q_col", {t,i,j}, (u(j, i-1)*(-d)+u(j, i)*(1.0+2.0*d) - u(j, i+1)*f-q(i, j-1)*a)/(p(i, j-1)*a+b));
    computation v_col_last("v_col_last", {t, i}, 1.0);

    computation v_col("v_col", {t,i,j_reversed}, p(i, -j_reversed) * v(1-j_reversed, i) + q(i, -j_reversed));

    computation u_comp("u_comp", {t, i}, 1.0);
    computation p_comp2("p_comp2", {t, i}, 0.0);
    computation q_comp2("q_comp2", {t, i}, u(i,0));

    computation p_row("p_row", {t,i,j}, expr(-f) / (p(i, j-1)*d+e));
    computation q_row("q_row", {t,i,j}, (v(i-1,j)*(-a)+v(i, j)*(1.0+2.0*a) - v(i+1, j)*c-q(i, j-1)*d)/(p(i, j-1)*d+e));

    computation u_row_last("u_row_last", {t,i}, 1.0);
    computation u_row("u_row", {t,i,j_reversed}, p(i, -j_reversed) * u(i, 1-j_reversed) + q(i, -j_reversed));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    v_comp.then(p_comp, 1)
     .then(q_comp, 1)
     .then(p_col, 1)
     .then(q_col, 2)
     .then(v_col_last, 1)
     .then(v_col, 1)
     .then(u_comp, 0)
     .then(p_comp2, 1)
     .then(q_comp2, 1)
     .then(p_row, 0)
     .then(q_row, 2)
     .then(u_row_last, 1)
     .then(u_row, 1);

    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_u("b_u", {200,200}, p_float64, a_output);    
    buffer b_p("b_p", {200,200}, p_float64, a_temporary);    
    buffer b_q("b_q", {200,200}, p_float64, a_temporary);    
    buffer b_v("b_v", {200,200}, p_float64, a_temporary);    
   

    //Store inputs
    u.store_in(&b_u);
    v.store_in(&b_v);
    q.store_in(&b_q);
    p.store_in(&b_p);

    //Store computations
    v_comp.store_in(&b_v, {0,i});
    p_comp.store_in(&b_p, {i,0});
    q_comp.store_in(&b_q, {i,0});
    p_col.store_in(&b_p,{i,j});
    q_col.store_in(&b_q,{i,j});
    v_col_last.store_in(&b_v, {200-1,i});
    v_col.store_in(&b_v,{-j_reversed,i});
    u_comp.store_in(&b_u, {i,0});
    p_comp2.store_in(&b_p, {i,0});
    q_comp2.store_in(&b_q, {i,0});
    p_row.store_in(&b_p,{i,j});
    q_row.store_in(&b_q,{i,j});
    u_row_last.store_in(&b_u,{i,200-1});
    u_row.store_in(&b_u,{i,-j_reversed});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_u}, "function_adi_MEDIUM.o", "./function_adi_MEDIUM_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_adi_MEDIUM_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_adi_MEDIUM_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
