#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include <tiramisu/auto_scheduler/optimization_info.h>
#include "function_adi_MINI_wrapper.h"



const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";



using namespace tiramisu;
int main(int argc, char **argv)

{
    double DX, DY, DT;
    double B1, B2;
    double mul1, mul2;
    double a, b, c, d, e, f;

    DX = 1.0/(double)17;
    DY = 1.0/(double)17;
    DT = 1.0/(double)15;
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
    
    tiramisu::init("function_adi_MINI");
    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 1, 17-1), j("j", 1, 17-1), t("t", 1, 15+1);
    var i_f("i_f", 0, 17), j_f("j_f", 0, 17);

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

    computation v_col("v_col", {t,i,j}, p(i, j) * v(j+1, i) + q(i, j));

    computation u_comp("u_comp", {t, i}, 1.0);
    computation p_comp2("p_comp2", {t, i}, 0.0);
    computation q_comp2("q_comp2", {t, i}, u(i,0));

    computation p_row("p_row", {t,i,j}, expr(-f) / (p(i, j-1)*d+e));
    computation q_row("q_row", {t,i,j}, (v(i-1,j)*(-a)+v(i, j)*(1.0+2.0*a) - v(i+1, j)*c-q(i, j-1)*d)/(p(i, j-1)*d+e));

    computation u_row_last("u_row_last", {t,i}, 1.0);
    computation u_row("u_row", {t,i,j}, p(i, j) * u(i, j+1) + q(i, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    // list of initial transformations
    std::vector<tiramisu::auto_scheduler::optimization_info> transformations; 
    // v_col.loop_reversal(2);
    tiramisu::auto_scheduler::optimization_info optim_info_reversal_1;
    optim_info_reversal_1.type = tiramisu::auto_scheduler::optimization_type::MATRIX;
    
    std::vector <  std::vector<int> >  matrix_reversal_1(3);
    for(int l = 0; l<matrix_reversal_1.size(); l++){
        matrix_reversal_1.at(l)= std::vector<int>(3);
        for(int c = 0; c<matrix_reversal_1.size(); c++){
            if (l!=c ){
                matrix_reversal_1.at(l).at(c) = 0;
            }else{
                matrix_reversal_1.at(l).at(c) = 1;
            }
        }
    }
    optim_info_reversal_1.l0 = 2;
    matrix_reversal_1.at(2).at(2) = -1; 
    optim_info_reversal_1.comps = {&v_col};
    optim_info_reversal_1.matrix = matrix_reversal_1;
    optim_info_reversal_1.nb_l = 1;
    optim_info_reversal_1.unimodular_transformation_type = 2;
    transformations.push_back(optim_info_reversal_1);

    // u_row.loop_reversal(2);
    tiramisu::auto_scheduler::optimization_info optim_info_reversal_2;
    optim_info_reversal_2.type = tiramisu::auto_scheduler::optimization_type::MATRIX;
    
    std::vector <  std::vector<int> >  matrix_reversal_2(3);
    for(int l = 0; l<matrix_reversal_2.size(); l++){
        matrix_reversal_2.at(l)= std::vector<int>(3);
        for(int c = 0; c<matrix_reversal_2.size(); c++){
            if (l!=c ){
                matrix_reversal_2.at(l).at(c) = 0;
            }else{
                matrix_reversal_2.at(l).at(c) = 1;
            }
        }
    }
    optim_info_reversal_2.l0 = 2;
    matrix_reversal_2.at(2).at(2) = -1; 
    optim_info_reversal_2.nb_l = 1;
    optim_info_reversal_2.comps = {&u_row};
    optim_info_reversal_2.matrix = matrix_reversal_2;
    optim_info_reversal_2.unimodular_transformation_type = 2;
    transformations.push_back(optim_info_reversal_2);

    // order computations
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
    buffer b_u("b_u", {17,17}, p_float64, a_output);    
    buffer b_p("b_p", {17,17}, p_float64, a_temporary);    
    buffer b_q("b_q", {17,17}, p_float64, a_temporary);    
    buffer b_v("b_v", {17,17}, p_float64, a_temporary);    
   

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
    v_col_last.store_in(&b_v, {17-1,i});
    v_col.store_in(&b_v,{j,i});
    u_comp.store_in(&b_u, {i,0});
    p_comp2.store_in(&b_p, {i,0});
    q_comp2.store_in(&b_q, {i,0});
    p_row.store_in(&b_p,{i,j});
    q_row.store_in(&b_q,{i,j});
    u_row_last.store_in(&b_u,{i,17-1});
    u_row.store_in(&b_u,{i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_u}, "function_adi_MINI.o", "./function_adi_MINI_wrapper");
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
    auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, model_eval, scheds_gen);
    auto_scheduler::auto_scheduler as(bs, model_eval, transformations);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_adi_MINI_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
    delete model_eval;
	delete bs;
	return 0;
}
