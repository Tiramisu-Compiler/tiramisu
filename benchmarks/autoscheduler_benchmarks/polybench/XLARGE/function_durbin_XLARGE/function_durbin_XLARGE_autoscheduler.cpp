#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_durbin_XLARGE_wrapper.h"


const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = get_py_interface_path();



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_durbin_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 4000), j("j", 0, 4000), k("k", 1, 4000);
    var d("d",0, 1);
    
    //inputs
    input r("r", {i}, p_float64);
    input y("y", {i}, p_float64);
    input alpha("alpha", {}, p_float64);
    input beta("beta", {}, p_float64);
    input sum("sum", {}, p_float64);

    computation y_0("y_0", {d}, -r(0));
    computation beta_0("beta_0", {d}, 1.0);
    computation alpha_0("alpha_0", {d}, -r(0));

    //Computations
    computation beta_1("beta_1", {k}, (1-alpha(0)*alpha(0))*beta(0));
    computation sum_1("sum_1", {k}, 0.0);
    computation sum_2("{sum_2[k,i]: 1<=k<4000 and 0<=i<k}", expr(), true, p_float64, global::get_implicit_function());
    sum_2.set_expression(sum(0)+r(k-i-1)*y(i));
    computation alpha_1("alpha_1", {k}, - (r(k) + sum(0))/beta(0));
    computation z("{z[k,i]: 1<=k<4000 and 0<=i<k}", expr(), true, p_float64, global::get_implicit_function());
    z.set_expression(y(i) + alpha(0)*y(k-i-1));
    computation y_1("{y_1[k,i]: 1<=k<4000 and 0<=i<k}", expr(), true, p_float64, global::get_implicit_function());
    y_1.set_expression(z(0,i));
    computation y_2("y_2", {k}, alpha(0));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    y_0.then(beta_0, d)
       .then(alpha_0, d)
       .then(beta_1, computation::root)
       .then(sum_1, k)
       .then(sum_2, k)
       .then(alpha_1, k)
       .then(z, k)
       .then(y_1, k)
       .then(y_2, k);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_r("b_r", {4000}, p_float64, a_input);    
    buffer b_y("b_y", {4000}, p_float64, a_output);    
    buffer b_z("b_z", {4000}, p_float64, a_temporary);    
    buffer b_alpha("b_alpha", {1}, p_float64, a_temporary);    
    buffer b_beta("b_beta", {1}, p_float64, a_temporary);    
    buffer b_sum("b_sum", {1}, p_float64, a_temporary);    

    //Store inputs
    y_0.store_in(&b_y, {});
    beta_0.store_in(&b_beta, {});
    alpha_0.store_in(&b_alpha, {});
    r.store_in(&b_r);  
    y.store_in(&b_y);  
    alpha.store_in(&b_alpha);  
    beta.store_in(&b_beta);
    sum.store_in(&b_sum);

    //Store computations
    beta_1.store_in(&b_beta, {});
    sum_1.store_in(&b_sum, {});
    sum_2.store_in(&b_sum, {});
    alpha_1.store_in(&b_alpha, {});
    z.store_in(&b_z, {i});
    y_1.store_in(&b_y, {i});
    y_2.store_in(&b_y, {k});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_r, &b_y}, "function_durbin_XLARGE.o", "./function_durbin_XLARGE_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_durbin_XLARGE_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_durbin_XLARGE_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
