#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_fdtd_2d_SMALL_wrapper.h"



const std::string TIRAMISU_ROOT = get_tiramisu_root_path();
const std::string py_cmd_path = get_python_bin_path();
const std::string py_interface_path = TIRAMISU_ROOT + "/tutorials/tutorial_autoscheduler/model/main.py";;



using namespace tiramisu;
int main(int argc, char **argv)

{
    tiramisu::init("function_fdtd_2d_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i_f("i_f", 0, 60), j_f("j_f", 0, 80), i_m("i_m", 0, 60-1), j_m("j_m", 0, 80-1);
    var t("t", 0, 40), i("i", 1, 60), j("j", 1, 80);
    
    //inputs
    input fict("fict", {t}, p_float64);
    input ey("ey", {i_f, j_f}, p_float64);
    input ex("ex", {i_f, j_f}, p_float64);
    input hz("hz", {i_f, j_f}, p_float64);

    //Computations
    computation ey_slice("ey_slice", {t,j_f}, fict(t));
    computation ey_out("ey_out", {t, i, j_f}, ey(i, j_f) - (hz(i, j_f) - hz(i-1, j_f))*0.5);
    computation ex_out("ex_out", {t, i_f, j}, ex(i_f, j) - (hz(i_f, j) - hz(i_f, j - 1))*0.5);
    computation hz_out("hz_out", {t, i_m, j_m}, hz(i_m, j_m) - (ex(i_m, j_m + 1) - ex(i_m, j_m) + ey(i_m + 1, j_m) - ey(i_m, j_m))*0.7);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    ey_slice.then(ey_out, t)
            .then(ex_out, t)
            .then(hz_out, t);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_fict("b_fict", {40}, p_float64, a_input);    
    buffer b_ey("b_ey", {60,80}, p_float64, a_output);    
    buffer b_ex("b_ex", {60,80}, p_float64, a_output);    
    buffer b_hz("b_hz", {60,80}, p_float64, a_output);    

    //Store inputs
    fict.store_in(&b_fict);
    ey.store_in(&b_ey);
    ex.store_in(&b_ex);
    hz.store_in(&b_hz);


    //Store computations
    ey_slice.set_access("{ey_slice[t,j_f]->b_ey[0,j_f]}");
    ey_out.store_in(&b_ey, {i,j_f});
    ex_out.store_in(&b_ex, {i_f,j});
    hz_out.store_in(&b_hz, {i_m, j_m});



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    prepare_schedules_for_legality_checks();
	perform_full_dependency_analysis();

	const int beam_size = get_beam_size();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_ex, &b_ey, &b_hz, &b_fict}, "function_fdtd_2d_SMALL.o", "./function_fdtd_2d_SMALL_wrapper");
	int explore_by_execution =  get_exploration_mode();
	if (explore_by_execution){
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, exec_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_fdtd_2d_SMALL_explored_schedules.json", true);
	}else{
		auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
		auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);
		auto_scheduler::auto_scheduler as(bs, model_eval);
		as.set_exec_evaluator(exec_eval);
		as.sample_search_space("./function_fdtd_2d_SMALL_explored_schedules.json", true);
	}
	delete scheds_gen;
	delete exec_eval;
	return 0;
}
