#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function000299_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000299");
	var i0("i0", 1, 65), i1("i1", 1, 129), i2("i2", 0, 192), i3("i3", 0, 192), i1_p0("i1_p0", 0, 129), i0_p0("i0_p0", 0, 65), i0_p1("i0_p1", 0, 66), i1_p1("i1_p1", 0, 130);
	input input01("input01", {i0_p0,i1_p0,i2}, p_float64);
	input input02("input02", {i0_p0,i2}, p_float64);
	input input03("input03", {i0_p0,i2}, p_float64);
	input icomp01("icomp01", {i0_p1,i1_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2}, input01(i0, i1, i2) + input02(i0, i2) - input03(i0, i2));
	computation comp01("comp01", {i0,i1,i3},  p_float64);
	comp01.set_expression((icomp01(i0, i1) + icomp01(i0 + 1, i1))*icomp01(i0 - 1, i1) + icomp01(i0, i1 - 1)*icomp01(i0, i1 + 1) + 3.340);
	comp00.then(comp01, i1);
	buffer buf00("buf00", {65,129,192}, p_float64, a_output);
	buffer buf01("buf01", {65,129,192}, p_float64, a_input);
	buffer buf02("buf02", {65,192}, p_float64, a_input);
	buffer buf03("buf03", {65,192}, p_float64, a_input);
	buffer buf04("buf04", {66,130}, p_float64, a_output);
	input01.store_in(&buf01);
	input02.store_in(&buf02);
	input03.store_in(&buf03);
	icomp01.store_in(&buf04);
	comp00.store_in(&buf00);
	comp01.store_in(&buf04, {i0,i1});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03,&buf04}, "function000299.o", "./function000299_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function000299_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}