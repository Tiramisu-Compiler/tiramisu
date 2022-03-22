#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function014571_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function014571");
	var i0("i0", 1, 769), i1("i1", 1, 513), i2("i2", 0, 256), i3("i3", 1, 257), i4("i4", 0, 256), i5("i5", 0, 256), i0_p0("i0_p0", 0, 769), i1_p1("i1_p1", 0, 514), i3_p1("i3_p1", 0, 258), i0_p1("i0_p1", 0, 770), i1_p0("i1_p0", 0, 513);
	input icomp00("icomp00", {i0_p0,i1_p0}, p_float64);
	input input01("input01", {i1_p0,i2}, p_float64);
	input icomp01("icomp01", {i1_p1,i3_p1}, p_float64);
	input input03("input03", {i1_p0,i4,i5}, p_float64);
	input input04("input04", {i1_p0}, p_float64);
	input icomp03("icomp03", {i0_p1,i1_p0}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression((input01(i1, i2) - 1)*icomp00(i0, i1));
	computation comp01("comp01", {i0,i1,i3},  p_float64);
	comp01.set_expression((expr(0.0) - expr(2.120)*(expr(2.632)*icomp01(i1, i3) - icomp01(i1, i3 + 1)*icomp01(i1 + 1, i3 + 1) + expr(0.162)*icomp01(i1 + 1, i3) + icomp01(i1 + 1, i3 - 1) - 0.773)*icomp01(i1 - 1, i3 + 1) - expr(5.350)*icomp00(i0, i1) + expr(5.350)*icomp01(i1, i3 - 1) + expr(5.350)*icomp01(i1 - 1, i3) - icomp01(i1 - 1, i3 - 1))/(expr(2.632)*icomp01(i1, i3) - icomp01(i1, i3 + 1)*icomp01(i1 + 1, i3 + 1) + expr(0.162)*icomp01(i1 + 1, i3) + icomp01(i1 + 1, i3 - 1) - 0.773));
	computation comp02("comp02", {i0,i1,i4,i5}, icomp00(i0, i1) - input04(i1) + input03(i1, i4, i5)/icomp00(i0, i1));
	computation comp03("comp03", {i0,i1,i4,i5},  p_float64);
	comp03.set_expression((expr(0.0) - icomp03(i0 - 1, i1)*icomp03(i0 + 1, i1) + 1)*icomp03(i0, i1)*input04(i1)/icomp03(i0 - 1, i1));
	comp00.then(comp01, i1)
		.then(comp02, i1)
		.then(comp03, i5);
	buffer buf00("buf00", {769,513}, p_float64, a_output);
	buffer buf01("buf01", {513,256}, p_float64, a_input);
	buffer buf02("buf02", {514,258}, p_float64, a_output);
	buffer buf03("buf03", {513,256,256}, p_float64, a_input);
	buffer buf04("buf04", {513}, p_float64, a_input);
	buffer buf05("buf05", {770,513}, p_float64, a_output);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	input03.store_in(&buf03);
	input04.store_in(&buf04);
	icomp03.store_in(&buf05);
	comp00.store_in(&buf00, {i0,i1});
	comp01.store_in(&buf02, {i1,i3});
	comp02.store_in(&buf00, {i0,i1});
	comp03.store_in(&buf05, {i0,i1});

	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&buf00,&buf01,&buf02,&buf03,&buf04,&buf05}, "function014571.o", "./function014571_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, exec_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function014571_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}