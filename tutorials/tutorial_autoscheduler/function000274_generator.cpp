#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function000274_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function000274");
	var i0("i0", 1, 33), i1("i1", 1, 33), i0_p1("i0_p1", 0, 34), i1_p1("i1_p1", 0, 34), i0_p0("i0_p0", 0, 33), i1_p0("i1_p0", 0, 33);
	input input01("input01", {i0_p1,i1_p1}, p_float64);
	input icomp01("icomp01", {i0_p0,i1_p0}, p_float64);
	input input03("input03", {i0_p0,i1_p0}, p_float64);
	input input04("input04", {i0_p0,i1_p0}, p_float64);
	computation comp00("comp00", {i0,i1}, expr(0.0) - input01(i0, i1)/input01(i0, i1 - 1) + input01(i0, i1 + 1)/input01(i0 - 1, i1) - input01(i0 + 1, i1) - 5.800);
	computation comp01("comp01", {i0,i1},  p_float64);
	comp01.set_expression(icomp01(i0, i1) + input03(i0, i1)*input04(i0, i1));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {33,33}, p_float64, a_output);
	buffer buf01("buf01", {34,34}, p_float64, a_input);
	buffer buf02("buf02", {33,33}, p_float64, a_output);
	buffer buf03("buf03", {33,33}, p_float64, a_input);
	buffer buf04("buf04", {33,33}, p_float64, a_input);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	input03.store_in(&buf03);
	input04.store_in(&buf04);
	comp00.store_in(&buf00);
	comp01.store_in(&buf02);
	tiramisu::codegen({&buf00,&buf01,&buf02,&buf03,&buf04}, "function000274.o"); 
	return 0; 
}