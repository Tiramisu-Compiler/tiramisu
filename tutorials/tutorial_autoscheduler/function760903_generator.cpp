#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function760903_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function760903");
	var i0("i0", 0, 64), i1("i1", 0, 64), i2("i2", 0, 320), i3("i3", 0, 320);
	input icomp00("icomp00", {i0,i1}, p_float64);
	input input01("input01", {i2}, p_float64);
	input input03("input03", {i0,i1,i2}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression((1 - expr(1.260)*input01(i2))*icomp00(i0, i1));
	computation comp01("comp01", {i0,i1,i2}, icomp00(i0, i1) + input03(i0, i1, i2));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {64,64}, p_float64, a_output);
	buffer buf01("buf01", {320}, p_float64, a_input);
	buffer buf02("buf02", {64,64,320}, p_float64, a_output);
	buffer buf03("buf03", {64,64,320}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	input03.store_in(&buf03);
	comp00.store_in(&buf00, {i0,i1});
	comp01.store_in(&buf02);
	tiramisu::codegen({&buf00,&buf01,&buf02,&buf03}, "function760903.o"); 
	return 0; 
}