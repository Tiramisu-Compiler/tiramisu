#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function760051_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function760051");
	var i0("i0", 0, 256), i1("i1", 0, 256), i2("i2", 0, 256), i3("i3", 0, 256);
	input input01("input01", {i0,i1,i2}, p_float64);
	input icomp01("icomp01", {i0,i1}, p_float64);
	computation comp00("comp00", {i0,i1,i2}, input01(i0, i1, i2));
	computation comp01("comp01", {i0,i1,i3},  p_float64);
	comp01.set_expression(expr(0.202)*icomp01(i0, i1));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {256,256}, p_float64, a_output);
	buffer buf01("buf01", {256,256,256}, p_float64, a_input);
	input01.store_in(&buf01);
	icomp01.store_in(&buf00);
	comp00.store_in(&buf00, {i0,i1});
	comp01.store_in(&buf00, {i0,i1});
	tiramisu::codegen({&buf00,&buf01}, "function760051.o"); 
	return 0; 
}