#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function760413_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function760413");
	var i0("i0", 0, 256), i1("i1", 1, 257), i2("i2", 0, 512), i3("i3", 0, 512), i4("i4", 1, 17), i1_p0("i1_p0", 0, 257), i1_p1("i1_p1", 0, 258), i4_p1("i4_p1", 0, 18);
	input icomp00("icomp00", {i2,i3}, p_float64);
	input input01("input01", {i1_p0,i3}, p_float64);
	input icomp01("icomp01", {i1_p1,i4_p1}, p_float64);
	computation comp00("comp00", {i0,i1,i2,i3},  p_float64);
	comp00.set_expression(expr(4.580)*icomp00(i2, i3)*icomp00(i2, i3) + input01(i1, i3));
	computation comp01("comp01", {i0,i1,i4},  p_float64);
	comp01.set_expression(expr(0.0) - expr(4.420)*icomp01(i1, i4)*icomp01(i1 + 1, i4) + icomp01(i1, i4) + expr(5.240)*icomp01(i1, i4 + 1) + icomp01(i1 - 1, i4 + 1) + expr(5.397)*icomp01(i1 + 1, i4 - 1) - icomp01(i1 - 1, i4)*icomp01(i1 - 1, i4 - 1)*icomp01(i1 + 1, i4 + 1)/icomp01(i1, i4 - 1));
	comp00.then(comp01, i1);
	buffer buf00("buf00", {512,512}, p_float64, a_output);
	buffer buf01("buf01", {257,512}, p_float64, a_input);
	buffer buf02("buf02", {258,18}, p_float64, a_output);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	comp00.store_in(&buf00, {i2,i3});
	comp01.store_in(&buf02, {i1,i4});
	tiramisu::codegen({&buf00,&buf01,&buf02}, "function760413.o"); 
	return 0; 
}