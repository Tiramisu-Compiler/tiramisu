#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function1050000_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function1050000");
	var i0("i0"), i1("i1"), i2("i2"), i3("i3"), i2_p0("i2_p0"), i2_p1("i2_p1");
	input icomp00("icomp00", {i1,i2_p0,i3}, p_float64);
	input input01("input01", {i2_p1}, p_float64);
	computation comp00("{comp00[i0,i1,i2,i3]: 0<=i0<64 and 1<=i1<384 and 1<=i2<769 and 0<=i3<32}",  expr(), true, p_float64, global::get_implicit_function());
	comp00.set_expression(icomp00(i1, i2, i3)*input01(i2 - 1)/input01(i2 + 1) + input01(i2));
	computation comp01("{comp01[i0,i1,i2,i3]: 0<=i0<64 and 1<=i1<384 and 1<=i2<769 and 0<=i3<32}", icomp00(i1, i2, i3) - input01(i2), true, p_float64, global::get_implicit_function());
	comp00.then(comp01, i3);
	buffer buf00("buf00", {384,769,64}, p_float64, a_output);
	buffer buf01("buf01", {770}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	comp00.store_in(&buf00, {i1,i2,i3});
	comp01.store_in(&buf00, {i1,i2,i3});
	tiramisu::codegen({&buf00,&buf01}, "function1050000.o"); 
	return 0; 
}
