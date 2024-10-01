#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_seidel2d_SMALL_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_seidel2d_SMALL");
   
    var i_f("i_f", 0, 120), j_f("j_f", 0, 120);
    var t("t", 0, 40), i("i", 1, 120-1), j("j", 1, 120-1);

    input A("A", {i_f, j_f}, p_float64);

    computation comp_A_out("comp_A_out", {t,i,j}, (A(i-1, j-1) + A(i-1, j) + A(i-1, j+1)
		                               + A(i, j-1) + A(i, j) + A(i, j+1)
		                               + A(i+1, j-1) + A(i+1, j) + A(i+1, j+1))/9.0);

    buffer b_A("b_A", {120,120}, p_float64, a_output);    

    A.store_in(&b_A);
    comp_A_out.store_in(&b_A, {i,j});

    tiramisu::codegen({&b_A}, "function_seidel2d_SMALL.o");

    return 0;
}