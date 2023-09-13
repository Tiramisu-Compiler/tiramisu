#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gesummv_SMALL_wrapper.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_gesummv_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 90), j("j", 0, 90);
    

    //inputs
    input A("A", {i, j}, p_float64);
    input B("B", {i, j}, p_float64);
    input x("x", {i}, p_float64);
    input y("y", {i}, p_float64);
    input tmp("tmp", {i}, p_float64);

    //Computations
    computation tmp_init("tmp_init", {i}, 0.0);
    computation y_init("y_init", {i}, 0.0);
    computation tmp_comp("tmp_comp", {i,j}, tmp(i)+A(i,j)*x(j));
    computation y_comp1("y_comp1", {i,j}, y(i)+B(i,j)*x(j));
    computation y_comp2("y_comp2", {i}, tmp(i)*1.5 + y(i)*1.2);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    tmp_init.then(y_init, i)
            .then(tmp_comp,i)
            .then(y_comp1,{j})
            .then(y_comp2,i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_tmp("b_tmp", {90}, p_float64, a_temporary);
    buffer b_A("b_A", {90,90}, p_float64, a_input);
    buffer b_B("b_B", {90,90}, p_float64, a_input);
    buffer b_x("b_x", {90}, p_float64, a_input);
    buffer b_y("b_y", {90}, p_float64, a_output);     

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    x.store_in(&b_x);
    y.store_in(&b_y);
    tmp.store_in(&b_tmp);
    

    //Store computations
    tmp_init.store_in(&b_tmp);
    tmp_comp.store_in(&b_tmp,{i});
    y_init.store_in(&b_y);
    y_comp1.store_in(&b_y,{i});
    y_comp2.store_in(&b_y);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B, &b_x, &b_y}, "function_gesummv_SMALL.o");

    return 0;
}

