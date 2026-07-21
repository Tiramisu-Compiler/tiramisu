#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gemver_MINI_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_gemver_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 40), j("j", 0, 40);
    

    //inputs
    input A("A", {i, j}, p_float64);
    input u1("u1", {i}, p_float64);
    input u2("u2", {i}, p_float64);
    input v1("v1", {i}, p_float64);
    input v2("v2", {i}, p_float64);
    input y("y", {i}, p_float64);
    input z("z", {i}, p_float64);
    input x_inp("x_inp", {i}, p_float64);
    input w_inp("w_inp", {i}, p_float64);


    //Computations
    
    computation A_hat("A_hat", {i,j}, A(i, j) + u1(i)*v1(j) + u2(i)*v2(j));
    computation x_temp("x_temp", {i,j}, x_inp(i) + A_hat(j, i)*y(j)*1.2);
    computation x("x", {i}, x_inp(i) + z(i));
    computation w("w", {i,j}, w_inp(i) + A_hat(i, j) * x(j)*1.5);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_hat.then(x_temp, computation::root)
         .then(x, computation::root)
         .then(w, computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {40,40}, p_float64, a_input);
    buffer b_u1("b_u1", {40}, p_float64, a_input);
    buffer b_u2("b_u2", {40}, p_float64, a_input);
    buffer b_v1("b_v1", {40}, p_float64, a_input);
    buffer b_v2("b_v2", {40}, p_float64, a_input);
    buffer b_z("b_z", {40}, p_float64, a_input);
    buffer b_y("b_y", {40}, p_float64, a_input);
    buffer b_A_hat("b_A_hat", {40,40}, p_float64, a_output);
    buffer b_x("b_x", {40}, p_float64, a_output);
    buffer b_w("b_w", {40}, p_float64, a_output);

    //Store inputs
    A.store_in(&b_A);
    u1.store_in(&b_u1);
    u2.store_in(&b_u2);
    v1.store_in(&b_v1);
    v2.store_in(&b_v2);
    y.store_in(&b_y);
    z.store_in(&b_z);
    x_inp.store_in(&b_x);
    w_inp.store_in(&b_z);

    
    //Store computations
    A_hat.store_in(&b_A_hat);
    x_temp.store_in(&b_x, {i});
    x.store_in(&b_x);
    w.store_in(&b_w, {i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_u1, &b_u2, &b_v1, &b_v2, &b_y, &b_z, &b_A_hat, &b_x, &b_w}, "function_gemver_MINI.o");

    return 0;
}
