#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_covariance_SMALL_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_covariance_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 80), j("j", 0, 80), k("k", 0, 100), l("l", 0, 100);
    

     //inputs
    input data("data", {l, j}, p_float64);
    input mean("mean", {j}, p_float64);
    input cov("cov", {i,j}, p_float64);


    //Computations
    
    computation mean_init("mean_init", {j}, 0.0);
    computation mean_sum("mean_sum", {j,l}, mean(j) + data(l,j));

    computation mean_div("mean_div", {j}, mean(j) /expr(cast(p_float64, 100)));

    computation data_sub("data_sub", {l,j}, data(l,j)-mean(j));

    computation cov_init("{cov_init[i,j]: 0<=i<80 and i<=j<80}", expr(0.0), true, p_float64, global::get_implicit_function());
    
    computation cov_prod("{cov_prod[i,j,k]: 0<=i<80 and i<=j<80 and 0<=k<100}", expr(), true, p_float64, global::get_implicit_function());
    cov_prod.set_expression(cov(i,j) + data(k,i)*data(k,j));

    computation cov_div("{cov_div[i,j]: 0<=i<80 and i<=j<80}", expr(0.0), true, p_float64, global::get_implicit_function());
    cov_div.set_expression(cov(i,j)/expr(cast(p_float64, 100-1)));

    computation cov_sym("{cov_sym[i,j]: 0<=i<80 and i<=j<80}", expr(0.0), true, p_float64, global::get_implicit_function());
    cov_sym.set_expression(cov(i,j));
    

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    mean_init.then(mean_sum, j)
             .then(mean_div,j)
             .then(data_sub,computation::root)
             .then(cov_init, computation::root)
             .then(cov_prod, j)
             .then(cov_div, j)
             .then(cov_sym, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_data("b_data", {100,80}, p_float64, a_input);
    buffer b_mean("b_mean", {80}, p_float64, a_temporary);
    buffer b_cov("b_cov", {80,80}, p_float64, a_output);   
    

    //Store inputs
    data.store_in(&b_data);
    mean.store_in(&b_mean);
    cov.store_in(&b_cov);
    

    //Store computations
    mean_init.store_in(&b_mean);
    mean_sum.store_in(&b_mean, {j});
    mean_div.store_in(&b_mean, {j});
    data_sub.store_in(&b_data);
    cov_init.store_in(&b_cov);
    cov_prod.store_in(&b_cov, {i,j});
    cov_div.store_in(&b_cov, {i,j});
    cov_sym.store_in(&b_cov, {j,i});


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_data, &b_cov}, "./function_covariance_SMALL.o");
    return 0;
}