#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_2mm_SMALL_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_2mm_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
     var i("i", 0, 40), j("j", 0, 80), k("k", 0, 70), l("l", 0, 50);

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, l}, p_float64);
    input C("C", {l, j}, p_float64);
    input D("D", {i, j}, p_float64);
    input tmp("tmp", {i,l}, p_float64);
    
    //Computations
    computation tmp_init("tmp_init",{i,l}, 0.0);
    computation tmp_prod("tmp_prod",{i,l,k}, tmp(i,l) + A(i,k)*B(k,l)*1.5);

    computation D_beta("D_beta", {i,j}, D(i,j)*1.2);
    computation D_prod("D_prod", {i,j,l}, D(i,j)+tmp(i,l)*C(l,j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    tmp_init.then(tmp_prod,l)
            .then(D_beta, computation::root)
            .then(D_prod, j);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {40,70}, p_float64, a_input);
    buffer b_B("b_B", {70,50}, p_float64, a_input);
    buffer b_C("b_C", {50,80}, p_float64, a_input);
    buffer b_D("b_D", {40,80}, p_float64, a_output);
    buffer b_tmp("b_tmp", {40,50}, p_float64, a_temporary);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    D.store_in(&b_D);
    tmp.store_in(&b_tmp);

    //Store computations
    tmp_init.store_in(&b_tmp);
    tmp_prod.store_in(&b_tmp, {i,l});
    D_beta.store_in(&b_D);
    D_prod.store_in(&b_D, {i,j});
   

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B, &b_C, &b_D}, "function_2mm_SMALL.o");
    
    return 0;
}