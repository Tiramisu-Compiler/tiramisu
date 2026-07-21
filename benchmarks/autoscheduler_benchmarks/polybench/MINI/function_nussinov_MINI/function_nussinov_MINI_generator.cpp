#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_nussinov_MINI_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_nussinov_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 60), j("j", 0, 60), k("k");
    var i_reversed("i_reversed");

    //inputs
    input table("table", {i, j}, p_int32);
    input seq("seq", {i}, p_int32);


    //Computations
    computation table_1("{table_1[i,j]: -60+1<=i<1 and 1-i<=j<60 and 0<=j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_1.set_expression(expr(o_max, table(-i, j), table(-i, j-1)));
    computation table_2("{table_2[i,j]: -60+1<=i<1 and 1-i<=j<60 and 1-i<60}", expr(), true, p_int32, global::get_implicit_function());
    table_2.set_expression(expr(o_max, table(-i, j), table(1-i, j)));
    computation table_3("{table_3[i,j]: -60+1<=i<1 and 1-i<=j<60 and 0<=j-1 and 1-i<60 and -i<j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_3.set_expression(expr(o_max, table(-i, j), table(1-i, j-1)+cast(p_int32, ((seq(-i)+seq(j))==3))));
    computation table_4("{table_4[i,j]: -60+1<=i<1 and 1-i<=j<60 and 0<=j-1 and 1-i<60 and -i>=j-1}", expr(), true, p_int32, global::get_implicit_function());
    table_4.set_expression(expr(o_max, table(-i, j), table(1-i, j-1)));
    computation table_5("{table_5[i,j,k]: -60+1<=i<1 and 1-i<=j<60 and 1-i<=k<j}", expr(), true, p_int32, global::get_implicit_function());
    table_5.set_expression(expr(o_max, table(-i, j), table(-i, k) + table(k+1, j)));
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    table_1.then(table_2, j)
           .then(table_3, j)
           .then(table_4, j)
           .then(table_5, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_table("b_table", {60,60}, p_int32, a_output);    
    buffer b_seq("b_seq", {60}, p_int32, a_input);    

    //Store inputs
    table.store_in(&b_table);  
    seq.store_in(&b_seq);  

    //Store computations
    table_1.store_in(&b_table, {-i, j});
    table_2.store_in(&b_table, {-i, j});
    table_3.store_in(&b_table, {-i, j});
    table_4.store_in(&b_table, {-i, j});
    table_5.store_in(&b_table, {-i, j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_table, &b_seq}, "function_nussinov_MINI.o");

    return 0;
}
