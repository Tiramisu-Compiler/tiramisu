#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_floyd_warshall_LARGE_wrapper.h"


using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_floyd_warshall_LARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 2800), j("j", 0, 2800), k("k", 0, 2800);
    
    //inputs
    input paths("paths", {i, j}, p_int32);

    //Computations
    computation paths_update("paths_update", {k,i,j}, p_int32);
    paths_update.set_expression(expr(o_min, paths(i,j), paths(i,k) + paths(k,j)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    // no_schedule
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_paths("b_paths", {2800,2800}, p_int32, a_output);    

    //Store inputs
    paths.store_in(&b_paths);

    //Store computations
    paths_update.store_in(&b_paths, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_paths}, "function_floyd_warshall_LARGE.o");

    return 0;
}
