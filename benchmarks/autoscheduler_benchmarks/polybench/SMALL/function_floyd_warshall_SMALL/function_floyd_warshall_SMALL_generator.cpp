#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_floyd_warshall_SMALL_wrapper.h"


using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_floyd_warshall_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 180), j("j", 0, 180), k("k", 0, 180);
    
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
    buffer b_paths("b_paths", {180,180}, p_int32, a_output);    

    //Store inputs
    paths.store_in(&b_paths);

    //Store computations
    paths_update.store_in(&b_paths, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_paths}, "function_floyd_warshall_SMALL.o");

    return 0;
}
