#include <tiramisu/tiramisu.h>

int main(int argc, char **argv)
{

    tiramisu::init("function_0");

    tiramisu::var i("i",0,100) ;
    tiramisu::var j("j",0,100) ;

    tiramisu::computation C_init("C_init", {i,j}, (i+j));

    tiramisu::computation S0 ("S0",{i,j},tiramisu::p_int32) ;
    S0.set_expression( S0(i-1,j) + S0(i,j-2)) ;
    S0.parallelize(i);
       

    S0.after(C_init,tiramisu::computation::root) ;


    tiramisu::buffer b_A("b_A", {100,100}, tiramisu::p_int32, tiramisu::a_temporary);
    tiramisu::buffer b_output("b_output", {100,100}, tiramisu::p_int32, tiramisu::a_output);

    C_init.store_in(&b_A);
    S0.store_in(&b_output);


    tiramisu::codegen({&b_output}, "build/new.o");

    return 0;
    
}