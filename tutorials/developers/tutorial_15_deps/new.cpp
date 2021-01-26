#include <tiramisu/tiramisu.h>
#include <isl/set.h>
#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/id.h>
#include <isl/flow.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <iostream>

int main(int argc, char **argv)
{

    tiramisu::init("function_0");

    /*
    ==============================================================

    In this tutorial we will try to highlight the new some public methods regarding :

    Data analysis & legality check :
          this includes the memory dependencies analysis with buffers
          legality of unrolling / vectorisation / parallelization
          legality of a schedule ( case of tiling,skewing, shifting, interchange )
          legality of loop fuzing ( respect of dependecies between 2 computations)
          legality of all the schedules of the function

    New optimization methods
          skewing (angle_skew) that takes 2 params and perform general transformation i' = a i + b j 
              and autocomplete j' to have Det (transformation) = 1

          loop_reversal that is helpful in enabling and correcting the schedule i' = - i
            i.e start from 100 to 1 instead of 1 to 100 

    
    
    ===============================================================
    */

    // declaring variables


    tiramisu::var i("i",0,100) ;
    tiramisu::var j("j",0,100) ;
    tiramisu::var k("k",0,10) ;
    tiramisu::var i0("i0",1,99) ;
    tiramisu::var j0("j0",1,99) ;
    tiramisu::var m("m"),l("l") ;

    tiramisu::var i1("i1");
    tiramisu::var i2("i2");
    tiramisu::var j1("j1");
    tiramisu::var j2("j2");


    // declaring computations

    

    /*
    code :
      for i :
          c_init : b_A[i] = i

      for i
          for J0
              S0: b_A[j0] = b_A[j0] + b_A[j0+1] + b_A[j0-1] 

      for i
          S1 : b_output[i] = b_A[i]+10
    
    
    */


    /*
      declaring the computations
    */
    tiramisu::computation C_init("C_init", {i}, (i));

    tiramisu::computation S0 ("S0",{i,j0},tiramisu::p_int32) ;
    S0.set_expression( C_init(j0+1)+C_init(j0-1)+C_init(j0) ) ;

    tiramisu::computation S1("S1",{i}, S0(0,i)+S0(0,i)+10 );
    

    // exec order
    S0.after(C_init,tiramisu::computation::root) ;
    S1.after(S0,tiramisu::computation::root) ;


    // declaring buffers
    tiramisu::buffer b_A("b_A", {100}, tiramisu::p_int32, tiramisu::a_temporary);
    tiramisu::buffer b_A2("b_A2", {100}, tiramisu::p_int32, tiramisu::a_temporary);
    tiramisu::buffer b_output("b_output", {100}, tiramisu::p_int32, tiramisu::a_output);

    // mapping the computations to buffers
    // for S0 the outer most variable i is free, allowing repetitive calculations 

    C_init.store_in(&b_A,{i});
    S0.store_in(&b_A,{j0});
    S1.store_in(&b_output,{i});


    // get the function object

    tiramisu::function * fct = tiramisu::global::get_implicit_function();

    // dependency analysis

    fct->performe_full_dependecy_analysis();


    S0.angle_skew(i,j0,2,1,false,i1,j1) ;

    if(S0.parallelization_is_legal(j1)){

      S0.parallelize(j1);

    }
  
    //S0.loop_reversal(j1,j2) ;
    
    // full check of legality 
    fct->check_legality_for_function() ;

    S0.applied_schedule_is_legal() ;

    S0.applied_schedule_is_legal(&S1) ;

    

     //S0.angle_skew(i,j0,1,1,false,i0,j0);
     //S0.tile(i0,j0,32,32,i1,i2,j1,j2);

    // S0.loop_reversal(j0,j1) ;

     


       // S0.after(C_init,j0) ;
   
    //S0.angle_skew(i,j,2,1,false,i0,j0) ;
    //S0.tile(i,j,4,4,i1,i2,j1,j2);
   // S0.vectorize(j0,20) ;
   /* if(S0.applied_schedule_is_legal())
    {
      std::cout<<" schid is legal ";
    }
    else
    {
      std::cout<<" schid is notlegal ";
    } */

   
   

        
    //  isl_union_map *map = S0.get_function()->compute_dep_graph() ; 

    //std::cout<<(isl_union_map_to_str(map)) ; 

    /*std::cout<<"\nfesdfchjksbdfchljubsdhjbjjj\n" ;

    std::cout<<(isl_map_to_str(S0.get_access_relation())) ;
    */

    std::cout<<"\nfesdfchjksbdfchljubsdhjbjjj\n" ;
  

   // S0.get_function()->gen_ordering_schedules() ;
   // S0.get_function()->align_schedules() ;
/*

    std::cout<<(isl_union_map_to_str(S0.get_function()->get_schedule())) ;


    S0.get_function()->gen_ordering_schedules() ;
    S0.get_function()->align_schedules() ;

   
    */

     
    //S0.get_function()->save_computation_default_schedules() ;
   // S0.get_function()->save_computations_levels() ;

    // S0.get_function()->calculate_dep_flow();

   //  S0.get_function()->get_live_out_computations_from_buffers_deps() ;

     S0.applied_schedule_is_legal() ;

    

    //S1.after_change(S0,k) ;

    //S0.get_function()->gen_ordering_schedules() ;
    //S0.get_function()->align_schedules() ;

    //S0.applied_schedule_is_legal(S1) ;


     /* if(S0.parallelization_is_legal(k)){
          std::cout<<"legal on ";
      }
      else{
          std::cout<<"legal off ";
      }*/

      /*S0.tile(i0,j0,32,32,i1,i2,j1,j2);
     if(S0.unrolling_is_legal(j2)){
          std::cout<<"legal on ";
      }
      else{
          std::cout<<"legal off ";
      }*/


    //S1.after_change(S0,j) ;
   // S1.shift(i,2);

    //S0.angle_skew(i,j,1,1,false,i0,j0) ;
   // S0.parallelize(j0) ;
   //S0.vectorize(j0,32) ;
   // S0.unroll(j0,32) ;
   // C_init.tile(i,j,32,32) ;
    //S1.interchange(i,j) ;
    

    //S0.get_function()->restore_function_to_no_optimisations() ;
    
    //S0.get_function()->restore_computations_levels() ;

    
    

    tiramisu::codegen({&b_output}, "new_1.o");


    //tiramisu::codegen({&b_output}, "build/new.o");

    return 0;
    
}