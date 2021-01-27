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
    tiramisu::buffer b_output("b_output", {100}, tiramisu::p_int32, tiramisu::a_output);

    // mapping the computations to buffers
    // for S0 the outer most variable i is free, allowing repetitive calculations 

    C_init.store_in(&b_A,{i});
    S0.store_in(&b_A,{j0});
    S1.store_in(&b_output,{i});


    // get the function object

    tiramisu::function * fct = tiramisu::global::get_implicit_function();

   /*
   
    performe a full dependecy analysis RAW/WAR/WAW and the result is stored in attributes inside the function
    to invoke this method user must : define computations order & define the buffer mapped to each computation

   */

    fct->performe_full_dependecy_analysis();


/*
  the list of legality checks are methods that return a boolean :
        True if it's legal & false if not

  [they must be invoked after performe_full_dependecy_analysis() since they use these results ]
*/

    // full check of legality for this function

    fct->check_legality_for_function() ;

  // legality check for reflexive dependencies S0 -> S0
    S0.applied_schedule_is_legal() ;

  // legality check Previous_Comp -> Next_Comp  

    S0.applied_schedule_is_legal(&S1) ;

   
   /*
      also for high level optimizations there methods that checks the legality ()
   */

  // only check for reflexive deps
  S0.parallelization_is_legal(j0) ;

  S0.vectorization_is_legal(j0) ;

  S0.unrolling_is_legal(j0) ;

  /*
    in case of fuzed computations , using this method and including the list of computations fuzed together helps determine
    if that loop level is indeed parallel or not by checking all the dependencies inside the loop 
    
  */

  //C_init.parallelization_is_legal(j0,{&S0,&S1}) ;

    /*
      warning : 
      if tiling is called then user cannot use legality checks untill he calls 
        1-gen_ordering_schedules
        2-align_schedules
        these are function methods
    
    */


   /*
      We also introduce live_out_access i.e last computations to write to thier buffers ,

      Remark : sometime not all the iteration domain of a computation is a live_out access (maybe only the last iteration for example)
      to get this kind of information check the live_out_access private attribute (isl_union_map) that contain detailed information 
   */

  std::vector<tiramisu::computation * > live_out = fct->get_live_out_computations_from_buffers_deps() ;



  /*
  
      For new optimizations introduced he have general skewing
      S0.angle_skew 
  */

    //S0.angle_skew(i,j0,2,1,false,i1,j1) ;

    /*
      Loop reversal : changes iterations direction 0->100 to 100->0
      
    */
   //S0.loop_reversal(j,j2) ;

    

    

    tiramisu::codegen({&b_output}, "new_1.o");



    return 0;
    
}