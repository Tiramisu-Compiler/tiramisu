

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

    
    tiramisu::var i("i",0,100) ;
    tiramisu::var j("j",0,100) ;
    tiramisu::var k("k",0,10) ;
    tiramisu::var i0("i0",0,100) ;
    tiramisu::var j0("j0",0,100) ;
    tiramisu::var m("m"),l("l") ;

    tiramisu::var i1("i1");
    tiramisu::var i2("i2");
    tiramisu::var j1("j1");
    tiramisu::var j2("j2");

    tiramisu::computation C_init("C_init", {i,j,k}, (i+j));

    tiramisu::computation S0 ("S0",{i,j,k},tiramisu::p_int32) ;
    S0.set_expression( S0(i-1,j+1,k) + S0(i,j-1,k)+C_init(i,j,k) ) ;

    //tiramisu::computation S0("[M,b0,b1]->{c_x[j]: b0<=j<b1}", tiramisu::expr(), false, tiramisu::p_uint8 ,C_init.get_function()) ;

    tiramisu::computation S1("S1",{i,j,k}, S0(i,j,k)+S0(i,j,k+1)+10 );
    

    
    //S0.get_access_relation();


     //S0.angle_skew(i,j,2,1,true,i0,j0);

     
    //S0.parallelize(i0);


    S0.after(C_init,tiramisu::computation::root) ;
       // S0.after(C_init,j0) ;
   
    //S0.angle_skew(i,j,2,1,false,i0,j0) ;
    //S0.tile(i,j,4,4,i1,i2,j1,j2);
   // S0.vectorize(j0,20) ;
    /*if(S0.applied_schedule_is_legal())
    {
      std::cout<<" schid is legal ";
    }
    else
    {
      std::cout<<" schid is notlegal ";
    } */

    S1.after(S0,tiramisu::computation::root) ;
   

        
      //  isl_union_map *map = S0.get_function()->compute_dep_graph() ; 

    //std::cout<<(isl_union_map_to_str(map)) ; 

    /*std::cout<<"\nfesdfchjksbdfchljubsdhjbjjj\n" ;

    std::cout<<(isl_map_to_str(S0.get_access_relation())) ;
    */

    std::cout<<"\nfesdfchjksbdfchljubsdhjbjjj\n" ;
  

    S0.get_function()->gen_ordering_schedules() ;
    S0.get_function()->align_schedules() ;
/*

    std::cout<<(isl_union_map_to_str(S0.get_function()->get_schedule())) ;


    S0.get_function()->gen_ordering_schedules() ;
    S0.get_function()->align_schedules() ;

    std::cout<<(isl_union_map_to_str(S0.get_function()->get_schedule())) ;
    */

      std::cout<<"\nfesdfchjksbdfchljubsdhjbjjj\n" ;

    tiramisu::buffer b_A("b_A", {100,100,10}, tiramisu::p_int32, tiramisu::a_temporary);
    tiramisu::buffer b_A2("b_A2", {100,100,10}, tiramisu::p_int32, tiramisu::a_temporary);
    tiramisu::buffer b_output("b_output", {100,100,10}, tiramisu::p_int32, tiramisu::a_output);

    S0.get_function()->save_computation_default_schedules() ;
    S0.get_function()->save_computations_levels() ;

    S1.after_change(S0,k) ;

    S0.get_function()->gen_ordering_schedules() ;
    S0.get_function()->align_schedules() ;

    S0.applied_schedule_is_legal(S1) ;

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

    C_init.store_in(&b_A);
    S0.store_in(&b_A2);
    //S0.store_in(&b_output);
    S1.store_in(&b_output);
    std::string t ="test2.txt" ;
    tiramisu::codegen_write_potential_schedules(t,{&b_output}, "new_1.o");

    tiramisu::codegen({&b_output}, "new_1.o");


    //tiramisu::codegen({&b_output}, "build/new.o");

    return 0;
    
}