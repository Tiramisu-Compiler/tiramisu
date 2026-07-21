#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_fdtd_2d_MINI_wrapper.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_fdtd_2d_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i_f("i_f", 0, 20), j_f("j_f", 0, 30), i_m("i_m", 0, 20-1), j_m("j_m", 0, 30-1);
    var t("t", 0, 20), i("i", 1, 20), j("j", 1, 30);
    
    //inputs
    input fict("fict", {t}, p_float64);
    input ey("ey", {i_f, j_f}, p_float64);
    input ex("ex", {i_f, j_f}, p_float64);
    input hz("hz", {i_f, j_f}, p_float64);

    //Computations
    computation ey_slice("ey_slice", {t,j_f}, fict(t));
    computation ey_out("ey_out", {t, i, j_f}, ey(i, j_f) - (hz(i, j_f) - hz(i-1, j_f))*0.5);
    computation ex_out("ex_out", {t, i_f, j}, ex(i_f, j) - (hz(i_f, j) - hz(i_f, j - 1))*0.5);
    computation hz_out("hz_out", {t, i_m, j_m}, hz(i_m, j_m) - (ex(i_m, j_m + 1) - ex(i_m, j_m) + ey(i_m + 1, j_m) - ey(i_m, j_m))*0.7);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    ey_slice.then(ey_out, t)
            .then(ex_out, t)
            .then(hz_out, t);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_fict("b_fict", {20}, p_float64, a_input);    
    buffer b_ey("b_ey", {20,30}, p_float64, a_output);    
    buffer b_ex("b_ex", {20,30}, p_float64, a_output);    
    buffer b_hz("b_hz", {20,30}, p_float64, a_output);    

    //Store inputs
    fict.store_in(&b_fict);
    ey.store_in(&b_ey);
    ex.store_in(&b_ex);
    hz.store_in(&b_hz);


    //Store computations
    ey_slice.set_access("{ey_slice[t,j_f]->b_ey[0,j_f]}");
    ey_out.store_in(&b_ey, {i,j_f});
    ex_out.store_in(&b_ex, {i_f,j});
    hz_out.store_in(&b_hz, {i_m, j_m});



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_ex, &b_ey, &b_hz, &b_fict}, "function_fdtd_2d_MINI.o");

    return 0;
}
