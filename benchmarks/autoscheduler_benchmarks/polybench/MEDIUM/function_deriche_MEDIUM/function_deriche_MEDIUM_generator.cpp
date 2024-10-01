#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_deriche_MEDIUM_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv) {
    float k;
    float a1, a2, a3, a4, a5, a6, a7, a8;
    float b1, b2, c1, c2;

    k = (1.0-exp(-0.25))*(1.0-exp(-0.25))/(1.0+2.0*0.25*exp(-0.25)-exp(2.0*0.25));
    a1 = a5 = k;
    a2 = a6 = k*exp(-0.25)*(0.25-1.0);
    a3 = a7 = k*exp(-0.25)*(0.25+1.0);
    a4 = a8 = -k*exp((-2.0)*0.25);
    b1 = pow(2.0,-0.25);
    b2 = -exp((-2.0)*0.25);
    c1 = c2 = 1;

    tiramisu::init("function_deriche_MEDIUM");


    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant HH("HH", 480), WW("WW",720);


    //Iteration variables    
    var i("i", 0, 720), j("j", 0, 480);
    var j_reversed("j_reversed", -480+1, 1), i_reversed("i_reversed", -720+1, 1);

    //inputs
    input imgIn("imgIn", {i, j}, p_float32);
    input y1("y1", {i,j}, p_float32); //used as a wrapper for buf y1
    input y2("y2", {i,j}, p_float32); //used as a wrapper for buf y2

 
    //Computations
    computation ym1_w_init("ym1_w_init", {i}, expr((float)0.0));
    computation ym2_w_init("ym2_w_init", {i}, expr((float)0.0));
    computation xm1_w_init("xm1_w_init", {i}, expr((float)0.0));
    computation y1_1("y1_1", {i,j}, imgIn(i, j)*a1 + xm1_w_init(0)*a2 + ym1_w_init(0)*b1 + ym2_w_init(0)*b2);
    computation xm1_w("xm1_w", {i,j}, imgIn(i, j));
    computation ym2_w("ym2_w", {i,j}, ym1_w_init(0));
    computation ym1_w("ym1_w", {i,j}, y1(i, j));

    computation yp1_w_init("yp1_w_init", {i}, expr((float)0.0));
    computation yp2_w_init("yp2_w_init", {i}, expr((float)0.0));
    computation xp1_w_init("xp1_w_init", {i}, expr((float)0.0));
    computation xp2_w_init("xp2_w_init", {i}, expr((float)0.0));
    computation y2_reverse_j("y2_reverse_j", {i,j_reversed}, xp1_w_init(0)*a3 + xp2_w_init(0)*a4 + yp1_w_init(0)*b1 + yp2_w_init(0)*b2);
    computation xp2_w("xp2_w", {i,j_reversed}, xp1_w_init(0));
    computation xp1_w("xp1_w", {i,j_reversed}, imgIn(i, -j_reversed));
    computation yp2_w("yp2_w", {i,j_reversed}, yp1_w_init(0));
    computation yp1_w("yp1_w", {i,j_reversed}, y2(i, -j_reversed));//

    computation imgOut_r("imgOut_r", {i,j}, (y1(i, j) + y2(i, j))*c1);//

    computation tm1_h_init("tm1_h_init" ,{j}, expr((float)0.0));
    computation ym1_h_init("ym1_h_init" ,{j}, expr((float)0.0));
    computation ym2_h_init("ym2_h_init" ,{j}, expr((float)0.0));
    computation y1_transpose("y1_transpose", {j,i}, imgOut_r(i, j)*a5 + tm1_h_init(0)*a6 + ym1_h_init(0)*b1 + ym2_h_init(0)*b2);///
    computation tm1_h("tm1_h", {j,i}, imgOut_r(i, j));
    computation ym2_h("ym2_h", {j,i}, ym1_h_init(0));
    computation ym1_h("ym1_h", {j,i}, y1(i, j));

    computation tp1_h_init("tp1_h_init", {j}, expr((float)0.0));
    computation tp2_h_init("tp2_h_init", {j}, expr((float)0.0));
    computation yp1_h_init("yp1_h_init", {j}, expr((float)0.0));
    computation yp2_h_init("yp2_h_init", {j}, expr((float)0.0));
    computation y2_reverse_i("y2_reverse_i", {j,i_reversed}, tp1_h_init(0)*a7 + tp2_h_init(0)*a8 + yp1_h_init(0)*b1 + yp2_h_init(0)*b2);
    computation tp2_h("tp2_h", {j,i_reversed}, tp1_h_init(0));
    computation tp1_h("tp1_h", {j,i_reversed}, imgOut_r(-i_reversed, j));
    computation yp2_h("yp2_h", {j,i_reversed}, yp1_h_init(0));
    computation yp1_h("yp1_h", {j,i_reversed}, y2(-i_reversed, j));//

    computation imgOut("imgOut", {i,j}, (y1(i, j) + y2(i, j))*c2);//
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    
    ym1_w_init.then(ym2_w_init, i)
              .then(xm1_w_init, i)
              .then(y1_1, i)
              .then(xm1_w, j)
              .then(ym2_w, j)
              .then(ym1_w, j)
              .then(yp1_w_init, computation::root)
              .then(yp2_w_init, i)
              .then(xp1_w_init, i)
              .then(xp2_w_init, i)
              .then(y2_reverse_j, i)
              .then(xp2_w, j_reversed)
              .then(xp1_w, j_reversed)
              .then(yp2_w, j_reversed)
              .then(yp1_w, j_reversed)
              .then(imgOut_r, computation::root)
              .then(tm1_h_init, computation::root)
              .then(ym1_h_init, j)
              .then(ym2_h_init, j)
              .then(y1_transpose, j)
              .then(tm1_h,i)
              .then(ym2_h,i)
              .then(ym1_h,i)
              .then(tp1_h_init, computation::root)
              .then(tp2_h_init, j)
              .then(yp1_h_init, j)
              .then(yp2_h_init, j)
              .then(y2_reverse_i, j)
              .then(tp2_h, i_reversed)
              .then(tp1_h, i_reversed)
              .then(yp2_h, i_reversed)
              .then(yp1_h, i_reversed)
              .then(imgOut, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_imgIn("b_imgIn", {720,480}, p_float32, a_input);    
    buffer b_imgOut("b_imgOut", {720,480}, p_float32, a_output);    
    buffer b_y1("b_y1", {720,480}, p_float32, a_temporary);    
    buffer b_y2("b_y2", {720,480}, p_float32, a_temporary);  
    buffer b_xm1("b_xm1", {1}, p_float32, a_temporary);
    buffer b_ym2("b_ym2", {1}, p_float32, a_temporary);
    buffer b_ym1("b_ym1", {1}, p_float32, a_temporary);
    buffer b_xp2("b_xp2", {1}, p_float32, a_temporary);
    buffer b_xp1("b_xp1", {1}, p_float32, a_temporary);
    buffer b_yp2("b_yp2", {1}, p_float32, a_temporary);
    buffer b_yp1("b_yp1", {1}, p_float32, a_temporary);
    buffer b_tp2("b_tp2", {1}, p_float32, a_temporary);
    buffer b_tp1("b_tp1", {1}, p_float32, a_temporary);
    buffer b_tm1("b_tm1", {1}, p_float32, a_temporary);

    //Store inputs
    imgIn.store_in(&b_imgIn);    
    y2.store_in(&b_y2);    
    y1.store_in(&b_y1);      

    //Store computations
    ym1_w_init.store_in(&b_ym1,{0});
    ym2_w_init.store_in(&b_ym2,{0});
    xm1_w_init.store_in(&b_xm1,{0});
    y1_1.store_in(&b_y1);
    xm1_w.store_in(&b_xm1,{0});
    ym2_w.store_in(&b_ym2,{0});
    ym1_w.store_in(&b_ym1,{0});
    yp1_w_init.store_in(&b_yp1,{0});
    yp2_w_init.store_in(&b_yp2,{0});
    xp1_w_init.store_in(&b_xp1,{0});
    xp2_w_init.store_in(&b_xp2,{0});
    y2_reverse_j.store_in(&b_y2, {i, -j_reversed});
    xp2_w.store_in(&b_xp2,{0});
    xp1_w.store_in(&b_xp1,{0});
    yp2_w.store_in(&b_yp2,{0});
    yp1_w.store_in(&b_yp1,{0});
    imgOut_r.store_in(&b_imgOut);
    tm1_h_init.store_in(&b_tm1,{0});
    ym1_h_init.store_in(&b_ym1,{0});
    ym2_h_init.store_in(&b_ym2,{0});
    y1_transpose.store_in(&b_y1,{i,j});
    tm1_h.store_in(&b_tm1,{0});
    ym2_h.store_in(&b_ym2,{0});
    ym1_h.store_in(&b_ym1,{0});
    tp1_h_init.store_in(&b_tp1,{0});
    tp2_h_init.store_in(&b_tp2,{0});
    yp1_h_init.store_in(&b_yp1,{0});
    yp2_h_init.store_in(&b_yp2,{0});
    y2_reverse_i.store_in(&b_y2,{-i_reversed,j});
    tp2_h.store_in(&b_tp2,{0});
    tp1_h.store_in(&b_tp1,{0});
    yp2_h.store_in(&b_yp2,{0});
    yp1_h.store_in(&b_yp1,{0});
    imgOut.store_in(&b_imgOut);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_imgIn, &b_imgOut}, "function_deriche_MEDIUM.o");

    return 0;
}
