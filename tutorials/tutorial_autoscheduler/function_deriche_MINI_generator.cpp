#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_deriche_MINI_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv) {
    double k;
    double a1, a2, a3, a4, a5, a6, a7, a8;
    double b1, b2, c1, c2;

    k = (1.0-exp(-0.25))*(1.0-exp(-0.25))/(1.0+2.0*0.25*exp(-0.25)-exp(2.0*0.25));
    a1 = a5 = k;
    a2 = a6 = k*exp(-0.25)*(0.25-1.0);
    a3 = a7 = k*exp(-0.25)*(0.25+1.0);
    a4 = a8 = -k*exp((-2.0)*0.25);
    b1 = pow(2.0,-0.25);
    b2 = -exp((-2.0)*0.25);
    c1 = c2 = 1;

    tiramisu::init("function_deriche_MINI");


    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant HH("HH", 64), WW("WW",64);


    //Iteration variables    
    var i("i", 0, 64), j("j", 0, 64);
    

    //inputs
    input imgIn("imgIn", {i, j}, p_float64);
    input y1("y1", {i,j}, p_float64); //used as a wrapper for buf y1
    input y2("y2", {i,j}, p_float64); //used as a wrapper for buf y2

 
    //Computations
    computation ym1_w_init("ym1_w_init", {i}, 0.0);
    computation ym2_w_init("ym2_w_init", {i}, 0.0);
    computation xm1_w_init("xm1_w_init", {i}, 0.0);
    computation y1_1("y1", {i,j}, imgIn(i, j)*a1 + xm1_w_init(i)*a2 + ym1_w_init(i)*b1 + ym2_w_init(i)*b2);
    computation xm1_w("xm1_w", {i,j}, imgIn(i, j));
    computation ym2_w("ym2_w", {i,j}, ym1_w_init(i));
    computation ym1_w("ym1_w", {i,j}, y1(i, j));

    computation yp1_w_init("yp1_w_init", {i}, 0.0);
    computation yp2_w_init("yp2_w_init", {i}, 0.0);
    computation xp1_w_init("xp1_w_init", {i}, 0.0);
    computation xp2_w_init("xp2_w_init", {i}, 0.0);
    computation y2_reverse_j("y2_reverse_j", {i,j}, xp1_w_init(i)*a3 + xp2_w_init(i)*a4 + yp1_w_init(i)*b1 + yp2_w_init(i)*b2);
    // computation y2_reverse_j("y2_reverse_j", {i,j}, 5.0);
    computation xp2_w("xp2_w", {i,j}, xp1_w_init(i));
    computation xp1_w("xp1_w", {i,j}, imgIn(i, 64-1-j));
    computation yp2_w("yp2_w", {i,j}, yp1_w_init(i));
    computation yp1_w("yp1_w", {i,j}, y2(i, 64-1-j));//

    computation imgOut_r("imgOut_r", {i,j}, (y1(i, j) + y2(i, j))*c1);//

    computation tm1_h_init("tm1_h_init" ,{j}, 0.0);
    computation ym1_h_init("ym1_h_init" ,{j}, 0.0);
    computation ym2_h_init("ym2_h_init" ,{j}, 0.0);
    computation y1_transpose("y1_transpose", {j,i}, imgOut_r(i, j)*a5 + tm1_h_init(j)*a6 + ym1_h_init(j)*b1 + ym2_h_init(j)*b2);///
    computation tm1_h("tm1_h", {j,i}, imgOut_r(i, j));
    computation ym2_h("ym2_h", {j,i}, ym1_h_init(j));
    computation ym1_h("ym1_h", {j,i}, y1(i, j));

    computation tp1_h_init("tp1_h_init", {j}, 0.0);
    computation tp2_h_init("tp2_h_init", {j}, 0.0);
    computation yp1_h_init("yp1_h_init", {j}, 0.0);
    computation yp2_h_init("yp2_h_init", {j}, 0.0);
    computation y2_reverse_i("y2_reverse_i", {j,i}, tp1_h_init(j)*a7 + tp2_h_init(j)*a8 + yp1_h_init(j)*b1 + yp2_h_init(j)*b2);
    computation tp2_h("tp2_h", {j,i}, tp1_h_init(j));
    computation tp1_h("tp1_h", {j,i}, imgOut_r(64-1-i, j));
    computation yp2_h("yp2_h", {j,i}, yp1_h_init(j));
    computation yp1_h("yp1_h", {j,i}, y2(64-1-i, j));//

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
              .then(xp2_w, j)
              .then(xp1_w, j)
              .then(yp2_w, j)
              .then(yp1_w, j)
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
              .then(tp2_h, i)
              .then(tp1_h, i)
              .then(yp2_h, i)
              .then(yp1_h, i)
              .then(imgOut, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_imgIn("b_imgIn", {64,64}, p_float64, a_input);    
    buffer b_imgOut("b_imgOut", {64,64}, p_float64, a_output);    
    buffer b_y1("b_y1", {64,64}, p_float64, a_temporary);    
    buffer b_y2("b_y2", {64,64}, p_float64, a_temporary);  
    buffer b_xm1_w("b_xm1_w", {64}, p_float64, a_temporary);
    buffer b_ym2_w("b_ym2_w", {64}, p_float64, a_temporary);
    buffer b_ym1_w("b_ym1_w", {64}, p_float64, a_temporary);
    buffer b_xp2_w("b_xp2_w", {64}, p_float64, a_temporary);
    buffer b_xp1_w("b_xp1_w", {64}, p_float64, a_temporary);
    buffer b_yp2_w("b_yp2_w", {64}, p_float64, a_temporary);
    buffer b_yp1_w("b_yp1_w", {64}, p_float64, a_temporary);
    buffer b_tp2_h("b_tp2_h", {64}, p_float64, a_temporary);
    buffer b_tp1_h("b_tp1_h", {64}, p_float64, a_temporary);
    buffer b_yp2_h("b_yp2_h", {64}, p_float64, a_temporary);
    buffer b_yp1_h("b_yp1_h", {64}, p_float64, a_temporary);  
    buffer b_tm1_h("b_tm1_h", {64}, p_float64, a_temporary);
    buffer b_ym1_h("b_ym1_h", {64}, p_float64, a_temporary);
    buffer b_ym2_h("b_ym2_h", {64}, p_float64, a_temporary);

    //Store inputs
    imgIn.store_in(&b_imgIn);    
    y2.store_in(&b_y2);    
    y1.store_in(&b_y1);    

    //Store computations
    ym1_w_init.store_in(&b_ym1_w);
    ym2_w_init.store_in(&b_ym2_w);
    xm1_w_init.store_in(&b_xm1_w);
    y1_1.store_in(&b_y1);
    xm1_w.store_in(&b_xm1_w, {i});
    ym2_w.store_in(&b_ym2_w, {i});
    ym1_w.store_in(&b_ym1_w, {i});
    yp1_w_init.store_in(&b_yp1_w);
    yp2_w_init.store_in(&b_yp2_w);
    xp1_w_init.store_in(&b_xp1_w);
    xp2_w_init.store_in(&b_xp2_w);
    y2_reverse_j.set_access("[HH]->{y2_reverse_j[i,j]->b_y2[i,HH-1-j]}");
    xp2_w.store_in(&b_xp2_w, {i});
    xp1_w.store_in(&b_xp1_w, {i});
    yp2_w.store_in(&b_yp2_w, {i});
    yp1_w.store_in(&b_yp1_w, {i});
    imgOut_r.store_in(&b_imgOut);
    tm1_h_init.store_in(&b_tm1_h);
    ym1_h_init.store_in(&b_ym1_h);
    ym2_h_init.store_in(&b_ym2_h);
    y1_transpose.set_access("[HH]->{y1_transpose[j,i]->b_y1[i,j]}");
    tm1_h.store_in(&b_tm1_h, {j});
    ym2_h.store_in(&b_ym2_h, {j});
    ym1_h.store_in(&b_ym1_h, {j});
    tp1_h_init.store_in(&b_tp1_h);
    tp2_h_init.store_in(&b_tp2_h);
    yp1_h_init.store_in(&b_yp1_h);
    yp2_h_init.store_in(&b_yp2_h);
    y2_reverse_i.set_access("[WW]->{y2_reverse_i[j,i]->b_y2[WW-1-i,j]}");
    tp2_h.store_in(&b_tp2_h, {j});
    tp1_h.store_in(&b_tp1_h, {j});
    yp2_h.store_in(&b_yp2_h, {j});
    yp1_h.store_in(&b_yp1_h, {j});
    imgOut.store_in(&b_imgOut);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_imgIn, &b_imgOut}, "function_deriche_MINI.o");

    return 0;
}
