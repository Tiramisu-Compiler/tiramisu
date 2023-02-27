#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_adi_MINI_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    double DX, DY, DT;
    double B1, B2;
    double mul1, mul2;
    double a, b, c, d, e, f;

    DX = 1.0/(double)17;
    DY = 1.0/(double)17;
    DT = 1.0/(double)15;
    B1 = 2.0;
    B2 = 1.0;
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 /  2.0;
    b = 1.0+mul1;
    c = a;
    d = -mul2 / 2.0;
    e = 1.0+mul2;
    f = d;
    
    tiramisu::init("function_adi_MINI");
    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 1, 17-1), j("j", 1, 17-1), t("t", 1, 15+1);
    var i_f("i_f", 0, 17), j_f("j_f", 0, 17);

    //inputs
    input u("u", {i, j}, p_float64);
    input v("v", {i, j}, p_float64);
    input p("p", {i, j}, p_float64);
    input q("q", {i, j}, p_float64);

    //Computations
    computation v_comp("v_comp", {t, i}, 1.0);
    computation p_comp("p_comp", {t, i}, 0.0);
    computation q_comp("q_comp", {t, i}, v(0,i));

    computation p_col("p_col", {t,i,j}, expr(-c) / (p(i, j-1)*a+b));
    computation q_col("q_col", {t,i,j}, (u(j, i-1)*(-d)+u(j, i)*(1.0+2.0*d) - u(j, i+1)*f-q(i, j-1)*a)/(p(i, j-1)*a+b));
    computation v_col_last("v_col_last", {t, i}, 1.0);

    computation v_col("v_col", {t,i,j}, p(i, j) * v(j+1, i) + q(i, j));

    computation u_comp("u_comp", {t, i}, 1.0);
    computation p_comp2("p_comp2", {t, i}, 0.0);
    computation q_comp2("q_comp2", {t, i}, u(i,0));

    computation p_row("p_row", {t,i,j}, expr(-f) / (p(i, j-1)*d+e));
    computation q_row("q_row", {t,i,j}, (v(i-1,j)*(-a)+v(i, j)*(1.0+2.0*a) - v(i+1, j)*c-q(i, j-1)*d)/(p(i, j-1)*d+e));

    computation u_row_last("u_row_last", {t,i}, 1.0);
    computation u_row("u_row", {t,i,j}, p(i, j) * u(i, j+1) + q(i, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    v_col.loop_reversal(2);
    u_row.loop_reversal(2);

    v_comp.then(p_comp, 1)
     .then(q_comp, 1)
     .then(p_col, 1)
     .then(q_col, 2)
     .then(v_col_last, 1)
     .then(v_col, 1)
     .then(u_comp, 0)
     .then(p_comp2, 1)
     .then(q_comp2, 1)
     .then(p_row, 0)
     .then(q_row, 2)
     .then(u_row_last, 1)
     .then(u_row, 1);

    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_u("b_u", {17,17}, p_float64, a_output);    
    buffer b_p("b_p", {17,17}, p_float64, a_temporary);    
    buffer b_q("b_q", {17,17}, p_float64, a_temporary);    
    buffer b_v("b_v", {17,17}, p_float64, a_temporary);    
   

    //Store inputs
    u.store_in(&b_u);
    v.store_in(&b_v);
    q.store_in(&b_q);
    p.store_in(&b_p);

    //Store computations
    v_comp.store_in(&b_v, {0,i});
    p_comp.store_in(&b_p, {i,0});
    q_comp.store_in(&b_q, {i,0});
    p_col.store_in(&b_p,{i,j});
    q_col.store_in(&b_q,{i,j});
    v_col_last.store_in(&b_v, {17-1,i});
    v_col.store_in(&b_v,{j,i});
    u_comp.store_in(&b_u, {i,0});
    p_comp2.store_in(&b_p, {i,0});
    q_comp2.store_in(&b_q, {i,0});
    p_row.store_in(&b_p,{i,j});
    q_row.store_in(&b_q,{i,j});
    u_row_last.store_in(&b_u,{i,17-1});
    u_row.store_in(&b_u,{i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_u}, "function_adi_MINI.o");

    return 0;
}
