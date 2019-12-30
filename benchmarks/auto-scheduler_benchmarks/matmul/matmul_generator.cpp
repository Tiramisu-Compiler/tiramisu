#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant l_size("l_size", LL), m_size("m_size", MM), n_size("n_size", NN) ;
    var l("l", 0, l_size), m("m", 0, m_size), n("n", 0, n_size);
    //inputs
    input A("A", {l, m}, p_float64);
    input B("B", {m, n}, p_float64);
    
    //Computations
    computation matmul_init("matmul_init", {l, n}, expr(cast(p_float64, 0)));
    computation matmul("matmul", {l, n, m}, p_float64);
    matmul.set_expression(matmul(l, n, m) + A(l, m)*B(m, n));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    matmul_init.then(matmul, n);
    matmul_init.parallelize(l);
    matmul.parallelize(l);
    
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer buf_A("A", {l_size, m_size}, p_float64, a_input);
    buffer buf_B("B", { m_size, n_size}, p_float64, a_input);
    buffer buf_matmul("matmul", {l_size, n_size}, p_float64, a_output);
    
    //Store inputs
    A.store_in(&buf_A);
    B.store_in(&buf_B);

    //Store computations
    matmul_init.store_in(&buf_matmul, {l, n});
    matmul.store_in(&buf_matmul, {l, n});
   

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_A, &buf_B, &buf_matmul}, "generated_matmul.o");
    
    return 0;
}

