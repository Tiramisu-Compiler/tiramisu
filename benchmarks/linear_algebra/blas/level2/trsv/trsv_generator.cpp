#include <tiramisu/tiramisu.h>
#include <string>
#include "benchmarks.h"

using namespace tiramisu;

/*
 * Benchmark for BLAS DTRSV
 *
 * Resolve the linear system :
 *     AX = b
 * where A is an upper triangular matrix.
 * 
 * inputs:
 * --------
 * - n: order of matrix A.
 * - A: matrix of size nxn.
 * - b: a vector of size n.
 *      right-hand side of the linear system.
 *
 * outputs:
 * ---------
 * - X: a vector of size n.
 *      solution of the linear system.
 *
 * Algorithm:
 * for (i = 0; i < n; ++i)
 *     forward = 0;
 *     for (j = 0; j < i; ++j)
 *          forward += A[i][j] * X[j]
 *
 *     X[i] = (b[i] - forward) / A[i][i]
 */

void generate_function(std::string name)
{
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    function trsv(name);

    // Inputs
    computation SIZES("{SIZES[0]}", expr(), false, p_int32, &trsv);

    computation A("[N]->{A[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_float64, &trsv);
    computation b("[N]->{b[i]: 0<=i<N}", expr(), false, p_float64, &trsv);
    
    constant N_cst("N", SIZES(0), p_int32, true, NULL, 0, &trsv);

    // Outputs
    computation X("[N]->{X[i]: 0<=i<N}", expr(), true, p_float64, &trsv);
    computation forward_init("[N]->{forward_init[i]: 0<=i<N}",
			       expr((double)0), true, p_float64, &trsv);
    
    computation forward("[N]->{forward[i,j]: 0<=i<N and 0<=j<i}",
			  expr(), true, p_float64, &trsv);
    
    var i("i"), j("j");
    forward.set_expression(forward(i, j-1) + A(i, j)*X(j));
    X.set_expression((b(i) - forward(i, i)) / A(i, i));
  
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    forward.after(forward_init, i);
    X.after(forward, i);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &trsv);
    buffer b_A("b_A", {N_cst, N_cst}, p_float64, a_input, &trsv);
    buffer b_b("b_b", {N_cst}, p_float64, a_input, &trsv);

    buffer b_X("b_X", {N_cst}, p_float64, a_output, &trsv);
    buffer b_forward("b_forward", {expr(1)}, p_float64, a_temporary, &trsv);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    A.set_access("{A[i,j]->b_A[i,j]}");
    b.set_access("{b[i]->b_b[i]}");
    
    X.set_access("{X[i]->b_X[i]}");
    forward_init.set_access("{forward_init[i]->b_forward[0]}");
    forward.set_access("{forward[i,j]->b_forward[0]}");
  
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    trsv.set_arguments({&b_SIZES, &b_A, &b_b, &b_X});
    trsv.gen_time_space_domain();
    trsv.gen_isl_ast();
    trsv.gen_halide_stmt();
    trsv.gen_halide_obj("generated_" + name + ".o");
}

int main(int argc, char** argv)
{
    generate_function("trsv");
    return 0;
}
