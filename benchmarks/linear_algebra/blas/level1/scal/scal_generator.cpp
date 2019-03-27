#include <tiramisu/tiramisu.h>
#include <string>
#include "benchmarks.h"

using namespace tiramisu;

/*
 * Benchmark for BLAS DSCAL
 *     X = alpha*X
 * 
 * inputs:
 * --------
 * - n: size of vector X.
 * - alpha: scale factor.
 * - X: vector to scale.
 *
 * outputs:
 * ---------
 * The operation is done inplace.
 */

#if TIRAMISU_XLARGE || TIRAMISU_LARGE
    #define VECTORIZE_V 512
#elif TIRAMISU_MEDIUM
    #define VECTORIZE_V 256
#else
    #define VECTORIZE_V 16
#endif

void generate_function(std::string name)
{
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    function scal(name);
  
    computation SIZES("{SIZES[0]}", expr(), false, p_int32, &scal);
    computation alpha("{alpha[0]}", expr(), false, p_float64, &scal);
    computation X("[M]->{X[i]: 0<=i<M}", expr(), false, p_float64, &scal);

    constant M_cst("M", SIZES(0), p_int32, true, NULL, 0, &scal);
  
    var i("i");
    computation result("[M]->{result[i]: 0<=i<M}", alpha(0)*X(i), true, p_float64, &scal);
  
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    result.vectorize(i, VECTORIZE_V);
  
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &scal);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input, &scal);
    buffer b_X("b_X", {M_cst}, p_float64, a_output, &scal);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    X.set_access("{X[i]->b_X[i]}");
  
    result.set_access("{result[i]->b_X[i]}");
  
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    scal.set_arguments({&b_SIZES, &b_alpha, &b_X});
    scal.gen_time_space_domain();
    scal.gen_isl_ast();
    scal.gen_halide_stmt();
    scal.gen_halide_obj("generated_" + name + ".o");
}

int main(int argc, char** argv)
{
    generate_function("scal");
    return 0;
}
