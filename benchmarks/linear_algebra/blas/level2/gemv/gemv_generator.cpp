#include <tiramisu/tiramisu.h>

using namespace tiramisu;


/**
 * Benchmark for BLAS GEMV
 *     out = a*A*x + b*y
 *  
 *
 *     A : is a M x N matrix
 *     x : is a size N vector
 *     y : is a size M vector
 *     a,b : are scalars
 *
 *     out : is a size M vector
 */
/**
We will make a tiramisu implementation of this code :
  for(int i=0;i<M;i++){
    tmp=0;
    for(int j=0;j<N;j++){
      tmp += A(i,j)*x(j)
    }
	tmp *= alpha
    result(i) = tmp + beta * y(i)
  }


*/
void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function func(name);

    
    //Declare both iteration variables
    tiramisu::var i("i");
    tiramisu::var j("j");
	//Matrix Dimensions
    tiramisu::computation SIZES("{SIZES[k]: 0<=k<2}", tiramisu::expr(), false, p_float64, &func);
    //The constant M represents the number of rows of the matrix A
    tiramisu::constant M("M", SIZES(0), p_int32, true, NULL, 0, &func);
    //The constant M represents the number of columns of the matrix A
    tiramisu::constant N("N", SIZES(1), p_int32, true, NULL, 0, &func);


    tiramisu::computation A("[M,N]->{A[i,j]: 0<=i<M and 0<=j<N}", tiramisu::expr(), false, p_float64, &func);
    tiramisu::computation x("[N]->{x[j]: 0<=j<N}", tiramisu::expr(), false, p_float64, &func);
    tiramisu::computation y("[M]->{y[i]: 0<=i<M}", tiramisu::expr(), false, p_float64, &func);
    tiramisu::computation alpha("{alpha[0]}", tiramisu::expr(), false, p_float64, &func);
    tiramisu::computation beta("{beta[0]}", tiramisu::expr(), false, p_float64, &func);

    

    //Initialize with A(i,0)  x(0)
    tiramisu::computation result_init("[M]->{result_init[i]: 0<=i<M}",tiramisu::expr(A(i,0)*x(0)),true,p_float64,&func);
    //Perform A(i,:)*x(:)
    tiramisu::computation sumLine("[M,N]->{sumLine[i,j]: 0<=i<M and 1<=j<N}",tiramisu::expr() , true, p_float64, &func);
    //Multiply the sum by alpha
    tiramisu::computation multAlpha("[M]->{multAlpha[i]:0<=i<M}",tiramisu::expr(),true,p_float64,&func);
    //Add beta*y(j)
    tiramisu::computation addY("[M]->{addY[i]: 0<=i<M}",tiramisu::expr(),true,p_float64,&func);


    sumLine.set_expression(tiramisu::expr(sumLine(i,j-1) + A(i,j) * x(j)));
    multAlpha.set_expression(tiramisu::expr(alpha(0)*sumLine(i,N-1)));
    addY.set_expression(tiramisu::expr(multAlpha(i)+beta(0)*y(i)));


    addY.after(multAlpha,i);
    multAlpha.after(sumLine,i);
    sumLine.after(result_init,i);
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
	

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    //INPUT BUFFERS
    tiramisu::buffer buf_SIZES("buf_SIZES", {2}, tiramisu::p_int32, a_input, &func);

    tiramisu::buffer buf_A("buf_A", {M,N}, tiramisu::p_float64, a_input, &func);
    tiramisu::buffer buf_x("buf_x", {N}, tiramisu::p_float64, a_input, &func);
    tiramisu::buffer buf_y("buf_y", {M}, tiramisu::p_float64, a_input, &func);
    tiramisu::buffer buf_alpha("buf_alpha", {1}, tiramisu::p_float64, a_input, &func);
    tiramisu::buffer buf_beta("buf_beta", {1}, tiramisu::p_float64, a_input, &func);

    //OUTPUT BUFFERS
    tiramisu::buffer buf_result("buf_result",{M}, tiramisu::p_float64,a_output,&func);

	SIZES.set_access("{SIZES[k]->buf_SIZES[k] : 0<=k<2}");
    alpha.set_access("{alpha[0]->buf_alpha[0]}");
    beta.set_access("{beta[0]->buf_beta[0]}");
    A.set_access("{A[i,j]->buf_A[i,j]}");
    x.set_access("{x[j]->buf_x[j]}");
    y.set_access("{y[i]->buf_y[i]}");


    result_init.set_access("[M]->{result_init[i]->buf_result[i]}");
    sumLine.set_access("[M,N]->{sumLine[i,j]->buf_result[i]}");
    multAlpha.set_access("[M]->{multAlpha[i]->buf_result[i]}");
    addY.set_access("[M]->{addY[i]->buf_result[i]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    func.set_arguments({&buf_SIZES,&buf_A, &buf_x, &buf_y,&buf_alpha,&buf_beta,&buf_result});
    func.gen_time_space_domain();
    func.gen_isl_ast();
    func.gen_halide_stmt();
    func.gen_halide_obj("generated_" + name + ".o");
}

int main(int argc, char **argv)
{
    generate_function("gemv");

    return 0;
}
