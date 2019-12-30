#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("convolution");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant batch_size("batch_size", BATCH_SIZE), nb_channels("nb_channels", CHANNELS), input_height("input_height", INP_Y), input_width("input_width", INP_X),  nb_kernels("nb_kernels", NB_K), kernel_height("kernel_height", K_Y), kernel_width("kernel_width", K_X), output_height("output_height", INP_Y-(K_Y-1)), output_width("output_width", INP_X-(K_X-1));
    var b("b", 0, batch_size), c("c", 0, nb_channels), i_y("i_y", 0, input_height), i_x("i_x", 0, input_width), o_y("o_y", 0, output_height), o_x("o_x", 0, output_width), k_y("k_y", 0, kernel_height), k_x("k_x", 0, kernel_width), nb_k("nb_k", 0, nb_kernels);

    //inputs
    input kernel("kernel", {nb_k, c, k_y, k_x}, p_float64);
    input bias("bias", {nb_k}, p_float64);
    input input("input", {b, c, i_y, i_x}, p_float64);
    
    //Computations
    computation convolution_init("convolution_init", {b, nb_k, o_y, o_x}, bias(nb_k));
    computation convolution("convolution", {b, nb_k, o_y, o_x, c, k_y, k_x}, p_float64);
    convolution.set_expression(convolution(b, nb_k, o_y, o_x, c, k_y, k_x) + input(b, c, o_y + k_y, o_x + k_x) * kernel(nb_k, c, k_y, k_x));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    convolution_init.then(convolution, o_x);
    convolution_init.parallelize(b);
    convolution.parallelize(b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer buf_input("buf_input", {batch_size, nb_channels, input_height, input_width}, p_float64, a_input);
    buffer buf_bias("buf_bias", {nb_kernels}, p_float64, a_input);
    buffer buf_kernel("buf_kernel", { nb_kernels, kernel_height, kernel_width}, p_float64, a_input);
    buffer buf_convolution("buf_convolution", {batch_size, nb_kernels, output_height, output_width}, p_float64, a_output);
    
    //Store inputs
    input.store_in(&buf_input);
    bias.store_in(&buf_bias);
    kernel.store_in(&buf_kernel);

    //Store computations
    convolution_init.store_in(&buf_convolution, {b, nb_k, o_y, o_x});
    convolution.store_in(&buf_convolution, {b, nb_k, o_y, o_x});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_input, &buf_kernel, &buf_bias, &buf_convolution}, "generated_convolution.o");
    
    return 0;
}

