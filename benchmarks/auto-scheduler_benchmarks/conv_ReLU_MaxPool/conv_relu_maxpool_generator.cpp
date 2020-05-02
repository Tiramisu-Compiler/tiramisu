#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("conv_relu_maxpool");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant batch_size("batch_size", BATCH_SIZE), nb_channels("nb_channels", CHANNELS), input_height("input_height", INP_Y), input_width("input_width", INP_X),  nb_kernels("nb_kernels", NB_K), kernel_height("kernel_height", K_Y), kernel_width("kernel_width", K_X), conv_height("conv_height", INP_Y-(K_Y-1)), conv_width("conv_width", INP_X-(K_X-1));
    var b("b", 0, batch_size), c("c", 0, nb_channels), i_y("i_y", 0, input_height), i_x("i_x", 0, input_width), o_y("o_y", 0, conv_height), o_x("o_x", 0, conv_width), k_y("k_y", 0, kernel_height), k_x("k_x", 0, kernel_width), nb_k("nb_k", 0, nb_kernels);
    constant pool_width("pool_width", POOL_WIDTH), pool_height("pool_height", POOL_HEIGHT);
    var pool_x("pool_x", 0, pool_width), pool_y("pool_y", 0, pool_height);

    //inputs
    input kernel("kernel", {nb_k, c, k_y, k_x}, p_float64);
    input bias("bias", {nb_k}, p_float64);
    input input("input", {b, c, i_y, i_x}, p_float64);
    
    //Computations
    computation convolution_init("convolution_init", {b, nb_k, o_y, o_x}, bias(nb_k));
    computation convolution("convolution", {b, nb_k, o_y, o_x, c, k_y, k_x}, p_float64);
    convolution.set_expression(convolution(b, nb_k, o_y, o_x, c, k_y, k_x) + input(b, c, o_y + k_y, o_x + k_x) * kernel(nb_k, c, k_y, k_x));
    computation relu("relu", {b, nb_k, o_y, o_x}, expr(o_max, convolution(b, nb_k, o_y, o_x, 0, 0, 0), cast(p_float64, 0)));
    computation maxpool("maxpool", {b, nb_k, o_y, o_x}, p_float64);
    maxpool.set_expression(expr(o_max, maxpool(b, nb_k, o_y, o_x),  relu(b, nb_k, o_y, o_x)));


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    convolution_init.then(convolution, o_x)
                    .then(relu, o_x)
                    .then(maxpool, o_x);
    convolution_init.parallelize(b);
    convolution.parallelize(b);
    maxpool.parallelize(b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer buf_input("buf_input", {batch_size, nb_channels, input_height, input_width}, p_float64, a_input);
    buffer buf_bias("buf_bias", {nb_kernels}, p_float64, a_input);
    buffer buf_kernel("buf_kernel", { nb_kernels, kernel_height, kernel_width}, p_float64, a_input);
    buffer buf_convolution("buf_convolution", {batch_size, nb_kernels, conv_height, conv_width}, p_float64, a_temporary);
    buffer buf_relu("buf_relu", {batch_size, nb_kernels, conv_height, conv_width}, p_float64, a_temporary);
    buffer buf_maxpool("buf_maxpool",  {batch_size, nb_kernels, conv_height/POOL_HEIGHT, conv_width/POOL_WIDTH}, p_float64, a_output);

    
    //Store inputs
    input.store_in(&buf_input);
    bias.store_in(&buf_bias);
    kernel.store_in(&buf_kernel);

    //Store computations
    convolution_init.store_in(&buf_convolution, {b, nb_k, o_y, o_x});
    convolution.store_in(&buf_convolution, {b, nb_k, o_y, o_x});
    relu.store_in(&buf_relu, {b, nb_k, o_y, o_x});
    maxpool.store_in(&buf_maxpool, {b, nb_k, o_y/POOL_HEIGHT, o_x/POOL_WIDTH});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_input, &buf_kernel, &buf_bias, &buf_maxpool}, "generated_conv_relu_maxpool.o");
    
    return 0;
}

