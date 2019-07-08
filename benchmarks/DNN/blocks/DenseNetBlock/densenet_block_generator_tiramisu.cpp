#include <tiramisu/tiramisu.h>
#include <vector>
#include "configure.h"

using namespace tiramisu;

int main()
{
    init("densenet_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), z("z", 0, 4*GR), n("n", 0, BATCH_SIZE);
    var _y("_y", 0, N - 1);

    var x_pad("x", 0, N + 2), y_pad("y", 0, N + 2);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y), fout("fout", 0, GR);

    var z_b("z_b", 0, Z_NB_BLOCKS), zz("zz", 0, Z_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, z_b, y, x, zz}, p_float32);
    input bn_scale("bn_scale", {z_b, zz}, p_float32);
    input bn_shift("bn_shift", {z_b, zz}, p_float32);

    input conv_filter("conv_filter", {z_b, fout_b, k_y, k_x, ffout, zz}, p_float32);
    input conv_bias("conv_bias", {fout_b, ffout}, p_float32);

    // Batch normalization followed by ReLU
    // Compute the sum over the features dimension (z)
    computation input_sum_init("input_sum_init", {z_b, zz}, cast(p_float32, 0));
    computation input_sum(
        "input_sum", 
        {n, z_b, y, x, zz}, 
        input_sum_init(z_b, zz) + c_input(n, z_b, y, x, zz)
    );

    // Compute the sum of squares over the features dimension (z)
    computation input_sum_squares_init("input_sum_squares_init", {z_b, zz}, cast(p_float32, 0));
    computation input_sum_squares(
        "input_sum_squares", 
        {n, z_b, y, x, zz}, 
        input_sum_squares_init(z_b, zz) + c_input(n, z_b, y, x, zz) * c_input(n, z_b, y, x, zz)
    );

    computation input_mean(
        "input_mean", 
        {z_b, zz}, 
        input_sum(BATCH_SIZE - 1, z_b, N - 1, N - 1, zz) / cast(p_float32, BATCH_SIZE*N*N)
    );

    computation input_sd(
        "input_sd", 
        {z_b, zz}, 
        expr(
            o_sqrt, 
            input_sum_squares(BATCH_SIZE - 1, z_b, N - 1, N - 1, zz) / cast(p_float32, BATCH_SIZE*N*N) - input_mean(z_b, zz) * input_mean(z_b, zz) + cast(p_float32, EPSILON)
        )
    );
    
    // Compute BN followed by ReLU
    std::vector<var> bn_relu_iter_vars = {n, z_b, y, x, zz};
    if (SCHEDULE_FUSION)
        bn_relu_iter_vars = {n, z_b, _y, x, zz};
        
    computation init_workspace("init_workspace", {n, z_b, y_pad, x_pad, zz}, cast(p_float32, 0));
    computation bn("bn", bn_relu_iter_vars, expr(p_float32));
    computation relu("relu", bn_relu_iter_vars, expr(p_float32));
    
    // These two computations are not scheduled when fusion is disabled
    computation bn_prelude("bn_prelude", {n, z_b, x, zz}, expr(p_float32), SCHEDULE_FUSION);
    computation relu_prelude("relu_prelude", {n, z_b, x, zz}, expr(p_float32), SCHEDULE_FUSION);

    if (!SCHEDULE_FUSION) {
        bn.set_expression(bn_scale(z_b, zz) * ((c_input(n, z_b, y, x, zz) - input_mean(z_b, zz)) / input_sd(z_b, zz)) + bn_shift(z_b, zz));
        relu.set_expression(
            expr(
                o_max, 
                cast(p_float32, 0), bn(n, z_b, y, x, zz)
            )
        );
    }

    else {
        bn_prelude.set_expression(bn_scale(z_b, zz) * ((c_input(n, z_b, 0, x, zz) - input_mean(z_b, zz)) / input_sd(z_b, zz)) + bn_shift(z_b, zz));
        relu_prelude.set_expression(
            expr(
                o_max, 
                cast(p_float32, 0), bn_prelude(n, z_b, x, zz)
            )
        );


        bn.set_expression(bn_scale(z_b, zz) * ((c_input(n, z_b, _y + 1, x, zz) - input_mean(z_b, zz)) / input_sd(z_b, zz)) + bn_shift(z_b, zz));
        relu.set_expression(
            expr(
                o_max, 
                cast(p_float32, 0), bn(n, z_b, _y, x, zz)
            )
        );
    }

    view relu_padded("relu_padded", {n, z_b, y_pad, x_pad, zz}, p_float32);

    // Convolution computation
    std::vector<var> conv_iter_vars = {n, z_b, fout_b, y, x, k_y, k_x, ffout, zz};
    if (SCHEDULE_FUSION)
        conv_iter_vars = {n, z_b, fout_b, _y, x, k_y, k_x, ffout, zz};

    computation init_output("init_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv("conv", conv_iter_vars, expr(p_float32));

    // This operation is not scheduled when fusion is disabled
    computation conv_conclude(
        "conv_conclude", 
        {n, z_b, fout_b, x, k_y, k_x, ffout, zz}, 
        expr(p_float32), 
        SCHEDULE_FUSION
    );

    if (!SCHEDULE_FUSION)
        conv.set_expression(init_output(n, fout_b, y, x, ffout) + relu_padded(n, z_b, y + k_y, x + k_x, zz)*conv_filter(z_b, fout_b, k_y, k_x, ffout, zz));

    else {
        conv.set_expression(init_output(n, fout_b, _y, x, ffout) + relu_padded(n, z_b, _y + k_y, x + k_x, zz)*conv_filter(z_b, fout_b, k_y, k_x, ffout, zz));
        conv_conclude.set_expression(init_output(n, fout_b, N - 1, x, ffout) + relu_padded(n, z_b, N + k_y - 1, x + k_x, zz)*conv_filter(z_b, fout_b, k_y, k_x, ffout, zz));
    }
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    if (!SCHEDULE_FUSION) {
        init_output.then(input_sum_init, computation::root)
                   .then(input_sum_squares_init, zz)
                   .then(input_sum, computation::root)
                   .then(input_sum_squares, zz)
                   .then(input_mean, computation::root)
                   .then(input_sd, zz)
                   .then(init_workspace, computation::root)
                   .then(bn, z_b)
                   .then(relu, zz)
                   .then(conv, computation::root);

        input_sum.vectorize(zz, Z_BLOCKING);
        
        bn.tag_parallel_level(n);
        bn.vectorize(zz, Z_BLOCKING);

        //n, z_b, fout_b, y, x, k_y, k_x, ffout, z_b
        conv.interchange(x, k_y);
        conv.interchange(x, k_x);
        //n, z_b, fout_b, y, k_y, k_x, x, ffout, z_b

        if (BLOCK_NUMBER >= 1) {
            //n, z_b, fout_b, y, k_y, k_x, x, ffout, z_b
            conv.interchange(y, k_y);
            conv.interchange(y, k_x);
            //n, z_b, fout_b, k_y, k_x, y, x, ffout, z_b
        }
        
        conv.tag_parallel_level(n);
        conv.vectorize(ffout, FOUT_BLOCKING);
    }

    else {
        conv.interchange(fout_b, _y);
        conv_conclude.interchange(fout_b, x);

        init_output.then(input_sum_init, computation::root)
                   .then(input_sum_squares_init, zz)
                   .then(input_sum, computation::root)
                   .then(input_sum_squares, zz)
                   .then(input_mean, computation::root)
                   .then(input_sd, zz)
                   .then(init_workspace, computation::root)
                   .then(bn_prelude, z_b)
                   .then(relu_prelude, z_b)
                   .then(bn, z_b)
                   .then(relu, zz)
                   .then(conv, _y)
                   .then(conv_conclude, z_b);

        input_sum.vectorize(zz, Z_BLOCKING);
        bn_prelude.vectorize(zz, Z_BLOCKING);
        bn.vectorize(zz, Z_BLOCKING);

        //n, z_b, y, fout_b, x, k_y, k_x, ffout, zz
        conv.interchange(x, k_y);
        conv.interchange(x, k_x);

        conv_conclude.interchange(x, k_y);
        conv_conclude.interchange(x, k_x);
        //n, z_b, y, fout_b, k_y, k_x, x, ffout, zz
        
        conv.tag_parallel_level(n);
        conv.vectorize(ffout, FOUT_BLOCKING);
        conv_conclude.vectorize(ffout, FOUT_BLOCKING);
    }

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    buffer input_mean_buf("input_mean_buf", {Z_NB_BLOCKS, Z_BLOCKING}, p_float32, a_temporary);
    buffer input_sd_buf("input_sd_buf", {Z_NB_BLOCKS, Z_BLOCKING}, p_float32, a_temporary);

    // We use this buffer as a temporary buffer to store BN and ReLU results
    // This buffer is padded (its width and height are C_N + 2) so as to prepare it for convolution
    std::vector<expr> workspace_dim_sizes = {BATCH_SIZE, Z_NB_BLOCKS, N + 2, N + 2, Z_BLOCKING};

    // When fusion is enabled, we don't need the second dimension of workspace_buf.
    // This is because we consume all computed values of this buffer before moving
    // on to the next z_block.
    if (SCHEDULE_FUSION)
        workspace_dim_sizes = {BATCH_SIZE, N + 2, N + 2, Z_BLOCKING};

    buffer workspace_buf(
        "workspace_buf", 
        workspace_dim_sizes, 
        p_float32, 
        a_temporary
    );

    if (!SCHEDULE_FUSION)
        init_workspace.store_in(&workspace_buf);
    else
        init_workspace.store_in(&workspace_buf, {n, y_pad, x_pad, zz});

    input_sum_init.store_in(&input_mean_buf);
    input_sum.store_in(&input_mean_buf, {z_b, zz});
    input_mean.store_in(&input_mean_buf);
    
    input_sum_squares_init.store_in(&input_sd_buf);
    input_sum_squares.store_in(&input_sd_buf, {z_b, zz});
    input_sd.store_in(&input_sd_buf);

    if (!SCHEDULE_FUSION) {
        // We shift the BN and ReLU computations, so as to avoid to
        // compute on the padding region of workspace_buf.
        bn.store_in(&workspace_buf, {n, z_b, y + 1, x + 1, zz});
        relu.store_in(&workspace_buf, {n, z_b, y + 1, x + 1, zz});
        relu_padded.store_in(&workspace_buf);

        init_output.store_in(&output_buf);
        conv.store_in(&output_buf, {n, fout_b, y, x, ffout});
    }

    else {
        // We shift the BN and ReLU computations, so as to avoid to
        // compute on the padding region of workspace_buf.
        bn_prelude.store_in(&workspace_buf, {n, 1, x + 1, zz});
        relu_prelude.store_in(&workspace_buf, {n, 1, x + 1, zz});

        bn.store_in(&workspace_buf, {n, _y + 2, x + 1, zz});
        relu.store_in(&workspace_buf, {n, _y + 2, x + 1, zz});

        relu_padded.store_in(&workspace_buf, {n, y_pad, x_pad, zz});

        init_output.store_in(&output_buf);
        conv.store_in(&output_buf, {n, fout_b, _y, x, ffout});
        conv_conclude.store_in(&output_buf, {n, fout_b, N - 1, x, ffout});
    }

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        bn_scale.get_buffer(), 
        bn_shift.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(), 
        &output_buf
    }, "densenet_block_tiramisu.o");

    return 0;
}
