#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

expr mixf(expr x, expr y, expr a)
{
    return x + (y - x) * a;
}

int main()
{
    global::set_default_tiramisu_options();

    function resize_spconv_relu_maxpool_block("resize_spconv_relu_maxpool_block");
    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    //Resize inputs
    computation c_input("[IMG_HEIGHT, IMG_WIDTH, F_In, BATCHSIZE]->{c_input[b, fin, o_y, o_x]: 0<=b<BATCHSIZE and 0<=o_y<IMG_HEIGHT and 0<=o_x<IMG_WIDTH and 0<=fin<F_In}", expr(), false, p_float32, &resize_spconv_relu_maxpool_block);

    // Convolution inputs
    computation SIZES("{SIZES[e]: 0<=e<1}", expr(), false, p_int32, &resize_spconv_relu_maxpool_block);
    computation c_filter_values("[FNNZ]->{c_filter_values[k]: 0<=k<FNNZ}", expr(), false, p_float32, &resize_spconv_relu_maxpool_block);
    computation c_filter_idx("[FNNZ]->{c_filter_idx[k]: 0<=k<FNNZ}", expr(), false, p_int32, &resize_spconv_relu_maxpool_block);
    computation c_filter_finptr("[filter_finptr_size]->{c_filter_finptr[fin]: 0<=fin<filter_finptr_size}", expr(), false, p_int32, &resize_spconv_relu_maxpool_block);
    computation c_bias("[F_Out]->{c_bias[fout]: 0<=fout<F_Out}", expr(), false, p_float32, &resize_spconv_relu_maxpool_block);

    // Resize constants

    constant BATCHSIZE("BATCHSIZE", BATCH_SIZE, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant F_In("F_In", FIn, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant IMG_HEIGHT("IMG_HEIGHT", IMG_H, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant IMG_WIDTH("IMG_WIDTH", IMG_W, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant NEW_HEIGHT("NEW_HEIGHT", N + 2, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant NEW_WIDTH("NEW_WIDTH", N + 2, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant UNPADDED_NEW_HEIGHT("UNPADDED_NEW_HEIGHT", N, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant UNPADDED_NEW_WIDTH("UNPADDED_NEW_WIDTH", N, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    // Convolution constants

    constant IND_RANGE("IND_RANGE",(N + 2) * (N + 2) * FIn, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant FNNZ("FNNZ", SIZES(0), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant F_Out("F_Out", FOut, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant KK("KK", K, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant filter_finptr_size("filter_finptr_size", expr(FOut + 1), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant X_BLOCKING("X_BLOCKING", expr(X_BL), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant X_NB_BLOCKS("X_NB_BLOCKS", expr(X_NB_BL), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    constant Y_BLOCKING("Y_BLOCKING", expr(Y_BL), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant Y_NB_BLOCKS("Y_NB_BLOCKS", expr(Y_NB_BL), p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    // Maxpool constants
    constant maxpool_height("maxpool_height", N / 2, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);
    constant maxpool_width("maxpool_width", N / 2, p_int32, true, NULL, 0, &resize_spconv_relu_maxpool_block);

    // Variables
    // Resize
    var o_x("o_x"), o_y("o_y"), b("b"); // o_x, o_y go till IMG_WIDTH/IMG_HEIGHT
    var x("x"), y("y"); // Go till N
    var x_pad("x_pad"), y_pad("y_pad"); // Go till N + 2
    var fin("fin"); // Go till FIn

    // Convolution
    var k("k"), l("l"), fout("fout");
    var x_b("x_b"), y_b("y_b"), yy("yy"), xx("xx");

    // COMPUTATIONS
    // Resize computation :
    expr o_r((cast(p_float32, y + 2) + 0.5f) * (cast(p_float32, IMG_H) / cast(p_float32, N + 2)) - 0.5f);
    expr o_c((cast(p_float32, x_pad) + 0.5f) * (cast(p_float32, IMG_W) / cast(p_float32, N + 2)) - 0.5f);

    expr r_coeff(expr(o_r) - expr(o_floor, o_r));
    expr c_coeff(expr(o_c) - expr(o_floor, o_c));

    expr A00_r(cast(p_int32, expr(o_floor, o_r)));
    expr A00_c(cast(p_int32, expr(o_floor, o_c)));
    computation resize("[BATCHSIZE, F_In, NEW_WIDTH, UNPADDED_NEW_HEIGHT]->{resize[b, fin, y, x_pad]: 0<=b<BATCHSIZE and 0<=y<UNPADDED_NEW_HEIGHT and 0<=x_pad<NEW_WIDTH and 0<=fin<F_In}",
      mixf(
        mixf(
            c_input(b, fin, A00_r, A00_c),
            c_input(b, fin, A00_r + 1, A00_c),
            r_coeff
        ),

        mixf(
            c_input(b, fin, A00_r, A00_c + 1),
            c_input(b, fin, A00_r + 1, A00_c + 1),
            r_coeff
        ),

        c_coeff
      ),
      true,
      p_float32,
      &resize_spconv_relu_maxpool_block
    );

    // For fusion
    var y_prelude("y_prelude"); // 0 to 2

    expr o_r_prelude((cast(p_float32, y_prelude) + 0.5f) * (cast(p_float32, IMG_H) / cast(p_float32, N + 2)) - 0.5f);
    expr r_coeff_prelude(expr(o_r_prelude) - expr(o_floor, o_r_prelude));
    expr A00_r_prelude(cast(p_int32, expr(o_floor, o_r_prelude)));

    computation resize_prelude("[BATCHSIZE, F_In, NEW_WIDTH]->{resize_prelude[b, fin, y_prelude, x_pad]: 0<=b<BATCHSIZE and 0<=y_prelude<2 and 0<=x_pad<NEW_WIDTH and 0<=fin<F_In}",
      mixf(
        mixf(
            c_input(b, fin, A00_r_prelude, A00_c),
            c_input(b, fin, A00_r_prelude + 1, A00_c),
            r_coeff_prelude
        ),

        mixf(
            c_input(b, fin, A00_r_prelude, A00_c + 1),
            c_input(b, fin, A00_r_prelude + 1, A00_c + 1),
            r_coeff_prelude
        ),

        c_coeff
      ),
      true,
      p_float32,
      &resize_spconv_relu_maxpool_block
    );

    // Convolution part
    computation resized_view("[BATCHSIZE,IND_RANGE]->{resized_view[b,ind]: 0<=b<BATCHSIZE and 0<=ind<IND_RANGE}", expr(), false, p_float32, &resize_spconv_relu_maxpool_block);
    computation init_conv("[BATCHSIZE, F_Out, Y_BLOCKING, Y_NB_BLOCKS, X_BLOCKING, X_NB_BLOCKS]->{init_conv[b,y_b,fout,x_b,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING}", c_bias(fout), true, p_float32, &resize_spconv_relu_maxpool_block);
    computation maxpool("[BATCHSIZE, F_Out, Y_BLOCKING, Y_NB_BLOCKS, X_BLOCKING, X_NB_BLOCKS]->{maxpool[b,y_b,fout,x_b, yy, xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING}", expr(), true, p_float32, &resize_spconv_relu_maxpool_block);

    computation convolve("[BATCHSIZE, F_Out, Y_BLOCKING, Y_NB_BLOCKS, X_BLOCKING, X_NB_BLOCKS, k_range0, k_range1]->{convolve[b,y_b,fout,x_b,k,yy,xx]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=y_b<Y_NB_BLOCKS and 0<=yy<Y_BLOCKING and 0<=x_b<X_NB_BLOCKS and 0<=xx<X_BLOCKING and k_range0<=k<k_range1}", expr(), true, p_float32, &resize_spconv_relu_maxpool_block);

    constant filter_k("filter_k", c_filter_idx(k), p_int32, false, &convolve, 4, &resize_spconv_relu_maxpool_block);

    constant k_range0("k_range0", c_filter_finptr(fout), p_int32, false, &convolve, 2, &resize_spconv_relu_maxpool_block);
    constant k_range1("k_range1", c_filter_finptr(fout + 1), p_int32, false, &convolve, 2, &resize_spconv_relu_maxpool_block);

    convolve.set_expression(convolve(b, y_b, fout, x_b, k, yy, xx) + resized_view(b, (y_b * Y_BL + yy) * (N + 2) + x_b * X_BL + xx) * c_filter_values(k));

    maxpool.set_expression(expr(o_max, maxpool(b, y_b, fout, x_b, yy, xx), convolve(b, y_b, fout, x_b, 0, yy, xx)));

    // Maxpool init
    computation init_output("[BATCHSIZE, F_Out, maxpool_height, maxpool_width]->{init_output[b,fout,output_y,output_x]: 0<=b<BATCHSIZE and 0<=fout<F_Out and 0<=output_y<maxpool_height and 0<=output_x<maxpool_width}",  cast(p_float32, 0), true, p_float32, &resize_spconv_relu_maxpool_block);

    resize_spconv_relu_maxpool_block.set_context_set("[k_range0,k_range1]->{: k_range0>0 and k_range1>0 and k_range1>k_range0}");

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    resize.split(y, Y_BL, y_b, yy);
    resize.interchange(y_b, fin);

    buffer b_workspace("b_workspace", {Y_BL, X_BL}, p_float32, a_temporary, &resize_spconv_relu_maxpool_block);
    computation *alloc = b_workspace.allocate_at(init_conv, x_b);

    resize_prelude.after_low_level(init_output, 0);
    resize.after_low_level(resize_prelude, 0);
    k_range0.after_low_level(resize, 1);
    k_range1.after_low_level(k_range0, 2);
    alloc->after_low_level(k_range1, 3);
    init_conv.after_low_level(*alloc, 3);
    filter_k.after_low_level(init_conv, 3);
    convolve.after_low_level(filter_k, 4);
    maxpool.after_low_level(convolve, 3);

    convolve.parallelize(b);

    convolve.tag_vector_level(xx, X_BL);
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer b_input("b_input", {BATCH_SIZE, FIn, IMG_H, IMG_W}, p_float32, a_input, &resize_spconv_relu_maxpool_block);
    buffer b_resized("b_resized", {BATCH_SIZE, FIn, N + 2, N + 2}, p_float32, a_input, &resize_spconv_relu_maxpool_block);
    //
    //convolution
    buffer b_SIZES("b_SIZES", {expr(1)}, p_int32, a_input, &resize_spconv_relu_maxpool_block);
    buffer b_filter_values("b_filter_values", {expr(FNNZ)}, p_float32, a_input, &resize_spconv_relu_maxpool_block);
    buffer b_filter_idx("b_filter_idx", {FNNZ}, p_int32, a_input, &resize_spconv_relu_maxpool_block);
    buffer b_filter_finptr("b_filter_finptr", {expr(filter_finptr_size)}, p_int32, a_input, &resize_spconv_relu_maxpool_block);
    buffer b_bias("b_bias", {F_Out}, p_float32, a_input, &resize_spconv_relu_maxpool_block);

    buffer output_buf("output_buf", {BATCH_SIZE, FOut, N/2, N/2}, p_float32, a_output, &resize_spconv_relu_maxpool_block);

    //Inputs
    c_input.set_access("{c_input[b,fin,o_y,o_x]->b_input[b, fin, o_y, o_x]}");
    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    resized_view.set_access("[filter_k]->{resized_view[b,ind]->b_resized[b,0,0,ind + filter_k]}");

    c_filter_values.set_access("{c_filter_values[k]->b_filter_values[k]}");
    c_filter_idx.set_access("{c_filter_idx[k]->b_filter_idx[k]}");
    c_filter_finptr.set_access("{c_filter_finptr[fin]->b_filter_finptr[fin]}");

    c_bias.set_access("{c_bias[fout]->b_bias[fout]}");

    resize.set_access("{resize[b,fin,y,x]->b_resized[b, fin, y + 2, x]}");
    resize_prelude.set_access("{resize_prelude[b,fin,y,x]->b_resized[b, fin, y, x]}");

    std::stringstream maxpool_access;

    init_conv.set_access("{init_conv[b, y_b, fout, x_b, yy, xx]->b_workspace[yy,xx]}");

    init_output.set_access("{init_output[b, fout, output_y, output_x]->output_buf[b,fout,output_y,output_x]}");

    maxpool_access << "{maxpool[b, y_b, fout, x_b, yy, xx]->output_buf[b, fout,(y_b * " << Y_BL <<" + yy)/2,(x_b * "<< X_BL <<" + xx)/2]}";
    maxpool.set_access(maxpool_access.str());

    convolve.set_access("{convolve[b, y_b, fout, x_b, k, yy, xx]->b_workspace[yy,xx]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    resize_spconv_relu_maxpool_block.set_arguments({&b_input, &b_SIZES, &b_resized, &b_filter_values, &b_filter_idx, &b_filter_finptr, &b_bias, &output_buf});
    resize_spconv_relu_maxpool_block.gen_time_space_domain();
    resize_spconv_relu_maxpool_block.gen_isl_ast();
    resize_spconv_relu_maxpool_block.gen_halide_stmt();
    resize_spconv_relu_maxpool_block.gen_halide_obj("resize_spconv_relu_maxpool_tiramisu.o");
    return 0;
}
