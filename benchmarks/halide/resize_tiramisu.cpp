#include <tiramisu/tiramisu.h>
#include "halide_image_io.h"

using namespace tiramisu;

expr mixf(expr x, expr y, expr a)
{
    return cast(p_float32, x) * (1 - a) + cast(p_float32, y) * a;
}

int main(int argc, char **argv)
{
    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./utils/images/gray.png");
    int IMG_WIDTH = in_image.width();
    int IMG_HEIGHT = in_image.height();

    int OUT_WIDTH = in_image.width() / 1.5f;
    int OUT_HEIGHT = in_image.height() / 1.5f;

    init("resize_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var o_x("o_x", 0, IMG_WIDTH), o_y("o_y", 0, IMG_HEIGHT);
    var x("x", 0, OUT_WIDTH), y("y", 0, OUT_HEIGHT);
    input c_input("c_input", {o_y, o_x}, p_uint8);

    expr o_r((cast(p_float32, y) + 0.5f) * (cast(p_float32, IMG_HEIGHT) / cast(p_float32, OUT_HEIGHT)) - 0.5f);
    expr o_c((cast(p_float32, x) + 0.5f) * (cast(p_float32, IMG_WIDTH) / cast(p_float32, OUT_WIDTH)) - 0.5f);

    expr r_coeff(expr(o_r) - expr(o_floor, o_r));
    expr c_coeff(expr(o_c) - expr(o_floor, o_c));

    expr A00_r(cast(p_int32, expr(o_floor, o_r)));
    expr A00_c(cast(p_int32, expr(o_floor, o_c)));

    computation resize(
        "resize",
        {y, x},
        mixf(
            mixf(
                c_input(A00_r, A00_c), 
                c_input(A00_r + 1, A00_c), 
                r_coeff
            ),

            mixf(
                c_input(A00_r, A00_c + 1), 
                c_input(A00_r + 1, A00_c + 1), 
                r_coeff
            ),
    
            c_coeff
        )
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    resize.tag_parallel_level(y);
    resize.vectorize(x, 8);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {OUT_HEIGHT, OUT_WIDTH}, p_float32, a_output);
    resize.store_in(&output_buf);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        &output_buf
    }, "build/generated_fct_resize.o");

    return 0;
}
