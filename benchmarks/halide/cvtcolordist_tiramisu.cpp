#include <isl/set.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/utils.h>
#include "halide_image_io.h"
#include "wrapper_cvtcolordist.h"

using namespace tiramisu;
int main() {
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    function cvtcolor_dist("cvtcolordist_tiramisu");
    std::string img = "/utils/images/rgb.png";
    Halide::Buffer<uint8_t> image = Halide::Tools::load_image(std::getenv("TIRAMISU") + img);

    int _rows = image.extent(1);
    int _cols = image.extent(0);

    constant channels("channels", expr(3), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant rows("rows", expr(_rows), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant cols("cols", expr(_cols), p_int32, true, NULL, 0, &cvtcolor_dist);
    var i("i"), j("j"), c("c");
    computation input("[channels, rows, cols]->{input[c, i, j]: 0<=c<channels and 0<=i<rows and 0<=j<cols}",
                      expr(), false, p_uint8, &cvtcolor_dist);
    expr rgb_expr(tiramisu::o_cast, tiramisu::p_uint8, (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                                                                            input(tiramisu::expr((int32_t)2), i, j)) *
                                                             tiramisu::expr((uint32_t)1868)) +
                                                            (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                                                                            input(tiramisu::expr((int32_t)1), i, j)) *
                                                             tiramisu::expr((uint32_t)9617))) +
                                                           (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32,
                                                                           input(tiramisu::expr((int32_t)0), i, j)) *
                                                            tiramisu::expr((uint32_t)4899))) +
                                                          tiramisu::expr((uint32_t)8192)) /
                                                         tiramisu::expr((uint32_t)16384)));


    computation RGB2Gray_s0("[rows, cols]->{RGB2Gray_s0[i, j]: 0<=i<rows and 0<=j<cols}",
                            expr(o_cast, p_uint8, rgb_expr), true, p_uint8,
                            &cvtcolor_dist);

    // distribute stuff
    var i1("i1"), i2("i2");
    RGB2Gray_s0.split(i, _rows / NODES, i1, i2);
    RGB2Gray_s0.tag_distribute_level(i1);
    RGB2Gray_s0.drop_rank_iter(i1);
    RGB2Gray_s0.tag_parallel_level(i2);

    buffer buff_input("buff_input", {3, rows / NODES, cols}, p_uint8, a_input,
                      &cvtcolor_dist);
    buffer buff_RGB2Gray("buff_RGB2Gray", {rows / NODES, cols}, p_uint8, a_output,
                         &cvtcolor_dist);
    input.set_access("{input[c, i, j]->buff_input[c, i, j]}");
    RGB2Gray_s0.set_access("{RGB2Gray_s0[i,j]->buff_RGB2Gray[i,j]}");

    std::string bdir = "/build/generated_fct_cvtcolordist.o";
    cvtcolor_dist.codegen({&buff_input, &buff_RGB2Gray}, std::getenv("TIRAMISU") + bdir);

    return 0;
}

