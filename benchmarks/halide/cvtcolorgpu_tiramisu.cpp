//
// Created by malek on 3/6/18.
//

#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace tiramisu;

constexpr auto it_type = p_int32;
constexpr auto comp_type = p_uint32;
constexpr auto d_type = p_uint8;

#define one cast(comp_type, 1)

#define cv_descale(x, n) ((x) + (one << ((n) - one))) >> (n)

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(it_type);

    tiramisu::function fct("cvtcolorgpu_tiramisu");

    // Contains the sizes of the buffers
    computation sizes{"{sizes[i]: 0 <= i < 2}", expr(), false, it_type, &fct};

    const expr yuv_shift = value_cast(comp_type, 14);
    const expr R2Y = value_cast(comp_type, 4899);
    const expr G2Y = value_cast(comp_type, 9617);
    const expr B2Y = value_cast(comp_type, 1868);

    // Image is NxM
    constant N{"N", sizes(0), it_type, true, nullptr, computation::root_dimension, &fct};
    constant M{"M", sizes(1), it_type, true, nullptr, computation::root_dimension, &fct};

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");

    computation input{"[N, M] -> {input[i, j, c] : 0 <= i < N  and 0 <= j < M and 0 <= c < 3}",
                      expr(), false, d_type, &fct};
    computation output{"[N, M] -> {output[i, j, c] : 0 <= i < N  and 0 <= j < M and 0 <= c < 3}",
                       cast(d_type, cv_descale(
                               cast(comp_type, input(i, j, 0)) * R2Y +
                               cast(comp_type, input(i, j, 1)) * G2Y +
                               cast(comp_type, input(i, j, 2)) * B2Y,
                               yuv_shift)),
                       true, d_type, &fct};

    buffer in_buffer{"in_buffer", {3, M, N}, d_type, a_input, &fct};
    buffer sizes_buffer{"sizes_buffer", {2}, it_type, a_input, &fct};
    buffer out_buffer{"out_buffer", {3, M, N}, d_type, a_output, &fct};

    input.set_access("{input[i, j, c] -> in_buffer[c, j, i]}");
    output.set_access("{output[i, j, c] -> out_buffer[c, j, i]}");
    sizes.set_access("{sizes[i] -> sizes_buffer[i]}");

    buffer in_gpu{"in_gpu", {3, M, N}, d_type, a_temporary, &fct};
    buffer out_gpu{"out_gpu", {3, M, N}, d_type, a_temporary, &fct};

    in_gpu.tag_gpu_global();
    out_gpu.tag_gpu_global();

    computation copy_input("{copy_input[0]}", expr(o_memcpy, var(in_buffer.get_name()), var(in_gpu.get_name())), true, p_none, &fct);
    computation copy_output("{copy_output[0]}", expr(o_memcpy, var(out_gpu.get_name()), var(out_buffer.get_name())), true, p_none, &fct);


    input.set_access("{input[i, j, c] -> in_gpu[c, j, i]}");
    output.set_access("{output[i, j, c] -> out_gpu[c, j, i]}");
    output.gpu_tile(i, j, 16, 16);

    output.between(copy_input, computation::root, copy_output, computation::root);


    fct.set_arguments({&sizes_buffer, &in_buffer, &out_buffer});
    fct.gen_time_space_domain();
    fct.gen_isl_ast();
    fct.gen_cuda_stmt();
    fct.gen_halide_stmt();
    fct.dump_halide_stmt();
    fct.gen_halide_obj("build/generated_fct_cvtcolorgpu.o");

    return 0;
}
