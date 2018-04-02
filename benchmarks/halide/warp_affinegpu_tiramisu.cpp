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

expr clamp(expr n, expr min, expr max) {
    return expr{o_select, n < min, min, expr{o_select, n > max, max, n}};
}

expr mixf(expr x, expr y, expr a) {
    return x * (expr{1.0f} -a) + y * a;
}

constexpr auto work_type = p_float32;
constexpr auto it_type = p_int32;
constexpr auto data_type = p_uint8;
constexpr auto block_size = 16;
#define BS "16"

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();


    tiramisu::function affine_tiramisu("warp_affinegpu_tiramisu");

    var i0("i0"), i1("i1"), j0("j0"), j1("j1");

    // Layer I

    // Input params.
    float a00 = 0.1;
    float a01 = 0.1;
    float a10 = 0.1;
    float a11 = 0.1;
    float b00 = 0.1;
    float b10 = 0.1;

    computation sizes{"[N, M] -> {sizes[i]}", expr{}, false, data_type, &affine_tiramisu};
    constant N{"N", sizes(0), it_type, true, nullptr, 0, &affine_tiramisu};
    constant M{"M", sizes(1), it_type, true, nullptr, 0, &affine_tiramisu};

    computation in{"[N, M] -> {in[i, j]}", expr{}, false, data_type, &affine_tiramisu};

    computation out{"[N, M] -> {out[i0, j0, i1, j1] : 0 <= i0 * " BS " + i1 < N and 0 <= i0 and 0 <= i1 < " BS " and 0 <= j0 * " BS " + j1 < M and 0 <= j0 and 0 <= j1 < " BS "}", expr(), true, data_type, &affine_tiramisu};

    auto i = i0 * expr{block_size} + i1, j = j0 * expr{block_size} + j1;

    int level = 3;
    constant o_r{"o_r", expr{a11} * cast(work_type, i) + expr{a10} * cast(work_type, j) + expr{b00}, work_type, false, &out, level, &affine_tiramisu};
    constant o_c{"o_c", expr{a01} * cast(work_type, i) + expr{a00} * cast(work_type, j) + expr{b10}, work_type, false, &out, level, &affine_tiramisu};
    constant r{"r", static_cast<expr>(o_r) - expr{o_floor, o_r}, work_type, false, &out, level, &affine_tiramisu};
    constant c{"c", static_cast<expr>(o_c) - expr{o_floor, o_c}, work_type, false, &out, level, &affine_tiramisu};
    constant coord_r{"coord_r", cast(it_type, expr{o_floor, o_r}), it_type, false, &out, level, &affine_tiramisu};
    constant coord_c{"coord_c", cast(it_type, expr{o_floor, o_r}), it_type, false, &out, level, &affine_tiramisu};
    constant coord_00r{"coord_00r", clamp(coord_r    , 0, N), it_type, false, &out, level, &affine_tiramisu};
    constant coord_00c{"coord_00c", clamp(coord_c    , 0, M), it_type, false, &out, level, &affine_tiramisu};
    constant coord_01r{"coord_01r", clamp(coord_r    , 0, N), it_type, false, &out, level, &affine_tiramisu};
    constant coord_01c{"coord_01c", clamp(coord_c + 1, 0, M), it_type, false, &out, level, &affine_tiramisu};
    constant coord_10r{"coord_10r", clamp(coord_r + 1, 0, N), it_type, false, &out, level, &affine_tiramisu};
    constant coord_10c{"coord_10c", clamp(coord_c    , 0, M), it_type, false, &out, level, &affine_tiramisu};
    constant coord_11r{"coord_11r", clamp(coord_r + 1, 0, N), it_type, false, &out, level, &affine_tiramisu};
    constant coord_11c{"coord_11c", clamp(coord_c + 1, 0, M), it_type, false, &out, level, &affine_tiramisu};

    expr A00 = cast(p_float32, in(coord_00r, coord_00c));
    expr A01 = cast(p_float32, in(coord_01r, coord_01c));
    expr A10 = cast(p_float32, in(coord_10r, coord_10c));
    expr A11 = cast(p_float32, in(coord_11r, coord_11c));

    out.set_expression(cast(data_type, mixf(mixf(A00, A01, r), mixf(A10, A11, r), c)));

    // Layer II & III

//    out.tile(i, j, block_size, block_size, i0, j0, i1, j1);
    out.tag_gpu_level(i0, j0, i1, j1);
    o_r.tag_gpu_level(i0, j0, i1, j1);



    buffer in_gpu{"in_gpu", {M, N}, data_type, a_temporary, &affine_tiramisu};
    in.set_access("{in[i, j] -> in_gpu[j, i]}");
    in_gpu.tag_gpu_global();
    buffer out_gpu{"out_gpu", {M, N}, data_type, a_temporary, &affine_tiramisu};
    out.set_access("{out[i0, j0, i1, j1] -> out_gpu[j0 * " BS " + j1, i0 * " BS " + i1]}");
    out_gpu.tag_gpu_global();

    buffer in_host{"in_host", {M, N}, data_type, a_input, &affine_tiramisu};
    buffer out_host{"out_host", {M, N}, data_type, a_output, &affine_tiramisu};
    buffer sizes_host{"sizes_host", {2}, it_type, a_input, &affine_tiramisu};
    sizes.set_access("{sizes[i] -> sizes_host[i]}");

    affine_tiramisu.dump_sched_graph();

    computation copy_to_device{"{copy_to_device[0]}", memcpy(in_host, in_gpu), true, p_none, &affine_tiramisu};
    // copy_to_device.before(*buffer_declares[0], computation::root);
    copy_to_device.before(o_r, computation::root);
    computation copy_to_host{"{copy_to_host[0]}", memcpy(out_gpu, out_host), true, p_none, &affine_tiramisu};
    copy_to_host.after(out, computation::root);



    affine_tiramisu.set_arguments({&sizes_host, &in_host, &out_host});
    affine_tiramisu.gen_time_space_domain();
    affine_tiramisu.gen_isl_ast();
    affine_tiramisu.gen_cuda_stmt();
    affine_tiramisu.gen_halide_stmt();
    affine_tiramisu.dump_halide_stmt();
    affine_tiramisu.gen_halide_obj("build/generated_fct_warp_affinegpu.o");

    return 0;
}
